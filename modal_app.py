"""
Modal Application — Serverless Orchestrator for ClippedAI.

Defines micro-containers with exact hardware + images.
Main orchestrator: process_video(url, settings)
Parallel map/spawn for chunk analysis, reduce for global merge.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import modal

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Modal App & Volume
# ─────────────────────────────────────────────

app = modal.App(config.MODAL_APP_NAME)
volume = modal.Volume.from_name(config.MODAL_VOLUME_NAME, create_if_missing=True)

# ─────────────────────────────────────────────
# Modal Image Definitions
# ─────────────────────────────────────────────

image_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "numpy", "scipy", "tqdm", "tenacity",
    )
    .add_local_python_source("config", copy=True)
    .add_local_python_source("pipeline", copy=True)
)

image_ingest = (
    image_base
    .apt_install("curl", "unzip")
    .pip_install("yt-dlp", "playwright")
    .run_commands("playwright install chromium --with-deps")
    .run_commands(
        "curl -fsSL https://deno.land/install.sh | sh"
    )
    .run_commands(
        "curl -L https://github.com/Brainicism/bgutil-ytdlp-pot-provider/releases/"
        "latest/download/bgutil-pot-server -o /usr/local/bin/bgutil-pot-server "
        "&& chmod +x /usr/local/bin/bgutil-pot-server"
    )
)

image_asr = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg")
    .run_commands(
        "pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118"
    )
    .pip_install("whisperx")
    .add_local_python_source("config", copy=True)
    .add_local_python_source("pipeline", copy=True)
)

image_vision = (
    image_base
    .pip_install(
        "ultralytics", "insightface", "onnxruntime-gpu",
        "filterpy", "mediapipe", "opencv-python-headless",
        "scikit-learn", "setuptools", "loguru",
        "lapx",  # drop-in replacement for lap (fixes Python 3.11 build)
        "huggingface_hub",
    )
    .run_commands(
        "git clone https://github.com/SJTUwxz/LoCoNet_ASD.git /opt/loconet || true"
    )
    .run_commands(
        "git clone https://github.com/bellhyeon/BoT-FaceSORT.git /opt/bot-facesort "
        "&& cd /opt/bot-facesort && pip install -r requirements.txt || true"
    )
    .run_commands(
        # Pre-download face-specific YOLOv8 model from HuggingFace
        'python -c "from huggingface_hub import hf_hub_download; '
        'hf_hub_download(repo_id=\"arnabdhar/YOLOv8-Face-Detection\", '
        'filename=\"model.pt\", local_dir=\"/opt\"); '
        'import shutil; shutil.move(\"/opt/model.pt\", \"/opt/yolov8n-face.pt\")" || true'
    )
)

image_scene = (
    image_base
    .pip_install("torch", "torchvision", "scenedetect[opencv]")
    .run_commands(
        "git clone https://github.com/wentaozhu/AutoShot.git /opt/autoshot || true"
    )
)

image_audio = (
    image_base
    .apt_install("wget")
    .pip_install(
        "panns-inference", "librosa",
        "torch", "torchaudio",
    )
    .run_commands(
        # Pre-download PANNs model + AudioSet class labels at build time
        "python -c \"from panns_inference import AudioTagging; AudioTagging(checkpoint_path=None, device='cpu')\" || true"
    )
)

image_render = (
    image_base
    .apt_install("libass-dev")
    .pip_install("opencv-python-headless")
)

image_llm = (
    image_base
    .pip_install(
        "groq", "cerebras-cloud-sdk",
        "opencv-python-headless",  # needed by pipeline.face_track, pipeline.render
        "scikit-learn",  # needed by pipeline.clip_selector
    )
    .apt_install("libass-dev")  # needed by pipeline.render (ffmpeg ass filter)
)

# ─────────────────────────────────────────────
# Secrets
# ─────────────────────────────────────────────

secrets = [
    modal.Secret.from_name("clippedai-secrets"),
    modal.Secret.from_name("cerebras-api-key"),
]


# ─────────────────────────────────────────────
# Container Classes
# ─────────────────────────────────────────────

@app.cls(
    image=image_ingest,
    volumes={config.MODAL_VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
class Ingester:
    """YouTube download + chunking container."""

    @modal.enter()
    def start_pot_server(self):
        """Start bgutil-pot-server for PO tokens."""
        try:
            self.pot_process = subprocess.Popen(
                ["/usr/local/bin/bgutil-pot-server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("PoT server started (PID: %d)", self.pot_process.pid)
        except Exception as e:
            self.pot_process = None
            logger.warning("bgutil-pot-server unavailable: %s", e)

    @modal.exit()
    def stop_pot_server(self):
        if hasattr(self, "pot_process") and self.pot_process:
            self.pot_process.terminate()

    @modal.method()
    def ingest(self, url: str) -> dict:
        from pipeline.ingest import ingest
        from pipeline.cookie_refresh import refresh_cookies, is_cookie_valid

        work_dir = Path(config.MODAL_VOLUME_MOUNT) / "_work"
        cookie_path = Path(config.MODAL_VOLUME_MOUNT) / "_cookies" / "youtube.txt"

        # Refresh cookies if needed
        if not is_cookie_valid(cookie_path):
            try:
                refresh_cookies(cookie_path)
            except Exception as e:
                logger.warning("Cookie refresh failed: %s", e)

        result = ingest(url, work_dir, cookie_path)
        volume.commit()
        return result


@app.cls(
    image=image_asr,
    gpu="A10G",
    volumes={config.MODAL_VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=300,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
class Transcriber:
    """WhisperX + Pyannote container (A10G GPU)."""

    @modal.method()
    def transcribe(self, chunk_meta: dict) -> dict:
        from pipeline.transcribe import transcribe_chunk
        return transcribe_chunk(
            audio_path=Path(chunk_meta["audio_path"]),
            chunk_start=chunk_meta["start"],
        )


@app.cls(
    image=image_vision,
    gpu="T4",
    volumes={config.MODAL_VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
class FaceTracker:
    """YOLO + BoT-FaceSORT + LoCoNet container (T4 GPU)."""

    @modal.method()
    def process_chunk(self, chunk_meta: dict) -> list[dict]:
        from pipeline.face_track import (
            detect_faces_in_chunk, track_faces, detect_active_speakers,
        )
        video_path = Path(chunk_meta["path"])
        audio_path = Path(chunk_meta["audio_path"])

        detections = detect_faces_in_chunk(video_path)
        detections = track_faces(detections, video_path)
        detections = detect_active_speakers(detections, video_path, audio_path)

        return detections


@app.cls(
    image=image_scene,
    gpu="T4",
    volumes={config.MODAL_VOLUME_MOUNT: volume},
    timeout=120,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
class SceneDetector:
    """AutoShot container (T4 GPU)."""

    @modal.method()
    def detect(self, chunk_meta: dict) -> list[dict]:
        from pipeline.scene_detect import detect_scenes
        return detect_scenes(
            Path(chunk_meta["path"]),
            chunk_start=chunk_meta["start"],
        )


@app.cls(
    image=image_audio,
    cpu=2,
    volumes={config.MODAL_VOLUME_MOUNT: volume},
    timeout=120,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
class AudioAnalyzer:
    """PANNs container (CPU)."""

    @modal.method()
    def analyze(self, chunk_meta: dict) -> list[dict]:
        from pipeline.audio_analysis import analyze_audio
        return analyze_audio(
            Path(chunk_meta["audio_path"]),
            chunk_start=chunk_meta["start"],
        )


# ─────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────

@app.function(
    image=image_llm,
    volumes={config.MODAL_VOLUME_MOUNT: volume},
    secrets=secrets,
    timeout=3600,
)
def process_video(
    url: str,
    settings: Optional[dict] = None,
    video_path: Optional[str] = None,
    video_hash: Optional[str] = None,
) -> list[str]:
    """
    Main pipeline orchestrator.

    1. Ingest (download + chunk) — or skip if video_path provided
    2. Parallel map analysis (ASR, face, scene, audio)
    3. Global reduce (re-ID, merge)
    4. Clip selection (LLM)
    5. Per-clip: reframe → captions → render
    6. Return clip paths
    """
    from pipeline.transcribe import merge_chunk_transcripts, save_transcript
    from pipeline.face_track import global_reid, save_face_tracks
    from pipeline.scene_detect import merge_chunk_scenes, save_scenes
    from pipeline.audio_analysis import merge_chunk_audio_events, save_audio_events
    from pipeline.clip_selector import generate_candidates, rank_with_llm, save_selected_clips
    from pipeline.reframe import compute_crop_plan, save_crop_plan
    from pipeline.captions import generate_captions
    from pipeline.render import render_clip, validate_render

    settings = settings or {}

    # ─── Step 1: Ingest ───
    logger.info("═══ Step 1: Ingest ═══")

    if video_hash:
        # Pre-chunked: load cached ingest data directly from Volume
        cache_dir = config.video_dir(video_hash)
        ingest_marker = cache_dir / ".ingest_complete"
        if ingest_marker.exists():
            logger.info("Loading pre-chunked ingest data for hash %s", video_hash)
            with open(cache_dir / "ingest_meta.json") as f:
                ingest_result = json.load(f)
        else:
            raise RuntimeError(
                f"video_hash {video_hash} provided but no .ingest_complete marker found at {cache_dir}"
            )
    elif video_path:
        # Pre-uploaded video: skip download, do chunking locally
        from pipeline.ingest import compute_video_hash, chunk_video, extract_audio, _get_duration
        source = Path(video_path)
        computed_hash = compute_video_hash(source)
        cache_dir = config.video_dir(computed_hash)
        chunk_dir = cache_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Check if already chunked
        ingest_marker = cache_dir / ".ingest_complete"
        if ingest_marker.exists():
            logger.info("Ingest cache hit — loading existing artifacts")
            with open(cache_dir / "ingest_meta.json") as f:
                ingest_result = json.load(f)
        else:
            logger.info("Chunking pre-uploaded video: %s", source.name)
            chunks = chunk_video(source, chunk_dir)
            for chunk in chunks:
                audio_path = Path(chunk["path"]).with_suffix(".wav")
                extract_audio(Path(chunk["path"]), audio_path)
                chunk["audio_path"] = str(audio_path)
            duration = _get_duration(source)
            ingest_result = {
                "video_hash": computed_hash,
                "url": url or "local",
                "source_path": str(source),
                "duration": duration,
                "num_chunks": len(chunks),
                "chunks": chunks,
            }
            with open(cache_dir / "ingest_meta.json", "w") as f:
                json.dump(ingest_result, f, indent=2)
            ingest_marker.touch()
            volume.commit()
            logger.info("Local ingest complete: %d chunks from %.1fs video", len(chunks), duration)
    else:
        ingester = Ingester()
        ingest_result = ingester.ingest.remote(url)

    video_hash = ingest_result["video_hash"]
    chunks = ingest_result["chunks"]
    video_dir = config.video_dir(video_hash)

    # Check if analysis is cached
    analysis_marker = video_dir / ".analysis_complete"
    if analysis_marker.exists() and not settings.get("force_reanalyze"):
        logger.info("Analysis cache hit — loading existing results")
        return _run_from_cache(video_dir, ingest_result, settings)

    # ─── Step 2: Parallel Map Analysis ───
    logger.info("═══ Step 2: Parallel Analysis (%d chunks) ═══", len(chunks))

    transcriber = Transcriber()
    face_tracker = FaceTracker()
    scene_detector = SceneDetector()
    audio_analyzer = AudioAnalyzer()

    # Launch all analysis in parallel using .map()
    transcript_futures = []
    face_futures = []
    scene_futures = []
    audio_futures = []

    for chunk in chunks:
        transcript_futures.append(transcriber.transcribe.spawn(chunk))
        face_futures.append(face_tracker.process_chunk.spawn(chunk))
        scene_futures.append(scene_detector.detect.spawn(chunk))
        audio_futures.append(audio_analyzer.analyze.spawn(chunk))

    # Collect results
    chunk_transcripts = [f.get() for f in transcript_futures]
    chunk_faces = [f.get() for f in face_futures]
    chunk_scenes = [f.get() for f in scene_futures]
    chunk_audio = [f.get() for f in audio_futures]

    # ─── Step 3: Reduce & Merge ───
    logger.info("═══ Step 3: Global Reduce ═══")

    # Merge transcripts
    transcript = merge_chunk_transcripts(chunk_transcripts, chunks)
    save_transcript(transcript, video_dir / "transcript.json")

    # Global face re-ID
    source_path = Path(ingest_result["source_path"])
    reid_mapping = global_reid(chunk_faces, source_path)

    # Apply re-ID mapping to face tracks
    all_face_tracks = []
    for chunk_idx, chunk_track in enumerate(chunk_faces):
        for frame_data in chunk_track:
            for face in frame_data.get("faces", []):
                local_tid = face.get("track_id", -1)
                face["global_face_id"] = reid_mapping.get(
                    (chunk_idx, local_tid), local_tid
                )
            all_face_tracks.append(frame_data)

    save_face_tracks(all_face_tracks, video_dir / "face_tracks.json")

    # Merge scenes and audio
    scenes = merge_chunk_scenes(chunk_scenes, chunks)
    save_scenes(scenes, video_dir / "scenes.json")

    audio_events = merge_chunk_audio_events(chunk_audio, chunks)
    save_audio_events(audio_events, video_dir / "audio_events.json")

    # Cache analysis
    analysis_marker.touch()
    volume.commit()

    # ─── Step 4: Clip Selection ───
    logger.info("═══ Step 4: Clip Selection ═══")

    candidates = generate_candidates(transcript, scenes, audio_events, settings)
    clips = rank_with_llm(candidates, transcript, settings.get("max_clips", config.MAX_CLIPS))
    save_selected_clips(clips, video_dir / "selected_clips.json")

    if not clips:
        logger.error("No clips selected!")
        return []

    # ─── Step 5: Per-clip Render ───
    logger.info("═══ Step 5: Rendering %d clips ═══", len(clips))

    # Get source video dimensions (with FPS guard)
    import cv2
    cap = cv2.VideoCapture(str(source_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not src_w or not src_h:
        logger.error("OpenCV could not read video dimensions")
        return []
    if not src_fps or src_fps <= 0:
        logger.warning("OpenCV returned invalid FPS (%.1f) — defaulting to 30", src_fps or 0)
        src_fps = 30.0

    # Build speaker map (simplified)
    speaker_map = {}

    render_dir = video_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = video_dir / "crop_plans"
    crop_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for clip in clips:
        clip_id = f"clip_{clip['rank']:02d}"

        # Compute crop plan
        crop_plan = compute_crop_plan(
            clip, all_face_tracks, speaker_map, scenes, transcript,
            src_w, src_h,
        )
        crop_path = save_crop_plan(crop_plan, crop_dir / f"{clip_id}.json")

        # Generate captions
        caption_path = render_dir / f"{clip_id}.ass"
        generate_captions(
            transcript, clip["start"], clip["end"], caption_path,
        )

        # Render
        output_path = render_dir / f"{clip_id}.mp4"
        render_clip(
            source_path, crop_path, caption_path, output_path,
            clip_start=clip["start"],
            clip_end=clip["end"],
        )

        # Validate
        validation = validate_render(output_path)
        if validation["valid"]:
            output_paths.append(str(output_path))
            logger.info("✓ %s (%.1f MB)", clip_id, validation.get("size_mb", 0))
        else:
            logger.error("✗ %s: %s", clip_id, validation.get("reason", "unknown"))

    volume.commit()
    logger.info("═══ Pipeline Complete: %d clips ═══", len(output_paths))

    # Write run report (breadcrumbs for debugging)
    import time as _time
    run_report = {
        "url": url,
        "video_hash": video_hash,
        "chunks_processed": len(chunks),
        "candidates_generated": len(candidates),
        "clips_rendered": len(output_paths),
        "clips_failed": len(clips) - len(output_paths),
        "asd_mode": "fallback",  # until LoCoNet is wired
        "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(video_dir / "run_report.json", "w") as f:
        json.dump(run_report, f, indent=2)
    volume.commit()

    return output_paths


def _run_from_cache(
    video_dir: Path,
    ingest_result: dict,
    settings: dict,
) -> list[str]:
    """Re-run clip selection and render from cached analysis."""
    from pipeline.clip_selector import generate_candidates, rank_with_llm
    from pipeline.reframe import compute_crop_plan, save_crop_plan
    from pipeline.captions import generate_captions
    from pipeline.render import render_clip, validate_render

    # Load cached data
    with open(video_dir / "transcript.json") as f:
        transcript = json.load(f)
    with open(video_dir / "scenes.json") as f:
        scenes = json.load(f)
    with open(video_dir / "audio_events.json") as f:
        audio_events = json.load(f)
    with open(video_dir / "face_tracks.json") as f:
        face_tracks = json.load(f)

    # Re-run selection + render
    candidates = generate_candidates(transcript, scenes, audio_events, settings)
    clips = rank_with_llm(candidates, transcript, settings.get("max_clips", config.MAX_CLIPS))

    source_path = Path(ingest_result["source_path"])

    import cv2
    cap = cv2.VideoCapture(str(source_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    render_dir = video_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = video_dir / "crop_plans"
    crop_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for clip in clips:
        clip_id = f"clip_{clip['rank']:02d}"

        crop_plan = compute_crop_plan(
            clip, face_tracks, {}, scenes, transcript, src_w, src_h,
        )
        crop_path = save_crop_plan(crop_plan, crop_dir / f"{clip_id}.json")

        caption_path = render_dir / f"{clip_id}.ass"
        generate_captions(transcript, clip["start"], clip["end"], caption_path)

        output_path = render_dir / f"{clip_id}.mp4"
        render_clip(
            source_path, crop_path, caption_path, output_path,
            clip_start=clip["start"], clip_end=clip["end"],
        )

        validation = validate_render(output_path)
        if validation["valid"]:
            output_paths.append(str(output_path))

    volume.commit()
    return output_paths


# ─────────────────────────────────────────────
# Local Entrypoint (for `modal run modal_app.py`)
# ─────────────────────────────────────────────

@app.local_entrypoint()
def main(
    url: str = "",
    max_clips: int = 5,
    min_duration: int = 15,
    max_duration: int = 60,
    dry_run: bool = False,
    force_reanalyze: bool = False,
    video_path: str = "",
    video_hash: str = "",
):
    """
    ClippedAI — Generate viral short-form clips from YouTube videos.

    Usage:
        modal run modal_app.py -- "URL"
        modal run modal_app.py -- "URL" --max-clips 3
    """
    import shutil
    import time as _time

    if not url and not video_path and not video_hash:
        print("❌ No URL, video path, or video hash provided.")
        print("   Usage: modal run modal_app.py --url \"URL\"")
        print("     or:  modal run modal_app.py --video-hash \"abc123\" --max-clips 3")
        return

    display_source = url if url else video_path
    print(f"""
╔══════════════════════════════════════╗
║         🎬  ClippedAI  🎬           ║
║   Viral Clips from YouTube Videos    ║
╚══════════════════════════════════════╝
    """)
    print(f"  Source:    {display_source}")
    print(f"  Max clips: {max_clips}")
    print(f"  Duration:  {min_duration}-{max_duration}s")
    if video_path:
        print(f"  Mode:      PRE-UPLOADED ({video_path})")
    if dry_run:
        print("  Mode:      DRY RUN")
    print()

    settings = {
        "max_clips": max_clips,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "ideal_duration": (15, 45),
        "dry_run": dry_run,
        "force_reanalyze": force_reanalyze,
    }

    start = _time.time()

    # Build volume path if video_path is relative (within modal volume)
    full_video_path = ""
    if video_path:
        full_video_path = f"{config.MODAL_VOLUME_MOUNT}/{video_path}"

    clip_paths = process_video.remote(
        url or "local",
        settings,
        video_path=full_video_path if full_video_path else None,
        video_hash=video_hash if video_hash else None,
    )
    elapsed = _time.time() - start

    if not clip_paths:
        print(f"\n❌ No clips generated after {elapsed:.0f}s")
        return

    # Download clips locally
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print(f"\n📥 Downloading {len(clip_paths)} clips...\n")

    downloaded = []
    for remote_path in clip_paths:
        filename = Path(remote_path).name
        local_path = output_dir / filename
        try:
            mount_prefix = config.MODAL_VOLUME_MOUNT + "/"
            relative_path = remote_path.replace(mount_prefix, "")
            data = volume.read_file(relative_path)
            with open(local_path, "wb") as f:
                for chunk in data:
                    f.write(chunk)
            downloaded.append(local_path)
        except Exception as e:
            print(f"  ⚠️ Could not download {filename}: {e}")

    # Summary
    print(f"\n{'═' * 50}")
    print(f"  ✅ Pipeline Complete — {elapsed:.1f}s")
    print(f"{'═' * 50}\n")

    if downloaded:
        print(f"  {'#':<4} {'Filename':<25} {'Size':>8}")
        print(f"  {'─' * 4} {'─' * 25} {'─' * 8}")
        for i, clip in enumerate(downloaded, 1):
            size = clip.stat().st_size / 1e6
            print(f"  {i:<4} {clip.name:<25} {size:>6.1f}MB")
        print(f"\n  📂 Output: {output_dir}/")
    else:
        print("  ⚠️ No clips downloaded.")
    print()
