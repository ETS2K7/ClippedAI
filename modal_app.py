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
)

image_ingest = (
    image_base
    .apt_install("curl")
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
    image_base
    .pip_install(
        "whisperx", "pyannote.audio==3.1",
        "torch", "torchaudio",
    )
)

image_vision = (
    image_base
    .pip_install(
        "ultralytics", "insightface", "onnxruntime-gpu",
        "filterpy", "mediapipe", "opencv-python-headless",
        "scikit-learn", "setuptools",
    )
    .run_commands(
        "git clone https://github.com/SJTUwxz/LoCoNet_ASD.git /opt/loconet "
        "&& cd /opt/loconet && git checkout 8a3f3d2"
    )
    .run_commands(
        "git clone https://github.com/bellhyeon/BoT-FaceSORT.git /opt/bot-facesort "
        "&& cd /opt/bot-facesort && git checkout 3e1c5b9 && pip install -r requirements.txt"
    )
)

image_scene = (
    image_base
    .pip_install("torch", "torchvision", "scenedetect[opencv]")
    .run_commands(
        "git clone https://github.com/wentaozhu/AutoShot.git /opt/autoshot "
        "&& cd /opt/autoshot && git checkout a4b7e1f"
    )
)

image_audio = (
    image_base
    .pip_install(
        "panns-inference", "librosa",
        "torch", "torchaudio",
    )
)

image_render = (
    image_base
    .apt_install("libass-dev")
    .pip_install("opencv-python-headless")
)

image_llm = (
    image_base
    .pip_install("groq", "openai")
)

# ─────────────────────────────────────────────
# Secrets
# ─────────────────────────────────────────────

secrets = [modal.Secret.from_name("clippedai-secrets")]


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
        except FileNotFoundError:
            self.pot_process = None
            logger.warning("bgutil-pot-server not found, PoT tokens unavailable")

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
    timeout=900,
)
def process_video(
    url: str,
    settings: Optional[dict] = None,
) -> list[str]:
    """
    Main pipeline orchestrator.

    1. Ingest (download + chunk)
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
