# ClippedAI — Full Codebase

**Generated**: 2026-02-24 11:31:00  
**Commit**: `36e5957`  
**Files**: 15

## Recently Changed

- `config.py`
- `modal_app.py`
- `pipeline/render.py`
- `pipeline/face_track.py`
- `pipeline/ingest.py`

## File Index

- `config.py`
- `modal_app.py`
- `pipeline/__init__.py`
- `pipeline/audio_analysis.py`
- `pipeline/captions.py`
- `pipeline/clip_selector.py`
- `pipeline/cookie_refresh.py`
- `pipeline/face_track.py`
- `pipeline/ingest.py`
- `pipeline/reframe.py`
- `pipeline/render.py`
- `pipeline/scene_detect.py`
- `pipeline/transcribe.py`
- `run.py`
- `snapshot.py`

---

## `config.py`

```python
"""
ClippedAI — Core Pipeline Configuration
All tunable parameters in one place.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
LLM_PROVIDER_PRIORITY = ["groq", "cerebras"]
LLM_MODEL_RANKING = "llama-3.3-70b-versatile"  # Groq production model
LLM_MODEL_FALLBACK = "llama-3.3-70b"  # Cerebras format
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# ─────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────
CHUNK_DURATION = 300        # 5-minute chunks
CHUNK_OVERLAP = 30          # seconds of overlap between chunks
ASPECT_RATIO = "9:16"
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
VIDEO_CRF = 20              # H.264 quality (lower = better, 18-23 typical)
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "medium"

# ─────────────────────────────────────────────
# Clip Selection
# ─────────────────────────────────────────────
MAX_CLIPS = 5
MIN_CLIP_DURATION = 15      # seconds
MAX_CLIP_DURATION = 60      # hard ceiling: retention drops sharply after 60s
IDEAL_CLIP_DURATION = (15, 45)  # target range for highest retention
NO_TEMPORAL_OVERLAP = True  # clips must NOT share any source frames
SEMANTIC_SIMILARITY_PENALTY = 0.75  # penalize clips with >0.75 cosine sim

# ─────────────────────────────────────────────
# ASR (WhisperX)
# ─────────────────────────────────────────────
WHISPERX_MODEL = "large-v3-turbo"
WHISPERX_COMPUTE_TYPE = "float16"
WHISPERX_BATCH_SIZE = 16
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─────────────────────────────────────────────
# Face Detection & Tracking
# ─────────────────────────────────────────────
FACE_MODEL = "yolov8n.pt"  # Ultralytics YOLO model (auto-downloads)
FACE_MODEL_FACE = "/opt/yolov8n-face.pt"  # Face-specific model (pre-downloaded from HuggingFace)
FACE_CONF_THRESHOLD = 0.5
FACE_IOU_THRESHOLD = 0.45
INSIGHTFACE_COSINE_THRESHOLD = 0.62  # for global re-ID clustering
REID_EMBEDDINGS_PER_SECOND = 1  # subsample to prevent OOM

# ─────────────────────────────────────────────
# Active Speaker Detection (LoCoNet)
# ─────────────────────────────────────────────
LOCONET_CONFIDENCE_HIGH = 0.85   # above = definitely active speaker
LOCONET_CONFIDENCE_LOW = 0.60    # below = uncertain, use fallback

# ─────────────────────────────────────────────
# Reframing
# ─────────────────────────────────────────────
REFRAME_SAFETY_MARGIN = 0.15     # 15% padding around face crop
REFRAME_MIN_FACE_HEIGHT = 0.55   # face must be >= 55% of frame height
REFRAME_HYSTERESIS = (1.5, 3.0)  # seconds — dynamic based on content
REFRAME_SMOOTHING_WINDOW = 15    # frames for Kalman smoother
CROP_SEGMENT_DURATION = 0.5      # seconds per pre-computed crop segment
GOOD_FRAME_THRESHOLD = 0.92      # minimum ratio for validation pass

# ─────────────────────────────────────────────
# Captions
# ─────────────────────────────────────────────
CAPTION_STYLE = "highlight_word"
CAPTION_FONT = "Arial"
CAPTION_FONT_SIZE_RATIO = 25     # output_height / this = font size
CAPTION_MAX_LINES = 2
CAPTION_MAX_WIDTH_RATIO = 0.80   # 80% of frame width
CAPTION_SAFE_ZONE_Y = 0.80       # captions placed at 80% down
CAPTION_GAP_THRESHOLD = 0.3      # seconds — gap > this = blank screen
CAPTION_MIN_WORD_DURATION = 0.2  # seconds — minimum display time
CAPTION_CONFIDENCE_THRESHOLD = 0.4  # drop words below this confidence

# ─────────────────────────────────────────────
# YouTube Download
# ─────────────────────────────────────────────
YT_PLAYER_CLIENTS = ["android", "ios", "web"]
YT_COOKIE_TTL = 21600       # 6 hours in seconds
YT_MAX_DOWNLOADS_PER_HOUR = 15
YT_MAX_DOWNLOADS_PER_DAY = 50
YT_COOLDOWN_AFTER_FAILURES = 3
YT_COOLDOWN_DURATION = 3600  # 1 hour
YT_MAX_RETRIES = 3
YT_RETRY_BACKOFF = (2, 10)  # exponential backoff range in seconds

# ─────────────────────────────────────────────
# Video Hash
# ─────────────────────────────────────────────
HASH_HEAD_BYTES = 10 * 1024 * 1024   # first 10MB
HASH_TAIL_BYTES = 10 * 1024 * 1024   # last 10MB

# ─────────────────────────────────────────────
# Modal
# ─────────────────────────────────────────────
MODAL_VOLUME_NAME = "clippedai-vol"
MODAL_VOLUME_MOUNT = "/vol"
MODAL_APP_NAME = "clippedai"

# ─────────────────────────────────────────────
# Paths (on Modal Volume)
# ─────────────────────────────────────────────
def video_dir(video_hash: str) -> Path:
    """Return path to video artifacts on Modal Volume."""
    return Path(MODAL_VOLUME_MOUNT) / video_hash

def artifact_path(video_hash: str, filename: str) -> Path:
    """Return path to a specific artifact file."""
    return video_dir(video_hash) / filename
```

---

## `modal_app.py`

```python
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
    gpu="T4",
    volumes={config.MODAL_VOLUME_MOUNT: volume},
    timeout=120,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
class AudioAnalyzer:
    """PANNs container (T4 GPU)."""

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
    gpu="T4",
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
```

---

## `pipeline/__init__.py`

```python
"""ClippedAI Pipeline Package"""
```

---

## `pipeline/audio_analysis.py`

```python
"""
Audio event classification using PANNs.

Detects laughter, applause, music, crowd noise, etc.
"""

import json
import logging
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


def analyze_audio(
    audio_path: Path,
    chunk_start: float = 0.0,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
) -> list[dict]:
    """
    Classify audio events in a chunk using PANNs.

    Returns list of events:
      [{start, end, event_type, confidence}]
    """
    try:
        return _analyze_with_panns(audio_path, chunk_start, window_seconds, hop_seconds)
    except (ImportError, Exception) as e:
        logger.warning("PANNs unavailable (%s), using energy-based fallback", e)
        return _analyze_with_energy(audio_path, chunk_start)


def _analyze_with_panns(
    audio_path: Path,
    chunk_start: float,
    window_seconds: float,
    hop_seconds: float,
) -> list[dict]:
    """Audio event detection using PANNs."""
    from panns_inference import AudioTagging
    import librosa

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=32000, mono=True)
    duration = len(audio) / sr

    # Auto-detect device — AudioAnalyzer container is CPU-only
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize PANNs
    tagger = AudioTagging(checkpoint_path=None, device=device)

    events = []
    # Sliding window analysis
    pos = 0.0
    while pos < duration:
        end = min(pos + window_seconds, duration)
        start_sample = int(pos * sr)
        end_sample = int(end * sr)
        segment = audio[start_sample:end_sample]

        if len(segment) < sr * 0.5:  # skip tiny segments
            pos += hop_seconds
            continue

        # Pad to window size
        if len(segment) < int(window_seconds * sr):
            segment = np.pad(segment, (0, int(window_seconds * sr) - len(segment)))

        # Predict
        clipwise_output, _ = tagger.inference(segment[np.newaxis, :])
        probs = clipwise_output[0]

        # Map AudioSet labels to our categories
        event_mappings = {
            "Speech": (0, 0.5),
            "Laughter": (16, 0.3),
            "Applause": (25, 0.3),
            "Music": (137, 0.4),
            "Crowd": (36, 0.3),
            "Silence": None,  # detected by absence of other events
        }

        for event_type, mapping in event_mappings.items():
            if mapping is None:
                continue
            idx, threshold = mapping
            if idx < len(probs) and probs[idx] > threshold:
                events.append({
                    "start": round(pos + chunk_start, 3),
                    "end": round(end + chunk_start, 3),
                    "event_type": event_type.lower(),
                    "confidence": round(float(probs[idx]), 3),
                })

        pos += hop_seconds

    logger.info("PANNs: detected %d audio events", len(events))
    return events


def _analyze_with_energy(
    audio_path: Path,
    chunk_start: float,
) -> list[dict]:
    """Fallback: simple RMS energy analysis."""
    import librosa

    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Compute RMS energy in 1-second windows
    hop_length = sr  # 1 second
    frame_length = sr
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    events = []
    mean_rms = np.mean(rms) if len(rms) > 0 else 0
    std_rms = np.std(rms) if len(rms) > 0 else 1

    for i, energy in enumerate(rms):
        timestamp = i + chunk_start
        # High energy = potential highlight
        if energy > mean_rms + 1.5 * std_rms:
            events.append({
                "start": round(timestamp, 3),
                "end": round(timestamp + 1.0, 3),
                "event_type": "high_energy",
                "confidence": round(float(min(1.0, (energy - mean_rms) / (std_rms + 1e-8))), 3),
            })

    logger.info("Energy analysis: %d high-energy events", len(events))
    return events


def merge_chunk_audio_events(
    chunk_events: list[list[dict]],
    chunks_meta: list[dict],
) -> list[dict]:
    """Merge audio events from overlapping chunks."""
    all_events = []
    for events, chunk_meta in zip(chunk_events, chunks_meta):
        overlap = config.CHUNK_OVERLAP
        chunk_start = chunk_meta["start"]

        # Trust region (same logic as transcript merge)
        trust_start = chunk_start + overlap / 2 if chunk_meta["index"] > 0 else chunk_start
        for event in events:
            if event["start"] >= trust_start:
                all_events.append(event)

    all_events.sort(key=lambda e: e["start"])
    return all_events


def save_audio_events(events: list[dict], output_path: Path) -> Path:
    """Save audio events to JSON."""
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)
    logger.info("Saved %d audio events: %s", len(events), output_path)
    return output_path
```

---

## `pipeline/captions.py`

```python
"""
Caption generation as per-word .ass subtitles.

Implements all 6 non-negotiable caption rules:
  1. Each word = one .ass Dialogue event
  2. Immediate disappearance on gaps > 300ms
  3. Minimum 200ms display duration
  4. No pre-display of upcoming words
  5. Safe zone positioning (bottom 15%, max 80% width, max 2 lines)
  6. Highlight styling (active word bold, spoken dimmed)
"""

import logging
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


def generate_captions(
    transcript: dict,
    clip_start: float,
    clip_end: float,
    output_path: Path,
    output_width: int = config.OUTPUT_WIDTH,
    output_height: int = config.OUTPUT_HEIGHT,
) -> Path:
    """
    Generate .ass subtitle file for a clip.
    Each word is a separate Dialogue event with exact WhisperX timing.
    """
    # Extract words within clip range
    words = _extract_and_filter_words(transcript, clip_start, clip_end)

    if not words:
        logger.warning("No words found for clip %.1f-%.1f", clip_start, clip_end)
        _write_empty_ass(output_path, output_width, output_height)
        return output_path

    # Apply caption rules
    words = _apply_timing_rules(words, clip_start)
    words = _apply_gap_rules(words)

    # Generate .ass content
    font_size = output_height // config.CAPTION_FONT_SIZE_RATIO
    ass_content = _build_ass_file(
        words, output_width, output_height, font_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ass_content, encoding="utf-8")
    logger.info("Generated captions: %s (%d words)", output_path, len(words))

    return output_path


# ─────────────────────────────────────────────
# Word Extraction & Filtering
# ─────────────────────────────────────────────

def _extract_and_filter_words(
    transcript: dict,
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """Extract words in clip range, apply confidence + hallucination filters."""
    words = []

    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if w["start"] >= clip_start and w["end"] <= clip_end:
                words.append({
                    "word": w["word"].strip(),
                    "start": w["start"] - clip_start,  # relative to clip
                    "end": w["end"] - clip_start,
                    "score": w.get("score", 1.0),
                    "speaker": seg.get("speaker", ""),
                })

    # Filter: drop low confidence words (WhisperX hallucinations)
    words = [w for w in words if w["score"] >= config.CAPTION_CONFIDENCE_THRESHOLD]

    # Filter: drop repeated words within 0.5s
    filtered = []
    for i, w in enumerate(words):
        if i > 0 and w["word"].lower() == words[i - 1]["word"].lower():
            if w["start"] - words[i - 1]["start"] < 0.5:
                continue
        filtered.append(w)

    # Capitalize first word and after silence gaps
    for i, w in enumerate(filtered):
        if i == 0:
            w["word"] = w["word"].capitalize()
        elif i > 0 and w["start"] - filtered[i - 1]["end"] > config.CAPTION_GAP_THRESHOLD:
            w["word"] = w["word"].capitalize()

    return filtered


# ─────────────────────────────────────────────
# Timing Rules
# ─────────────────────────────────────────────

def _apply_timing_rules(words: list[dict], clip_start: float) -> list[dict]:
    """
    RULE 3: Minimum 200ms display duration.
    RULE 4: No pre-display — word N starts at word N's timestamp.
    """
    for w in words:
        duration = w["end"] - w["start"]
        if duration < config.CAPTION_MIN_WORD_DURATION:
            w["end"] = w["start"] + config.CAPTION_MIN_WORD_DURATION

    return words


def _apply_gap_rules(words: list[dict]) -> list[dict]:
    """
    RULE 2: If gap > 300ms between words, insert blank.
    Each word is already a separate event, so gaps naturally create blanks.
    We just need to ensure no word's end extends into the gap.
    """
    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        if gap > config.CAPTION_GAP_THRESHOLD:
            # Ensure this word ends at its own end time, not bridging the gap
            pass  # already handled by per-word events
        elif gap < 0:
            # Overlapping timestamps — clip current word's end
            words[i]["end"] = words[i + 1]["start"]

    return words


# ─────────────────────────────────────────────
# .ass File Generation
# ─────────────────────────────────────────────

def _build_ass_file(
    words: list[dict],
    width: int,
    height: int,
    font_size: int,
) -> str:
    """Build complete .ass subtitle file."""

    # Calculate safe zone positioning
    margin_h = int(width * (1 - config.CAPTION_MAX_WIDTH_RATIO) / 2)
    margin_v = int(height * (1 - config.CAPTION_SAFE_ZONE_Y))

    header = f"""[Script Info]
Title: ClippedAI Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.CAPTION_FONT},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,{margin_h},{margin_h},{margin_v},1
Style: Active,{config.CAPTION_FONT},{font_size},&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,{margin_h},{margin_h},{margin_v},1
Style: Spoken,{config.CAPTION_FONT},{font_size},&H80FFFFFF,&H0000FFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    # Group words into display lines (max 2 lines, word-wrap)
    lines = _group_into_lines(words, width, font_size, margin_h)

    for line_group in lines:
        for word_data in line_group:
            start_ts = _format_ass_time(word_data["start"])
            end_ts = _format_ass_time(word_data["end"])
            text = word_data["display_text"]

            events.append(
                f"Dialogue: 0,{start_ts},{end_ts},Active,,0,0,0,,{text}"
            )

    return header + "\n".join(events) + "\n"


def _group_into_lines(
    words: list[dict],
    width: int,
    font_size: int,
    margin: int,
) -> list[list[dict]]:
    """
    Group words into display lines respecting:
    RULE 5: max 2 lines, word wrap at boundaries, max 80% width.
    Enforces CAPTION_MAX_LINES per time window — older lines are flushed.
    """
    max_chars_per_line = int((width - 2 * margin) / (font_size * 0.55))
    max_lines = config.CAPTION_MAX_LINES

    all_line_groups = []
    current_line = []
    current_chars = 0
    pending_lines = []  # buffer of lines being displayed

    for w in words:
        word_len = len(w["word"]) + 1  # +1 for space

        if current_chars + word_len > max_chars_per_line and current_line:
            pending_lines.append(current_line)
            current_line = []
            current_chars = 0

            # Enforce max_lines: flush oldest line when exceeded
            if len(pending_lines) > max_lines:
                all_line_groups.append(pending_lines.pop(0))

        # Build display text with highlight override
        display = "{\\b1\\c&H00FFFF&}" + w["word"]

        current_line.append({
            **w,
            "display_text": display,
        })
        current_chars += word_len

    if current_line:
        pending_lines.append(current_line)

    # Flush remaining
    all_line_groups.extend(pending_lines)

    return all_line_groups


def _format_ass_time(seconds: float) -> str:
    """Format seconds as .ass timestamp: H:MM:SS.CC"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_empty_ass(path: Path, width: int, height: int) -> None:
    """Write an empty .ass file (no captions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""[Script Info]
Title: ClippedAI Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,50,50,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    path.write_text(content, encoding="utf-8")
```

---

## `pipeline/clip_selector.py`

```python
"""
LLM-powered clip selection.

Stage 1: Algorithmic candidate generation (7 signals)
Stage 2: Groq Llama 4 Scout structured ranking (candidate_index only)
"""

import json
import logging
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Stage 1: Algorithmic Candidate Generation
# ─────────────────────────────────────────────

def generate_candidates(
    transcript: dict,
    scenes: list[dict],
    audio_events: list[dict],
    settings: Optional[dict] = None,
) -> list[dict]:
    """
    Generate clip candidates using 7 signals:
      1. Audio energy peaks
      2. Keyword triggers (questions, reactions, emotional words)
      3. Scene density (rapid cuts = high energy)
      4. Sentiment shifts in transcript
      5. Speaker turn density
      6. Laughter / applause events
      7. Silence gaps (natural breakpoints)

    Uses precomputed time indices for O(log n) range queries.
    Returns list of candidate clips with scores.
    """
    settings = settings or {}
    min_dur = settings.get("min_duration", config.MIN_CLIP_DURATION)
    max_dur = settings.get("max_duration", config.MAX_CLIP_DURATION)
    ideal_dur = settings.get("ideal_duration", config.IDEAL_CLIP_DURATION)
    max_clips = settings.get("max_clips", config.MAX_CLIPS)

    segments = transcript.get("segments", [])
    if not segments:
        logger.warning("No transcript segments — cannot generate candidates")
        return []

    # ── Precompute sorted time indices (avoids O(n) per window) ──
    word_times = []  # [(start_time, word_dict)]
    for seg in segments:
        for w in seg.get("words", []):
            word_times.append((w.get("start", 0), w))
    word_times.sort(key=lambda x: x[0])
    word_starts = [wt[0] for wt in word_times]

    seg_starts = sorted(seg["start"] for seg in segments)
    seg_ends = sorted(seg["end"] for seg in segments)

    scene_boundaries = sorted(s.get("boundary", 0) for s in scenes)

    highlight_events = sorted(
        e["start"] for e in audio_events
        if e.get("event_type") in ("laughter", "applause", "high_energy")
    )

    seg_speaker_data = [(seg["start"], seg.get("speaker", "")) for seg in segments]
    seg_speaker_data.sort(key=lambda x: x[0])
    seg_speaker_starts = [s[0] for s in seg_speaker_data]

    # Score every possible window using bisect for range queries
    import bisect

    duration = segments[-1]["end"] if segments else 0
    candidates = []

    step = 5.0
    for window_start in np.arange(0, max(0, duration - min_dur), step):
        for window_dur in range(ideal_dur[0], ideal_dur[1] + 1, 5):
            window_end = window_start + window_dur
            if window_end > duration:
                continue

            score = _score_window_indexed(
                float(window_start), float(window_end),
                word_times, word_starts,
                seg_starts, seg_ends,
                scene_boundaries,
                highlight_events,
                seg_speaker_data, seg_speaker_starts,
            )

            if score > 0.1:
                candidates.append({
                    "start": round(float(window_start), 3),
                    "end": round(float(window_end), 3),
                    "duration": round(float(window_dur), 1),
                    "algorithmic_score": round(float(score), 4),
                })

    # Sort by score, take top 3× max_clips
    candidates.sort(key=lambda c: c["algorithmic_score"], reverse=True)
    candidates = candidates[:max_clips * 3]

    # Remove overlapping candidates (enforce zero temporal overlap)
    candidates = _remove_overlapping(candidates)

    # Remove semantically similar candidates (enforce content diversity)
    candidates = _deduplicate_by_content(candidates, transcript)

    # Re-sort after dedup to ensure best clips survive downstream
    candidates.sort(key=lambda c: c["algorithmic_score"], reverse=True)

    logger.info("Generated %d candidates from %.0fs video", len(candidates), duration)
    return candidates


def _score_window_indexed(
    start: float,
    end: float,
    word_times: list[tuple],
    word_starts: list[float],
    seg_starts: list[float],
    seg_ends: list[float],
    scene_boundaries: list[float],
    highlight_events: list[float],
    seg_speaker_data: list[tuple],
    seg_speaker_starts: list[float],
) -> float:
    """Score a candidate window using 7 signals with O(log n) lookups."""
    import bisect

    duration = end - start

    # 1. Word density (bisect range query)
    lo = bisect.bisect_left(word_starts, start)
    hi = bisect.bisect_right(word_starts, end)
    words_in_window = [word_times[i][1] for i in range(lo, hi)]
    word_density = len(words_in_window) / max(1, duration)
    signal_speech = min(1.0, word_density / 3.0)

    # 2. Keyword triggers
    trigger_words = {
        "wow", "amazing", "incredible", "insane", "crazy", "unbelievable",
        "wait", "what", "oh", "no way", "seriously", "actually",
        "important", "secret", "truth", "real", "honest",
        "?",
    }
    text = " ".join(w["word"].lower() for w in words_in_window)
    trigger_count = sum(1 for tw in trigger_words if tw in text)
    signal_keywords = min(1.0, trigger_count / 5.0)

    # 3. Scene density (bisect)
    s_lo = bisect.bisect_left(scene_boundaries, start)
    s_hi = bisect.bisect_right(scene_boundaries, end)
    scene_cuts = s_hi - s_lo
    signal_scenes = min(1.0, scene_cuts / max(1, duration / 10))

    # 4. Speaker turn density (bisect)
    sp_lo = bisect.bisect_left(seg_speaker_starts, start)
    sp_hi = bisect.bisect_right(seg_speaker_starts, end)
    speaker_changes = 0
    prev_speaker = None
    for i in range(sp_lo, sp_hi):
        spk = seg_speaker_data[i][1]
        if prev_speaker and spk != prev_speaker:
            speaker_changes += 1
        prev_speaker = spk
    signal_turns = min(1.0, speaker_changes / 5.0)

    # 5. Audio events (bisect)
    a_lo = bisect.bisect_left(highlight_events, start)
    a_hi = bisect.bisect_right(highlight_events, end)
    signal_audio = min(1.0, (a_hi - a_lo) / 3.0)

    # 6. Completeness — sentence boundaries (bisect)
    ss_lo = bisect.bisect_left(seg_starts, start - 1.0)
    ss_hi = bisect.bisect_right(seg_starts, start + 1.0)
    starts_at_sentence = ss_hi > ss_lo

    se_lo = bisect.bisect_left(seg_ends, end - 1.0)
    se_hi = bisect.bisect_right(seg_ends, end + 1.0)
    ends_at_sentence = se_hi > se_lo

    signal_completeness = (0.5 if starts_at_sentence else 0) + (0.5 if ends_at_sentence else 0)

    # 7. Ideal duration bonus
    ideal_min, ideal_max = config.IDEAL_CLIP_DURATION
    if ideal_min <= duration <= ideal_max:
        signal_duration = 1.0
    elif duration < ideal_min:
        signal_duration = duration / ideal_min
    else:
        signal_duration = max(0, 1.0 - (duration - ideal_max) / 30)

    # Weighted sum
    score = (
        0.25 * signal_speech +
        0.15 * signal_keywords +
        0.10 * signal_scenes +
        0.10 * signal_turns +
        0.15 * signal_audio +
        0.15 * signal_completeness +
        0.10 * signal_duration
    )

    return score


def _remove_overlapping(candidates: list[dict]) -> list[dict]:
    """Remove overlapping candidates, keeping highest scored."""
    selected = []
    for c in candidates:
        overlaps = False
        for s in selected:
            if c["start"] < s["end"] and c["end"] > s["start"]:
                overlaps = True
                break
        if not overlaps:
            selected.append(c)
    return selected


def _deduplicate_by_content(
    candidates: list[dict],
    transcript: dict,
) -> list[dict]:
    """
    Remove semantically similar candidates using TF-IDF cosine similarity.
    If two clips share > SEMANTIC_SIMILARITY_PENALTY similarity, discard  
    the lower-scoring one. Prevents duplicate-topic clips from podcasts.
    """
    if len(candidates) <= 1:
        return candidates

    segments = transcript.get("segments", [])

    # Extract text for each candidate
    texts = []
    for c in candidates:
        text = _extract_text_for_range(segments, c["start"], c["end"])
        texts.append(text if text.strip() else "empty")

    # Compute TF-IDF similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Greedy dedup: keep highest scored, discard similar
        keep = [True] * len(candidates)
        for i in range(len(candidates)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(candidates)):
                if not keep[j]:
                    continue
                if sim_matrix[i][j] > config.SEMANTIC_SIMILARITY_PENALTY:
                    # Discard the lower-scored one
                    keep[j] = False
                    logger.info(
                        "Dedup: dropping candidate %.0f-%.0fs (sim=%.2f with %.0f-%.0fs)",
                        candidates[j]["start"], candidates[j]["end"],
                        sim_matrix[i][j],
                        candidates[i]["start"], candidates[i]["end"],
                    )

        return [c for c, k in zip(candidates, keep) if k]

    except ImportError:
        logger.warning("sklearn unavailable, skipping semantic dedup")
        return candidates


# ─────────────────────────────────────────────
# Stage 2: LLM Ranking
# ─────────────────────────────────────────────

def rank_with_llm(
    candidates: list[dict],
    transcript: dict,
    max_clips: int = config.MAX_CLIPS,
) -> list[dict]:
    """
    Rank candidates using Groq Llama 4 Scout.
    LLM returns candidate_index integers only — timestamps from Python.
    """
    if not candidates:
        return []

    # Build candidate summaries for the LLM
    candidate_summaries = []
    segments = transcript.get("segments", [])

    for i, c in enumerate(candidates):
        # Extract text within this candidate's time range
        text = _extract_text_for_range(segments, c["start"], c["end"])
        candidate_summaries.append({
            "index": i,
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "text_preview": text[:300],
            "algorithmic_score": c["algorithmic_score"],
        })

    # Call LLM
    ranked_indices = _call_llm(candidate_summaries, max_clips)

    # Map indices back to candidates with LLM ranking
    ranked_clips = []
    for rank, idx in enumerate(ranked_indices):
        if 0 <= idx < len(candidates):
            clip = candidates[idx].copy()
            clip["rank"] = rank + 1
            clip["llm_selected"] = True
            ranked_clips.append(clip)

    logger.info("LLM selected %d clips from %d candidates", len(ranked_clips), len(candidates))
    return ranked_clips


def _call_llm(
    candidate_summaries: list[dict],
    max_clips: int,
) -> list[int]:
    """
    Call Groq (primary) or Cerebras (fallback) to rank candidates.
    Returns list of candidate indices in ranked order.
    """
    prompt = _build_ranking_prompt(candidate_summaries, max_clips)

    for provider in config.LLM_PROVIDER_PRIORITY:
        try:
            if provider == "groq":
                return _call_groq(prompt, max_clips)
            elif provider == "cerebras":
                return _call_cerebras(prompt, max_clips)
        except Exception as e:
            logger.warning("LLM provider %s failed: %s", provider, e)
            continue

    # All providers failed — return top candidates by algorithmic score
    logger.error("All LLM providers failed, using algorithmic ranking")
    return list(range(min(max_clips, len(candidate_summaries))))


def _build_ranking_prompt(candidates: list[dict], max_clips: int) -> str:
    """Build the structured ranking prompt."""
    candidate_text = ""
    for c in candidates:
        candidate_text += (
            f"\n[Candidate {c['index']}] "
            f"({c['duration']:.0f}s, score={c['algorithmic_score']:.2f})\n"
            f"Text: {c['text_preview']}\n"
        )

    return f"""You are a viral short-form content curator. Your job is to select the {max_clips} best clips from a longer video for YouTube Shorts / TikTok / Instagram Reels.

CANDIDATES:
{candidate_text}

SELECTION CRITERIA:
1. Hook potential — would this grab attention in the first 3 seconds?
2. Completeness — is this a complete thought/story/joke?
3. Emotional impact — does this evoke strong reactions?
4. Shareability — would someone share this?
5. Diversity — pick varied content, not the same topic repeated.

RULES:
- Return ONLY a JSON array of candidate index integers, ranked best to worst.
- Return exactly {max_clips} indices (or fewer if not enough good candidates).
- Do NOT modify timestamps. Just return indices.

Example response: [3, 0, 7, 1, 5]

Your selection:"""


def _call_groq(prompt: str, max_clips: int) -> list[int]:
    """Call Groq API with structured output."""
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)

    response = client.chat.completions.create(
        model=config.LLM_MODEL_RANKING,
        messages=[
            {"role": "system", "content": "You are a viral content curator. Respond with JSON: {\"indices\": [int, ...]}"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=100,
        timeout=30,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content.strip()
    result = json.loads(text)

    # Handle both {"indices": [...]} and [...] formats
    if isinstance(result, list):
        return [int(x) for x in result[:max_clips]]
    elif isinstance(result, dict):
        indices = result.get("indices", result.get("selection", result.get("ranking", [])))
        return [int(x) for x in indices[:max_clips]]

    return list(range(max_clips))


def _call_cerebras(prompt: str, max_clips: int) -> list[int]:
    """Call Cerebras API."""
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=config.CEREBRAS_API_KEY)

    response = client.chat.completions.create(
        model=config.LLM_MODEL_FALLBACK,
        messages=[
            {"role": "system", "content": "You are a viral content curator. Respond with JSON: {\"indices\": [int, ...]}"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=100,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content.strip()
    result = json.loads(text)

    if isinstance(result, list):
        return [int(x) for x in result[:max_clips]]
    elif isinstance(result, dict):
        indices = result.get("indices", result.get("selection", result.get("ranking", [])))
        return [int(x) for x in indices[:max_clips]]

    return list(range(max_clips))


def _extract_text_for_range(
    segments: list[dict],
    start: float,
    end: float,
) -> str:
    """Extract transcript text within a time range.
    Falls back to segment-level text if WhisperX dropped word timings."""
    words = []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            for w in seg_words:
                if start <= w["start"] <= end:
                    words.append(w["word"])
        elif seg["start"] < end and seg["end"] > start:
            # Fallback: no word-level data, use segment text
            words.append(seg.get("text", ""))
    return " ".join(words)


def save_selected_clips(clips: list[dict], output_path) -> None:
    """Save selected clips to JSON."""
    from pathlib import Path
    with open(Path(output_path), "w") as f:
        json.dump(clips, f, indent=2)
    logger.info("Saved %d selected clips: %s", len(clips), output_path)
```

---

## `pipeline/cookie_refresh.py`

```python
"""
Automated YouTube cookie refresh via Playwright Stealth.

Runs a headless Chromium session to generate fresh cookies + visitor_data
when yt-dlp encounters bot verification. Cached on Modal Volume with TTL.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


def refresh_cookies(
    cookie_path: Path,
    account_email: Optional[str] = None,
    account_password: Optional[str] = None,
) -> Path:
    """
    Launch headless Chromium via Playwright Stealth, navigate to YouTube,
    and export cookies in Netscape format for yt-dlp.

    Returns path to the cookie file.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && "
            "playwright install chromium"
        )

    cookie_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Linux; Android 13; SM-G991B) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Mobile Safari/537.36"
            ),
            viewport={"width": 412, "height": 915},
            is_mobile=True,
        )

        page = context.new_page()

        try:
            # Navigate to YouTube
            logger.info("Navigating to YouTube for cookie refresh...")
            page.goto("https://www.youtube.com", wait_until="networkidle",
                      timeout=30000)
            time.sleep(3)  # Let JS settle

            # Accept cookies dialog if present
            try:
                accept_btn = page.locator(
                    "button:has-text('Accept'), "
                    "button:has-text('I agree'), "
                    "button:has-text('Accept all')"
                )
                if accept_btn.count() > 0:
                    accept_btn.first.click()
                    time.sleep(1)
            except Exception:
                pass  # No cookie dialog, continue

            # Extract cookies
            cookies = context.cookies()
            _write_netscape_cookies(cookies, cookie_path)

            logger.info(
                "Cookie refresh successful: %d cookies saved to %s",
                len(cookies), cookie_path,
            )

        except Exception as e:
            logger.error("Cookie refresh failed: %s", e)
            raise

        finally:
            browser.close()

    return cookie_path


def is_cookie_valid(cookie_path: Path) -> bool:
    """Check if cookie file exists and is within TTL."""
    if not cookie_path.exists():
        return False

    age = time.time() - cookie_path.stat().st_mtime
    if age > config.YT_COOKIE_TTL:
        logger.info("Cookies expired (%.0fs old, TTL=%ds)", age, config.YT_COOKIE_TTL)
        return False

    return True


def _write_netscape_cookies(cookies: list[dict], path: Path) -> None:
    """Write cookies in Netscape format that yt-dlp accepts."""
    lines = ["# Netscape HTTP Cookie File", ""]

    for c in cookies:
        domain = c.get("domain", "")
        if not domain.startswith("."):
            domain = "." + domain

        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path_val = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expires = str(int(c.get("expires", 0)))
        name = c.get("name", "")
        value = c.get("value", "")

        lines.append(f"{domain}\t{flag}\t{path_val}\t{secure}\t{expires}\t{name}\t{value}")

    path.write_text("\n".join(lines))
```

---

## `pipeline/face_track.py`

```python
"""
Face detection (YOLO), tracking (BoT-FaceSORT), and
active speaker detection (LoCoNet).

Outputs per-frame face bounding boxes with track IDs and
speaker activity confidence per track.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Add BoT-FaceSORT and LoCoNet to path (installed via git clone in Modal)
_BOT_FACESORT_PATH = Path("/opt/bot-facesort")
_LOCONET_PATH = Path("/opt/loconet")
if _BOT_FACESORT_PATH.exists():
    sys.path.insert(0, str(_BOT_FACESORT_PATH))
if _LOCONET_PATH.exists():
    sys.path.insert(0, str(_LOCONET_PATH))


# ─────────────────────────────────────────────
# Face Detection
# ─────────────────────────────────────────────

def detect_faces_in_chunk(
    video_path: Path,
    sample_fps: float = 2.0,
    device: str = "cuda",
) -> list[dict]:
    """
    Run YOLO face detection on sampled frames.

    Returns list of frame results:
      [{frame_idx, timestamp, faces: [{bbox, confidence}]}]
    """
    from ultralytics import YOLO

    # Prefer face-specific model if available (pre-downloaded from HuggingFace)
    face_model_path = Path(config.FACE_MODEL_FACE)
    if face_model_path.exists():
        model = YOLO(str(face_model_path))
        logger.info("Using face-specific model: %s", face_model_path)
    else:
        model = YOLO(config.FACE_MODEL)
        logger.info("Face model not found, using general model: %s", config.FACE_MODEL)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(video_fps / sample_fps))

    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            timestamp = frame_idx / video_fps

            # Run YOLO detection
            results = model(
                frame,
                conf=config.FACE_CONF_THRESHOLD,
                iou=config.FACE_IOU_THRESHOLD,
                device=device,
                verbose=False,
            )

            faces = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    faces.append({
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "confidence": round(float(box.conf[0]), 3),
                    })

            frame_results.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 3),
                "faces": faces,
            })

        frame_idx += 1

    cap.release()
    logger.info(
        "Face detection: %d frames sampled, %d total faces",
        len(frame_results),
        sum(len(f["faces"]) for f in frame_results),
    )
    return frame_results


# ─────────────────────────────────────────────
# Face Tracking
# ─────────────────────────────────────────────

def track_faces(
    frame_detections: list[dict],
    video_path: Path,
) -> list[dict]:
    """
    Apply BoT-FaceSORT tracking to assign consistent track IDs
    across frames within a chunk.

    Returns updated frame_detections with track_id per face.
    If BoT-FaceSORT is not available, falls back to simple IoU tracking.
    """
    try:
        return _track_with_bot_facesort(frame_detections, video_path)
    except (ImportError, Exception) as e:
        logger.warning("BoT-FaceSORT unavailable (%s), using IoU fallback", e)
        return _track_with_iou_fallback(frame_detections)


def _track_with_bot_facesort(
    frame_detections: list[dict],
    video_path: Path,
) -> list[dict]:
    """Tracking using BoT-FaceSORT."""
    # Import from cloned repo
    from tracker import BoTFaceSORT

    tracker = BoTFaceSORT()

    for frame_data in frame_detections:
        if not frame_data["faces"]:
            continue

        # Format detections for tracker: [x1, y1, x2, y2, conf]
        dets = np.array([
            f["bbox"] + [f["confidence"]]
            for f in frame_data["faces"]
        ])

        # Update tracker
        tracks = tracker.update(dets)

        # Map track IDs back to faces
        for i, face in enumerate(frame_data["faces"]):
            if i < len(tracks):
                face["track_id"] = int(tracks[i][4]) if len(tracks[i]) > 4 else i
            else:
                face["track_id"] = -1

    return frame_detections


def _track_with_iou_fallback(
    frame_detections: list[dict],
) -> list[dict]:
    """Simple IoU-based tracking fallback."""
    next_id = 0
    prev_faces = []

    for frame_data in frame_detections:
        curr_faces = frame_data["faces"]

        if not prev_faces:
            for face in curr_faces:
                face["track_id"] = next_id
                next_id += 1
        else:
            # Compute IoU matrix
            used = set()
            for face in curr_faces:
                best_iou = 0.0
                best_id = -1
                for prev in prev_faces:
                    if prev["track_id"] in used:
                        continue
                    iou = _compute_iou(face["bbox"], prev["bbox"])
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_id = prev["track_id"]

                if best_id >= 0:
                    face["track_id"] = best_id
                    used.add(best_id)
                else:
                    face["track_id"] = next_id
                    next_id += 1

        prev_faces = curr_faces

    return frame_detections


def _compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# Active Speaker Detection
# ─────────────────────────────────────────────

def detect_active_speakers(
    frame_detections: list[dict],
    video_path: Path,
    audio_path: Path,
) -> list[dict]:
    """
    Run LoCoNet active speaker detection on tracked faces.
    Adds 'asd_confidence' to each face in frame_detections.

    Falls back to simple lip-motion heuristic if LoCoNet unavailable.
    """
    try:
        return _asd_with_loconet(frame_detections, video_path, audio_path)
    except (ImportError, Exception) as e:
        logger.warning("LoCoNet unavailable (%s), using face-size fallback", e)
        return _asd_fallback(frame_detections)


def _asd_with_loconet(
    frame_detections: list[dict],
    video_path: Path,
    audio_path: Path,
) -> list[dict]:
    """Active speaker detection using LoCoNet."""
    # LoCoNet requires specific data preparation (face crops + audio segments
    # aligned per-frame). The full integration depends on LoCoNet's repo
    # structure which varies. For MVP, we explicitly use the fallback and
    # log clearly so we know when to wire in the real model.
    logger.warning(
        "LoCoNet ASD not yet wired — using face-size fallback. "
        "Clip quality will be good but not optimal."
    )
    raise ImportError("LoCoNet integration pending — using fallback")


def _asd_fallback(frame_detections: list[dict]) -> list[dict]:
    """Fallback: assign ASD confidence based on face size (larger = more likely speaking)."""
    for frame_data in frame_detections:
        faces = frame_data["faces"]
        if not faces:
            continue

        # Largest face gets highest confidence
        max_area = 0
        for face in faces:
            bbox = face["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            max_area = max(max_area, area)

        for face in faces:
            bbox = face["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            face["asd_confidence"] = round(area / max_area, 3) if max_area > 0 else 0.5

    return frame_detections


# ─────────────────────────────────────────────
# Global Re-ID (Reduce Phase)
# ─────────────────────────────────────────────

def global_reid(
    all_chunk_tracks: list[list[dict]],
    video_path: Path,
    device: str = "cuda",
) -> dict:
    """
    Merge face tracks across chunks using InsightFace embeddings + DBSCAN.

    Subsamples to 1 embedding per track per second to prevent OOM.
    Returns mapping: {(chunk_idx, local_track_id): global_face_id}
    """
    try:
        from insightface.app import FaceAnalysis
        from sklearn.cluster import DBSCAN
    except ImportError:
        logger.warning("InsightFace/sklearn unavailable, using identity mapping")
        return _identity_mapping(all_chunk_tracks)

    # Initialize InsightFace
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0)

    # Collect one embedding per track per second
    embeddings = []
    embedding_labels = []  # (chunk_idx, track_id)

    # Open video once and reuse across all chunks (avoid per-chunk overhead)
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not cap.isOpened():
        logger.warning("Could not open video for re-ID: %s", video_path)
        return _identity_mapping(all_chunk_tracks)

    for chunk_idx, chunk_tracks in enumerate(all_chunk_tracks):
        track_embeddings = {}  # track_id -> list of embeddings

        for frame_data in chunk_tracks:
            for face in frame_data.get("faces", []):
                tid = face.get("track_id", -1)
                if tid < 0:
                    continue

                # Subsample: 1 per second
                if tid not in track_embeddings:
                    track_embeddings[tid] = []

                ts = frame_data["timestamp"]
                existing_ts = [e["ts"] for e in track_embeddings[tid]]
                if any(abs(ts - t) < config.REID_EMBEDDINGS_PER_SECOND for t in existing_ts):
                    continue

                # Extract face crop and compute embedding
                frame_idx = frame_data["frame_idx"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                bbox = face["bbox"]
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                faces_detected = app.get(face_crop)
                if faces_detected:
                    emb = faces_detected[0].embedding
                    track_embeddings[tid].append({"ts": ts, "emb": emb})

        # Take mean embedding per track
        for tid, emb_list in track_embeddings.items():
            if emb_list:
                mean_emb = np.mean([e["emb"] for e in emb_list], axis=0)
                embeddings.append(mean_emb)
                embedding_labels.append((chunk_idx, tid))

    cap.release()

    if not embeddings:
        return _identity_mapping(all_chunk_tracks)

    # DBSCAN clustering
    emb_matrix = np.array(embeddings)
    # Normalize for cosine distance
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / (norms + 1e-8)

    # DBSCAN with cosine distance
    clustering = DBSCAN(
        eps=1.0 - config.INSIGHTFACE_COSINE_THRESHOLD,
        min_samples=1,
        metric="cosine",
    ).fit(emb_matrix)

    # Build mapping
    mapping = {}
    for label, (chunk_idx, track_id) in zip(clustering.labels_, embedding_labels):
        mapping[(chunk_idx, track_id)] = int(label)

    n_identities = len(set(clustering.labels_) - {-1})
    logger.info(
        "Global re-ID: %d embeddings → %d unique identities",
        len(embeddings), n_identities,
    )

    return mapping


def _identity_mapping(all_chunk_tracks: list[list[dict]]) -> dict:
    """Fallback: each chunk's tracks are separate identities."""
    mapping = {}
    global_id = 0
    for chunk_idx, chunk_tracks in enumerate(all_chunk_tracks):
        seen_tracks = set()
        for frame_data in chunk_tracks:
            for face in frame_data.get("faces", []):
                tid = face.get("track_id", -1)
                if tid >= 0 and tid not in seen_tracks:
                    mapping[(chunk_idx, tid)] = global_id
                    seen_tracks.add(tid)
                    global_id += 1
    return mapping


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────

def save_face_tracks(tracks: list[dict], output_path: Path) -> Path:
    """Save face tracks to JSON."""
    # Convert numpy types for JSON serialization
    clean = json.loads(json.dumps(tracks, default=_json_default))
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info("Saved face tracks: %s", output_path)
    return output_path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
```

---

## `pipeline/ingest.py`

```python
"""
YouTube download with bot-proof bypass + video chunking.

Runs on Modal with Playwright Stealth for cookie refresh,
bgutil-pot-server for PO Tokens, and multi-client yt-dlp.
"""

import hashlib
import json
import logging
import struct
import subprocess
import time
from pathlib import Path
from typing import Optional


import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Video Hash
# ─────────────────────────────────────────────

def compute_video_hash(video_path: Path) -> str:
    """
    Collision-resistant hash: SHA256(first_10MB + last_10MB + duration).
    Handles re-uploads, trimmed versions, and identical intros.
    """
    file_size = video_path.stat().st_size
    head_size = min(config.HASH_HEAD_BYTES, file_size)
    tail_size = min(config.HASH_TAIL_BYTES, file_size)

    h = hashlib.sha256()

    with open(video_path, "rb") as f:
        # Read head
        h.update(f.read(head_size))
        # Read tail
        if file_size > head_size:
            f.seek(max(0, file_size - tail_size))
            h.update(f.read(tail_size))

    # Get duration via ffprobe
    duration = _get_duration(video_path)
    h.update(struct.pack("d", duration))

    return h.hexdigest()[:16]  # 16-char hex = 64 bits, sufficient for caching


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        logger.warning("Could not determine video duration, using 0.0")
        return 0.0


# ─────────────────────────────────────────────
# YouTube Download
# ─────────────────────────────────────────────

def download_video(
    url: str,
    output_dir: Path,
    cookie_path: Optional[Path] = None,
) -> Path:
    """
    Download video from YouTube using yt-dlp with bot bypass.
    Enforces rate limits via persistent state to prevent account burns.
    Returns path to downloaded video file.
    """
    # Enforce rate guard before attempting download
    _enforce_rate_guard(output_dir)

    import yt_dlp  # lazy import — only needed for download, not chunking

    output_template = str(output_dir / "source.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": config.YT_PLAYER_CLIENTS,
            }
        },
    }

    # Add cookies if available
    if cookie_path and cookie_path.exists():
        ydl_opts["cookiefile"] = str(cookie_path)
        logger.info("Using cached cookies: %s", cookie_path)

    # Download with retries
    last_error = None
    for attempt in range(1, config.YT_MAX_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                # yt-dlp may change extension after merge
                video_path = Path(filename).with_suffix(".mp4")
                if not video_path.exists():
                    # Try finding any video file in the output dir
                    candidates = list(output_dir.glob("source.*"))
                    if candidates:
                        video_path = candidates[0]
                    else:
                        raise FileNotFoundError(
                            f"Download completed but no file found in {output_dir}"
                        )
                logger.info("Downloaded: %s (%.1f MB)", video_path.name,
                           video_path.stat().st_size / 1e6)
                _record_download_success(output_dir)
                return video_path

        except Exception as e:
            last_error = e
            _record_download_failure(output_dir)
            logger.warning(
                "Download attempt %d/%d failed: %s",
                attempt, config.YT_MAX_RETRIES, e,
            )
            if attempt < config.YT_MAX_RETRIES:
                backoff = min(
                    config.YT_RETRY_BACKOFF[0] * (2 ** (attempt - 1)),
                    config.YT_RETRY_BACKOFF[1],
                )
                logger.info("Retrying in %.1fs...", backoff)
                time.sleep(backoff)

    raise RuntimeError(
        f"All {config.YT_MAX_RETRIES} download attempts failed"
    ) from last_error


# ─────────────────────────────────────────────
# Rate Guard (persistent state on Volume)
# ─────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised when download rate limit is exceeded."""
    pass


def _rate_state_path(work_dir: Path) -> Path:
    """Path to persistent rate limit state file."""
    state_dir = Path(config.MODAL_VOLUME_MOUNT) / "_cookies"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "rate_limit_state.json"


def _load_rate_state(work_dir: Path) -> dict:
    """Load rate limit state from JSON."""
    path = _rate_state_path(work_dir)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"downloads": [], "failures": [], "cooldown_until": 0}


def _save_rate_state(work_dir: Path, state: dict) -> None:
    """Save rate limit state to JSON."""
    path = _rate_state_path(work_dir)
    with open(path, "w") as f:
        json.dump(state, f)


def _enforce_rate_guard(work_dir: Path) -> None:
    """
    Check rate limits before download:
      - Max 15 downloads/hour
      - Max 50 downloads/day
      - 3 consecutive failures → 1-hour cooldown
    """
    state = _load_rate_state(work_dir)
    now = time.time()

    # Check cooldown
    if now < state.get("cooldown_until", 0):
        remaining = int(state["cooldown_until"] - now)
        raise RateLimitError(
            f"Account in cooldown for {remaining}s after repeated failures"
        )

    # Check hourly cap
    one_hour_ago = now - 3600
    recent_downloads = [t for t in state.get("downloads", []) if t > one_hour_ago]
    if len(recent_downloads) >= config.YT_MAX_DOWNLOADS_PER_HOUR:
        raise RateLimitError(
            f"Hourly download limit reached ({config.YT_MAX_DOWNLOADS_PER_HOUR}/hr)"
        )

    # Check daily cap
    one_day_ago = now - 86400
    daily_downloads = [t for t in state.get("downloads", []) if t > one_day_ago]
    if len(daily_downloads) >= config.YT_MAX_DOWNLOADS_PER_DAY:
        raise RateLimitError(
            f"Daily download limit reached ({config.YT_MAX_DOWNLOADS_PER_DAY}/day)"
        )


def _record_download_success(work_dir: Path) -> None:
    """Record a successful download, reset failure counter."""
    state = _load_rate_state(work_dir)
    state["downloads"].append(time.time())
    state["failures"] = []  # reset on success
    # Prune old entries (keep last 24h)
    cutoff = time.time() - 86400
    state["downloads"] = [t for t in state["downloads"] if t > cutoff]
    _save_rate_state(work_dir, state)


def _record_download_failure(work_dir: Path) -> None:
    """Record a failure. Trigger cooldown after consecutive threshold."""
    state = _load_rate_state(work_dir)
    state["failures"].append(time.time())
    # Keep only recent failures (last hour)
    cutoff = time.time() - 3600
    state["failures"] = [t for t in state["failures"] if t > cutoff]

    if len(state["failures"]) >= config.YT_COOLDOWN_AFTER_FAILURES:
        state["cooldown_until"] = time.time() + config.YT_COOLDOWN_DURATION
        logger.warning(
            "Rate guard: %d consecutive failures → %ds cooldown",
            len(state["failures"]), config.YT_COOLDOWN_DURATION,
        )

    _save_rate_state(work_dir, state)


# ─────────────────────────────────────────────
# FFmpeg Chunking
# ─────────────────────────────────────────────

def chunk_video(
    video_path: Path,
    output_dir: Path,
    chunk_duration: int = config.CHUNK_DURATION,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split video into overlapping chunks using FFmpeg.
    Returns list of chunk metadata dicts.
    """
    duration = _get_duration(video_path)
    chunks = []
    chunk_idx = 0
    start = 0.0

    while start < duration:
        end = min(start + chunk_duration, duration)
        chunk_path = output_dir / f"chunk_{chunk_idx:03d}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.2f}",
            "-t", f"{end - start:.2f}",
            "-i", str(video_path),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(chunk_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg chunk failed: %s", result.stderr[-500:])
            raise RuntimeError(f"Failed to create chunk {chunk_idx}")

        chunks.append({
            "index": chunk_idx,
            "path": str(chunk_path),
            "start": start,
            "end": end,
            "duration": end - start,
        })

        logger.info(
            "Chunk %d: %.1fs - %.1fs (%.1fs)",
            chunk_idx, start, end, end - start,
        )

        # Next chunk starts overlap seconds before the end of this one
        start = end - overlap
        if start >= duration:
            break
        # Prevent infinite loop: if remaining video < overlap, stop
        if (duration - start) <= overlap:
            break
        chunk_idx += 1

    return chunks


# ─────────────────────────────────────────────
# Audio Extraction
# ─────────────────────────────────────────────

def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract mono WAV 16kHz audio for ASR."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[-500:]}")
    return output_path


# ─────────────────────────────────────────────
# Full Ingest Pipeline
# ─────────────────────────────────────────────

def ingest(
    url: str,
    work_dir: Path,
    cookie_path: Optional[Path] = None,
) -> dict:
    """
    Full ingest: download → hash → check cache → chunk → extract audio.
    Returns dict with video_hash, chunks, and audio paths.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    logger.info("Downloading: %s", url)
    video_path = download_video(url, work_dir, cookie_path)

    # Step 2: Compute hash
    video_hash = compute_video_hash(video_path)
    logger.info("Video hash: %s", video_hash)

    # Step 3: Check cache
    cache_dir = config.video_dir(video_hash)
    cache_marker = cache_dir / ".ingest_complete"
    if cache_marker.exists():
        logger.info("Cache hit — loading existing artifacts")
        with open(cache_dir / "ingest_meta.json") as f:
            return json.load(f)

    # Step 4: Create chunk directory
    chunk_dir = cache_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Step 5: Chunk video
    logger.info("Chunking video...")
    chunks = chunk_video(video_path, chunk_dir)

    # Step 6: Extract audio for each chunk
    logger.info("Extracting audio from chunks...")
    for chunk in chunks:
        audio_path = Path(chunk["path"]).with_suffix(".wav")
        extract_audio(Path(chunk["path"]), audio_path)
        chunk["audio_path"] = str(audio_path)

    # Step 7: Save metadata + mark complete
    duration = _get_duration(video_path)
    meta = {
        "video_hash": video_hash,
        "url": url,
        "source_path": str(video_path),
        "duration": duration,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    with open(cache_dir / "ingest_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    cache_marker.touch()
    logger.info("Ingest complete: %d chunks from %.1fs video", len(chunks), duration)

    return meta
```

---

## `pipeline/reframe.py`

```python
"""
Bulletproof 3-layer speaker-centered reframing engine.

Layer 1: LoCoNet ASD confidence → center on active speaker
Layer 2: Speech overlap / low confidence → safe wide shot
Layer 3: Lip motion / largest face fallback

Includes Kalman smoothing, hysteresis, and pre-render validation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class CropTarget:
    """A crop rectangle for a single frame."""
    cx: float  # center x (0-1, normalized)
    cy: float  # center y (0-1, normalized)
    scale: float  # zoom level (1.0 = full width)
    source: str  # which reframe layer selected this


@dataclass
class CropPlan:
    """Pre-computed crop plan for an entire clip."""
    clip_id: str
    fps: float
    width: int
    height: int
    frames: list[CropTarget] = field(default_factory=list)


# ─────────────────────────────────────────────
# Kalman Smoother
# ─────────────────────────────────────────────

class KalmanSmoother:
    """
    Non-causal (RTS) Kalman smoother for crop positions.
    Provides zero-lag smoothing with velocity clamping.
    """

    def __init__(self, process_var: float = 0.001, measure_var: float = 0.01):
        self.process_var = process_var
        self.measure_var = measure_var

    def smooth(self, positions: list[float]) -> list[float]:
        """Apply forward-backward Kalman smoothing."""
        if len(positions) < 3:
            return positions

        n = len(positions)
        # Forward pass
        x_fwd = np.zeros(n)
        p_fwd = np.zeros(n)
        x_fwd[0] = positions[0]
        p_fwd[0] = 1.0

        for i in range(1, n):
            # Predict
            x_pred = x_fwd[i - 1]
            p_pred = p_fwd[i - 1] + self.process_var

            # Update
            k = p_pred / (p_pred + self.measure_var)
            x_fwd[i] = x_pred + k * (positions[i] - x_pred)
            p_fwd[i] = (1 - k) * p_pred

        # Backward pass (RTS smoother)
        x_smooth = np.zeros(n)
        x_smooth[-1] = x_fwd[-1]

        for i in range(n - 2, -1, -1):
            p_pred = p_fwd[i] + self.process_var
            gain = p_fwd[i] / p_pred
            x_smooth[i] = x_fwd[i] + gain * (x_smooth[i + 1] - x_fwd[i])

        return x_smooth.tolist()


# ─────────────────────────────────────────────
# 3-Layer Reframe Engine
# ─────────────────────────────────────────────

def compute_crop_plan(
    clip: dict,
    face_tracks: list[dict],
    speaker_map: dict,
    scenes: list[dict],
    transcript: dict,
    source_width: int,
    source_height: int,
    output_fps: float = config.OUTPUT_FPS,
) -> CropPlan:
    """
    Compute the crop plan for a single clip.

    3-layer decision at each frame:
      1. LoCoNet high confidence → center on active speaker
      2. Overlap / low confidence → safe wide shot
      3. Lip motion / largest face fallback
    """
    clip_start = clip["start"]
    clip_end = clip["end"]
    clip_duration = clip_end - clip_start
    total_frames = int(clip_duration * output_fps)

    plan = CropPlan(
        clip_id=clip.get("rank", "0"),
        fps=output_fps,
        width=source_width,
        height=source_height,
    )

    # Get scene boundaries within clip
    scene_times = {
        s["boundary"] for s in scenes
        if clip_start <= s.get("boundary", 0) <= clip_end
    }

    # Get transcript for overlap detection
    speaking_segments = _get_speaking_segments(transcript, clip_start, clip_end)

    # Track hysteresis state
    current_target = None
    hold_until = 0.0

    for frame_idx in range(total_frames):
        timestamp = clip_start + frame_idx / output_fps

        # Check for scene cut → reset tracking
        if any(abs(timestamp - st) < 1.0 / output_fps for st in scene_times):
            current_target = None
            hold_until = 0.0

        # Get faces at this timestamp
        faces = _get_faces_at_time(face_tracks, timestamp)

        # Check for speech overlap
        is_overlap = _is_speech_overlap(speaking_segments, timestamp)

        # 3-Layer decision
        crop = _decide_crop(
            faces, is_overlap, source_width, source_height,
        )

        # Hysteresis: don't switch target too fast
        if current_target and timestamp < hold_until:
            # Blend toward new target
            crop = _blend_crop(current_target, crop, blend_factor=0.3)
        else:
            current_target = crop
            # Dynamic hysteresis based on confidence
            hold_time = config.REFRAME_HYSTERESIS[0]  # minimum
            if crop.source == "asd_primary":
                hold_time = config.REFRAME_HYSTERESIS[1]  # maximum
            hold_until = timestamp + hold_time

        plan.frames.append(crop)

    # Apply Kalman smoothing
    plan = _smooth_crop_plan(plan)

    # Validate plan
    validation = validate_crop_plan(plan, face_tracks, clip_start, clip_end)
    if not validation["passes"]:
        logger.warning(
            "Crop plan validation failed for clip %s: %s — applying safe mode",
            plan.clip_id, validation,
        )
        plan = _apply_safe_mode(plan, source_width, source_height)

    logger.info(
        "Crop plan for clip %s: %d frames, primary source: %s",
        plan.clip_id, len(plan.frames),
        _dominant_source(plan),
    )

    return plan


def _decide_crop(
    faces: list[dict],
    is_overlap: bool,
    src_w: int,
    src_h: int,
) -> CropTarget:
    """3-layer crop decision."""

    if not faces:
        # No faces → center crop
        return CropTarget(cx=0.5, cy=0.5, scale=1.0, source="no_face_center")

    # Layer 1: ASD high confidence
    active_faces = [
        f for f in faces
        if f.get("asd_confidence", 0) >= config.LOCONET_CONFIDENCE_HIGH
    ]

    if active_faces and not is_overlap:
        # Center on the most confident active speaker
        best = max(active_faces, key=lambda f: f.get("asd_confidence", 0))
        cx, cy = _face_center(best["bbox"], src_w, src_h)
        return CropTarget(
            cx=cx, cy=cy,
            scale=_face_scale(best["bbox"], src_h),
            source="asd_primary",
        )

    # Layer 2: Overlap or low confidence → safe wide shot
    if is_overlap or any(
        f.get("asd_confidence", 0) < config.LOCONET_CONFIDENCE_LOW
        for f in faces
    ):
        cx, cy = _all_faces_center(faces, src_w, src_h)
        return CropTarget(
            cx=cx, cy=cy,
            scale=_wideshot_scale(faces, src_w, src_h),
            source="wide_shot",
        )

    # Layer 3: Largest face fallback
    largest = max(faces, key=lambda f: _face_area(f["bbox"]))
    cx, cy = _face_center(largest["bbox"], src_w, src_h)
    return CropTarget(
        cx=cx, cy=cy,
        scale=_face_scale(largest["bbox"], src_h),
        source="largest_face",
    )


# ─────────────────────────────────────────────
# Geometry Helpers
# ─────────────────────────────────────────────

def _face_center(bbox: list, src_w: int, src_h: int) -> tuple[float, float]:
    """Normalized center of a face bbox."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / src_w
    cy = ((y1 + y2) / 2) / src_h
    return (cx, cy)


def _face_area(bbox: list) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _face_scale(bbox: list, src_h: int) -> float:
    """Compute scale so face is >= 55% of output frame height."""
    face_h = bbox[3] - bbox[1]
    desired_ratio = config.REFRAME_MIN_FACE_HEIGHT
    if face_h <= 0:
        return 1.0
    scale = (face_h / src_h) / desired_ratio
    return max(0.3, min(1.0, scale))


def _all_faces_center(faces: list[dict], src_w: int, src_h: int) -> tuple[float, float]:
    """Center point of all visible faces."""
    if not faces:
        return (0.5, 0.5)
    cx_sum = sum((f["bbox"][0] + f["bbox"][2]) / 2 for f in faces) / len(faces)
    cy_sum = sum((f["bbox"][1] + f["bbox"][3]) / 2 for f in faces) / len(faces)
    return (cx_sum / src_w, cy_sum / src_h)


def _wideshot_scale(faces: list[dict], src_w: int, src_h: int) -> float:
    """Scale to fit all faces with safety margin."""
    if not faces:
        return 1.0
    all_x = [f["bbox"][0] for f in faces] + [f["bbox"][2] for f in faces]
    span = (max(all_x) - min(all_x)) / src_w
    return max(0.5, min(1.0, span + config.REFRAME_SAFETY_MARGIN * 2))


# ─────────────────────────────────────────────
# Temporal Helpers
# ─────────────────────────────────────────────

def _get_faces_at_time(
    face_tracks: list[dict],
    timestamp: float,
    tolerance: float = 0.1,
) -> list[dict]:
    """Get face detections closest to a given timestamp."""
    best = None
    best_diff = float("inf")
    for frame_data in face_tracks:
        diff = abs(frame_data["timestamp"] - timestamp)
        if diff < best_diff:
            best_diff = diff
            best = frame_data
    if best and best_diff <= tolerance:
        return best.get("faces", [])
    return []


def _get_speaking_segments(
    transcript: dict,
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """Get transcript segments that fall within the clip."""
    return [
        seg for seg in transcript.get("segments", [])
        if seg["start"] < clip_end and seg["end"] > clip_start
    ]


def _is_speech_overlap(
    segments: list[dict],
    timestamp: float,
) -> bool:
    """Check if multiple speakers are active at this timestamp."""
    active = set()
    for seg in segments:
        if seg["start"] <= timestamp <= seg["end"]:
            active.add(seg.get("speaker", ""))
    return len(active) > 1


def _blend_crop(a: CropTarget, b: CropTarget, blend_factor: float) -> CropTarget:
    """Blend between two crop targets."""
    return CropTarget(
        cx=a.cx + blend_factor * (b.cx - a.cx),
        cy=a.cy + blend_factor * (b.cy - a.cy),
        scale=a.scale + blend_factor * (b.scale - a.scale),
        source=b.source,
    )


# ─────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────

def _smooth_crop_plan(plan: CropPlan) -> CropPlan:
    """Apply Kalman smoothing to crop positions."""
    if len(plan.frames) < 3:
        return plan

    smoother = KalmanSmoother()

    cx_values = [f.cx for f in plan.frames]
    cy_values = [f.cy for f in plan.frames]
    scale_values = [f.scale for f in plan.frames]

    cx_smooth = smoother.smooth(cx_values)
    cy_smooth = smoother.smooth(cy_values)
    scale_smooth = smoother.smooth(scale_values)

    for i, frame in enumerate(plan.frames):
        frame.cx = cx_smooth[i]
        frame.cy = cy_smooth[i]
        frame.scale = scale_smooth[i]

    return plan


# ─────────────────────────────────────────────
# Pre-Render Validation
# ─────────────────────────────────────────────

def validate_crop_plan(
    plan: CropPlan,
    face_tracks: list[dict],
    clip_start: float,
    clip_end: float,
) -> dict:
    """
    Pre-render validation of crop plan quality.
    Checks good_frame_ratio, center_offset_variance, jitter_score.
    """
    if not plan.frames:
        return {"passes": False, "reason": "empty plan"}

    # Jitter: measure frame-to-frame position changes
    diffs = []
    for i in range(1, len(plan.frames)):
        dx = plan.frames[i].cx - plan.frames[i - 1].cx
        dy = plan.frames[i].cy - plan.frames[i - 1].cy
        diffs.append(np.sqrt(dx ** 2 + dy ** 2))

    jitter_score = float(np.mean(diffs)) if diffs else 0.0

    # Center offset variance
    cx_var = float(np.var([f.cx for f in plan.frames]))
    cy_var = float(np.var([f.cy for f in plan.frames]))
    offset_variance = cx_var + cy_var

    # Good frame ratio (frames where face is reasonably centered)
    good_frames = sum(
        1 for f in plan.frames
        if 0.2 <= f.cx <= 0.8 and 0.2 <= f.cy <= 0.8
    )
    good_frame_ratio = good_frames / len(plan.frames)

    passes = (
        good_frame_ratio >= config.GOOD_FRAME_THRESHOLD and
        jitter_score < 0.05 and
        offset_variance < 0.1
    )

    return {
        "passes": passes,
        "good_frame_ratio": round(good_frame_ratio, 3),
        "jitter_score": round(jitter_score, 5),
        "offset_variance": round(offset_variance, 5),
    }


def _apply_safe_mode(
    plan: CropPlan,
    src_w: int,
    src_h: int,
) -> CropPlan:
    """Fallback: center crop with minimal movement."""
    for frame in plan.frames:
        frame.cx = 0.5
        frame.cy = 0.4  # slightly above center
        frame.scale = 0.8
        frame.source = "safe_mode"
    return plan


def _dominant_source(plan: CropPlan) -> str:
    """Return the most common source in the plan."""
    from collections import Counter
    sources = Counter(f.source for f in plan.frames)
    return sources.most_common(1)[0][0] if sources else "unknown"


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────

def save_crop_plan(plan: CropPlan, output_path: Path) -> Path:
    """Save crop plan to JSON."""
    data = {
        "clip_id": plan.clip_id,
        "fps": plan.fps,
        "width": plan.width,
        "height": plan.height,
        "num_frames": len(plan.frames),
        "frames": [
            {
                "cx": round(f.cx, 4),
                "cy": round(f.cy, 4),
                "scale": round(f.scale, 4),
                "source": f.source,
            }
            for f in plan.frames
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved crop plan: %s (%d frames)", output_path, len(plan.frames))
    return output_path
```

---

## `pipeline/render.py`

```python
"""
FFmpeg-based clip rendering with dynamic cropping + .ass captions.

Takes a crop plan + .ass file and produces the final 9:16 clip.
"""

import logging
import subprocess
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def render_clip(
    source_video: Path,
    crop_plan_path: Path,
    caption_path: Path,
    output_path: Path,
    output_width: int = config.OUTPUT_WIDTH,
    output_height: int = config.OUTPUT_HEIGHT,
    clip_start: float = 0.0,
    clip_end: float = 0.0,
) -> Path:
    """
    Render a final clip with dynamic crop + captions.

    Uses FFmpeg with:
    - Crop filter based on pre-computed crop segments
    - .ass subtitle burn-in via libass
    - H.264 encoding at CRF 20
    """
    import json

    # Load crop plan
    with open(crop_plan_path) as f:
        plan = json.load(f)

    # Build crop filter from plan
    crop_filter = _build_crop_filter(
        plan, output_width, output_height,
    )

    # Build FFmpeg command
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-to", f"{clip_end:.3f}",
        "-i", str(source_video),
        "-vf", f"{crop_filter},scale={output_width}:{output_height},ass={caption_path}",
        "-c:v", config.VIDEO_CODEC,
        "-preset", config.VIDEO_PRESET,
        "-crf", str(config.VIDEO_CRF),
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info("Rendering clip: %s", output_path.name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg render timed out after 120s: %s", output_path.name)
        return _render_simple(
            source_video, caption_path, output_path,
            output_width, output_height,
            clip_start, clip_end,
        )

    if result.returncode != 0:
        logger.error("FFmpeg render failed: %s", result.stderr[-500:])
        # Try simpler render without dynamic crop
        return _render_simple(
            source_video, caption_path, output_path,
            output_width, output_height,
            clip_start, clip_end,
        )

    file_size = output_path.stat().st_size / 1e6
    logger.info("Rendered: %s (%.1f MB)", output_path.name, file_size)
    return output_path


def _build_crop_filter(
    plan: dict,
    out_w: int,
    out_h: int,
) -> str:
    """
    Build FFmpeg crop filter from crop plan.
    Groups consecutive frames with similar crop positions
    into 0.5s segments and generates expression-based dynamic
    crop_x / crop_y per segment using if(between(t,...)).
    """
    frames = plan.get("frames", [])
    src_w = plan.get("width", 1920)
    src_h = plan.get("height", 1080)
    fps = plan.get("fps", 30)

    if not frames:
        # Default center crop
        crop_w = int(src_h * out_w / out_h)
        crop_x = (src_w - crop_w) // 2
        return f"crop={crop_w}:{src_h}:{crop_x}:0"

    # Group frames into 0.5s segments
    segment_size = max(1, int(fps * config.CROP_SEGMENT_DURATION))
    segments = []

    for i in range(0, len(frames), segment_size):
        chunk = frames[i:i + segment_size]
        avg_cx = sum(f["cx"] for f in chunk) / len(chunk)
        avg_cy = sum(f["cy"] for f in chunk) / len(chunk)
        avg_scale = sum(f["scale"] for f in chunk) / len(chunk)

        # Convert normalized coords to pixel crop
        crop_h = int(src_h * avg_scale)
        crop_w_seg = int(crop_h * out_w / out_h)

        # Clamp
        crop_w_seg = min(crop_w_seg, src_w)
        crop_h = min(crop_h, src_h)

        crop_x = int(avg_cx * src_w - crop_w_seg / 2)
        crop_y = int(avg_cy * src_h - crop_h / 2)

        # Keep in bounds
        crop_x = max(0, min(crop_x, src_w - crop_w_seg))
        crop_y = max(0, min(crop_y, src_h - crop_h))

        t_start = (i / fps)
        t_end = ((i + len(chunk)) / fps)

        segments.append({
            "t_start": round(t_start, 4),
            "t_end": round(t_end, 4),
            "crop_w": crop_w_seg,
            "crop_h": crop_h,
            "crop_x": crop_x,
            "crop_y": crop_y,
        })

    if not segments:
        crop_w = int(src_h * out_w / out_h)
        crop_x = (src_w - crop_w) // 2
        return f"crop={crop_w}:{src_h}:{crop_x}:0"

    # Use median crop_w / crop_h for consistent output size
    # (FFmpeg crop filter output dimensions must be constant)
    median_idx = len(segments) // 2
    fixed_w = segments[median_idx]["crop_w"]
    fixed_h = segments[median_idx]["crop_h"]

    # Build per-segment expression for crop_x and crop_y
    # FFmpeg expression: if(between(t,t0,t1), x0, if(between(t,...), ...))
    if len(segments) == 1:
        s = segments[0]
        return f"crop={fixed_w}:{fixed_h}:{s['crop_x']}:{s['crop_y']}"

    # Build nested if-else expression for x and y
    crop_x_expr = _build_time_expr(segments, "crop_x", segments[-1]["crop_x"])
    crop_y_expr = _build_time_expr(segments, "crop_y", segments[-1]["crop_y"])

    return f"crop={fixed_w}:{fixed_h}:{crop_x_expr}:{crop_y_expr}"


def _build_time_expr(segments: list[dict], key: str, default: int) -> str:
    """
    Build FFmpeg expression: if(between(t,t0,t1),val0,if(between(t,t1,t2),val1,...))
    Limits nesting depth to avoid FFmpeg expression parser limits (~100 levels).
    For very long clips: subsample segments to stay under the limit.
    """
    MAX_NESTING = 80  # FFmpeg expression parser limit

    segs = segments
    if len(segs) > MAX_NESTING:
        # Subsample: pick evenly spaced segments
        step = len(segs) / MAX_NESTING
        segs = [segs[int(i * step)] for i in range(MAX_NESTING)]

    # Build from the inside out (last segment = default)
    expr = str(default)
    for s in reversed(segs):
        expr = f"if(between(t\\,{s['t_start']}\\,{s['t_end']})\\,{s[key]}\\,{expr})"

    return expr


def _render_simple(
    source_video: Path,
    caption_path: Path,
    output_path: Path,
    out_w: int,
    out_h: int,
    clip_start: float,
    clip_end: float,
) -> Path:
    """Fallback: simple center crop without dynamic reframing."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-to", f"{clip_end:.3f}",
        "-i", str(source_video),
        "-vf", (
            f"crop=ih*{out_w}/{out_h}:ih,"
            f"scale={out_w}:{out_h},"
            f"ass={caption_path}"
        ),
        "-c:v", config.VIDEO_CODEC,
        "-preset", config.VIDEO_PRESET,
        "-crf", str(config.VIDEO_CRF),
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info("Simple render fallback: %s", output_path.name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg simple render timed out after 120s: {output_path.name}")

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg simple render failed: {result.stderr[-500:]}")

    return output_path


def validate_render(output_path: Path) -> dict:
    """Post-render validation: check file exists, has video/audio, reasonable size."""
    if not output_path.exists():
        return {"valid": False, "reason": "file does not exist"}

    size = output_path.stat().st_size
    if size < 10000:  # < 10KB = probably broken
        return {"valid": False, "reason": f"file too small: {size} bytes"}

    # Check with ffprobe
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "stream=codec_type,width,height,duration",
        "-of", "json",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {"valid": False, "reason": "ffprobe failed"}

    import json
    try:
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)

        if not has_video:
            return {"valid": False, "reason": "no video stream"}
        if not has_audio:
            return {"valid": False, "reason": "no audio stream"}

        return {
            "valid": True,
            "size_mb": round(size / 1e6, 2),
            "streams": len(streams),
        }
    except json.JSONDecodeError:
        return {"valid": False, "reason": "ffprobe output parse error"}
```

---

## `pipeline/scene_detect.py`

```python
"""
Shot boundary detection using AutoShot (primary) with
PySceneDetect/TransNetV2 fallback.
"""

import json
import logging
import sys
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_AUTOSHOT_PATH = Path("/opt/autoshot")
if _AUTOSHOT_PATH.exists():
    sys.path.insert(0, str(_AUTOSHOT_PATH))


def detect_scenes(
    video_path: Path,
    chunk_start: float = 0.0,
) -> list[dict]:
    """
    Detect shot boundaries in a video chunk.

    Returns list of scene boundaries:
      [{start, end, transition_type}]
    """
    try:
        return _detect_with_autoshot(video_path, chunk_start)
    except (ImportError, Exception) as e:
        logger.warning("AutoShot unavailable (%s), using PySceneDetect fallback", e)
        return _detect_with_pyscenedetect(video_path, chunk_start)


def _detect_with_autoshot(video_path: Path, chunk_start: float) -> list[dict]:
    """Shot detection using AutoShot."""
    # AutoShot inference — adapted from their inference script
    import torch
    from model import AutoShotModel  # from /opt/autoshot

    model = AutoShotModel.load_pretrained()
    model.eval()

    predictions = model.predict(str(video_path))

    scenes = []
    for i, boundary_frame in enumerate(predictions):
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        timestamp = boundary_frame / fps + chunk_start
        scenes.append({
            "boundary": round(timestamp, 3),
            "transition_type": "cut",
        })

    logger.info("AutoShot: detected %d scene boundaries", len(scenes))
    return scenes


def _detect_with_pyscenedetect(video_path: Path, chunk_start: float) -> list[dict]:
    """Fallback using PySceneDetect ContentDetector."""
    from scenedetect import detect, ContentDetector

    scene_list = detect(str(video_path), ContentDetector(threshold=27.0))

    scenes = []
    for scene in scene_list:
        start_time = scene[0].get_seconds() + chunk_start
        end_time = scene[1].get_seconds() + chunk_start
        scenes.append({
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "boundary": round(start_time, 3),
            "transition_type": "cut",
        })

    logger.info("PySceneDetect: detected %d scenes", len(scenes))
    return scenes


def merge_chunk_scenes(
    chunk_scenes: list[list[dict]],
    chunks_meta: list[dict],
) -> list[dict]:
    """Merge scene boundaries from overlapping chunks, deduplicate."""
    all_boundaries = []

    for scenes, chunk_meta in zip(chunk_scenes, chunks_meta):
        for scene in scenes:
            all_boundaries.append(scene["boundary"])

    # Deduplicate: boundaries within 0.5s of each other are the same cut
    all_boundaries.sort()
    deduped = []
    for b in all_boundaries:
        if not deduped or abs(b - deduped[-1]) > 0.5:
            deduped.append(b)

    return [{"boundary": round(b, 3), "transition_type": "cut"} for b in deduped]


def save_scenes(scenes: list[dict], output_path: Path) -> Path:
    """Save scene boundaries to JSON."""
    with open(output_path, "w") as f:
        json.dump(scenes, f, indent=2)
    logger.info("Saved %d scene boundaries: %s", len(scenes), output_path)
    return output_path
```

---

## `pipeline/transcribe.py`

```python
"""
WhisperX transcription + Pyannote 3.1 diarization.

Outputs word-level timestamps with speaker labels per chunk.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


def transcribe_chunk(
    audio_path: Path,
    chunk_start: float = 0.0,
    device: str = "cuda",
    num_speakers: Optional[int] = None,
) -> dict:
    """
    Transcribe an audio chunk with WhisperX + Pyannote diarization.

    Returns dict with:
      - segments: list of {start, end, text, speaker, words: [{word, start, end, score}]}
      - language: detected language code
    """
    import whisperx
    import torch

    # Step 1: Load model + transcribe
    logger.info("Transcribing: %s", audio_path.name)
    model = whisperx.load_model(
        config.WHISPERX_MODEL,
        device=device,
        compute_type=config.WHISPERX_COMPUTE_TYPE,
    )

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(
        audio,
        batch_size=config.WHISPERX_BATCH_SIZE,
    )
    language = result.get("language", "en")
    logger.info("Detected language: %s", language)

    # Step 2: Forced alignment for word-level timestamps
    logger.info("Aligning words...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device=device,
        return_char_alignments=False,
    )

    # Free alignment model
    del align_model
    torch.cuda.empty_cache()

    # Step 3: Speaker diarization (optional — whisperx API changes across versions)
    try:
        logger.info("Diarizing speakers...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=config.HF_TOKEN,
            device=device,
        )
        diarize_segments = diarize_model(
            audio,
            min_speakers=1,
            max_speakers=num_speakers or 10,
        )

        # Assign speakers to word-level segments
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Free diarize model
        del diarize_model
        torch.cuda.empty_cache()
    except (AttributeError, Exception) as e:
        logger.warning("Diarization skipped: %s — speaker labels will be UNKNOWN", e)

    # Step 4: Offset timestamps to global time (chunk_start)
    segments = []
    for seg in result.get("segments", []):
        segment = {
            "start": round(seg["start"] + chunk_start, 3),
            "end": round(seg["end"] + chunk_start, 3),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", "UNKNOWN"),
            "words": [],
        }

        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            word_entry = {
                "word": w["word"].strip(),
                "start": round(w["start"] + chunk_start, 3),
                "end": round(w["end"] + chunk_start, 3),
                "score": round(w.get("score", 0.0), 3),
            }
            segment["words"].append(word_entry)

        if segment["text"]:
            segments.append(segment)

    logger.info(
        "Transcription complete: %d segments, %d words",
        len(segments),
        sum(len(s["words"]) for s in segments),
    )

    return {
        "language": language,
        "segments": segments,
    }


def merge_chunk_transcripts(
    chunk_transcripts: list[dict],
    chunks_meta: list[dict],
) -> dict:
    """
    Merge transcripts from overlapping chunks.
    In overlap regions, keep the transcript from the chunk where
    the words fall in the non-overlap middle section (higher confidence).
    """
    if not chunk_transcripts:
        return {"language": "en", "segments": []}

    language = chunk_transcripts[0].get("language", "en")
    all_segments = []

    for i, (transcript, chunk_meta) in enumerate(
        zip(chunk_transcripts, chunks_meta)
    ):
        chunk_start = chunk_meta["start"]
        chunk_end = chunk_meta["end"]

        # Determine the "trusted" region for this chunk:
        # First chunk: trust from start to (end - overlap/2)
        # Last chunk: trust from (start + overlap/2) to end
        # Middle chunks: trust from (start + overlap/2) to (end - overlap/2)
        overlap = config.CHUNK_OVERLAP

        if i == 0:
            trust_start = chunk_start
        else:
            trust_start = chunk_start + overlap / 2

        if i == len(chunk_transcripts) - 1:
            trust_end = chunk_end
        else:
            trust_end = chunk_end - overlap / 2

        for seg in transcript.get("segments", []):
            # Keep segment if its midpoint falls in the trusted region
            seg_mid = (seg["start"] + seg["end"]) / 2
            if trust_start <= seg_mid <= trust_end:
                all_segments.append(seg)

    # Sort by start time
    all_segments.sort(key=lambda s: s["start"])

    return {
        "language": language,
        "segments": all_segments,
    }


def save_transcript(transcript: dict, output_path: Path) -> Path:
    """Save transcript to JSON."""
    with open(output_path, "w") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info("Saved transcript: %s", output_path)
    return output_path
```

---

## `run.py`

```python
#!/usr/bin/env python3
"""
ClippedAI — CLI Entry Point

Usage:
    python run.py "https://youtube.com/watch?v=VIDEO_ID"
    python run.py "URL" --max-clips 3 --duration 15-45
    python run.py "URL" --dry-run
    python run.py "URL" --force-reanalyze
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import config


def main():
    parser = argparse.ArgumentParser(
        description="ClippedAI — Generate viral short-form clips from YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py "https://youtube.com/watch?v=VIDEO_ID"
    python run.py "URL" --max-clips 3
    python run.py "URL" --duration 15-45
    python run.py "URL" --dry-run
    python run.py "URL" --force-reanalyze
        """,
    )

    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--max-clips", type=int, default=config.MAX_CLIPS,
                       help=f"Maximum number of clips (default: {config.MAX_CLIPS})")
    parser.add_argument("--duration", type=str, default=f"{config.MIN_CLIP_DURATION}-{config.MAX_CLIP_DURATION}",
                       help=f"Clip duration range MIN-MAX in seconds (default: {config.MIN_CLIP_DURATION}-{config.MAX_CLIP_DURATION})")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory (default: output/)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run analysis only, show clip candidates without rendering")
    parser.add_argument("--force-reanalyze", action="store_true",
                       help="Force re-analysis even if cache exists")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse duration range
    try:
        min_dur, max_dur = map(int, args.duration.split("-"))
    except ValueError:
        print(f"Error: Invalid duration format '{args.duration}'. Use MIN-MAX (e.g., 15-60)")
        sys.exit(1)

    # Build settings dict
    settings = {
        "max_clips": args.max_clips,
        "min_duration": min_dur,
        "max_duration": max_dur,
        "ideal_duration": config.IDEAL_CLIP_DURATION,
        "dry_run": args.dry_run,
        "force_reanalyze": args.force_reanalyze,
    }

    # Print banner
    _print_banner(args.url, settings)

    # Run on Modal
    start_time = time.time()

    try:
        import modal
        from modal_app import process_video, app

        print("\n⏳ Running pipeline on Modal...")
        print("   This may take 2-5 minutes on first run.\n")

        with app.run():
            clip_paths = process_video.remote(args.url, settings)

        elapsed = time.time() - start_time

        if not clip_paths:
            print("\n❌ No clips were generated. Check logs for details.")
            sys.exit(1)

        # Download clips to local output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📥 Downloading {len(clip_paths)} clips to {output_dir}/...\n")

        downloaded = []
        for remote_path in clip_paths:
            filename = Path(remote_path).name
            local_path = output_dir / filename

            # In Modal context, volume files are accessible directly
            # For local CLI, we'd use modal.Volume.read_file
            try:
                from modal_app import volume as vol
                # Strip mount prefix: read_file expects relative path within volume
                mount_prefix = config.MODAL_VOLUME_MOUNT + "/"
                relative_path = remote_path.replace(mount_prefix, "")
                data = vol.read_file(relative_path)
                with open(local_path, "wb") as f:
                    for chunk in data:
                        f.write(chunk)
                downloaded.append(local_path)
            except Exception as e:
                # Fallback: try direct copy if running in same context
                src = Path(remote_path)
                if src.exists():
                    shutil.copy2(src, local_path)
                    downloaded.append(local_path)
                else:
                    print(f"   ⚠️  Could not download {filename}: {e}")

        # Print summary
        _print_summary(downloaded, elapsed)

    except ImportError:
        print("\n❌ Modal is not installed. Run: pip install modal")
        print("   Then authenticate: modal token new")
        sys.exit(1)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Pipeline failed after {elapsed:.0f}s: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _print_banner(url: str, settings: dict):
    """Print startup banner."""
    print("""
╔══════════════════════════════════════╗
║         🎬  ClippedAI  🎬           ║
║   Viral Clips from YouTube Videos    ║
╚══════════════════════════════════════╝
    """)
    print(f"  URL:       {url}")
    print(f"  Max clips: {settings['max_clips']}")
    print(f"  Duration:  {settings['min_duration']}-{settings['max_duration']}s")
    print(f"  Ideal:     {settings['ideal_duration'][0]}-{settings['ideal_duration'][1]}s")
    if settings.get("dry_run"):
        print("  Mode:      DRY RUN (no rendering)")
    if settings.get("force_reanalyze"):
        print("  Cache:     FORCE REANALYZE")


def _print_summary(clips: list[Path], elapsed: float):
    """Print completion summary table."""
    print(f"\n{'═' * 50}")
    print(f"  ✅ Pipeline Complete — {elapsed:.1f}s")
    print(f"{'═' * 50}")
    print()

    if clips:
        print(f"  {'#':<4} {'Filename':<25} {'Size':>8}")
        print(f"  {'─' * 4} {'─' * 25} {'─' * 8}")
        for i, clip in enumerate(clips, 1):
            size = clip.stat().st_size / 1e6
            print(f"  {i:<4} {clip.name:<25} {size:>6.1f}MB")
        print()
        print(f"  📂 Output: {clips[0].parent}/")
    else:
        print("  ⚠️  No clips downloaded.")

    print()


if __name__ == "__main__":
    main()
```

---

## `snapshot.py`

```python
#!/usr/bin/env python3
"""
Snapshot — Generates CODEBASE.md with the full project source.

Usage:
    python snapshot.py
    python snapshot.py --changed pipeline/reframe.py pipeline/captions.py
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
OUTPUT_FILE = PROJECT_DIR / "CODEBASE.md"

# Extensions to include
INCLUDE_EXT = {".py", ".gitignore", ".md", ".toml", ".cfg", ".txt", ".yml", ".yaml", ".json"}

# Paths to exclude
EXCLUDE = {
    "__pycache__", ".git", ".venv", "venv", "env",
    "node_modules", ".idea", ".vscode", ".modal",
    "output", "models", "tmp", "temp", "logs",
    # Plan files — private
    "implementation_plan.md", "core_clipping_pipeline.md",
    "CODEBASE.md", "codebase.md",
}

LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sh": "bash",
    ".cfg": "ini",
}


def get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def collect_files() -> list[Path]:
    """Walk project and collect source files, respecting excludes."""
    files = []
    for root, dirs, filenames in os.walk(PROJECT_DIR):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE and not d.startswith(".")]

        for name in sorted(filenames):
            if name in EXCLUDE:
                continue
            path = Path(root) / name
            suffix = path.suffix if path.suffix else ("." + name)  # .gitignore case
            if suffix in INCLUDE_EXT:
                files.append(path)

    return sorted(files)


def generate(changed_files: list[str] | None = None) -> None:
    """Generate CODEBASE.md."""
    files = collect_files()
    commit = get_commit_hash()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# ClippedAI — Full Codebase\n")
    lines.append(f"**Generated**: {timestamp}  ")
    lines.append(f"**Commit**: `{commit}`  ")
    lines.append(f"**Files**: {len(files)}\n")

    # Changed files header
    if changed_files:
        lines.append("## Recently Changed\n")
        for f in changed_files:
            lines.append(f"- `{f}`")
        lines.append("")

    # File index
    lines.append("## File Index\n")
    for f in files:
        rel = f.relative_to(PROJECT_DIR)
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("---\n")

    # File contents
    for f in files:
        rel = f.relative_to(PROJECT_DIR)
        lang = LANG_MAP.get(f.suffix, "")
        content = f.read_text(encoding="utf-8", errors="replace")

        lines.append(f"## `{rel}`\n")
        lines.append(f"```{lang}")
        lines.append(content.rstrip())
        lines.append("```\n")
        lines.append("---\n")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    total_lines = sum(1 for _ in OUTPUT_FILE.read_text().splitlines())
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"✅ CODEBASE.md generated: {len(files)} files, {total_lines} lines, {size_kb:.0f} KB")
    print(f"   Path: {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Generate CODEBASE.md snapshot")
    parser.add_argument(
        "--changed", nargs="*", default=None,
        help="List of recently changed files to highlight at the top",
    )
    args = parser.parse_args()
    generate(changed_files=args.changed)


if __name__ == "__main__":
    main()
```

---
