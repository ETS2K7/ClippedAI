"""
ClippedAI — Modal App Definition (Milestone 3: Speaker Framing)

Micro-container architecture:
- image_ingest: FFmpeg only (dev mode — video uploaded from Mac)
- image_asr: WhisperX + Pyannote 4.0 (GPU: A10G)
- image_vision: YOLO26 + BoT-FaceSORT + LoCoNet + InsightFace (GPU: T4)

Later milestones will add:
- image_scene: AutoShot
- image_audio: BEATs
"""
import json
import os
from pathlib import Path

import modal

from config import (
    APP_NAME,
    VOLUME_NAME,
    VOLUME_MOUNT,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    HF_SECRET_NAME,
    GPU_TYPE_ASR,
    GPU_TYPE_VISION,
    FACE_CONF_THRESHOLD,
    FACE_NMS_THRESHOLD,
    REID_COSINE_THRESHOLD,
    BOTFACESORT_COMMIT,
    LOCONET_COMMIT,
)

# --- Modal App ---
app = modal.App(APP_NAME)

# --- Volume ---
volume = modal.Volume.from_name(VOLUME_NAME)

# --- Secrets ---
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

# --- Images ---
image_ingest = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("modal", "numpy")
    .add_local_file("config.py", "/root/config.py")
    .add_local_dir("pipeline", "/root/pipeline")
)

image_asr = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch", "torchaudio",
        "torchcodec",
        "soundfile",
        "pandas",
        "whisperx @ git+https://github.com/m-bain/whisperX.git",
        "pyannote.audio>=3.1",
    )
    .add_local_file("config.py", "/root/config.py")
    .add_local_dir("pipeline", "/root/pipeline")
)

image_vision = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics",
        "insightface",
        "onnxruntime-gpu",
        "filterpy",
        "mediapipe",
        "opencv-python-headless",
        "numpy", "setuptools", "scikit-learn",
        "torch", "torchvision",
        "lapx",  # pre-built wheel replacement for 'lap' (BoT-FaceSORT dep)
    )
    .run_commands(
        f"git clone https://github.com/SJTUwxz/LoCoNet_ASD.git /opt/loconet "
        f"&& cd /opt/loconet && git checkout {LOCONET_COMMIT}",
    )
    .run_commands(
        f"git clone https://github.com/bellhyeon/BoT-FaceSORT.git /opt/bot-facesort "
        f"&& cd /opt/bot-facesort && git checkout {BOTFACESORT_COMMIT} "
        # Replace lap with lapx (lap has no pre-built wheel)
        f"&& sed -i 's/^lap==.*/lapx/' requirements.txt "
        # Remove conflicting version-pinned packages already installed
        f"&& sed -i '/^torch==/d' requirements.txt "
        f"&& sed -i '/^torchvision==/d' requirements.txt "
        f"&& sed -i '/^torchaudio==/d' requirements.txt "
        f"&& sed -i '/^numpy==/d' requirements.txt "
        f"&& sed -i '/^onnxruntime-gpu==/d' requirements.txt "
        f"&& sed -i '/^triton==/d' requirements.txt "
        f"&& sed -i '/^opencv-python==/d' requirements.txt "
        f"&& sed -i '/^--extra-index-url/d' requirements.txt "
        f"&& pip install -r requirements.txt",
    )
    .add_local_file("config.py", "/root/config.py")
    .add_local_dir("pipeline", "/root/pipeline")
)


# ============================================================
# CHUNKING
# ============================================================

@app.function(
    image=image_ingest,
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
def chunk_video(video_hash: str) -> list[dict]:
    """Chunk a video on the volume into 5-min segments with 30s overlaps."""
    import sys
    sys.path.insert(0, "/root")
    from pipeline.ingest import chunk_video as _chunk_video

    video_path = f"{VOLUME_MOUNT}/{video_hash}/source.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found at {video_path}")

    chunks = _chunk_video(video_path, video_hash)
    volume.commit()
    return chunks


# ============================================================
# TRANSCRIPTION (GPU)
# ============================================================

@app.function(
    image=image_asr,
    secrets=[hf_secret],
    gpu=GPU_TYPE_ASR,
    volumes={VOLUME_MOUNT: volume},
    timeout=900,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
def transcribe_chunk(video_hash: str, chunk: dict) -> dict:
    """Transcribe a chunk with WhisperX + Pyannote on GPU."""
    import sys
    sys.path.insert(0, "/root")
    from pipeline.transcribe import transcribe_chunk as _transcribe_chunk

    output_dir = f"{VOLUME_MOUNT}/{video_hash}/transcripts"
    hf_token = os.environ["HF_TOKEN"]

    transcript = _transcribe_chunk(chunk["path"], output_dir, hf_token)
    volume.commit()
    return transcript


# ============================================================
# FACE TRACKING (GPU)
# ============================================================

@app.function(
    image=image_vision,
    secrets=[hf_secret],
    gpu=GPU_TYPE_VISION,
    volumes={VOLUME_MOUNT: volume},
    timeout=1200,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def face_track_chunk(video_hash: str, chunk: dict) -> dict:
    """
    Run face detection + tracking + ASD on a chunk.
    YOLO26 → BoT-FaceSORT → LoCoNet → InsightFace re-ID.
    """
    import sys
    sys.path.insert(0, "/root")
    from pipeline.face_track import analyze_chunk

    output_dir = f"{VOLUME_MOUNT}/{video_hash}/face_analysis"
    analysis = analyze_chunk(
        chunk_path=chunk["path"],
        output_dir=output_dir,
        conf=FACE_CONF_THRESHOLD,
        nms=FACE_NMS_THRESHOLD,
        cosine_threshold=REID_COSINE_THRESHOLD,
    )
    volume.commit()
    return analysis


# ============================================================
# REFRAME (compute crop plan)
# ============================================================

@app.function(
    image=image_ingest,
    volumes={VOLUME_MOUNT: volume},
    timeout=300,
)
def compute_reframe(
    video_hash: str, face_analysis: dict, transcript: dict | None, clip_index: int
) -> str:
    """Compute the reframe crop plan from face analysis data."""
    import sys
    sys.path.insert(0, "/root")
    from pipeline.reframe import generate_crop_plan

    output_path = f"{VOLUME_MOUNT}/{video_hash}/crop_plans/clip_{clip_index:03d}.json"
    generate_crop_plan(
        face_analysis=face_analysis,
        output_path=output_path,
        transcript=transcript,
    )
    volume.commit()
    return output_path


# ============================================================
# CAPTION GENERATION
# ============================================================

@app.function(
    image=image_ingest,
    volumes={VOLUME_MOUNT: volume},
    timeout=300,
)
def generate_captions(video_hash: str, transcript: dict, clip_index: int) -> str:
    """Generate karaoke-style ASS captions from a transcript."""
    import sys
    sys.path.insert(0, "/root")
    from pipeline.captions import generate_captions as _generate_captions

    captions_dir = f"{VOLUME_MOUNT}/{video_hash}/captions"
    output_path = f"{captions_dir}/clip_{clip_index:03d}.ass"

    _generate_captions(
        transcript=transcript,
        output_path=output_path,
        video_width=OUTPUT_WIDTH,
        video_height=OUTPUT_HEIGHT,
    )

    volume.commit()
    return output_path


# ============================================================
# RENDERING
# ============================================================

@app.function(
    image=image_ingest,
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
def render_clip(
    video_hash: str,
    chunk: dict,
    clip_index: int,
    ass_path: str | None = None,
    crop_plan_path: str | None = None,
) -> str:
    """
    Render a vertical clip with speaker-centered reframing + captions.
    Falls back to center-crop if no crop plan is available.
    """
    import sys
    sys.path.insert(0, "/root")
    from pipeline.render import center_crop_render, dynamic_crop_render

    renders_dir = Path(VOLUME_MOUNT) / video_hash / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(renders_dir / f"clip_{clip_index:03d}.mp4")

    if crop_plan_path and os.path.exists(crop_plan_path):
        dynamic_crop_render(
            input_path=chunk["path"],
            output_path=output_path,
            crop_plan_path=crop_plan_path,
            ass_path=ass_path,
        )
    else:
        center_crop_render(
            input_path=chunk["path"],
            output_path=output_path,
            ass_path=ass_path,
        )

    volume.commit()
    return output_path


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

@app.function(
    image=image_ingest,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
)
def process_video(video_hash: str, max_clips: int = 1) -> list[str]:
    """
    Main orchestrator (Milestone 3).
    1. Check cache
    2. Chunk video
    3. Transcribe chunks (GPU — A10G)
    4. Face track chunks (GPU — T4)
    5. Compute reframe crop plans
    6. Generate captions
    7. Render speaker-centered clips with captions
    """
    video_dir = Path(VOLUME_MOUNT) / video_hash

    # Cache check
    renders_dir = video_dir / "renders"
    if renders_dir.exists():
        existing = sorted(renders_dir.glob("clip_*.mp4"))
        if existing:
            return [str(p) for p in existing[:max_clips]]

    # Step 1: Chunk video
    chunks_meta_path = video_dir / "chunks_meta.json"
    if chunks_meta_path.exists():
        with open(chunks_meta_path) as f:
            meta = json.load(f)
        chunks = meta["chunks"]
    else:
        chunks = chunk_video.remote(video_hash)

    # Step 2: Transcribe chunks in parallel (GPU — A10G)
    transcripts = []
    for chunk in chunks[:max_clips]:
        transcript = transcribe_chunk.remote(video_hash, chunk)
        transcripts.append(transcript)

    # Step 3: Face track chunks in parallel (GPU — T4)
    face_analyses = []
    for chunk in chunks[:max_clips]:
        analysis = face_track_chunk.remote(video_hash, chunk)
        face_analyses.append(analysis)

    # Step 4: Compute reframe + generate captions + render
    clip_paths = []
    for i, (chunk, transcript, face_analysis) in enumerate(
        zip(chunks[:max_clips], transcripts, face_analyses)
    ):
        # Compute crop plan from face analysis
        crop_plan_path = compute_reframe.remote(
            video_hash, face_analysis, transcript, i
        )

        # Generate captions
        ass_path = generate_captions.remote(video_hash, transcript, i)

        # Render with dynamic crop + captions
        clip_path = render_clip.remote(
            video_hash, chunk, i,
            ass_path=ass_path,
            crop_plan_path=crop_plan_path,
        )
        clip_paths.append(clip_path)

    return clip_paths
