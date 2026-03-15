"""
ClippedAI — Modal App Definition (Milestone 2: Speech + Captions)

Micro-container architecture:
- image_ingest: FFmpeg only (dev mode — video uploaded from Mac)
- image_asr: WhisperX + Pyannote 4.0 (GPU: A10G)

Later milestones will add:
- image_vision: YOLO26 + BoT-FaceSORT + LoCoNet
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
def render_clip(video_hash: str, chunk: dict, clip_index: int, ass_path: str | None = None) -> str:
    """
    Render a center-cropped vertical clip with optional caption burn-in.
    """
    import sys
    sys.path.insert(0, "/root")
    from pipeline.render import center_crop_render

    renders_dir = Path(VOLUME_MOUNT) / video_hash / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(renders_dir / f"clip_{clip_index:03d}.mp4")

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
    timeout=1800,
)
def process_video(video_hash: str, max_clips: int = 1) -> list[str]:
    """
    Main orchestrator (Milestone 2).
    1. Check cache
    2. Chunk video
    3. Transcribe chunks (GPU)
    4. Generate captions
    5. Render center-crop clips with captions
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

    # Step 2: Transcribe chunks in parallel (GPU)
    transcripts = []
    for chunk in chunks[:max_clips]:
        transcript = transcribe_chunk.remote(video_hash, chunk)
        transcripts.append(transcript)

    # Step 3: Generate captions + render clips
    clip_paths = []
    for i, (chunk, transcript) in enumerate(zip(chunks[:max_clips], transcripts)):
        ass_path = generate_captions.remote(video_hash, transcript, i)
        clip_path = render_clip.remote(video_hash, chunk, i, ass_path=ass_path)
        clip_paths.append(clip_path)

    return clip_paths
