"""
ClippedAI — Modal App Definition (Milestone 1: Ingest + Raw Crop)

Micro-container architecture:
- image_ingest: FFmpeg only (dev mode — video uploaded from Mac)
- process_video: Main orchestrator

Later milestones will add:
- image_asr: WhisperX + Pyannote 4.0
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
    VIDEO_CRF,
)

# --- Modal App ---
app = modal.App(APP_NAME)

# --- Volume ---
volume = modal.Volume.from_name(VOLUME_NAME)

# --- Images ---
image_ingest = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("modal")
    .add_local_file("config.py", "/root/config.py")
    .add_local_dir("pipeline", "/root/pipeline")
)


@app.function(
    image=image_ingest,
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
def chunk_video(video_hash: str) -> list[dict]:
    """
    Chunk a video that has been uploaded to the volume.
    Expects the source video at /vol/{video_hash}/source.mp4
    """
    import sys
    sys.path.insert(0, "/root")
    from pipeline.ingest import chunk_video as _chunk_video

    video_path = f"{VOLUME_MOUNT}/{video_hash}/source.mp4"

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found at {video_path}")

    chunks = _chunk_video(video_path, video_hash)
    volume.commit()
    return chunks


@app.function(
    image=image_ingest,
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
def render_center_crop(video_hash: str, chunk: dict, clip_index: int) -> str:
    """
    Render a center-cropped vertical clip from a chunk.
    Returns the path to the rendered clip on the volume.
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
    )

    volume.commit()
    return output_path


@app.function(
    image=image_ingest,
    volumes={VOLUME_MOUNT: volume},
    timeout=900,
)
def process_video(video_hash: str, max_clips: int = 1) -> list[str]:
    """
    Main orchestrator for Milestone 1.
    1. Check cache
    2. Chunk video
    3. Render center-crop clips from first chunk(s)
    Returns list of rendered clip paths on the volume.
    """
    video_dir = Path(VOLUME_MOUNT) / video_hash

    # Cache check: if renders already exist, return them
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

    # Step 2: Render center-crop clips (one from each chunk, up to max_clips)
    clip_paths = []
    for i, chunk in enumerate(chunks[:max_clips]):
        clip_path = render_center_crop.remote(video_hash, chunk, i)
        clip_paths.append(clip_path)

    return clip_paths
