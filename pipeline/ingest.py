"""
ClippedAI — Pipeline ingestion: video hashing + chunking.
Runs on Modal after the video has been uploaded to the volume from the local Mac.
"""
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

from config import CHUNK_DURATION, CHUNK_OVERLAP, VOLUME_MOUNT


def compute_video_hash(video_path: str) -> str:
    """
    Collision-resistant video hash.
    sha256(first_10MB + last_10MB + duration)
    """
    chunk_size = 10 * 1024 * 1024  # 10 MB

    with open(video_path, "rb") as f:
        head = f.read(chunk_size)
        f.seek(0, 2)
        file_size = f.tell()
        if file_size > chunk_size:
            f.seek(-chunk_size, 2)
            tail = f.read(chunk_size)
        else:
            tail = head

    # Get duration via ffprobe
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())

    h = hashlib.sha256()
    h.update(head)
    h.update(tail)
    h.update(struct.pack("d", duration))
    return h.hexdigest()[:16]


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def chunk_video(video_path: str, video_hash: str) -> list[dict]:
    """
    Split video into 5-minute chunks with 30-second overlaps.
    Chunks are written to /vol/{video_hash}/chunks/

    Returns list of chunk metadata dicts:
        [{"index": 0, "path": "/vol/.../chunk_000.mp4", "start": 0.0, "end": 300.0}, ...]
    """
    duration = get_video_duration(video_path)
    chunk_dir = Path(VOLUME_MOUNT) / video_hash / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    start = 0.0
    index = 0

    while start < duration:
        end = min(start + CHUNK_DURATION, duration)
        chunk_path = str(chunk_dir / f"chunk_{index:03d}.mp4")

        # Use FFmpeg to extract chunk (stream copy, no re-encode)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(end - start),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                chunk_path,
            ],
            capture_output=True, check=True,
        )

        chunks.append({
            "index": index,
            "path": chunk_path,
            "start": start,
            "end": end,
        })

        # Next chunk starts (CHUNK_DURATION - CHUNK_OVERLAP) after this one
        start += CHUNK_DURATION - CHUNK_OVERLAP
        index += 1

    # Save chunk metadata
    meta_path = Path(VOLUME_MOUNT) / video_hash / "chunks_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"video_hash": video_hash, "duration": duration, "chunks": chunks}, f, indent=2)

    return chunks
