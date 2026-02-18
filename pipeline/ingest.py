"""
Pipeline Step 1 — Ingest
Handles both local video files and remote URLs.
On Modal: receives pre-downloaded video bytes, writes to disk, probes metadata.
Locally: downloads via yt-dlp (has browser cookie access).
"""

import subprocess
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YTDLP_FORMAT, MAX_VIDEO_DURATION, MAX_RESOLUTION


WORK_DIR = "/tmp/clipped"


def save_uploaded_video(video_bytes: bytes, title: str = "Untitled") -> dict:
    """
    Save pre-downloaded video bytes to disk and probe metadata.
    Used when video is downloaded locally and uploaded to Modal.

    Returns:
        {
            "video_path": str,
            "title": str,
            "duration": float,
            "width": int,
            "height": int,
            "fps": float,
        }
    """
    os.makedirs(WORK_DIR, exist_ok=True)
    video_path = os.path.join(WORK_DIR, "source.mp4")

    print(f"  📥 Writing uploaded video ({len(video_bytes) / 1024 / 1024:.1f} MB)...")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # Probe for metadata
    probe = _ffprobe_full(video_path)

    if probe["duration"] > MAX_VIDEO_DURATION:
        raise ValueError(
            f"Video too long: {probe['duration']/3600:.1f}h "
            f"(max {MAX_VIDEO_DURATION/3600:.0f}h)"
        )

    return {
        "video_path": video_path,
        "title": title,
        "duration": probe["duration"],
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["fps"],
    }


def download_video_locally(url: str) -> tuple:
    """
    Download video on the LOCAL machine using yt-dlp.
    This works because the local machine has browser cookie access.

    Returns: (video_bytes, title)
    """
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="clipped_")
    video_path = os.path.join(tmp_dir, "source.mp4")

    # Get metadata first
    meta_cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        url,
    ]
    print(f"  📥 Fetching video metadata...")
    result = subprocess.run(meta_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {result.stderr[:500]}")

    meta = json.loads(result.stdout)
    duration = meta.get("duration", 0)
    title = meta.get("title", "Untitled")

    if duration > MAX_VIDEO_DURATION:
        raise ValueError(
            f"Video too long: {duration/3600:.1f}h "
            f"(max {MAX_VIDEO_DURATION/3600:.0f}h)"
        )

    print(f"  📥 Downloading: {title} ({duration:.0f}s)")

    # Download video
    dl_cmd = [
        "yt-dlp",
        "-f", YTDLP_FORMAT,
        "--merge-output-format", "mp4",
        "-o", video_path,
        "--no-playlist",
        "--retries", "3",
        "--no-check-certificates",
        url,
    ]
    result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr[:500]}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Downloaded file not found at {video_path}")

    # Read bytes
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    # Cleanup
    try:
        os.remove(video_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    size_mb = len(video_bytes) / (1024 * 1024)
    print(f"  ✅ Downloaded: {size_mb:.1f} MB")

    return video_bytes, title


def extract_audio(video_path: str) -> str:
    """
    Extract mono 16kHz WAV audio for WhisperX.

    Returns: path to audio.wav
    """
    audio_path = os.path.join(WORK_DIR, "audio.wav")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz
        "-ac", "1",               # Mono
        "-y",                     # Overwrite
        audio_path,
    ]
    print(f"  🔊 Extracting audio...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    return audio_path


def _ffprobe_full(video_path: str) -> dict:
    """Get video dimensions, fps, and duration via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "v:0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]
    fmt = data.get("format", {})

    # Parse fps from r_frame_rate (e.g., "30000/1001")
    fps_parts = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    # Duration from format (more reliable) or stream
    duration = float(fmt.get("duration", stream.get("duration", 0)))

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(fps, 2),
        "duration": round(duration, 2),
    }
