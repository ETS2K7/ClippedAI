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

import yt_dlp

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
    Returns path to downloaded video file.
    """
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
                return video_path

        except Exception as e:
            last_error = e
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
