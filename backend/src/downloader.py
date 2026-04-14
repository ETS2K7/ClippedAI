"""
Local-only YouTube downloader using yt-dlp.

⚠️  This module is used ONLY by `run_local.py` for local development.
The production Modal pipeline (main.py) uses its own _download_youtube()
function which downloads via the Apify actor.
"""
import os
import subprocess
import glob
from config import get_logger, MASTER_VIDEO_TMPL, MASTER_VIDEO_FILE

logger = get_logger(__name__)


def download_video(url: str) -> str:
    """
    Downloads the best video and audio streams from the specified URL using `yt-dlp`.
    Transcodes to H264 MP4 if necessary.
    """
    logger.info("==================== PHASE 1: DOWNLOAD ====================")

    # Security: Strict validation to prevent yt-dlp parameter injection
    is_valid_url = isinstance(url, str) and (
        url.startswith("https://www.youtube.com/") or url.startswith("https://youtu.be/")
    )
    if not is_valid_url:
        raise ValueError("Invalid URL: Must be a standard YouTube URL.")

    logger.info(f"Downloading {url}...")

    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o",
        MASTER_VIDEO_TMPL,
        url,
    ]

    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp failed to download {url}")
        raise RuntimeError(f"Download failed: {e}") from e

    if not os.path.exists(MASTER_VIDEO_FILE):
        # yt-dlp may output a different extension like webm or mkv
        files = glob.glob(MASTER_VIDEO_TMPL.replace("%(ext)s", "*"))
        if files:
            actual = files[0]
            if actual != MASTER_VIDEO_FILE:
                logger.info(f"Transcoding {actual} to {MASTER_VIDEO_FILE}...")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    actual,
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    MASTER_VIDEO_FILE,
                ]
                try:
                    subprocess.run(
                        ffmpeg_cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )
                    os.remove(actual)
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg transcoding failed for {actual}")
                    raise RuntimeError(f"Transcoding failed: {e}") from e
        else:
            raise FileNotFoundError("Video downloaded but output file not found.")

    return MASTER_VIDEO_FILE
