import os
import time
import subprocess
import requests
from typing import List, Dict, Any
from config import get_logger, ASSEMBLYAI_KEY

logger = get_logger(__name__)

# Maximum polls before timeout (~10 minutes at 3s intervals)
MAX_POLL_ATTEMPTS = 200


def _extract_audio(video_path: str) -> str:
    """
    Extracts audio-only from a video file using FFmpeg.
    Returns the path to the lightweight audio file (~5MB vs ~200MB video).
    This reduces AssemblyAI upload time by ~90%.
    """
    audio_path = video_path.rsplit(".", 1)[0] + "_audio.ogg"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # strip video
        "-acodec", "libopus",     # efficient audio codec
        "-b:a", "48k",            # 48kbps is plenty for speech
        "-ac", "1",               # mono (speech doesn't need stereo)
        audio_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        logger.info(
            f"Audio extracted: {os.path.getsize(video_path) / 1024 / 1024:.1f}MB video → "
            f"{os.path.getsize(audio_path) / 1024 / 1024:.1f}MB audio"
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Audio extraction failed ({e}), falling back to full video upload")
        return video_path
    return audio_path


def transcribe(video_path: str, _video_url: str = "") -> List[Dict[str, Any]]:
    """
    Transcribes video using AssemblyAI with speaker diarization.
    Returns a list of word-level dicts with timestamps and speaker labels.

    Optimization: Extracts audio-only before uploading to AssemblyAI,
    reducing upload payload by ~90% (e.g. 200MB video → 5MB audio).

    Args:
        video_path: Local path to the video/audio file to transcribe.
        _video_url: Unused — kept for call-site compatibility.
    """
    logger.info("==================== PHASE 2: TRANSCRIPTION ====================")

    # Extract lightweight audio for upload (P0 optimization)
    upload_path = _extract_audio(video_path)
    is_extracted_audio = upload_path != video_path

    logger.info(f"Uploading {'audio' if is_extracted_audio else 'video'} to AssemblyAI...")
    headers = {"authorization": ASSEMBLYAI_KEY()}

    try:
        with open(upload_path, "rb") as f:
            res = requests.post(
                "https://api.assemblyai.com/v2/upload", headers=headers, data=f,
                timeout=600,  # 10 min timeout for large uploads
            )

        if res.status_code != 200:
            raise RuntimeError(f"Upload failed: {res.text}")
    finally:
        # Clean up extracted audio file to free disk space
        if is_extracted_audio:
            try:
                os.remove(upload_path)
            except OSError:
                pass

    upload_url = res.json()["upload_url"]

    logger.info("Requesting transcription...")
    json_payload = {
        "audio_url": upload_url,
        # AssemblyAI v2: singular string key, value is the model name
        "speech_model": "universal_2",
        "speaker_labels": True,
    }

    res = requests.post(
        "https://api.assemblyai.com/v2/transcript", headers=headers, json=json_payload,
        timeout=60,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Transcription start failed: {res.text}")

    transcript_id = res.json()["id"]

    logger.info(f"Polling transcription {transcript_id}...")
    for attempt in range(MAX_POLL_ATTEMPTS):
        try:
            res = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers,
                timeout=30,
            )
            if res.status_code != 200:
                logger.warning(f"Poll returned HTTP {res.status_code}, retrying...")
                time.sleep(3)
                continue
            data = res.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error during poll: {e}. Retrying in 3s...")
            time.sleep(3)
            continue
        status = data["status"]

        if status == "completed":
            words = data["words"]
            logger.info("Transcription complete.")
            return words
        elif status == "error":
            raise RuntimeError(f"AssemblyAI Error: {data.get('error')}")

        time.sleep(3)

    raise RuntimeError(
        f"Transcription timed out after {MAX_POLL_ATTEMPTS * 3}s for transcript {transcript_id}"
    )
