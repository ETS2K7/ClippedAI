"""
Module responsible for extracting lightweight audio payloads and polling AssemblyAI for high-accuracy transcripts and speaker diarization.
"""

import hashlib
import os
import pathlib
import time
import subprocess
from typing import List, Dict, Any

import requests
from config import get_logger, ASSEMBLYAI_KEY

logger = get_logger(__name__)

# Base polling configuration
POLL_INTERVAL_SECONDS = 3
# Calculate max attempts based on video duration: ~1 minute per 10 minutes of video
def get_max_poll_attempts(video_duration_seconds: int) -> int:
    """Returns appropriate max poll attempts based on video duration."""
    # Minimum 200 attempts (10 minutes), plus additional for longer videos
    base_attempts = 200
    additional_attempts = (video_duration_seconds // 600) * 50  # 50 extra attempts per 10 min of video
    return base_attempts + additional_attempts


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

    # ── Transcript cache ───────────────────────────────────────────────────
    # Key: SHA-256 of the raw video file bytes.
    # Same video file → same key → same words list without calling AssemblyAI,
    # which also stabilises the downstream LLM cache key.
    import json as _json
    _tcache_dir = pathlib.Path.home() / ".clippedai" / "cache" / "transcript"
    _tcache_dir.mkdir(parents=True, exist_ok=True)

    try:
        _hasher = hashlib.sha256()
        with open(video_path, "rb") as _vf:
            for _chunk in iter(lambda: _vf.read(65536), b""):
                _hasher.update(_chunk)
        _video_hash = _hasher.hexdigest()
        _tcache_file = _tcache_dir / f"transcript_{_video_hash}.json"
    except OSError as _he:
        logger.warning(f"[Transcript] Could not hash video file ({_he}); cache disabled.")
        _tcache_file = None
        _video_hash = "<unknown>"

    if _tcache_file and _tcache_file.exists():
        try:
            words = _json.loads(_tcache_file.read_text("utf-8"))
            if isinstance(words, list) and len(words) > 0:
                logger.info(
                    f"[Transcript] 🟢 Cache hit — skipping AssemblyAI "
                    f"(key={_video_hash[:8]}, {len(words)} words)"
                )
                return words
            else:
                logger.warning("[Transcript] Cache entry empty/invalid, re-transcribing.")
        except Exception as _ce:
            logger.warning(f"[Transcript] Cache read failed ({_ce}), re-transcribing.")

    logger.info(f"[Transcript] 🔴 Cache miss (key={_video_hash[:8]}). Calling AssemblyAI...")

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
        "speech_models": ["universal-2"],
        "speaker_labels": True,
    }

    logger.info(f"Submitting actual payload to AAI: {json_payload}")

    res = requests.post(
        "https://api.assemblyai.com/v2/transcript", headers=headers, json=json_payload,
        timeout=60,
    )
    if res.status_code != 200:
        raise RuntimeError(f"Transcription start failed: {res.text}")

    transcript_id = res.json()["id"]

    # Get video duration for dynamic timeout calculation
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    video_duration_seconds = int(frame_count / fps) if fps > 0 else 600  # Default to 10 min if fps unavailable
    max_attempts = get_max_poll_attempts(video_duration_seconds)
    
    logger.info(f"Polling transcription {transcript_id} (max {max_attempts} attempts for {video_duration_seconds}s video)...")
    
    # Adaptive polling: start with 1s, exponentially increase to max 10s
    current_poll_interval = 1
    max_poll_interval = 10
    
    for attempt in range(max_attempts):
        try:
            res = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers,
                timeout=30,
            )
            if res.status_code != 200:
                logger.warning(f"Poll returned HTTP {res.status_code}, retrying...")
                time.sleep(current_poll_interval)
                continue
            data = res.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error during poll: {e}. Retrying in {current_poll_interval}s...")
            time.sleep(current_poll_interval)
            continue
        status = data["status"]

        if status == "completed":
            words = data["words"]
            logger.info(f"Transcription complete after {attempt + 1} polls.")

            # Write transcript cache
            if _tcache_file:
                try:
                    _tcache_file.write_text(_json.dumps(words), "utf-8")
                    logger.info(
                        f"[Transcript] Cached {len(words)} words to {_tcache_file.name}"
                    )
                except Exception as _we:
                    logger.warning(f"[Transcript] Cache write failed (non-fatal): {_we}")

            return words
        
        if status == "error":
            raise RuntimeError(f"AssemblyAI Error: {data.get('error')}")

        # Exponentially increase poll interval (capped at max_poll_interval)
        time.sleep(current_poll_interval)
        current_poll_interval = min(current_poll_interval * 2, max_poll_interval)

    raise RuntimeError(
        f"Transcription timed out after {max_attempts * POLL_INTERVAL_SECONDS}s for transcript {transcript_id}"
    )
