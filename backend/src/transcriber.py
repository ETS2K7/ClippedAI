import os
import time
import requests
from typing import List, Dict, Any
from config import get_logger, ASSEMBLYAI_KEY

logger = get_logger(__name__)

# Maximum polls before timeout (~10 minutes at 3s intervals)
MAX_POLL_ATTEMPTS = 200


def transcribe(video_path: str, _video_url: str = "") -> List[Dict[str, Any]]:
    """
    Transcribes video using AssemblyAI with speaker diarization.
    Returns a list of word-level dicts with timestamps and speaker labels.

    Args:
        video_path: Local path to the video/audio file to transcribe.
        _video_url: Unused — kept for call-site compatibility.
    """
    logger.info("==================== PHASE 2: TRANSCRIPTION ====================")

    logger.info("Uploading to AssemblyAI...")
    headers = {"authorization": ASSEMBLYAI_KEY}

    with open(video_path, "rb") as f:
        res = requests.post(
            "https://api.assemblyai.com/v2/upload", headers=headers, data=f
        )

    if res.status_code != 200:
        raise RuntimeError(f"Upload failed: {res.text}")

    upload_url = res.json()["upload_url"]

    logger.info("Requesting transcription...")
    json_payload = {
        "audio_url": upload_url,
        # AssemblyAI v2: singular string key, value is the model name
        "speech_model": "universal_2",
        "speaker_labels": True,
    }

    res = requests.post(
        "https://api.assemblyai.com/v2/transcript", headers=headers, json=json_payload
    )
    if res.status_code != 200:
        raise RuntimeError(f"Transcription start failed: {res.text}")

    transcript_id = res.json()["id"]

    logger.info(f"Polling transcription {transcript_id}...")
    for attempt in range(MAX_POLL_ATTEMPTS):
        res = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers
        )
        data = res.json()
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
