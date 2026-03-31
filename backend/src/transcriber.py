import os
import time
import json
import hashlib
import requests
from typing import List, Dict, Any
from config import get_logger, ASSEMBLYAI_KEY

logger = get_logger(__name__)


def transcribe(video_path: str, video_url: str) -> List[Dict[str, Any]]:
    """
    Transcribes video using AssemblyAI and caches the result locally.
    Enables speaker diarization via 'speaker_labels=True'.
    """
    logger.info("==================== PHASE 2: TRANSCRIPTION ====================")

    url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
    cache_file = f"assemblyai_diarized_cache_{url_hash}.json"

    if os.path.exists(cache_file):
        logger.info("Loading transcript from cache...")
        with open(cache_file, "r") as f:
            return json.load(f)

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
        "speech_models": ["universal-2"],
        "speaker_labels": True,
    }

    res = requests.post(
        "https://api.assemblyai.com/v2/transcript", headers=headers, json=json_payload
    )
    if res.status_code != 200:
        raise RuntimeError(f"Transcription start failed: {res.text}")

    transcript_id = res.json()["id"]

    logger.info(f"Polling transcription {transcript_id}...")
    while True:
        res = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers
        )
        data = res.json()
        status = data["status"]

        if status == "completed":
            words = data["words"]
            with open(cache_file, "w") as f:
                json.dump(words, f)
            logger.info("Transcription complete.")
            return words
        elif status == "error":
            raise RuntimeError(f"AssemblyAI Error: {data.get('error')}")

        time.sleep(3)
