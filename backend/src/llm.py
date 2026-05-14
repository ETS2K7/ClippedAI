"""
Module for LLM-powered viral clip selection.
Primary: Gemini 2.0-flash via Google Cloud Vertex AI
"""

import hashlib
import os
import json
import pathlib
import time
from typing import List, Dict, Any
from config import get_logger

logger = get_logger(__name__)

# ── Minimum duration (seconds) ──
_MIN_CLIP_DURATION = 30.0

def select_clips(words: List[Dict[str, Any]], specific_moments: str = None) -> List[Dict[str, Any]]:
    """Groups words and selects viral segments using Two-Pass architecture."""
    logger.info("==================== PHASE 3: VIRAL CLIP SELECTION ====================")
    if not words: return []

    # Auto-detect units (Seconds vs Milliseconds)
    is_ms = words[-1]["end"] > 10000 or words[0]["end"] > 500
    unit_factor = 1.0 if is_ms else 1000.0
    logger.info(f"[LLM] Detected time units: {'Milliseconds' if is_ms else 'Seconds'} (Factor: {unit_factor})")
    
    # Normalize ALL words to milliseconds for consistent internal math
    for w in words:
        w["start"] = w["start"] * unit_factor
        w["end"] = w["end"] * unit_factor

    # Pre-process into sentences with timestamps for the LLM
    sentences = []
    current_sentence = []
    start_time = None
    for w in words:
        if start_time is None: start_time = w["start"]
        current_sentence.append(w["text"])
        if any(w["text"].endswith(p) for p in [".", "?", "!", "।"]):
            sentences.append(f"[{start_time/1000.0:.1f}s - {w['end']/1000.0:.1f}s] {' '.join(current_sentence)}")
            current_sentence = []
            start_time = None
    
    transcript = "\n".join(sentences)
    prompt = (
        "TASK: Select 3 high-impact, viral segments from this transcript.\n"
        "RULES for 'Perfect Clips':\n"
        "1. START: Must begin with a 'Hook' (a strong, intriguing opening statement).\n"
        "2. END: Must conclude with a completed thought or a 'Loop' that leaves the viewer wanting more.\n"
        "3. COMPLETENESS: Never cut in the middle of a sentence or a punchline.\n"
        "4. DURATION: Each clip must be between 30 and 90 seconds.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    logger.info(f"[LLM] Selection Pass calling Gemini 2.0... (Specific: {specific_moments})")
    raw_clips = _call_gemini_selection(prompt, specific_moments)

    # ── SEMANTIC BOUNDARY SNAPPING ──
    validated_clips = []
    for clip in raw_clips:
        try:
            start_s = float(clip["start_time"])
            end_s = float(clip["end_time"])

            # Find the actual word closest to this start time (in milliseconds)
            start_ms = start_s * 1000
            end_ms = end_s * 1000
            
            # Snap to nearest word start
            snapped_start = min(words, key=lambda x: abs(x["start"] - start_ms))["start"] / 1000.0
            # Snap to nearest word end
            snapped_end = min(words, key=lambda x: abs(x["end"] - end_ms))["end"] / 1000.0

            # Adjust for 'breath buffer' (0.2s start padding, 0.3s end padding)
            snapped_start = max(0, snapped_start - 0.2)
            snapped_end = snapped_end + 0.3

            clip["start_time"] = snapped_start
            clip["end_time"] = snapped_end
            
            duration = snapped_end - snapped_start
            if duration >= _MIN_CLIP_DURATION and duration <= 120: # Allow up to 120s
                validated_clips.append(clip)
            else:
                logger.info(f"[LLM] Rejecting clip {clip.get('title')} - duration {duration:.1f}s outside limits.")
        except Exception as e:
            logger.warning(f"Failed to validate clip: {clip}. Error: {e}")

    # ── Parallel Transliteration ──
    from concurrent.futures import ThreadPoolExecutor
    
    def _process_transliteration(clip):
        start, end = clip["start_time"], clip["end_time"]
        clip_words = [w["text"] for w in words if w["start"]/1000.0 >= start and w["end"]/1000.0 <= end]
        clip_text = " ".join(clip_words)
        if any('\u0900' <= c <= '\u097f' for c in clip_text):
            clip["romanized_words"] = _call_gemini_transliterate(clip_text)
        else:
            clip["romanized_words"] = []

    if validated_clips:
        with ThreadPoolExecutor(max_workers=len(validated_clips)) as executor:
            executor.map(_process_transliteration, validated_clips)

    return validated_clips

def _call_gemini_selection(prompt: str, specific_moments: str = None) -> list:
    from google.genai import types
    client = _get_genai_client()
    
    system_instruction = (
        "You are a viral video editor. Your goal is to select 3 self-contained, high-retention clips. "
        "Ensure clips are semantically complete. Return JSON only."
    )
    if specific_moments:
        system_instruction += (
            f"\n\nPRIORITY: The user is specifically interested in the following moments: '{specific_moments}'. "
            "Prioritize selecting clips that match this description. If you find less than 3 matching clips, "
            "fill the remaining slots with the most viral moments from the rest of the transcript."
        )

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "clips": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "start_time": {"type": "NUMBER"},
                                        "end_time": {"type": "NUMBER"},
                                        "title": {"type": "STRING"},
                                        "viral_reason": {"type": "STRING"}
                                    },
                                    "required": ["start_time", "end_time", "title", "viral_reason"]
                                }
                            }
                        },
                        "required": ["clips"]
                    },
                    temperature=0.7,
                ),
            )
            logger.info(f"[LLM] Gemini Selection Result: {response.text}")
            return json.loads(response.text).get("clips", [])
        except Exception as e:
            logger.warning(f"Selection attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return []

def _call_gemini_transliterate(text: str) -> list:
    from google.genai import types
    client = _get_genai_client()
    
    prompt = f"Convert this Hindi text into Romanized Hinglish words (transliteration). Return a JSON list of strings.\n\nTEXT: {text}"
    
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "romanized_words": {"type": "ARRAY", "items": {"type": "STRING"}}
                        },
                        "required": ["romanized_words"]
                    },
                    temperature=0.1,
                ),
            )
            return json.loads(response.text).get("romanized_words", [])
        except: time.sleep(1)
    return []

def _get_genai_client():
    from google import genai
    import os
    import json

    credentials = None
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if gcp_json:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(gcp_json)
        ).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
    
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "clippedai-493912")
    return genai.Client(vertexai=True, project=gcp_project, location="us-central1", credentials=credentials) if credentials else genai.Client(vertexai=True, project=gcp_project, location="us-central1")
