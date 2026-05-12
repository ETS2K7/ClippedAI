"""
Module for LLM-powered viral clip selection.
Primary: Gemini 1.5-pro via Google Cloud Vertex AI
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

def select_clips(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Groups words and selects viral segments using Two-Pass architecture."""
    logger.info("==================== PHASE 3: VIRAL CLIP SELECTION ====================")
    if not words: return []

    sentences = []
    current_sentence = []
    start_time = None
    for w in words:
        if start_time is None: start_time = w["start"]
        current_sentence.append(w["text"])
        if any(w["text"].endswith(p) for p in [".", "?", "!"]):
            sentences.append(f"[{start_time/1000.0:.1f}s - {w['end']/1000.0:.1f}s] {' '.join(current_sentence)}")
            current_sentence = []
            start_time = None
    
    transcript = "\n".join(sentences)
    prompt = (
        "Analyze the transcript and select 3 viral clips (30-90s each).\n"
        "Ensure each clip is self-contained and high-value.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    # Cache Logic
    _cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    _cache_file = pathlib.Path.home() / ".clippedai" / "cache" / "llm" / f"llm_{_cache_key}.json"
    _cache_file.parent.mkdir(parents=True, exist_ok=True)

    if _cache_file.exists():
        try:
            cached = json.loads(_cache_file.read_text())
            if cached:
                logger.info(f"[LLM] 🟢 Cache hit (key={_cache_key[:8]})")
                return cached
        except: pass

    # Pass 1: Selection (Using stable Flash model)
    logger.info(f"[LLM] 🔴 Cache miss. Calling Gemini Flash...")
    validated_clips = _call_gemini_selection(prompt, words)

    if not validated_clips:
        logger.error("[LLM] Selection Pass failed to produce clips. This might be due to a short transcript or strict viral criteria.")
        return []

    # Pass 2: Transliterate (Using focused Flash requests)
    for i, clip in enumerate(validated_clips):
        start, end = clip["start_time"], clip["end_time"]
        clip_words = [w["text"] for w in words if w["start"]/1000.0 >= start and w["end"]/1000.0 <= end]
        clip_text = " ".join(clip_words)
        if any('\u0900' <= c <= '\u097f' for c in clip_text):
            logger.info(f"[LLM] Transliterating Clip {i}...")
            clip["romanized_words"] = _call_gemini_transliterate(clip_text)
        else:
            clip["romanized_words"] = []

    # Only cache if we have successful results
    try:
        _cache_file.write_text(json.dumps(validated_clips))
    except: pass

    return validated_clips

def _validate_clips(raw_clips: list, words: list) -> list:
    """Validate and filter clips for duration and timestamp bounds."""
    if not words: return []
    video_end_s = words[-1]["end"] / 1000.0
    validated = []
    for clip in raw_clips:
        start = float(clip.get("start_time") or clip.get("start") or 0)
        end   = float(clip.get("end_time")   or clip.get("end")   or 0)
        clip["start_time"] = start
        clip["end_time"]   = end
        duration = end - start
        if start < 0 or end <= start or start > video_end_s:
            continue
        if duration > 95 or duration < _MIN_CLIP_DURATION:
            continue
        validated.append(clip)
    return validated

def _get_genai_client():
    """Initializes and returns the Vertex AI GenAI client."""
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

def _call_gemini_selection(prompt: str, words: list) -> list:
    from google.genai import types
    client = _get_genai_client()
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="Select 3 viral segments (30-90s). Return JSON only.",
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "clips": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "reasoning": {"type": "STRING"},
                                        "title": {"type": "STRING"},
                                        "start_time": {"type": "NUMBER"},
                                        "end_time": {"type": "NUMBER"},
                                        "virality_score": {"type": "NUMBER"}
                                    },
                                    "required": ["reasoning", "title", "start_time", "end_time", "virality_score"]
                                }
                            }
                        },
                        "required": ["clips"]
                    },
                    temperature=0.7,
                ),
            )
            data = json.loads(response.text)
            return _validate_clips(data.get("clips", []), words)
        except Exception as e:
            logger.warning(f"Selection attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return []

def _call_gemini_transliterate(text: str) -> list:
    from google.genai import types
    client = _get_genai_client()
    prompt = (
        "Transliterate Hindi (Devanagari) to Roman Hindi (Latin Script).\n"
        "Format: Return an array of strings, each containing piped pairs 'Roman:Original'.\n"
        "Example: ['Namaste:नमस्ते|kaise:कैसे']\n\n"
        f"TEXT:\n{text}"
    )
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
