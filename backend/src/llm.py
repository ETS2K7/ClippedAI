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

def select_clips(words: List[Dict[str, Any]], specific_moments: str = None) -> List[Dict[str, Any]]:
    """Groups words and selects viral segments using Two-Pass architecture."""
    logger.info("==================== PHASE 3: VIRAL CLIP SELECTION ====================")
    if not words: return []

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

    # ... [Cache Logic Omitted for brevity] ...
    _cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    _cache_file = pathlib.Path.home() / ".clippedai" / "cache" / "llm" / f"llm_{_cache_key}.json"
    _cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Cache reads disabled per user request

    logger.info(f"[LLM] Selection Pass calling Gemini... (Specific: {specific_moments})")
    raw_clips = _call_gemini_selection(prompt, specific_moments)

    # ── SEMANTIC BOUNDARY SNAPPING ──
    # This ensures that even if the LLM rounds a timestamp, we snap it to the 
    # nearest actual word boundary in the transcript.
    validated_clips = []
    for clip in raw_clips:
        start_s = float(clip["start_time"])
        end_s = float(clip["end_time"])

        # Find the actual word closest to this start time
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
        if duration >= _MIN_CLIP_DURATION and duration <= 100:
            validated_clips.append(clip)

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

    # Cache writes disabled per user request

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
                model="gemini-1.5-flash-002",
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
            return data.get("clips", [])
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
                model="gemini-1.5-flash-002",
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
