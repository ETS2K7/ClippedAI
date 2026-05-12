"""
Module for LLM-powered viral clip selection.
Primary: Gemini 2.5-flash via Google Cloud Vertex AI
"""

import hashlib
import os
import json
import pathlib
import requests
import time
from typing import List, Dict, Any
from config import get_logger, ASSEMBLYAI_KEY
import re

logger = get_logger(__name__)


def select_clips(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups words into sentences and feeds them into Gemini 2.5-flash to
    select 3 high-retention viral segments between 10-60 seconds.
    """
    logger.info(
        "==================== PHASE 3: VIRAL CLIP SELECTION ===================="
    )

    if not words:
        raise ValueError("Cannot select clips from an empty transcript")

    sentences = []
    current_sentence = []
    start_time = None

    for w in words:
        if start_time is None:
            start_time = w["start"]
        current_sentence.append(w["text"])

        if any(w["text"].endswith(p) for p in [".", "?", "!"]):
            end_time = w["end"]
            text = " ".join(current_sentence)
            sentences.append(
                f"[{start_time / 1000.0:.1f}s - {end_time / 1000.0:.1f}s] {text}"
            )
            current_sentence = []
            start_time = None

    if current_sentence:
        end_time = words[-1]["end"]
        sentences.append(
            f"[{start_time / 1000.0:.1f}s - {end_time / 1000.0:.1f}s] {' '.join(current_sentence)}"
        )

    transcript = "\n".join(sentences)

    prompt = (
        "Extract up to 3 non-overlapping viral clips from this transcript.\n"
        "If the video is short, extract as many as mathematically possible.\n\n"
        "Each clip MUST be between 30 and 90 seconds long. No exceptions.\n"
        "1. PRIORITIZE DEPTH: Favor segments that allow for a complete explanation, deep insight, or full story. Do not aggressively trim for brevity.\n"
        "2. 100% SELF-CONTAINED: The clip MUST make complete sense to a viewer who has never seen the original video.\n"
        "3. NO UNRESOLVED PRONOUNS: The clip CANNOT start with words like 'He', 'This', 'That', or 'It' unless the subject is immediately clarified.\n"
        "4. FULL NARRATIVE ARC: Every clip must have a clear setup, escalation, and payoff/insight.\n"
        "5. PUNCHY TITLE: Create a viral, high-value title for each clip.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    # ── Transcript+prompt cache ──────────────────────────────────────────────
    _cache_dir = pathlib.Path.home() / ".clippedai" / "cache" / "llm"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_key = hashlib.sha256((prompt).encode("utf-8")).hexdigest()
    _cache_file = _cache_dir / f"llm_{_cache_key}.json"

    if _cache_file.exists():
        try:
            cached = json.loads(_cache_file.read_text("utf-8"))
            if isinstance(cached, list) and len(cached) >= 3:
                logger.info(
                    f"[LLM] 🟢 Cache hit — returning cached clip selection "
                    f"(key={_cache_key[:8]})"
                )
                return _validate_clips(cached[:3], words)
            else:
                logger.warning("[LLM] Cache entry invalid, re-running selection.")
        except Exception as _ce:
            logger.warning(f"[LLM] Cache read failed ({_ce}), re-running selection.")

    logger.info(f"[LLM] 🔴 Cache miss (key={_cache_key[:8]}). Calling Gemini...")

    validated_clips = _call_gemini(prompt, words)

    # Write to cache
    try:
        _cache_file.write_text(json.dumps(validated_clips), "utf-8")
        logger.info(f"[LLM] Cached clip selection to {_cache_file.name}")
    except Exception as _we:
        logger.warning(f"[LLM] Cache write failed (non-fatal): {_we}")

    return validated_clips


# ── Minimum duration (seconds) ──
_MIN_CLIP_DURATION = 30.0


def _validate_clips(raw_clips: list, words: list) -> list:
    """Validate and filter clips for duration and timestamp bounds."""
    video_end_s = words[-1]["end"] / 1000.0
    validated = []
    for clip in raw_clips:
        start = float(clip.get("start_time") or clip.get("start") or 0)
        end   = float(clip.get("end_time")   or clip.get("end")   or 0)
        clip["start_time"] = start
        clip["end_time"]   = end
        duration = end - start
        if start < 0 or end <= start or start > video_end_s:
            logger.warning(f"Skipping invalid clip: start={start}, end={end}")
            continue
        if duration > 95:  # Allow slight buffer over 90s
            logger.warning(f"Skipping clip with excessive duration ({duration:.1f}s)")
            continue
        # Discard clips that are too short (strict enforcement)
        if duration < _MIN_CLIP_DURATION:
            logger.warning(f"Skipping clip with insufficient duration ({duration:.1f}s < {_MIN_CLIP_DURATION}s)")
            continue
            
        validated.append(clip)
    return validated


def _call_gemini(prompt: str, words: list) -> list:
    """Primary: Gemini 2.5-flash via Google Cloud Vertex AI."""
    from google import genai
    from google.genai import types

    credentials = None
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    
    if gcp_json:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(gcp_json)
        ).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
    elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GCP credentials not set. Cannot call Vertex AI.")

    logger.info("[LLM] Calling Vertex AI (gemini-2.5-flash)...")
    MAX_RETRIES = 3
    
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "clippedai-493912")
    
    # Initialize Vertex AI client with explicit credentials or fallback to ADC
    if credentials:
        client = genai.Client(vertexai=True, project=gcp_project, location="us-central1", credentials=credentials)
    else:
        client = genai.Client(vertexai=True, project=gcp_project, location="us-central1")
        
    response = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an expert short-form video editor specializing in creating viral clips "
                        "for TikTok, YouTube Shorts, and Instagram Reels. Return only valid JSON."
                    ),
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "clips": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "reasoning": {"type": "STRING", "description": "Why this clip is highly engaging and viral."},
                                        "title": {"type": "STRING", "description": "A punchy, viral caption."},
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
                    temperature=0.2,
                ),
            )
            data = json.loads(response.text)
            raw_clips = data.get("clips") if isinstance(data.get("clips"), list) else []
            if not raw_clips:
                raise ValueError(f"Gemini returned unexpected JSON keys: {list(data.keys())}")

            validated = _validate_clips(raw_clips, words)
            if len(validated) < 1:
                raise ValueError(f"No valid clips after validation.")

            logger.info(f"[LLM] ✓ Gemini selected {len(validated)} clips.")
            return validated[:3]

        except Exception as e:
            wait = min(2 ** (attempt + 1), 16)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"[LLM Fallback] Gemini attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s...")
                if response:
                    logger.debug(f"Gemini output: {response.text}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Vertex AI failed after {MAX_RETRIES} attempts: {e}")



def transliterate_hinglish(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies Hindi (Devanagari) words and transliterates them to Romanized Hinglish
    (English letters) while preserving original timestamps.
    """
    # 1. Quick check: Is there any non-ASCII (likely Hindi) text?
    if not any(any(ord(c) > 127 for c in w["text"]) for w in words):
        return words

    logger.info("==================== PHASE 2.5: HINGLISH TRANSLITERATION ====================")
    logger.info(f"Detected Hindi script. Transliterating {len(words)} words...")

    # 2. Batch processing (Gemini handles ~100-200 words comfortably in one prompt)
    batch_size = 150
    batches = [words[i : i + batch_size] for i in range(0, len(words), batch_size)]
    
    transliterated_words = []
    
    for i, batch in enumerate(batches):
        texts = [w["text"] for w in batch]
        
        prompt = (
            "You are a transcription assistant specializing in Hinglish (Hindi + English).\n"
            "INPUT: A JSON list of words from a transcript.\n"
            "TASK: Transliterate any Hindi (Devanagari) words into Romanized Hinglish (English letters).\n"
            "RULES:\n"
            "1. KEEP English words EXACTLY as they are.\n"
            "2. Convert Hindi words to their standard phonetically spelled English equivalents (e.g., 'आज' -> 'Aaj', 'वीडियो' -> 'video').\n"
            "3. RETURN ONLY A JSON ARRAY of strings of the SAME LENGTH as the input.\n"
            "4. DO NOT add punctuation that wasn't there.\n\n"
            f"INPUT_WORDS: {json.dumps(texts)}"
        )
        
        try:
            # We use a smaller timeout for transliteration as it's a simpler task
            raw_response = _call_gemini_raw(prompt)
            # Find the JSON array in the response
            match = re.search(r"\[.*\]", raw_response, re.DOTALL)
            if match:
                new_texts = json.loads(match.group())
                if len(new_texts) == len(batch):
                    for idx, w in enumerate(batch):
                        new_w = w.copy()
                        new_w["text"] = new_texts[idx]
                        transliterated_words.append(new_w)
                    continue
            
            logger.warning(f"Transliteration batch {i} failed (length mismatch or parse error). Falling back to original.")
            transliterated_words.extend(batch)
        except Exception as e:
            logger.error(f"Transliteration batch {i} failed: {e}")
            transliterated_words.extend(batch)

    return transliterated_words


def _call_gemini_raw(prompt: str) -> str:
    """Raw wrapper for Gemini calls to handle text-in/text-out."""
    from google import genai
    from google.genai import types

    credentials = None
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if gcp_json:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(gcp_json)
        ).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])

    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "clippedai-493912")
    client = genai.Client(vertexai=True, project=gcp_project, location="us-central1", credentials=credentials)
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for precise transliteration
        )
    )
    return response.text
