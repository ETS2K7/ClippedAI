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
from config import get_logger

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
        "5. PUNCHY TITLE: Create a viral, high-value title for each clip.\n"
        "6. ROMANIZED HINDI: If the clip contains Hindi (Devanagari script), you MUST provide a 'romanized_transcript' field. "
        "Transliterate the Hindi words into Romanized Hindi (Latin script). Keep the exact same word count and order as the original so timing stays synced.\n\n"
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
                                        "virality_score": {"type": "NUMBER"},
                                        "romanized_words": {
                                            "type": "ARRAY",
                                            "description": "List of word pairs for transliteration. MUST match original word order.",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "original": {"type": "STRING"},
                                                    "romanized": {"type": "STRING"}
                                                },
                                                "required": ["original", "romanized"]
                                            }
                                        }
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



