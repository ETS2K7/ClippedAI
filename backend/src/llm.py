"""
Module for LLM-powered viral clip selection using Groq (primary) with Gemini fallback.
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
    Groups words into sentences and feeds them into Groq to automatically
    select 3 high-retention viral segments between 15-45 seconds.
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

    # ── Transcript-level cache ────────────────────────────────────────────────
    _cache_dir = pathlib.Path.home() / ".clippedai" / "cache" / "llm"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_key = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    _cache_file = _cache_dir / f"llm_{_cache_key}.json"

    if _cache_file.exists():
        try:
            cached = json.loads(_cache_file.read_text("utf-8"))
            if isinstance(cached, list) and len(cached) >= 3:
                logger.info(
                    f"[LLM] 🟢 Cache hit — returning cached clip selection "
                    f"(key={_cache_key[:8]})"
                )
                return cached[:3]
            else:
                logger.warning("[LLM] Cache entry invalid, re-running selection.")
        except Exception as _ce:
            logger.warning(f"[LLM] Cache read failed ({_ce}), re-running selection.")

    logger.info(f"[LLM] 🔴 Cache miss (key={_cache_key[:8]}). Calling Groq...")

    prompt = (
        "Analyze this transcript and extract exactly 3 clips optimized for maximum "
        "viral potential on short-form platforms (TikTok, YouTube Shorts, Instagram Reels).\n\n"
        "## CLIP SELECTION RULES\n"
        "1. Each clip MUST be between 15 and 45 seconds long.\n"
        "2. Each clip MUST begin with a strong hook — a surprising statement, bold claim, "
        "emotional moment, or curiosity-inducing question — within the first 3 seconds.\n"
        "3. Each clip MUST end on a complete thought. Never cut mid-sentence or mid-idea.\n"
        "4. Clips MUST NOT overlap with each other.\n"
        "5. Spread clips across different sections of the video. Do NOT cluster them together.\n\n"
        "## WHAT MAKES A CLIP VIRAL\n"
        "Prioritize moments that contain:\n"
        "- Controversial or counterintuitive opinions\n"
        "- Surprising facts or statistics\n"
        "- Emotional intensity (passion, humor, anger, vulnerability)\n"
        "- Universal relatability ('everyone has experienced this')\n"
        "- Actionable advice or 'life hack' energy\n"
        "- Storytelling with tension and payoff\n\n"
        "## WHAT TO AVOID\n"
        "- Generic introductions or 'welcome to the show' segments\n"
        "- Rambling or unfocused dialogue without a clear point\n"
        "- Segments that require prior context to understand\n"
        "- Moments where the speaker trails off or loses energy\n\n"
        "## OUTPUT FORMAT\n"
        'Return ONLY this exact JSON structure:\n'
        '{"clips": ['
        '{"start_time": 12.3, "end_time": 45.6, "title": "Short punchy hook title", "virality_score": 8.5}, '
        '...]}\\n'
        '- start_time/end_time: float in seconds\n'
        '- title: a short, attention-grabbing title (max 10 words) that could serve as a caption\n'
        '- virality_score: float from 0.0 to 10.0 representing viral potential\n\n'
        f"TRANSCRIPT:\n{transcript}"
    )

    validated_clips = _call_groq(prompt, words)

    # Write to cache
    try:
        _cache_file.write_text(json.dumps(validated_clips), "utf-8")
        logger.info(f"[LLM] Cached clip selection to {_cache_file.name}")
    except Exception as _we:
        logger.warning(f"[LLM] Cache write failed (non-fatal): {_we}")

    return validated_clips


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
        if duration < 15 or duration > 45:
            logger.warning(f"Skipping clip with unusual duration ({duration:.1f}s)")
            continue
        validated.append(clip)
    return validated


def _call_groq(prompt: str, words: list) -> list:
    """Primary: Groq (llama-3.3-70b-versatile) — fastest inference available."""
    groq_key = os.environ.get("GROQ_KEY")
    if not groq_key:
        logger.warning("[LLM] GROQ_KEY not set, falling back to Gemini.")
        return _call_gemini(prompt, words)

    logger.info("[LLM] Calling Groq (llama-3.3-70b-versatile)...")
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = json.loads(resp.json()["choices"][0]["message"]["content"])

            raw_clips = data.get("clips") if isinstance(data.get("clips"), list) else []
            if not raw_clips:
                raise ValueError(f"Groq returned unexpected JSON keys: {list(data.keys())}")

            validated = _validate_clips(raw_clips, words)
            if len(validated) < 3:
                raise ValueError(f"Only {len(validated)} valid clips after validation.")

            logger.info(f"[LLM] ✓ Groq selected {len(validated)} clips.")
            return validated[:3]

        except Exception as e:
            wait = 2 ** (attempt + 1)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"[LLM] Groq attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"[LLM] Groq failed after {MAX_RETRIES} attempts: {e}. Falling back to Gemini...")
                return _call_gemini(prompt, words)


def _call_gemini(prompt: str, words: list) -> list:
    """Fallback: Gemini 2.5-flash via standard Gemini API."""
    from google import genai
    from google.genai import types

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("Neither GROQ_KEY nor GEMINI_API_KEY is set. Cannot select clips.")

    logger.info("[LLM Fallback] Calling Gemini 2.5-flash...")
    MAX_RETRIES = 5
    client = genai.Client(api_key=gemini_key)
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
                    temperature=0.2,
                ),
            )
            data = json.loads(response.text)
            raw_clips = data.get("clips") if isinstance(data.get("clips"), list) else []
            if not raw_clips:
                raise ValueError(f"Gemini returned unexpected JSON: {list(data.keys())}")

            validated = _validate_clips(raw_clips, words)
            if len(validated) < 3:
                raise ValueError(f"Only {len(validated)} valid clips after validation.")

            logger.info(f"[LLM Fallback] ✓ Gemini selected {len(validated)} clips.")
            return validated[:3]

        except Exception as e:
            wait = min(2 ** (attempt + 1), 30)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"[LLM Fallback] Gemini attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s...")
                if response:
                    logger.debug(f"Gemini output: {response.text}")
                time.sleep(wait)
            else:
                logger.error(f"[LLM Fallback] Gemini also failed after {MAX_RETRIES} attempts: {e}")
                raise RuntimeError("Both Groq and Gemini failed to select clips.") from e
