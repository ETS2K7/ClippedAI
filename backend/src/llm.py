"""
Module for LLM-powered viral clip selection using Google Gemini.
"""

import functools
import hashlib
import os
import json
import pathlib
from typing import List, Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types
from config import get_logger

logger = get_logger(__name__)


class ClipSelection(BaseModel):
    start_time: float
    end_time: float
    title: str
    virality_score: float  # 0.0–10.0 viral potential score


class ClipList(BaseModel):
    clips: List[ClipSelection]


def select_clips(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups words into sentences and feeds them into Gemini to automatically
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

        # Determine sentence boundaries
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
    # Key: SHA-256 of the exact transcript text sent to the model.
    # Same video = same transcript = same key = same clips returned without
    # hitting Vertex AI, ensuring repeatable outputs for debugging/validation.
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

    logger.info(f"[LLM] 🔴 Cache miss (key={_cache_key[:8]}). Calling Vertex AI...")

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
        '...]}\n'
        '- start_time/end_time: float in seconds\n'
        '- title: a short, attention-grabbing title (max 10 words) that could serve as a caption\n'
        '- virality_score: float from 0.0 to 10.0 representing viral potential\n\n'
        f"TRANSCRIPT:\n{transcript}"
    )

    logger.info("Calling Vertex AI for clip selection...")

    import time as _time
    MAX_RETRIES = 3
    last_error = None
    response = None

    client = genai.Client(
        vertexai=True,
        project="clippedai-493912",
        location="us-central1",
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an expert short-form video editor specializing in creating viral clips "
                        "for TikTok, YouTube Shorts, and Instagram Reels. You have deep expertise in "
                        "audience retention, hook psychology, and narrative pacing. Return only valid JSON."
                    ),
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            data = json.loads(response.text)

            # Normalize response shape
            if "clips" in data and isinstance(data["clips"], list):
                raw_clips = data["clips"]
            else:
                raw_clips = [
                    v for k, v in data.items()
                    if isinstance(v, dict) and ("start" in v or "start_time" in v)
                ]

            if not raw_clips:
                raise ValueError(f"LLM returned unrecognized JSON structure: {list(data.keys())}")

            # Post-validate clip timestamps
            video_end_s = words[-1]["end"] / 1000.0
            validated_clips = []
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
                    logger.warning(f"Skipping clip with unusual duration ({duration:.1f}s): {clip}")
                    continue
                validated_clips.append(clip)

            if len(validated_clips) < 3:
                raise ValueError(f"Expected exactly 3 valid clips, but got {len(validated_clips)}.")

            validated_clips = validated_clips[:3]
            logger.info(f"Selected exactly {len(validated_clips)} clips (validated).")

            # Write to cache so the next run with the same video skips the API.
            try:
                _cache_file.write_text(json.dumps(validated_clips), "utf-8")
                logger.info(f"[LLM] Cached clip selection to {_cache_file.name}")
            except Exception as _we:
                logger.warning(f"[LLM] Cache write failed (non-fatal): {_we}")

            return validated_clips

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"Vertex AI API or validation failed (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"retrying in {wait}s: {e}"
                )
                if response:
                    logger.debug(f"Failed LLM output: {response.text}")
                _time.sleep(wait)
            else:
                logger.error(
                    f"Vertex AI API call failed after {MAX_RETRIES} attempts: {e}"
                )
                raise RuntimeError("Vertex AI max retries exceeded.") from e
