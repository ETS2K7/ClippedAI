"""
Module for interacting with Groq models for context-aware clip generation.
"""

import functools
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from groq import Groq
from config import get_logger, GROQ_KEY

logger = get_logger(__name__)

@functools.lru_cache(maxsize=1)
def _get_groq_client():
    return Groq(api_key=GROQ_KEY())


class ClipSelection(BaseModel):
    start_time: float
    end_time: float
    title: str
    virality_score: float  # 0.0–10.0 viral potential score


class ClipList(BaseModel):
    clips: List[ClipSelection]


def select_clips(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups words into sentences and feeds them into Groq to automatically
    select 3 high-retention viral segments between 30-60 seconds using Pydantic schemas.
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
    prompt = (
        "You are a master TikTok video editor. Review the following transcript. "
        "Find the 3 most viral, engaging clips. Each must be exactly 30 to 60 seconds long. "
        "They must have a strong hook at the start and conclude an interesting point. "
        'Return ONLY this exact JSON format with no extra keys:\n'
        '{"clips": [{"start_time": 12.3, "end_time": 45.6, "title": "Short punchy hook title", "virality_score": 8.5}, ...]}\n'
        'virality_score is a float from 0.0 to 10.0 representing viral potential.\n\n'
        f"TRANSCRIPT:\n{transcript}"
    )

    logger.info("Calling Groq for clip selection...")

    # Retry with exponential backoff for transient API failures
    import time as _time
    MAX_RETRIES = 3
    last_error = None
    response = None

    for attempt in range(MAX_RETRIES):
        try:
            response = _get_groq_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a master TikTok video editor. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=120.0,  # 2 minute timeout
            )
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"Groq API call failed (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"retrying in {wait}s: {e}"
                )
                _time.sleep(wait)
            else:
                logger.error(
                    f"Groq API call failed after {MAX_RETRIES} attempts: {e}"
                )
                raise RuntimeError("Groq max retries exceeded.") from e

    try:
        data = json.loads(response.choices[0].message.content)

        # Normalize response shape — LLM sometimes returns {clips:[...]} or {clip1:{...}, clip2:{...}}
        if "clips" in data and isinstance(data["clips"], list):
            raw_clips = data["clips"]
        else:
            # Flatten clip1/clip2/clip3 or any dict-of-dicts shape
            raw_clips = [
                v for k, v in data.items()
                if isinstance(v, dict) and ("start" in v or "start_time" in v)
            ]

        if not raw_clips:
            raise ValueError(f"LLM returned unrecognized JSON structure: {list(data.keys())}")

        # Post-validate clip timestamps — normalize start/end aliases
        video_end_s = words[-1]["end"] / 1000.0
        validated_clips = []
        for clip in raw_clips:
            start = float(clip.get("start_time") or clip.get("start") or 0)
            end   = float(clip.get("end_time")   or clip.get("end")   or 0)
            # Normalize to start_time/end_time keys for downstream consistency
            clip["start_time"] = start
            clip["end_time"]   = end
            duration = end - start
            if start < 0 or end <= start or start > video_end_s:
                logger.warning(f"Skipping invalid clip: start={start}, end={end}")
                continue
            if duration < 10 or duration > 120:
                logger.warning(f"Skipping clip with unusual duration ({duration:.1f}s): {clip}")
                continue
            validated_clips.append(clip)

        if not validated_clips:
            logger.error("No valid clips after timestamp validation")
            raise ValueError("LLM returned no valid clips")

        logger.info(f"Selected {len(validated_clips)} clips (validated).")
        return validated_clips
    except Exception as e:
        logger.error(f"Failed to parse or receive output from Groq: {e}")
        logger.error(f"Raw Output: {response.choices[0].message.content if response else 'No response'}")
        raise

