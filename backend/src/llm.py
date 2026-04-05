import json
from typing import List, Dict, Any
from pydantic import BaseModel
from google import genai
from config import get_logger, GEMINI_KEY

logger = get_logger(__name__)

# Lazy-initialised so importing this module doesn't crash when GEMINI_KEY is
# absent (e.g. during unit tests or static analysis).
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_KEY)
    return _gemini_client


class ClipSelection(BaseModel):
    start_time: float
    end_time: float
    title: str


class ClipList(BaseModel):
    clips: List[ClipSelection]


def select_clips(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups words into sentences and feeds them into Gemini 2.5 Flash to automatically
    select 3 high-retention viral segments between 30-60 seconds using Pydantic schemas.
    """
    logger.info(
        "==================== PHASE 3: VIRAL CLIP SELECTION ===================="
    )
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
        "Return ONLY valid JSON wrapping the start and end timestamps natively found in the text.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    logger.info("Calling Gemini 2.5 Flash for clip selection...")
    response = _get_gemini_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ClipList,
            "temperature": 0.2,
        },
    )

    try:
        data = json.loads(response.text)
        logger.info(f"Selected {len(data.get('clips', []))} clips.")
        return data["clips"]
    except Exception as e:
        logger.error(f"Failed to parse or receive output from Gemini: {e}")
        logger.error(f"Raw Output: {response.text}")
        raise
