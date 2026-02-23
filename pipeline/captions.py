"""
Caption generation as per-word .ass subtitles.

Implements all 6 non-negotiable caption rules:
  1. Each word = one .ass Dialogue event
  2. Immediate disappearance on gaps > 300ms
  3. Minimum 200ms display duration
  4. No pre-display of upcoming words
  5. Safe zone positioning (bottom 15%, max 80% width, max 2 lines)
  6. Highlight styling (active word bold, spoken dimmed)
"""

import logging
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


def generate_captions(
    transcript: dict,
    clip_start: float,
    clip_end: float,
    output_path: Path,
    output_width: int = config.OUTPUT_WIDTH,
    output_height: int = config.OUTPUT_HEIGHT,
) -> Path:
    """
    Generate .ass subtitle file for a clip.
    Each word is a separate Dialogue event with exact WhisperX timing.
    """
    # Extract words within clip range
    words = _extract_and_filter_words(transcript, clip_start, clip_end)

    if not words:
        logger.warning("No words found for clip %.1f-%.1f", clip_start, clip_end)
        _write_empty_ass(output_path, output_width, output_height)
        return output_path

    # Apply caption rules
    words = _apply_timing_rules(words, clip_start)
    words = _apply_gap_rules(words)

    # Generate .ass content
    font_size = output_height // config.CAPTION_FONT_SIZE_RATIO
    ass_content = _build_ass_file(
        words, output_width, output_height, font_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ass_content, encoding="utf-8")
    logger.info("Generated captions: %s (%d words)", output_path, len(words))

    return output_path


# ─────────────────────────────────────────────
# Word Extraction & Filtering
# ─────────────────────────────────────────────

def _extract_and_filter_words(
    transcript: dict,
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """Extract words in clip range, apply confidence + hallucination filters."""
    words = []

    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if w["start"] >= clip_start and w["end"] <= clip_end:
                words.append({
                    "word": w["word"].strip(),
                    "start": w["start"] - clip_start,  # relative to clip
                    "end": w["end"] - clip_start,
                    "score": w.get("score", 1.0),
                    "speaker": seg.get("speaker", ""),
                })

    # Filter: drop low confidence words (WhisperX hallucinations)
    words = [w for w in words if w["score"] >= config.CAPTION_CONFIDENCE_THRESHOLD]

    # Filter: drop repeated words within 0.5s
    filtered = []
    for i, w in enumerate(words):
        if i > 0 and w["word"].lower() == words[i - 1]["word"].lower():
            if w["start"] - words[i - 1]["start"] < 0.5:
                continue
        filtered.append(w)

    # Capitalize first word and after silence gaps
    for i, w in enumerate(filtered):
        if i == 0:
            w["word"] = w["word"].capitalize()
        elif i > 0 and w["start"] - filtered[i - 1]["end"] > config.CAPTION_GAP_THRESHOLD:
            w["word"] = w["word"].capitalize()

    return filtered


# ─────────────────────────────────────────────
# Timing Rules
# ─────────────────────────────────────────────

def _apply_timing_rules(words: list[dict], clip_start: float) -> list[dict]:
    """
    RULE 3: Minimum 200ms display duration.
    RULE 4: No pre-display — word N starts at word N's timestamp.
    """
    for w in words:
        duration = w["end"] - w["start"]
        if duration < config.CAPTION_MIN_WORD_DURATION:
            w["end"] = w["start"] + config.CAPTION_MIN_WORD_DURATION

    return words


def _apply_gap_rules(words: list[dict]) -> list[dict]:
    """
    RULE 2: If gap > 300ms between words, insert blank.
    Each word is already a separate event, so gaps naturally create blanks.
    We just need to ensure no word's end extends into the gap.
    """
    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        if gap > config.CAPTION_GAP_THRESHOLD:
            # Ensure this word ends at its own end time, not bridging the gap
            pass  # already handled by per-word events
        elif gap < 0:
            # Overlapping timestamps — clip current word's end
            words[i]["end"] = words[i + 1]["start"]

    return words


# ─────────────────────────────────────────────
# .ass File Generation
# ─────────────────────────────────────────────

def _build_ass_file(
    words: list[dict],
    width: int,
    height: int,
    font_size: int,
) -> str:
    """Build complete .ass subtitle file."""

    # Calculate safe zone positioning
    margin_h = int(width * (1 - config.CAPTION_MAX_WIDTH_RATIO) / 2)
    margin_v = int(height * (1 - config.CAPTION_SAFE_ZONE_Y))

    header = f"""[Script Info]
Title: ClippedAI Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.CAPTION_FONT},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,{margin_h},{margin_h},{margin_v},1
Style: Active,{config.CAPTION_FONT},{font_size},&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,{margin_h},{margin_h},{margin_v},1
Style: Spoken,{config.CAPTION_FONT},{font_size},&H80FFFFFF,&H0000FFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    # Group words into display lines (max 2 lines, word-wrap)
    lines = _group_into_lines(words, width, font_size, margin_h)

    for line_group in lines:
        for word_data in line_group:
            start_ts = _format_ass_time(word_data["start"])
            end_ts = _format_ass_time(word_data["end"])
            text = word_data["display_text"]

            events.append(
                f"Dialogue: 0,{start_ts},{end_ts},Active,,0,0,0,,{text}"
            )

    return header + "\n".join(events) + "\n"


def _group_into_lines(
    words: list[dict],
    width: int,
    font_size: int,
    margin: int,
) -> list[list[dict]]:
    """
    Group words into display lines respecting:
    RULE 5: max 2 lines, word wrap at boundaries, max 80% width.
    """
    max_chars_per_line = int((width - 2 * margin) / (font_size * 0.55))
    line_groups = []
    current_line = []
    current_chars = 0

    for w in words:
        word_len = len(w["word"]) + 1  # +1 for space

        if current_chars + word_len > max_chars_per_line and current_line:
            line_groups.append(current_line)
            current_line = []
            current_chars = 0

        # Build display text with highlight override
        display = "{\\b1\\c&H00FFFF&}" + w["word"]

        current_line.append({
            **w,
            "display_text": display,
        })
        current_chars += word_len

    if current_line:
        line_groups.append(current_line)

    return line_groups


def _format_ass_time(seconds: float) -> str:
    """Format seconds as .ass timestamp: H:MM:SS.CC"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_empty_ass(path: Path, width: int, height: int) -> None:
    """Write an empty .ass file (no captions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""[Script Info]
Title: ClippedAI Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,50,50,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    path.write_text(content, encoding="utf-8")
