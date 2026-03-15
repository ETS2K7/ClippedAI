"""
ClippedAI — Karaoke-style ASS Caption Generator

Phrase-based captions matching OpusClip style:
- Words grouped into 3-5 word phrases at natural boundaries
- Full phrase visible, active word highlights as spoken
- Smooth phrase transitions (no blank flash on short gaps)
"""
import os
import re
from pathlib import Path


# ASS file header template
ASS_HEADER = """[Script Info]
Title: ClippedAI Captions
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.cs)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _group_words_into_phrases(words: list[dict], max_words: int = 5) -> list[dict]:
    """
    Group words into phrases of 3-5 words based on natural speech boundaries.

    Uses punctuation and pauses as primary split points.
    Each phrase contains:
        - words: list of word dicts
        - start: start time of first word
        - end: end time of last word
        - speaker: speaker label
    """
    if not words:
        return []

    phrases = []
    current_phrase_words = []

    for i, word in enumerate(words):
        current_phrase_words.append(word)

        should_split = False

        # Split at punctuation boundaries
        if re.search(r'[.!?,;:]$', word["word"]):
            should_split = True

        # Split at max word count
        if len(current_phrase_words) >= max_words:
            should_split = True

        # Split at speaker change
        if i + 1 < len(words) and words[i + 1].get("speaker") != word.get("speaker"):
            should_split = True

        # Split at natural pauses (>300ms gap to next word)
        if i + 1 < len(words) and words[i + 1]["start"] - word["end"] > 0.3:
            should_split = True

        # Don't split if we have fewer than 2 words (minimum phrase size)
        if should_split and len(current_phrase_words) < 2 and i + 1 < len(words):
            # Unless it's a speaker change or long pause
            if i + 1 < len(words):
                gap = words[i + 1]["start"] - word["end"]
                speaker_change = words[i + 1].get("speaker") != word.get("speaker")
                if not speaker_change and gap <= 0.5:
                    should_split = False

        if should_split or i == len(words) - 1:
            if current_phrase_words:
                phrases.append({
                    "words": current_phrase_words,
                    "start": current_phrase_words[0]["start"],
                    "end": current_phrase_words[-1]["end"],
                    "speaker": current_phrase_words[0].get("speaker", "SPEAKER_00"),
                })
                current_phrase_words = []

    return phrases


def _generate_karaoke_events(
    phrases: list[dict],
    highlight_color: str = "&H0000FFFF",  # Yellow in ASS BGR format
    base_color: str = "&H00FFFFFF",       # White
    min_phrase_duration: float = 0.5,
) -> list[str]:
    """
    Generate ASS Dialogue events for karaoke-style captions.

    Each phrase is one Dialogue event. Within the event, ASS override tags
    highlight the active word at the correct time using \\k (karaoke) tags.
    """
    events = []

    for i, phrase in enumerate(phrases):
        words = phrase["words"]
        phrase_start = phrase["start"]
        phrase_end = max(phrase["end"], phrase_start + min_phrase_duration)

        # Check gap to next phrase
        if i + 1 < len(phrases):
            next_start = phrases[i + 1]["start"]
            gap = next_start - phrase_end
            # If gap > 500ms, phrase ends at its natural end
            # If gap < 500ms, extend phrase slightly to avoid blank flash
            if gap < 0.5 and gap > 0:
                phrase_end = max(phrase_end, next_start - 0.05)

        start_ts = _format_ass_time(phrase_start)
        end_ts = _format_ass_time(phrase_end)

        # Build karaoke text using \k tags
        # \k<duration_cs> highlights the next syllable/word for that duration
        text_parts = []
        for j, word in enumerate(words):
            # Duration of this word's highlight in centiseconds
            word_duration = max(word["end"] - word["start"], 0.2)
            duration_cs = int(round(word_duration * 100))

            # \kf = smooth fill karaoke (fills character by character)
            # \k = hard switch karaoke
            text_parts.append(f"{{\\kf{duration_cs}}}{word['word']}")

        # Join words with spaces
        full_text = " ".join(text_parts)

        # Add style overrides for karaoke colors
        # SecondaryColour is the "before highlight" color (base)
        # PrimaryColour becomes the "after highlight" color
        styled_text = (
            f"{{\\1c{highlight_color}\\2c{base_color}}}"
            f"{full_text}"
        )

        event = f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{styled_text}"
        events.append(event)

    return events


def generate_captions(
    transcript: dict,
    output_path: str,
    video_width: int = 1080,
    video_height: int = 1920,
) -> str:
    """
    Generate karaoke-style ASS captions from a transcript.

    Args:
        transcript: dict with "word_segments" list
        output_path: path to write .ass file
        video_width: output video width
        video_height: output video height

    Returns:
        Path to the generated .ass file
    """
    words = transcript.get("word_segments", [])
    if not words:
        # Write empty ASS file
        font_size = video_height // 25
        margin_v = int(video_height * 0.12)
        header = ASS_HEADER.format(
            play_res_x=video_width,
            play_res_y=video_height,
            font_size=font_size,
            margin_v=margin_v,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
        return output_path

    # Font sizing per spec: base size = output_height / 25
    font_size = video_height // 25
    margin_v = int(video_height * 0.12)  # Bottom 15% positioning

    # Generate header
    header = ASS_HEADER.format(
        play_res_x=video_width,
        play_res_y=video_height,
        font_size=font_size,
        margin_v=margin_v,
    )

    # Group words into phrases
    phrases = _group_words_into_phrases(words)

    # Generate karaoke events
    events = _generate_karaoke_events(phrases)

    # Write ASS file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for event in events:
            f.write(event + "\n")

    return output_path
