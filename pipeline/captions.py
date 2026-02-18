"""
Pipeline Step 7 — Captions
Generate .ass subtitle files with strict word-level timing.

Each word appears ONLY when spoken and disappears IMMEDIATELY when it ends.
No grouping, no merging, no lookahead.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAPTION_FONT, CAPTION_FONT_SIZE, CAPTION_HIGHLIGHT_COLOR,
    CAPTION_DEFAULT_COLOR, CAPTION_OUTLINE_COLOR, CAPTION_OUTLINE_WIDTH,
    CAPTION_WORDS_PER_GROUP, CAPTION_MARGIN_BOTTOM,
    OUTPUT_WIDTH, OUTPUT_HEIGHT,
)

WORK_DIR = "/tmp/clipped"

# Maximum gap (seconds) between words before we consider it a new phrase.
# Words within this gap are shown together as a rolling phrase.
PHRASE_GAP_THRESHOLD = 0.4


def generate_ass(words: list, clip_index: int) -> str:
    """
    Generate an .ass subtitle file with strict word-level timing.

    Each word is highlighted exactly when spoken. Words are shown in
    phrases (up to CAPTION_WORDS_PER_GROUP), but only words that have
    ALREADY been spoken remain visible. Future words are never shown.

    A phrase ends (all text disappears) when the last spoken word ends
    or when there's a gap > PHRASE_GAP_THRESHOLD between words.

    Args:
        words: [{"word": "Hey", "start": 0.0, "end": 0.28}, ...]
               Timestamps are relative to clip start (0-based).
        clip_index: For unique filename.

    Returns: Path to .ass file
    """
    if not words:
        ass_path = os.path.join(WORK_DIR, f"clip_{clip_index:02d}.ass")
        with open(ass_path, "w") as f:
            f.write(_ass_header())
        return ass_path

    # Split words into phrases based on timing gaps
    phrases = _split_into_phrases(words)

    # Build ASS events — one event per word, showing accumulated words in phrase
    events = []
    for phrase in phrases:
        for word_idx in range(len(phrase)):
            active_word = phrase[word_idx]

            # Build display: show all words from phrase start up to (and including)
            # the current active word. Words already spoken are default color,
            # the current word is highlighted.
            parts = []
            for j in range(word_idx + 1):
                w = phrase[j]
                if j == word_idx:
                    # Active (current) word — highlighted
                    parts.append(
                        f"{{\\b1\\c&H{CAPTION_HIGHLIGHT_COLOR}&}}"
                        f"{w['word']}"
                        f"{{\\b0\\c&H{CAPTION_DEFAULT_COLOR}&}}"
                    )
                else:
                    # Already spoken — default color
                    parts.append(w["word"])

            line_text = " ".join(parts)

            # Start: exactly when this word begins
            t_start = active_word["start"]

            # End: exactly when this word ends (NOT when the next word starts)
            t_end = active_word["end"]

            # Ensure minimum display time of 50ms
            if t_end - t_start < 0.05:
                t_end = t_start + 0.05

            events.append({
                "start": t_start,
                "end": t_end,
                "text": line_text,
            })

    # Write .ass file
    ass_path = os.path.join(WORK_DIR, f"clip_{clip_index:02d}.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(_ass_header())
        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for event in events:
            start_str = _format_ass_time(event["start"])
            end_str = _format_ass_time(event["end"])
            f.write(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{event['text']}\n"
            )

    return ass_path


def _split_into_phrases(words: list) -> list:
    """
    Split words into phrases based on timing gaps and max group size.

    A new phrase starts when:
    - Gap between previous word end and next word start > PHRASE_GAP_THRESHOLD
    - Phrase reaches CAPTION_WORDS_PER_GROUP words
    """
    if not words:
        return []

    phrases = []
    current_phrase = [words[0]]

    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]

        gap = curr_word["start"] - prev_word["end"]
        phrase_full = len(current_phrase) >= CAPTION_WORDS_PER_GROUP

        if gap > PHRASE_GAP_THRESHOLD or phrase_full:
            # Start new phrase
            phrases.append(current_phrase)
            current_phrase = [curr_word]
        else:
            current_phrase.append(curr_word)

    if current_phrase:
        phrases.append(current_phrase)

    return phrases


def _ass_header() -> str:
    """Generate ASS file header with style definition."""
    return f"""[Script Info]
Title: ClippedAI Captions
ScriptType: v4.00+
PlayResX: {OUTPUT_WIDTH}
PlayResY: {OUTPUT_HEIGHT}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{CAPTION_FONT},{CAPTION_FONT_SIZE},&H00{CAPTION_DEFAULT_COLOR},&H000000FF,&H00{CAPTION_OUTLINE_COLOR},&H80000000,-1,0,0,0,100,100,0,0,1,{CAPTION_OUTLINE_WIDTH},0,2,40,40,{CAPTION_MARGIN_BOTTOM},1
"""


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time format: H:MM:SS.CC"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
