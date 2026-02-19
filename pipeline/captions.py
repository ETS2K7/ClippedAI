"""
Pipeline Step 7 — Captions
Generate .ass subtitle files with strict 1-to-1 word-level timing.

Each word appears ONLY when spoken and disappears IMMEDIATELY when it ends.
No grouping. No merging. No lookahead. One word at a time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAPTION_FONT, CAPTION_FONT_SIZE, CAPTION_HIGHLIGHT_COLOR,
    CAPTION_DEFAULT_COLOR, CAPTION_OUTLINE_COLOR, CAPTION_OUTLINE_WIDTH,
    CAPTION_MARGIN_BOTTOM,
    OUTPUT_WIDTH, OUTPUT_HEIGHT,
)

WORK_DIR = "/tmp/clipped"


def generate_ass(words: list, clip_index: int) -> str:
    """
    Generate an .ass subtitle file with strict word-level timing.

    One word on screen at a time. Appears when spoken. Disappears when
    the word ends. No previous words, no upcoming words.

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

    events = []
    for word in words:
        t_start = word["start"]
        t_end = word["end"]

        # Prevent 0-duration flickering
        if t_end - t_start < 0.05:
            t_end = t_start + 0.05

        # Only the current word is shown — bold + highlighted
        text = (
            f"{{\\b1\\c&H{CAPTION_HIGHLIGHT_COLOR}&}}"
            f"{word['word']}"
        )

        events.append({"start": t_start, "end": t_end, "text": text})

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
