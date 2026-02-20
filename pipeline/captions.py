"""
Pipeline Step 7 — Captions
Generate .ass subtitle files with grouped word display + sequential highlighting.

Words are displayed in groups of CAPTION_WORDS_PER_GROUP. The current spoken word
is highlighted (bold + cyan), while other words in the group are dimmed white.
Groups advance when the last word in a group finishes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAPTION_FONT, CAPTION_FONT_SIZE, CAPTION_HIGHLIGHT_COLOR,
    CAPTION_DEFAULT_COLOR, CAPTION_OUTLINE_COLOR, CAPTION_OUTLINE_WIDTH,
    CAPTION_MARGIN_BOTTOM, CAPTION_WORDS_PER_GROUP,
    OUTPUT_WIDTH, OUTPUT_HEIGHT,
)

WORK_DIR = "/tmp/clipped"


def generate_ass(words: list, clip_index: int) -> str:
    """
    Generate an .ass subtitle file with grouped captions.

    Words appear in groups (e.g. 4 at a time). The spoken word is highlighted
    cyan+bold, while context words are dimmed white. This is more readable
    than single-word flashing while still showing precise timing.

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

    # Split words into groups
    group_size = CAPTION_WORDS_PER_GROUP
    groups = []
    for i in range(0, len(words), group_size):
        groups.append(words[i:i + group_size])

    # Generate events: one event per word, showing the full group with active highlight
    events = []
    for group in groups:
        for active_idx, active_word in enumerate(group):
            t_start = active_word["start"]
            t_end = active_word["end"]

            # Prevent 0-duration flickering
            if t_end - t_start < 0.05:
                t_end = t_start + 0.05

            # Build the display text: all words in group, active one highlighted
            parts = []
            for j, w in enumerate(group):
                if j == active_idx:
                    # Active word: bold + highlight color
                    parts.append(
                        f"{{\\b1\\c&H{CAPTION_HIGHLIGHT_COLOR}&}}{w['word']}"
                    )
                else:
                    # Context word: normal weight + default color
                    parts.append(
                        f"{{\\b0\\c&H{CAPTION_DEFAULT_COLOR}&}}{w['word']}"
                    )

            text = " ".join(parts)
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
