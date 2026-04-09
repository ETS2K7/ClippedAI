from typing import List, Dict, Any
from config import get_logger

logger = get_logger(__name__)


def ms_to_ass_time(ms: float) -> str:
    """Converts milliseconds to ASS video format (H:MM:SS.CC)."""
    hours = int(ms // 3600000)
    minutes = int((ms % 3600000) // 60000)
    seconds = int((ms % 60000) // 1000)
    cents = int((ms % 1000) // 10)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cents:02d}"


def generate_subtitles(
    words: List[Dict[str, Any]],
    clip: Dict[str, Any],
    idx: int,
    framing_meta: List[Dict[str, Any]],
) -> str:
    """
    Generates an .ASS subtitle file dynamically mapping words iteratively to
    the bounding box framing logic dictating its positional styling.
    """
    logger.info(
        f"==================== PHASE 6: SUBTITLE GENERATION (Clip {idx}) ===================="
    )
    out = f"temp_subtitles_{idx}.ass"
    start_ms = clip["start_time"] * 1000
    end_ms = clip["end_time"] * 1000

    def get_layout_for_time(ms: float) -> str:
        for meta in framing_meta:
            if meta["start_ms"] <= ms <= meta["end_ms"]:
                return meta["flag"]
        return "STATIONARY"

    clip_words = [
        w for w in words if w.get("start", 0) >= start_ms and w.get("end", 0) <= end_ms
    ]

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hormozi,Arial Black,50,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,3,2,10,10,350,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []
    chunks = []
    current_chunk = []
    MAX_WORDS = 3
    MAX_PAUSE_MS = 300

    # 1. Group words chunks dynamically based on pause timers + string punctuation max length breaks
    for i, w in enumerate(clip_words):
        current_chunk.append(w)
        is_last = i == len(clip_words) - 1

        if not is_last:
            next_w = clip_words[i + 1]
            pause_dur = next_w.get("start", 0) - w.get("end", 0)
            ends_with_punct = any(str(w["text"]).endswith(p) for p in [".", "?", "!"])
            too_long = len(current_chunk) >= MAX_WORDS
            long_pause = pause_dur > MAX_PAUSE_MS

            if ends_with_punct or too_long or long_pause:
                chunks.append(current_chunk)
                current_chunk = []
        else:
            chunks.append(current_chunk)

    for c_idx, chunk in enumerate(chunks):
        next_chunk_start = (
            chunks[c_idx + 1][0].get("start", float("inf"))
            if c_idx + 1 < len(chunks)
            else float("inf")
        )

        for w_idx, w in enumerate(chunk):
            w_start = w.get("start", 0) - start_ms

            if w_idx < len(chunk) - 1:
                w_end = max(w_start + 10, chunk[w_idx + 1].get("start", 0) - start_ms)
            else:
                w_end = w.get("end", 0) - start_ms
                actual_end = w.get("end", 0)
                pause_to_next_chunk = next_chunk_start - actual_end

                # Dynamic soft padding
                if pause_to_next_chunk > 0:
                    pad = min(pause_to_next_chunk, 400)
                    w_end += pad

            w_start = max(0, w_start)
            w_end = max(0, min(w_end, end_ms - start_ms))

            if w_end <= w_start:
                continue

            ass_start = ms_to_ass_time(w_start)
            ass_end = ms_to_ass_time(w_end)
            layout = get_layout_for_time(w_start)

            text_parts = []
            if layout == "SPLIT":
                text_parts.append("{\\an5\\pos(540,960)}")

            for j, cw in enumerate(chunk):
                # Escape ASS special syntax characters to prevent subtitle corruption
                raw_txt = str(cw.get("text", "")).upper()
                clean_txt = raw_txt.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
                if j == w_idx:
                    text_parts.append(
                        f"{{\\c&H0000FF00&}}{clean_txt}{{\\c&H00FFFFFF&}}"
                    )  # Green karaoke
                else:
                    text_parts.append(clean_txt)

            full_text = " ".join(text_parts)
            line = f"Dialogue: 0,{ass_start},{ass_end},Hormozi,,0,0,0,,{full_text}"
            lines.append(line)

    with open(out, "w") as f:
        f.write(header)
        for line in lines:
            f.write(line + "\n")

    return out
