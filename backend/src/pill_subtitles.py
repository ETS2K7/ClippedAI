"""
Pill Subtitle Renderer — draws captions directly onto video frames using PIL.

Style: white rounded-rectangle pill, dark charcoal text,
       currently-spoken word highlighted in coral (#FF6B47).
"""

from typing import List, Dict, Any
import os
import subprocess
import tempfile
from pathlib import Path
from config import get_logger

logger = get_logger(__name__)

# ── Style constants ────────────────────────────────────────────────────────────
PILL_BG           = (255, 255, 255, 220)   # white, slight transparency
TEXT_COLOR        = (30, 30, 30)           # dark charcoal
HIGHLIGHT_COLOR   = (255, 107, 71)         # coral  #FF6B47
FONT_SIZE         = 52                     # px, tuned for 1080-wide 9:16 video
PILL_PADDING_X    = 36
PILL_PADDING_Y    = 18
PILL_RADIUS       = 28
WORDS_PER_PILL    = 5                      # words shown in one pill at a time
BOTTOM_MARGIN     = 280                    # px from bottom of frame


def _get_font(size: int):
    """Return a PIL font, falling back to default if custom font unavailable."""
    try:
        from PIL import ImageFont
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/custom/KomikaAxis.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for c in candidates:
            if os.path.exists(c):
                return ImageFont.truetype(c, size)
        return ImageFont.load_default()
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def _draw_pill(draw, x: int, y: int, w: int, h: int, r: int, fill):
    """Draw a rounded rectangle pill."""
    from PIL import ImageDraw
    draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill)


def _render_pill_frame(frame_rgb, chunk: List[Dict], active_idx: int, frame_w: int, frame_h: int):
    """
    Overlays a pill caption onto a single RGB numpy frame (H, W, 3).
    Returns a modified numpy frame.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _get_font(FONT_SIZE)

    # Build word list with widths
    words_text = [str(w.get("text", "")).upper() for w in chunk]
    word_widths = []
    for wt in words_text:
        try:
            bb = font.getbbox(wt)
            word_widths.append(bb[2] - bb[0])
        except Exception:
            word_widths.append(font.getlength(wt) if hasattr(font, "getlength") else len(wt) * FONT_SIZE * 0.6)

    SPACE_W = 14
    total_text_w = sum(word_widths) + SPACE_W * (len(words_text) - 1)
    pill_w = total_text_w + 2 * PILL_PADDING_X
    pill_h = FONT_SIZE + 2 * PILL_PADDING_Y

    pill_x = (frame_w - pill_w) // 2
    pill_y = frame_h - BOTTOM_MARGIN - pill_h

    # Draw pill background
    _draw_pill(draw, pill_x, pill_y, pill_w, pill_h, PILL_RADIUS, PILL_BG)

    # Draw each word
    cursor_x = pill_x + PILL_PADDING_X
    text_y = pill_y + PILL_PADDING_Y

    for i, (wt, ww) in enumerate(zip(words_text, word_widths)):
        color = HIGHLIGHT_COLOR if i == active_idx else TEXT_COLOR
        draw.text((cursor_x, text_y), wt, font=font, fill=color)
        cursor_x += ww + SPACE_W

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    return combined


def generate_pill_subtitles(
    video_path: str,
    words: List[Dict[str, Any]],
    clip: Dict[str, Any],
    idx: int,
    work_dir: str = "",
) -> str:
    """
    Burns pill-style captions directly into the video.
    Returns the path to the new video with captions baked in.
    """
    import cv2
    import numpy as np

    logger.info(f"==================== PHASE 6: PILL SUBTITLES (Clip {idx}) ====================")

    start_ms = clip["start_time"] * 1000
    end_ms   = clip["end_time"]   * 1000

    # Apply Romanized Hindi mapping (same logic as ASS renderer)
    clip_words = [
        w for w in words if w.get("start", 0) >= start_ms and w.get("end", 0) <= end_ms
    ]
    rom_input = clip.get("romanized_words")
    if isinstance(rom_input, list) and clip_words:
        word_idx = 0
        for segment in rom_input:
            segment_pairs = [p.strip() for p in segment.split("|") if ":" in p]
            for pair in segment_pairs:
                if word_idx >= len(clip_words): break
                parts = pair.split(":", 1)
                if len(parts) == 2:
                    clip_words[word_idx]["text"] = parts[0].strip()
                word_idx += 1

    # Build chunks of WORDS_PER_PILL
    chunks: List[List[Dict]] = []
    for i in range(0, len(clip_words), WORDS_PER_PILL):
        chunks.append(clip_words[i:i + WORDS_PER_PILL])

    # Build a lookup: for each millisecond, which chunk/word-within-chunk is active
    def get_active(t_ms: float):
        """Returns (chunk, active_word_idx) for time t_ms relative to clip start."""
        abs_t = t_ms + start_ms
        for ci, chunk in enumerate(chunks):
            chunk_start = chunk[0].get("start", 0)
            chunk_end   = chunk[-1].get("end", 0)
            if chunk_start <= abs_t <= chunk_end:
                for wi, w in enumerate(chunk):
                    if w.get("start", 0) <= abs_t <= w.get("end", 0):
                        return chunk, wi
                # In gap between words within chunk — highlight last active
                return chunk, max(0, len(chunk) - 1)
            # Between chunks — show previous chunk until next starts
        return None, -1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[Pill] Cannot open video: {video_path}")
        return video_path

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(work_dir, f"pill_{idx}.mp4") if work_dir else f"pill_{idx}.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = (frame_num / fps) * 1000.0  # relative to clip start
        chunk, active_wi = get_active(t_ms)

        if chunk is not None and active_wi >= 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rendered  = _render_pill_frame(frame_rgb, chunk, active_wi, fw, fh)
            frame     = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()

    # Re-mux with original audio using ffmpeg (mp4v loses audio)
    final_path = os.path.join(work_dir, f"pill_audio_{idx}.mp4") if work_dir else f"pill_audio_{idx}.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", out_path,
            "-i", video_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-shortest",
            final_path
        ], check=True, capture_output=True)
        os.remove(out_path)
        return final_path
    except Exception as e:
        logger.warning(f"[Pill] ffmpeg remux failed: {e}, returning raw output")
        return out_path
