"""
Pipeline Step 8 — Final Render
FFmpeg segment cutting + vertical reframe + caption burn + audio normalize.
"""

import subprocess
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, VIDEO_CODEC, VIDEO_CRF, VIDEO_PRESET,
    AUDIO_CODEC, AUDIO_BITRATE, AUDIO_SAMPLE_RATE, LOUDNORM_TARGET,
    TARGET_ASPECT_RATIO,
)

WORK_DIR = "/tmp/clipped"
SEGMENTS_DIR = os.path.join(WORK_DIR, "segments")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")


def cut_segment(video_path: str, start: float, end: float, clip_index: int) -> str:
    """
    Cut a raw segment from the source video without re-encoding.

    Returns: path to raw segment file
    """
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    segment_path = os.path.join(SEGMENTS_DIR, f"clip_{clip_index:02d}_raw.mp4")

    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-y",
        segment_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg segment cut failed: {result.stderr}")

    return segment_path


def final_render(
    segment_path: str,
    crop_x: int,
    source_w: int,
    source_h: int,
    ass_path: str,
    clip_index: int,
) -> str:
    """
    Single-pass FFmpeg render:
    1. Crop to 9:16 region (face-centered)
    2. Scale to 1080x1920
    3. Burn captions (.ass)
    4. Normalize audio (loudnorm -14 LUFS)
    5. Encode H.264 + AAC

    Returns: path to final rendered clip
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"clip_{clip_index:02d}.mp4")

    # Compute crop dimensions
    crop_w = int(source_h * TARGET_ASPECT_RATIO)
    if crop_w > source_w:
        # Source is narrower than 9:16 — use full width, crop height
        crop_w = source_w
        crop_h = int(source_w / TARGET_ASPECT_RATIO)
        crop_y = max(0, (source_h - crop_h) // 2)
        crop_filter = f"crop={crop_w}:{crop_h}:0:{crop_y}"
    else:
        crop_filter = f"crop={crop_w}:{source_h}:{crop_x}:0"

    # Build video filter chain
    vf_parts = [
        crop_filter,
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos",
    ]

    # Add captions if .ass file has content
    if ass_path and os.path.exists(ass_path):
        # Escape special characters in path for FFmpeg filter
        ass_escaped = ass_path.replace("\\", "/").replace(":", "\\\\:").replace("'", "\\\\'")
        vf_parts.append(f"ass='{ass_escaped}'")

    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-i", segment_path,
        "-vf", vf,
        "-af", f"loudnorm=I={LOUDNORM_TARGET}:TP=-1.5:LRA=11",
        "-c:v", VIDEO_CODEC,
        "-preset", VIDEO_PRESET,
        "-crf", str(VIDEO_CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-movflags", "+faststart",
        "-y",
        output_path,
    ]

    print(f"    🔧 Rendering clip {clip_index + 1}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg render failed for clip {clip_index}: {result.stderr[-500:]}"
        )

    # Verify output exists and has content
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Rendered file not found: {output_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    ✅ Clip {clip_index + 1} rendered ({size_mb:.1f} MB)")

    return output_path
