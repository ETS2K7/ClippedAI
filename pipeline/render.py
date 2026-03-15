"""
ClippedAI — Vertical render with center-crop or dynamic speaker-following crop.
Supports burning in ASS karaoke captions via FFmpeg's ass filter.
"""
import json
import os
import subprocess
import tempfile

from config import OUTPUT_WIDTH, OUTPUT_HEIGHT, VIDEO_CODEC, VIDEO_CRF, VIDEO_PRESET


def center_crop_render(
    input_path: str,
    output_path: str,
    start: float | None = None,
    duration: float | None = None,
    ass_path: str | None = None,
) -> str:
    """
    Render a center-cropped 9:16 vertical clip from a source video.
    Fallback for when no crop plan is available.
    """
    cmd = ["ffmpeg", "-y"]

    if start is not None:
        cmd += ["-ss", str(start)]

    cmd += ["-i", input_path]

    if duration is not None:
        cmd += ["-t", str(duration)]

    filters = [
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos",
    ]

    if ass_path:
        escaped_path = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filters.append(f"ass='{escaped_path}'")

    cmd += [
        "-vf", ",".join(filters),
        "-c:v", VIDEO_CODEC,
        "-crf", str(VIDEO_CRF),
        "-preset", VIDEO_PRESET,
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg render failed:\n{result.stderr[-2000:]}"
        )

    return output_path


def dynamic_crop_render(
    input_path: str,
    output_path: str,
    crop_plan_path: str,
    ass_path: str | None = None,
) -> str:
    """
    Render a speaker-following vertical clip using a pre-computed crop plan.

    The crop plan contains 0.5s segments, each with (x, y, w, h) crop coords.
    Uses FFmpeg sendcmd filter to dynamically change crop position per segment.
    """
    with open(crop_plan_path) as f:
        crop_data = json.load(f)

    segments = crop_data["segments"]
    if not segments:
        # Fallback to center crop
        return center_crop_render(input_path, output_path, ass_path=ass_path)

    src_w = crop_data["source_width"]
    src_h = crop_data["source_height"]

    # Generate FFmpeg sendcmd script for dynamic crop
    # sendcmd changes crop filter parameters at specific timestamps
    sendcmd_lines = []
    for i, seg in enumerate(segments):
        t = seg["start_time"]
        crop = seg["crop"]
        x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
        sendcmd_lines.append(f"{t:.3f} crop x {x};")
        sendcmd_lines.append(f"{t:.3f} crop y {y};")

    # Write sendcmd script to temp file
    sendcmd_fd, sendcmd_path = tempfile.mkstemp(suffix=".cmd")
    with os.fdopen(sendcmd_fd, "w") as f:
        f.write("\n".join(sendcmd_lines))

    # Use the first segment's crop dimensions as the initial values
    first_crop = segments[0]["crop"]
    init_x = first_crop["x"]
    init_y = first_crop["y"]
    crop_w = first_crop["w"]
    crop_h = first_crop["h"]

    # Build filter chain:
    # 1. sendcmd to update crop params over time
    # 2. crop with initial values + dynamic updates
    # 3. scale to output dimensions
    # 4. (optional) ASS subtitles
    filters = [
        f"sendcmd=f='{sendcmd_path}'",
        f"crop={crop_w}:{crop_h}:{init_x}:{init_y}",
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos",
    ]

    if ass_path:
        escaped_path = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filters.append(f"ass='{escaped_path}'")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", ",".join(filters),
        "-c:v", VIDEO_CODEC,
        "-crf", str(VIDEO_CRF),
        "-preset", VIDEO_PRESET,
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg dynamic crop render failed:\n{result.stderr[-2000:]}"
            )
    finally:
        os.unlink(sendcmd_path)

    return output_path
