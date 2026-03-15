"""
ClippedAI — Center-crop vertical render with optional caption burn-in.
Takes a source video segment and produces a 9:16 center-cropped vertical clip.
Supports burning in ASS karaoke captions via FFmpeg's ass filter.
"""
import subprocess

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

    Args:
        input_path: source video path
        output_path: rendered clip output path
        start: optional start time in seconds
        duration: optional clip duration in seconds
        ass_path: optional path to .ass file for caption burn-in
    """
    cmd = ["ffmpeg", "-y"]

    if start is not None:
        cmd += ["-ss", str(start)]

    cmd += ["-i", input_path]

    if duration is not None:
        cmd += ["-t", str(duration)]

    # Build filter chain: crop → scale → (optional) ASS subtitles
    filters = [
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos",
    ]

    if ass_path:
        # Escape special characters in path for FFmpeg filter
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

