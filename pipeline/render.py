"""
ClippedAI — Basic center-crop vertical render (Milestone 1).
Takes a source video segment and produces a 9:16 center-cropped vertical clip.
"""
import subprocess

from config import OUTPUT_WIDTH, OUTPUT_HEIGHT, VIDEO_CODEC, VIDEO_CRF, VIDEO_PRESET


def center_crop_render(
    input_path: str,
    output_path: str,
    start: float | None = None,
    duration: float | None = None,
) -> str:
    """
    Render a center-cropped 9:16 vertical clip from a source video.

    Uses FFmpeg crop filter to extract the center vertical slice.
    No speaker detection — just a basic center crop for Milestone 1.
    """
    # Build FFmpeg command
    cmd = ["ffmpeg", "-y"]

    # If start/duration specified, seek first
    if start is not None:
        cmd += ["-ss", str(start)]

    cmd += ["-i", input_path]

    if duration is not None:
        cmd += ["-t", str(duration)]

    # Center crop: extract a vertical 9:16 slice from the center of the frame.
    # crop=ih*9/16:ih:iw/2-ih*9/16/2:0 means:
    #   width  = input_height * 9/16
    #   height = input_height
    #   x      = center horizontally
    #   y      = 0 (top)
    # Then scale to exact output dimensions.
    crop_filter = (
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos"
    )

    cmd += [
        "-vf", crop_filter,
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
