"""
FFmpeg-based clip rendering with dynamic cropping + .ass captions.

Takes a crop plan + .ass file and produces the final 9:16 clip.
"""

import logging
import subprocess
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def render_clip(
    source_video: Path,
    crop_plan_path: Path,
    caption_path: Path,
    output_path: Path,
    output_width: int = config.OUTPUT_WIDTH,
    output_height: int = config.OUTPUT_HEIGHT,
    clip_start: float = 0.0,
    clip_end: float = 0.0,
) -> Path:
    """
    Render a final clip with dynamic crop + captions.

    Uses FFmpeg with:
    - Crop filter based on pre-computed crop segments
    - .ass subtitle burn-in via libass
    - H.264 encoding at CRF 20
    """
    import json

    # Load crop plan
    with open(crop_plan_path) as f:
        plan = json.load(f)

    # Build crop filter from plan
    crop_filter = _build_crop_filter(
        plan, output_width, output_height,
    )

    # Build FFmpeg command
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-to", f"{clip_end:.3f}",
        "-i", str(source_video),
        "-vf", f"{crop_filter},scale={output_width}:{output_height},ass={caption_path}",
        "-c:v", config.VIDEO_CODEC,
        "-preset", config.VIDEO_PRESET,
        "-crf", str(config.VIDEO_CRF),
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info("Rendering clip: %s", output_path.name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg render timed out after 120s: %s", output_path.name)
        return _render_simple(
            source_video, caption_path, output_path,
            output_width, output_height,
            clip_start, clip_end,
        )

    if result.returncode != 0:
        logger.error("FFmpeg render failed: %s", result.stderr[-500:])
        # Try simpler render without dynamic crop
        return _render_simple(
            source_video, caption_path, output_path,
            output_width, output_height,
            clip_start, clip_end,
        )

    file_size = output_path.stat().st_size / 1e6
    logger.info("Rendered: %s (%.1f MB)", output_path.name, file_size)
    return output_path


def _build_crop_filter(
    plan: dict,
    out_w: int,
    out_h: int,
) -> str:
    """
    Build FFmpeg crop filter from crop plan.
    Groups consecutive frames with similar crop positions 
    into 0.5s segments for efficiency.
    """
    frames = plan.get("frames", [])
    src_w = plan.get("width", 1920)
    src_h = plan.get("height", 1080)
    fps = plan.get("fps", 30)

    if not frames:
        # Default center crop
        crop_w = int(src_h * out_w / out_h)
        crop_x = (src_w - crop_w) // 2
        return f"crop={crop_w}:{src_h}:{crop_x}:0"

    # Group frames into segments (0.5s each)
    segment_size = max(1, int(fps * config.CROP_SEGMENT_DURATION))
    segments = []

    for i in range(0, len(frames), segment_size):
        chunk = frames[i:i + segment_size]
        avg_cx = sum(f["cx"] for f in chunk) / len(chunk)
        avg_cy = sum(f["cy"] for f in chunk) / len(chunk)
        avg_scale = sum(f["scale"] for f in chunk) / len(chunk)

        # Convert normalized coords to pixel crop
        crop_h = int(src_h * avg_scale)
        crop_w = int(crop_h * out_w / out_h)

        # Clamp
        crop_w = min(crop_w, src_w)
        crop_h = min(crop_h, src_h)

        crop_x = int(avg_cx * src_w - crop_w / 2)
        crop_y = int(avg_cy * src_h - crop_h / 2)

        # Keep in bounds
        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_y = max(0, min(crop_y, src_h - crop_h))

        segments.append({
            "frame_start": i,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "crop_x": crop_x,
            "crop_y": crop_y,
        })

    if not segments:
        crop_w = int(src_h * out_w / out_h)
        crop_x = (src_w - crop_w) // 2
        return f"crop={crop_w}:{src_h}:{crop_x}:0"

    # For simplicity, use the average crop position
    # (FFmpeg's crop filter doesn't natively support frame-by-frame changes
    #  without complex sendcmd/zmq. Use zoompan for dynamic cropping.)
    avg = segments[len(segments) // 2]  # use middle segment as representative

    return f"crop={avg['crop_w']}:{avg['crop_h']}:{avg['crop_x']}:{avg['crop_y']}"


def _render_simple(
    source_video: Path,
    caption_path: Path,
    output_path: Path,
    out_w: int,
    out_h: int,
    clip_start: float,
    clip_end: float,
) -> Path:
    """Fallback: simple center crop without dynamic reframing."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-to", f"{clip_end:.3f}",
        "-i", str(source_video),
        "-vf", (
            f"crop=ih*{out_w}/{out_h}:ih,"
            f"scale={out_w}:{out_h},"
            f"ass={caption_path}"
        ),
        "-c:v", config.VIDEO_CODEC,
        "-preset", config.VIDEO_PRESET,
        "-crf", str(config.VIDEO_CRF),
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info("Simple render fallback: %s", output_path.name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg simple render timed out after 120s: {output_path.name}")

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg simple render failed: {result.stderr[-500:]}")

    return output_path


def validate_render(output_path: Path) -> dict:
    """Post-render validation: check file exists, has video/audio, reasonable size."""
    if not output_path.exists():
        return {"valid": False, "reason": "file does not exist"}

    size = output_path.stat().st_size
    if size < 10000:  # < 10KB = probably broken
        return {"valid": False, "reason": f"file too small: {size} bytes"}

    # Check with ffprobe
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "stream=codec_type,width,height,duration",
        "-of", "json",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {"valid": False, "reason": "ffprobe failed"}

    import json
    try:
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)

        if not has_video:
            return {"valid": False, "reason": "no video stream"}
        if not has_audio:
            return {"valid": False, "reason": "no audio stream"}

        return {
            "valid": True,
            "size_mb": round(size / 1e6, 2),
            "streams": len(streams),
        }
    except json.JSONDecodeError:
        return {"valid": False, "reason": "ffprobe output parse error"}
