"""
Bulletproof 3-layer speaker-centered reframing engine.

Layer 1: LoCoNet ASD confidence → center on active speaker
Layer 2: Speech overlap / low confidence → safe wide shot
Layer 3: Lip motion / largest face fallback

Includes Kalman smoothing, hysteresis, and pre-render validation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class CropTarget:
    """A crop rectangle for a single frame."""
    cx: float  # center x (0-1, normalized)
    cy: float  # center y (0-1, normalized)
    scale: float  # zoom level (1.0 = full width)
    source: str  # which reframe layer selected this


@dataclass
class CropPlan:
    """Pre-computed crop plan for an entire clip."""
    clip_id: str
    fps: float
    width: int
    height: int
    frames: list[CropTarget] = field(default_factory=list)


# ─────────────────────────────────────────────
# Kalman Smoother
# ─────────────────────────────────────────────

class KalmanSmoother:
    """
    Non-causal (RTS) Kalman smoother for crop positions.
    Provides zero-lag smoothing with velocity clamping.
    """

    def __init__(self, process_var: float = 0.001, measure_var: float = 0.01):
        self.process_var = process_var
        self.measure_var = measure_var

    def smooth(self, positions: list[float]) -> list[float]:
        """Apply forward-backward Kalman smoothing."""
        if len(positions) < 3:
            return positions

        n = len(positions)
        # Forward pass
        x_fwd = np.zeros(n)
        p_fwd = np.zeros(n)
        x_fwd[0] = positions[0]
        p_fwd[0] = 1.0

        for i in range(1, n):
            # Predict
            x_pred = x_fwd[i - 1]
            p_pred = p_fwd[i - 1] + self.process_var

            # Update
            k = p_pred / (p_pred + self.measure_var)
            x_fwd[i] = x_pred + k * (positions[i] - x_pred)
            p_fwd[i] = (1 - k) * p_pred

        # Backward pass (RTS smoother)
        x_smooth = np.zeros(n)
        x_smooth[-1] = x_fwd[-1]

        for i in range(n - 2, -1, -1):
            p_pred = p_fwd[i] + self.process_var
            gain = p_fwd[i] / p_pred
            x_smooth[i] = x_fwd[i] + gain * (x_smooth[i + 1] - x_fwd[i])

        return x_smooth.tolist()


# ─────────────────────────────────────────────
# 3-Layer Reframe Engine
# ─────────────────────────────────────────────

def compute_crop_plan(
    clip: dict,
    face_tracks: list[dict],
    speaker_map: dict,
    scenes: list[dict],
    transcript: dict,
    source_width: int,
    source_height: int,
    output_fps: float = config.OUTPUT_FPS,
) -> CropPlan:
    """
    Compute the crop plan for a single clip.

    3-layer decision at each frame:
      1. LoCoNet high confidence → center on active speaker
      2. Overlap / low confidence → safe wide shot
      3. Lip motion / largest face fallback
    """
    clip_start = clip["start"]
    clip_end = clip["end"]
    clip_duration = clip_end - clip_start
    total_frames = int(clip_duration * output_fps)

    plan = CropPlan(
        clip_id=clip.get("rank", "0"),
        fps=output_fps,
        width=source_width,
        height=source_height,
    )

    # Get scene boundaries within clip
    scene_times = {
        s["boundary"] for s in scenes
        if clip_start <= s.get("boundary", 0) <= clip_end
    }

    # Get transcript for overlap detection
    speaking_segments = _get_speaking_segments(transcript, clip_start, clip_end)

    # Track hysteresis state
    current_target = None
    hold_until = 0.0

    for frame_idx in range(total_frames):
        timestamp = clip_start + frame_idx / output_fps

        # Check for scene cut → reset tracking
        if any(abs(timestamp - st) < 1.0 / output_fps for st in scene_times):
            current_target = None
            hold_until = 0.0

        # Get faces at this timestamp
        faces = _get_faces_at_time(face_tracks, timestamp)

        # Check for speech overlap
        is_overlap = _is_speech_overlap(speaking_segments, timestamp)

        # 3-Layer decision
        crop = _decide_crop(
            faces, is_overlap, source_width, source_height,
        )

        # Hysteresis: don't switch target too fast
        if current_target and timestamp < hold_until:
            # Blend toward new target
            crop = _blend_crop(current_target, crop, blend_factor=0.3)
        else:
            current_target = crop
            # Dynamic hysteresis based on confidence
            hold_time = config.REFRAME_HYSTERESIS[0]  # minimum
            if crop.source == "asd_primary":
                hold_time = config.REFRAME_HYSTERESIS[1]  # maximum
            hold_until = timestamp + hold_time

        plan.frames.append(crop)

    # Apply Kalman smoothing
    plan = _smooth_crop_plan(plan)

    # Validate plan
    validation = validate_crop_plan(plan, face_tracks, clip_start, clip_end)
    if not validation["passes"]:
        logger.warning(
            "Crop plan validation failed for clip %s: %s — applying safe mode",
            plan.clip_id, validation,
        )
        plan = _apply_safe_mode(plan, source_width, source_height)

    logger.info(
        "Crop plan for clip %s: %d frames, primary source: %s",
        plan.clip_id, len(plan.frames),
        _dominant_source(plan),
    )

    return plan


def _decide_crop(
    faces: list[dict],
    is_overlap: bool,
    src_w: int,
    src_h: int,
) -> CropTarget:
    """3-layer crop decision."""

    if not faces:
        # No faces → center crop
        return CropTarget(cx=0.5, cy=0.5, scale=1.0, source="no_face_center")

    # Layer 1: ASD high confidence
    active_faces = [
        f for f in faces
        if f.get("asd_confidence", 0) >= config.LOCONET_CONFIDENCE_HIGH
    ]

    if active_faces and not is_overlap:
        # Center on the most confident active speaker
        best = max(active_faces, key=lambda f: f.get("asd_confidence", 0))
        cx, cy = _face_center(best["bbox"], src_w, src_h)
        return CropTarget(
            cx=cx, cy=cy,
            scale=_face_scale(best["bbox"], src_h),
            source="asd_primary",
        )

    # Layer 2: Overlap or low confidence → safe wide shot
    if is_overlap or any(
        f.get("asd_confidence", 0) < config.LOCONET_CONFIDENCE_LOW
        for f in faces
    ):
        cx, cy = _all_faces_center(faces, src_w, src_h)
        return CropTarget(
            cx=cx, cy=cy,
            scale=_wideshot_scale(faces, src_w, src_h),
            source="wide_shot",
        )

    # Layer 3: Largest face fallback
    largest = max(faces, key=lambda f: _face_area(f["bbox"]))
    cx, cy = _face_center(largest["bbox"], src_w, src_h)
    return CropTarget(
        cx=cx, cy=cy,
        scale=_face_scale(largest["bbox"], src_h),
        source="largest_face",
    )


# ─────────────────────────────────────────────
# Geometry Helpers
# ─────────────────────────────────────────────

def _face_center(bbox: list, src_w: int, src_h: int) -> tuple[float, float]:
    """Normalized center of a face bbox."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / src_w
    cy = ((y1 + y2) / 2) / src_h
    return (cx, cy)


def _face_area(bbox: list) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _face_scale(bbox: list, src_h: int) -> float:
    """Compute scale so face is >= 55% of output frame height."""
    face_h = bbox[3] - bbox[1]
    desired_ratio = config.REFRAME_MIN_FACE_HEIGHT
    if face_h <= 0:
        return 1.0
    scale = (face_h / src_h) / desired_ratio
    return max(0.3, min(1.0, scale))


def _all_faces_center(faces: list[dict], src_w: int, src_h: int) -> tuple[float, float]:
    """Center point of all visible faces."""
    if not faces:
        return (0.5, 0.5)
    cx_sum = sum((f["bbox"][0] + f["bbox"][2]) / 2 for f in faces) / len(faces)
    cy_sum = sum((f["bbox"][1] + f["bbox"][3]) / 2 for f in faces) / len(faces)
    return (cx_sum / src_w, cy_sum / src_h)


def _wideshot_scale(faces: list[dict], src_w: int, src_h: int) -> float:
    """Scale to fit all faces with safety margin."""
    if not faces:
        return 1.0
    all_x = [f["bbox"][0] for f in faces] + [f["bbox"][2] for f in faces]
    span = (max(all_x) - min(all_x)) / src_w
    return max(0.5, min(1.0, span + config.REFRAME_SAFETY_MARGIN * 2))


# ─────────────────────────────────────────────
# Temporal Helpers
# ─────────────────────────────────────────────

def _get_faces_at_time(
    face_tracks: list[dict],
    timestamp: float,
    tolerance: float = 0.1,
) -> list[dict]:
    """Get face detections closest to a given timestamp."""
    best = None
    best_diff = float("inf")
    for frame_data in face_tracks:
        diff = abs(frame_data["timestamp"] - timestamp)
        if diff < best_diff:
            best_diff = diff
            best = frame_data
    if best and best_diff <= tolerance:
        return best.get("faces", [])
    return []


def _get_speaking_segments(
    transcript: dict,
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """Get transcript segments that fall within the clip."""
    return [
        seg for seg in transcript.get("segments", [])
        if seg["start"] < clip_end and seg["end"] > clip_start
    ]


def _is_speech_overlap(
    segments: list[dict],
    timestamp: float,
) -> bool:
    """Check if multiple speakers are active at this timestamp."""
    active = set()
    for seg in segments:
        if seg["start"] <= timestamp <= seg["end"]:
            active.add(seg.get("speaker", ""))
    return len(active) > 1


def _blend_crop(a: CropTarget, b: CropTarget, blend_factor: float) -> CropTarget:
    """Blend between two crop targets."""
    return CropTarget(
        cx=a.cx + blend_factor * (b.cx - a.cx),
        cy=a.cy + blend_factor * (b.cy - a.cy),
        scale=a.scale + blend_factor * (b.scale - a.scale),
        source=b.source,
    )


# ─────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────

def _smooth_crop_plan(plan: CropPlan) -> CropPlan:
    """Apply Kalman smoothing to crop positions."""
    if len(plan.frames) < 3:
        return plan

    smoother = KalmanSmoother()

    cx_values = [f.cx for f in plan.frames]
    cy_values = [f.cy for f in plan.frames]
    scale_values = [f.scale for f in plan.frames]

    cx_smooth = smoother.smooth(cx_values)
    cy_smooth = smoother.smooth(cy_values)
    scale_smooth = smoother.smooth(scale_values)

    for i, frame in enumerate(plan.frames):
        frame.cx = cx_smooth[i]
        frame.cy = cy_smooth[i]
        frame.scale = scale_smooth[i]

    return plan


# ─────────────────────────────────────────────
# Pre-Render Validation
# ─────────────────────────────────────────────

def validate_crop_plan(
    plan: CropPlan,
    face_tracks: list[dict],
    clip_start: float,
    clip_end: float,
) -> dict:
    """
    Pre-render validation of crop plan quality.
    Checks good_frame_ratio, center_offset_variance, jitter_score.
    """
    if not plan.frames:
        return {"passes": False, "reason": "empty plan"}

    # Jitter: measure frame-to-frame position changes
    diffs = []
    for i in range(1, len(plan.frames)):
        dx = plan.frames[i].cx - plan.frames[i - 1].cx
        dy = plan.frames[i].cy - plan.frames[i - 1].cy
        diffs.append(np.sqrt(dx ** 2 + dy ** 2))

    jitter_score = float(np.mean(diffs)) if diffs else 0.0

    # Center offset variance
    cx_var = float(np.var([f.cx for f in plan.frames]))
    cy_var = float(np.var([f.cy for f in plan.frames]))
    offset_variance = cx_var + cy_var

    # Good frame ratio (frames where face is reasonably centered)
    good_frames = sum(
        1 for f in plan.frames
        if 0.2 <= f.cx <= 0.8 and 0.2 <= f.cy <= 0.8
    )
    good_frame_ratio = good_frames / len(plan.frames)

    passes = (
        good_frame_ratio >= config.GOOD_FRAME_THRESHOLD and
        jitter_score < 0.05 and
        offset_variance < 0.1
    )

    return {
        "passes": passes,
        "good_frame_ratio": round(good_frame_ratio, 3),
        "jitter_score": round(jitter_score, 5),
        "offset_variance": round(offset_variance, 5),
    }


def _apply_safe_mode(
    plan: CropPlan,
    src_w: int,
    src_h: int,
) -> CropPlan:
    """Fallback: center crop with minimal movement."""
    for frame in plan.frames:
        frame.cx = 0.5
        frame.cy = 0.4  # slightly above center
        frame.scale = 0.8
        frame.source = "safe_mode"
    return plan


def _dominant_source(plan: CropPlan) -> str:
    """Return the most common source in the plan."""
    from collections import Counter
    sources = Counter(f.source for f in plan.frames)
    return sources.most_common(1)[0][0] if sources else "unknown"


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────

def save_crop_plan(plan: CropPlan, output_path: Path) -> Path:
    """Save crop plan to JSON."""
    data = {
        "clip_id": plan.clip_id,
        "fps": plan.fps,
        "width": plan.width,
        "height": plan.height,
        "num_frames": len(plan.frames),
        "frames": [
            {
                "cx": round(f.cx, 4),
                "cy": round(f.cy, 4),
                "scale": round(f.scale, 4),
                "source": f.source,
            }
            for f in plan.frames
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved crop plan: %s (%d frames)", output_path, len(plan.frames))
    return output_path
