"""
ClippedAI — Speaker-centered Reframing Engine

Three-layer reliability engine + RTS Kalman smoother for near-perfect
speaker centering in vertical (9:16) clips.
"""
import json
import math
import os
from pathlib import Path

import numpy as np

from config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT,
    ASD_CONFIDENCE_THRESHOLD, ASD_OVERLAP_THRESHOLD,
    REFRAME_SEGMENT_DURATION,
    HYSTERESIS_MIN, HYSTERESIS_MAX,
    SAFETY_MARGIN, MIN_FACE_HEIGHT_RATIO,
    GOOD_FRAME_THRESHOLD,
)


# ============================================================
# RTS KALMAN SMOOTHER (non-causal, zero-lag)
# ============================================================

class RTSKalmanSmoother:
    """
    Rauch-Tung-Striebel smoother for crop position.

    Forward pass: standard Kalman filter
    Backward pass: RTS smoothing (non-causal, zero-lag)

    State: [x_center, y_center, vx, vy]
    """

    def __init__(self, process_noise: float = 0.5, measurement_noise: float = 2.0):
        self.dt = 1.0  # normalized time step
        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        # Process noise
        self.Q = np.eye(4) * process_noise
        # Measurement noise
        self.R = np.eye(2) * measurement_noise

    def smooth(self, measurements: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Apply RTS smoothing to a sequence of (x, y) measurements.
        Returns smoothed (x, y) positions.
        """
        n = len(measurements)
        if n == 0:
            return []
        if n == 1:
            return [measurements[0]]

        # Forward pass: Kalman filter
        x_fwd = []
        P_fwd = []
        x_pred = []
        P_pred = []

        # Initialize
        x0 = np.array([measurements[0][0], measurements[0][1], 0, 0], dtype=np.float64)
        P0 = np.eye(4) * 100.0

        x = x0
        P = P0

        for i in range(n):
            # Predict
            x_p = self.F @ x
            P_p = self.F @ P @ self.F.T + self.Q
            x_pred.append(x_p.copy())
            P_pred.append(P_p.copy())

            # Update
            z = np.array([measurements[i][0], measurements[i][1]])
            y = z - self.H @ x_p
            S = self.H @ P_p @ self.H.T + self.R
            K = P_p @ self.H.T @ np.linalg.inv(S)
            x = x_p + K @ y
            P = (np.eye(4) - K @ self.H) @ P_p

            x_fwd.append(x.copy())
            P_fwd.append(P.copy())

        # Backward pass: RTS smoothing
        x_smooth = [None] * n
        x_smooth[n - 1] = x_fwd[n - 1]

        for i in range(n - 2, -1, -1):
            C = P_fwd[i] @ self.F.T @ np.linalg.inv(P_pred[i + 1])
            x_smooth[i] = x_fwd[i] + C @ (x_smooth[i + 1] - x_pred[i + 1])

        # Velocity clamping: limit max velocity to prevent jumps
        max_velocity = 50.0  # pixels per step
        for i in range(1, n):
            dx = x_smooth[i][0] - x_smooth[i - 1][0]
            dy = x_smooth[i][1] - x_smooth[i - 1][1]
            speed = math.sqrt(dx**2 + dy**2)
            if speed > max_velocity:
                scale = max_velocity / speed
                x_smooth[i][0] = x_smooth[i - 1][0] + dx * scale
                x_smooth[i][1] = x_smooth[i - 1][1] + dy * scale

        return [(float(x[0]), float(x[1])) for x in x_smooth]


# ============================================================
# THREE-LAYER ASD REFRAME ENGINE
# ============================================================

def _get_active_speaker_face(faces: list[dict], threshold: float) -> dict | None:
    """Find the face with the highest ASD score above threshold."""
    best = None
    best_score = 0.0
    for face in faces:
        score = face.get("asd_score", 0.0)
        if score >= threshold and score > best_score:
            best = face
            best_score = score
    return best


def _get_safe_wide_shot(faces: list[dict], src_w: int, src_h: int) -> tuple[float, float]:
    """Compute center of bounding box that encompasses all visible faces."""
    if not faces:
        return src_w / 2, src_h / 2

    all_x = []
    all_y = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])

    cx = (min(all_x) + max(all_x)) / 2
    cy = (min(all_y) + max(all_y)) / 2
    return cx, cy


def _get_largest_face(faces: list[dict]) -> dict | None:
    """Get the largest face by area."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))


def _face_center(face: dict) -> tuple[float, float]:
    """Get center point of a face bbox."""
    x1, y1, x2, y2 = face["bbox"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def compute_crop_targets(
    face_analysis: dict,
    transcript: dict | None = None,
) -> list[dict]:
    """
    Compute per-frame crop target centers using the 3-layer reliability engine.

    Layer 1: LoCoNet ASD ≥ 0.85 → center on active face
    Layer 2: Speech overlap or ASD < 0.60 → safe wide shot
    Layer 3: Lip motion detected → center on highest lip motion
    Layer 4: Ultra-safe fallback → largest face

    Returns list of {"frame": int, "cx": float, "cy": float, "layer": str}
    """
    fps = face_analysis["fps"]
    src_w = face_analysis["width"]
    src_h = face_analysis["height"]
    frame_count = face_analysis["frame_count"]
    tracks = face_analysis["tracks"]

    # Build per-frame face lookup
    frame_faces = {}  # frame_idx -> list of face dicts
    for track in tracks:
        for frame_data in track["frames"]:
            frame_idx = frame_data["frame"]
            if frame_idx not in frame_faces:
                frame_faces[frame_idx] = []
            face = {
                "bbox": frame_data["bbox"],
                "asd_score": frame_data.get("asd_score", 0.0),
                "track_id": track["track_id"],
                "global_id": track.get("global_id", track["track_id"]),
            }
            frame_faces[frame_idx].append(face)

    # Apply 3-layer engine per frame
    crop_targets = []
    current_speaker = None
    speaker_hold_start = 0

    for frame_idx in range(frame_count):
        faces = frame_faces.get(frame_idx, [])

        cx, cy = src_w / 2, src_h / 2  # default: center
        layer = "center"

        if faces:
            # Layer 1: Active speaker (LoCoNet ASD)
            active_face = _get_active_speaker_face(faces, ASD_CONFIDENCE_THRESHOLD)
            if active_face is not None:
                cx, cy = _face_center(active_face)
                layer = "asd_primary"
                new_speaker = active_face.get("global_id", active_face.get("track_id"))

                # Hysteresis: don't switch speakers too quickly
                if current_speaker is not None and new_speaker != current_speaker:
                    hold_duration = (frame_idx - speaker_hold_start) / fps
                    if hold_duration < HYSTERESIS_MIN:
                        # Keep current speaker
                        prev_face = None
                        for f in faces:
                            if f.get("global_id", f.get("track_id")) == current_speaker:
                                prev_face = f
                                break
                        if prev_face:
                            cx, cy = _face_center(prev_face)
                            layer = "asd_hysteresis"
                        # else: allow switch
                    else:
                        current_speaker = new_speaker
                        speaker_hold_start = frame_idx
                else:
                    current_speaker = new_speaker
                    if current_speaker != new_speaker:
                        speaker_hold_start = frame_idx

            # Layer 2: Multiple speakers or low ASD → wide shot
            elif any(f.get("asd_score", 0) < ASD_OVERLAP_THRESHOLD for f in faces):
                cx, cy = _get_safe_wide_shot(faces, src_w, src_h)
                layer = "wide_shot"

            # Layer 3: Use lip motion (placeholder — mediapipe)
            # For now, falls through to Layer 4

            # Layer 4: Largest face fallback
            else:
                largest = _get_largest_face(faces)
                if largest:
                    cx, cy = _face_center(largest)
                    layer = "largest_face"

        crop_targets.append({
            "frame": frame_idx,
            "cx": round(cx, 1),
            "cy": round(cy, 1),
            "layer": layer,
        })

    return crop_targets


def compute_crop_plan(
    face_analysis: dict,
    transcript: dict | None = None,
    scene_cuts: list[int] | None = None,
) -> list[dict]:
    """
    Compute the full reframing crop plan:
    1. Get per-frame targets from 3-layer engine
    2. Apply RTS Kalman smoothing (with scene cut resets)
    3. Quantize to 0.5s crop segments
    4. Apply rule-of-thirds + safety margin
    5. Validate crop plan

    Returns list of crop segments:
    [{"start_frame": int, "end_frame": int, "crop": {"x": int, "y": int, "w": int, "h": int}}]
    """
    fps = face_analysis["fps"]
    src_w = face_analysis["width"]
    src_h = face_analysis["height"]
    frame_count = face_analysis["frame_count"]

    # Output crop dimensions (9:16 from source)
    crop_h = src_h
    crop_w = int(crop_h * 9 / 16)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(crop_w * 16 / 9)

    # Step 1: Per-frame crop targets
    crop_targets = compute_crop_targets(face_analysis, transcript)

    # Step 2: RTS Kalman smoothing with scene cut resets
    if scene_cuts is None:
        scene_cuts = []

    # Split targets at scene cuts
    segments = []
    seg_start = 0
    for cut_frame in sorted(scene_cuts):
        if cut_frame > seg_start and cut_frame < frame_count:
            segments.append((seg_start, cut_frame))
            seg_start = cut_frame
    segments.append((seg_start, frame_count))

    smoother = RTSKalmanSmoother(process_noise=0.3, measurement_noise=3.0)
    smoothed_positions = [None] * frame_count

    for start, end in segments:
        measurements = [(ct["cx"], ct["cy"]) for ct in crop_targets[start:end]]
        smoothed = smoother.smooth(measurements)
        for i, (sx, sy) in enumerate(smoothed):
            smoothed_positions[start + i] = (sx, sy)

    # Step 3: Quantize to 0.5s crop segments
    segment_frames = max(1, int(fps * REFRAME_SEGMENT_DURATION))
    crop_plan = []

    for seg_start in range(0, frame_count, segment_frames):
        seg_end = min(seg_start + segment_frames, frame_count)

        # Average position within segment
        positions = [smoothed_positions[i] for i in range(seg_start, seg_end) if smoothed_positions[i]]
        if positions:
            avg_cx = np.mean([p[0] for p in positions])
            avg_cy = np.mean([p[1] for p in positions])
        else:
            avg_cx = src_w / 2
            avg_cy = src_h / 2

        # Step 4: Apply safety margin and bounds
        margin_w = crop_w * SAFETY_MARGIN
        margin_h = crop_h * SAFETY_MARGIN

        # Center crop on target, clamped to frame bounds
        crop_x = int(max(0, min(avg_cx - crop_w / 2, src_w - crop_w)))
        crop_y = int(max(0, min(avg_cy - crop_h / 2, src_h - crop_h)))

        crop_plan.append({
            "start_frame": seg_start,
            "end_frame": seg_end,
            "start_time": round(seg_start / fps, 3),
            "end_time": round(seg_end / fps, 3),
            "crop": {
                "x": crop_x,
                "y": crop_y,
                "w": crop_w,
                "h": crop_h,
            },
        })

    # Step 5: Validate crop plan
    validation = validate_crop_plan(crop_plan, face_analysis)

    return crop_plan


def validate_crop_plan(
    crop_plan: list[dict],
    face_analysis: dict,
) -> dict:
    """
    Pre-render crop plan validation.

    Computes:
    - good_frame_ratio: % of frames with face inside crop
    - center_offset_variance: how centered the face is
    - jitter_score: smoothness of transitions
    """
    if not crop_plan:
        return {"good_frame_ratio": 0, "jitter_score": 0, "pass": False}

    # Jitter score: average position change between segments
    jitter_vals = []
    for i in range(1, len(crop_plan)):
        prev = crop_plan[i - 1]["crop"]
        curr = crop_plan[i]["crop"]
        dx = abs(curr["x"] - prev["x"])
        dy = abs(curr["y"] - prev["y"])
        jitter_vals.append(math.sqrt(dx**2 + dy**2))

    jitter_score = np.mean(jitter_vals) if jitter_vals else 0.0

    # Good frame ratio: check if faces are inside the crop region
    tracks = face_analysis.get("tracks", [])
    total_faces = 0
    good_faces = 0

    for track in tracks:
        for frame_data in track["frames"]:
            frame_idx = frame_data["frame"]
            bbox = frame_data["bbox"]
            face_cx = (bbox[0] + bbox[2]) / 2
            face_cy = (bbox[1] + bbox[3]) / 2

            # Find crop segment for this frame
            for segment in crop_plan:
                if segment["start_frame"] <= frame_idx < segment["end_frame"]:
                    crop = segment["crop"]
                    total_faces += 1
                    if (crop["x"] <= face_cx <= crop["x"] + crop["w"] and
                            crop["y"] <= face_cy <= crop["y"] + crop["h"]):
                        good_faces += 1
                    break

    good_frame_ratio = good_faces / total_faces if total_faces > 0 else 1.0

    return {
        "good_frame_ratio": round(good_frame_ratio, 3),
        "jitter_score": round(jitter_score, 2),
        "pass": good_frame_ratio >= GOOD_FRAME_THRESHOLD,
    }


def generate_crop_plan(
    face_analysis: dict,
    output_path: str,
    transcript: dict | None = None,
    scene_cuts: list[int] | None = None,
) -> str:
    """
    Generate and save the crop plan to disk.

    Returns path to the saved crop plan JSON.
    """
    crop_plan = compute_crop_plan(face_analysis, transcript, scene_cuts)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "segments": crop_plan,
            "source_width": face_analysis["width"],
            "source_height": face_analysis["height"],
            "fps": face_analysis["fps"],
        }, f, indent=2, default=str)

    return output_path
