"""
Pure-logic signal processing helpers for video_processing.py.

Extracted to a separate module so they can be unit-tested without
the heavy cv2/modal/scenedetect dependency chain.
"""

import logging
import numpy as np
import scipy.ndimage as ndimage

_logger = logging.getLogger(__name__)

# (Constants STATIONARY_STD_THRESHOLD and CLUSTER_GAP removed in favour of EMA + Snap-Cuts)


def smooth_segment(raw: np.ndarray, default: float, sigma: int) -> np.ndarray:
    """
    Stabilises a 1-D position array (gaps = -1) using an Exponential Moving Average
    (EMA) tracker with Snap-Cuts.

    1. Continuous Micro-Tracking (EMA): Ignores high-frequency bounding box jitter 
       while gently pulling the camera back to true center for low-frequency shifts.
    2. Snap-Cuts: If the active face jumps by a massive distance in a single frame 
       (e.g. active speaker switched to someone across the room), the camera instantly 
       snaps to the new speaker to prevent nausea-inducing pans.
    """
    n = len(raw)
    out = np.full(n, default)
    if n == 0:
        return out

    valid_idx = np.where(raw != -1)[0]
    if len(valid_idx) == 0:
        return out

    # We can tune alpha based on the requested sigma if we wanted to,
    # but a fixed alpha of 0.1 works excellently for 25-30fps video.
    alpha = 0.1
    # 0.15 is roughly 192 pixels on a 1280px width, a safe threshold for a jump
    snap_threshold = 0.15 

    current_pos = float(raw[valid_idx[0]])

    for i in range(n):
        if raw[i] != -1:
            target = float(raw[i])
            if abs(target - current_pos) > snap_threshold:
                # Snap cut! Distance is too large to pan smoothly
                current_pos = target
                _logger.debug(f"  → SNAP CUT to {current_pos:.2f}")
            else:
                # Smooth continuous tracking to kill jitter
                current_pos = alpha * target + (1 - alpha) * current_pos
        
        # If raw[i] == -1, current_pos remains unchanged (holds last known position)
        out[i] = current_pos

    return out


def stabilize_segment(raw: np.ndarray, min_entry: int, min_gap: int) -> np.ndarray:
    """
    Two-pass stabiliser for a boolean signal within one scene segment.
      Pass 1: Remove True-runs shorter than min_entry frames (noise).
      Pass 2: Merge adjacent True-runs separated by < min_gap False frames (hysteresis).
    """
    n = len(raw)
    if n == 0:
        return raw.copy()
    runs = []
    i = 0
    while i < n:
        if raw[i]:
            j = i
            while j < n and raw[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_entry]
    if not runs:
        return np.zeros(n, dtype=bool)
    merged = [runs[0]]
    for s, e in runs[1:]:
        if (s - merged[-1][1] - 1) < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    result = np.zeros(n, dtype=bool)
    for s, e in merged:
        result[s: e + 1] = True
    return result


def stabilize_speaker_identity(
    raw_cx: np.ndarray,
    min_frames: int,
    px_threshold: float,
) -> np.ndarray:
    """
    Debounce a per-frame face-centre (cx) signal for a single speaker slot.

    Problem: when the active speaker switches to a different camera angle
    (a cut to a new face or a wide-to-close zoom), TalkNet immediately assigns
    speaking=True to the new face.  The raw cx signal therefore jumps by
    hundreds of pixels in a single frame.  Without this filter, the Gaussian
    smoother receives that jump and starts drifting toward the new position
    immediately, causing the crop to visibly slide during what might be a
    transient one-second insert that never returns.

    Solution: treat the raw cx signal as a "candidate".  The crop only commits
    to a new position when the candidate has been consistently different from
    the committed position for at least min_frames consecutive frames.  Until
    then, the output is filled with the last committed position.

    Args:
        raw_cx:       1-D array of face-centre x values, -1 where no face.
        min_frames:   Minimum consecutive frames the new position must be held
                      before the switch is committed (e.g. 15 = ~0.6 s @ 25 fps).
        px_threshold: Minimum pixel displacement between old and new centre that
                      counts as a "different position" worth debouncing.

    Returns:
        Stabilised cx array (same shape as raw_cx, -1 entries preserved).
    """
    n = len(raw_cx)
    if n == 0:
        return raw_cx.copy()

    out = raw_cx.copy()

    # Find the first valid sample as the initial committed position.
    committed_cx: float = -1.0
    for i in range(n):
        if raw_cx[i] != -1:
            committed_cx = float(raw_cx[i])
            break

    if committed_cx == -1.0:
        return out  # no valid samples at all

    candidate_cx: float = committed_cx
    candidate_run: int = 0

    for i in range(n):
        cx = float(raw_cx[i])

        if cx == -1.0:
            # Gap frame — hold the committed position in output.
            out[i] = committed_cx
            # Gap resets the candidate run (we lost sight of the new face).
            candidate_run = 0
            candidate_cx = committed_cx
            continue

        if abs(cx - committed_cx) < px_threshold:
            # Still on the committed face (or very close) — accept immediately.
            committed_cx = cx
            out[i] = committed_cx
            candidate_run = 0
            candidate_cx = committed_cx
        else:
            # Different position — start or continue a candidate run.
            if abs(cx - candidate_cx) < px_threshold:
                # Same candidate as before — extend the run.
                candidate_run += 1
            else:
                # Yet another new position — reset candidate.
                candidate_cx = cx
                candidate_run = 1

            if candidate_run >= min_frames:
                # Candidate has been stable long enough — commit the switch.
                # Back-fill the frames we were debouncing so the cut happens
                # exactly when the angle changed, not 'min_frames' later.
                start_of_run = i - min_frames + 1
                out[start_of_run : i + 1] = candidate_cx
                
                _logger.debug(
                    f"  stabilize_speaker_identity: committed switch "
                    f"{committed_cx:.0f}px → {candidate_cx:.0f}px "
                    f"(back-filled {min_frames} frames)"
                )
                committed_cx = candidate_cx
                candidate_run = 0
            else:
                # Not yet stable — hold the previous committed position.
                out[i] = committed_cx

    return out
