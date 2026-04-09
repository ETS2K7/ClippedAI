"""
Pure-logic signal processing helpers for video_processing.py.

Extracted to a separate module so they can be unit-tested without
the heavy cv2/modal/scenedetect dependency chain.
"""

import numpy as np
import scipy.ndimage as ndimage


def smooth_segment(raw: np.ndarray, default: float, sigma: int) -> np.ndarray:
    """
    Gap-fills a 1-D position array (gaps = -1) then Gaussian-smooths it.
    Leading/trailing gaps are filled with the nearest valid value.
    """
    seg_len = len(raw)
    filled = raw.copy()
    valid_mask = filled != -1
    if not np.any(valid_mask):
        filled[:] = default
        return filled
    idxs = np.arange(seg_len)
    first_valid = int(np.argmax(valid_mask))
    filled[:first_valid] = filled[first_valid]
    last_valid = int(seg_len - 1 - np.argmax(valid_mask[::-1]))
    filled[last_valid + 1:] = filled[last_valid]
    valid_mask = filled != -1
    if not np.all(valid_mask):
        filled[~valid_mask] = np.interp(
            idxs[~valid_mask], idxs[valid_mask], filled[valid_mask]
        )
    effective_sigma = min(sigma, max(1, seg_len // 4))
    return ndimage.gaussian_filter1d(filled, sigma=effective_sigma)


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
