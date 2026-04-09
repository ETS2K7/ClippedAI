"""Tests for signal_helpers.py — pure-logic helpers (no cv2/modal deps)."""

import numpy as np


def test_stabilize_segment_removes_short_runs():
    """stabilize_segment should remove True-runs shorter than min_entry."""
    from src.signal_helpers import stabilize_segment

    # 60-frame boolean array with a short True burst (5 frames) — too short for min_entry=20
    raw = np.zeros(60, dtype=bool)
    raw[10:15] = True  # 5-frame burst — should be removed

    result = stabilize_segment(raw, min_entry=20, min_gap=20)

    # Short burst should have been filtered out
    assert not np.any(result), "Short True-runs should be removed"
    assert len(result) == len(raw)


def test_stabilize_segment_keeps_long_runs():
    """stabilize_segment should keep True-runs >= min_entry."""
    from src.signal_helpers import stabilize_segment

    raw = np.zeros(100, dtype=bool)
    raw[10:50] = True  # 40-frame burst — well above min_entry=20

    result = stabilize_segment(raw, min_entry=20, min_gap=20)

    # The long run should survive
    assert np.any(result), "Long True-runs should be kept"
    assert len(result) == len(raw)


def test_stabilize_segment_merges_close_runs():
    """stabilize_segment should merge True-runs separated by < min_gap."""
    from src.signal_helpers import stabilize_segment

    raw = np.zeros(100, dtype=bool)
    raw[10:35] = True   # 25-frame burst
    raw[40:65] = True   # 25-frame burst, gap = 5 frames < min_gap=20

    result = stabilize_segment(raw, min_entry=20, min_gap=20)

    # Both runs should be merged into one contiguous block
    assert result[37], "Gap between close runs should be filled"
    assert len(result) == len(raw)


def test_stabilize_segment_empty():
    """Empty input should return empty output."""
    from src.signal_helpers import stabilize_segment

    raw = np.array([], dtype=bool)
    result = stabilize_segment(raw, min_entry=20, min_gap=20)
    assert len(result) == 0


def test_smooth_segment_basic():
    """smooth_segment should smooth noisy position data."""
    from src.signal_helpers import smooth_segment

    # Create noisy position data: steady signal at 500 with noise
    rng = np.random.default_rng(42)
    raw = 500.0 + rng.integers(-30, 30, size=60).astype(float)

    result = smooth_segment(raw, default=500.0, sigma=10)

    # Smoothed output should have less variance than input
    assert np.std(result) < np.std(raw), "Smoothing should reduce variance"
    assert len(result) == len(raw)


def test_smooth_segment_with_gaps():
    """smooth_segment should fill -1 gaps before smoothing."""
    from src.signal_helpers import smooth_segment

    raw = np.array([100.0, 200.0, -1.0, -1.0, 300.0, 400.0])
    result = smooth_segment(raw, default=250.0, sigma=1)

    # No -1 gaps should remain
    assert np.all(result >= 0), "Gaps should be filled"
    assert len(result) == len(raw)
