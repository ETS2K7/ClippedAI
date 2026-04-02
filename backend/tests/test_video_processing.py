"""Tests for video_processing.py — pure-logic helpers."""

import numpy as np


def test_stabilize_segment_basic():
    """_stabilize_segment should smooth noisy position data."""
    from src.video_processing import _stabilize_segment

    # Create noisy position data: a steady signal at 500 with noise
    rng = np.random.default_rng(42)
    noisy = 500 + rng.integers(-30, 30, size=60)
    result = _stabilize_segment(noisy)

    # Stabilized output should have less variance than input
    assert np.std(result) < np.std(noisy), "Stabilization should reduce variance"
    # Output length should match input length
    assert len(result) == len(noisy)


def test_stabilize_segment_short():
    """Short segments (< kernel) should still work without error."""
    from src.video_processing import _stabilize_segment

    short = np.array([100, 200, 300])
    result = _stabilize_segment(short)
    assert len(result) == 3
