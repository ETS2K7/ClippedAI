"""
Pipeline Step 4 — Audio Analysis
RMS energy curves + peak detection for finding high-energy moments.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUDIO_FRAME_LENGTH, AUDIO_HOP_LENGTH, PEAK_STD_MULTIPLIER


def analyze_audio(audio_path: str) -> dict:
    """
    Analyze audio for energy levels and peaks.

    Returns:
        {
            "energy_curve": [0.02, 0.03, ...],  # RMS per 0.5s frame
            "peaks": [
                {"time": 124.5, "energy": 0.92},
                ...
            ],
            "avg_energy": 0.12,
            "max_energy": 0.95,
        }
    """
    import numpy as np
    import librosa

    print(f"  📊 Analyzing audio energy...")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Compute RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=AUDIO_FRAME_LENGTH,
        hop_length=AUDIO_HOP_LENGTH,
    )[0]

    # Normalize to 0-1
    max_rms = float(np.max(rms)) if np.max(rms) > 0 else 1.0
    energy_curve = (rms / max_rms).tolist()

    # Compute stats
    avg_energy = float(np.mean(energy_curve))
    peak_threshold = avg_energy + PEAK_STD_MULTIPLIER * float(np.std(energy_curve))

    # Find peaks (frames where energy significantly exceeds average)
    peaks = []
    frame_duration = AUDIO_HOP_LENGTH / sr  # 0.5s per frame
    for i, energy in enumerate(energy_curve):
        if energy > peak_threshold:
            peaks.append({
                "time": round(i * frame_duration, 2),
                "energy": round(energy, 4),
            })

    # Merge nearby peaks (within 2 seconds)
    merged_peaks = _merge_nearby_peaks(peaks, min_gap=2.0)

    print(f"  📊 Found {len(merged_peaks)} energy peaks (threshold: {peak_threshold:.3f})")
    return {
        "energy_curve": [round(e, 4) for e in energy_curve],
        "peaks": merged_peaks,
        "avg_energy": round(avg_energy, 4),
        "max_energy": round(float(max(energy_curve)) if energy_curve else 0, 4),
        "frame_duration": round(frame_duration, 4),
    }


def _merge_nearby_peaks(peaks: list, min_gap: float = 2.0) -> list:
    """Merge peaks that are within min_gap seconds of each other."""
    if not peaks:
        return []

    merged = [peaks[0].copy()]
    for peak in peaks[1:]:
        if peak["time"] - merged[-1]["time"] < min_gap:
            # Keep the higher energy peak
            if peak["energy"] > merged[-1]["energy"]:
                merged[-1] = peak.copy()
        else:
            merged.append(peak.copy())

    return merged
