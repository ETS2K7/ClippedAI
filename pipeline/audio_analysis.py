"""
Audio event classification using PANNs.

Detects laughter, applause, music, crowd noise, etc.
"""

import json
import logging
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


def analyze_audio(
    audio_path: Path,
    chunk_start: float = 0.0,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
) -> list[dict]:
    """
    Classify audio events in a chunk using PANNs.

    Returns list of events:
      [{start, end, event_type, confidence}]
    """
    try:
        return _analyze_with_panns(audio_path, chunk_start, window_seconds, hop_seconds)
    except (ImportError, Exception) as e:
        logger.warning("PANNs unavailable (%s), using energy-based fallback", e)
        return _analyze_with_energy(audio_path, chunk_start)


def _analyze_with_panns(
    audio_path: Path,
    chunk_start: float,
    window_seconds: float,
    hop_seconds: float,
) -> list[dict]:
    """Audio event detection using PANNs."""
    from panns_inference import AudioTagging
    import librosa

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=32000, mono=True)
    duration = len(audio) / sr

    # Auto-detect device — AudioAnalyzer container is CPU-only
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize PANNs
    tagger = AudioTagging(checkpoint_path=None, device=device)

    events = []
    # Sliding window analysis
    pos = 0.0
    while pos < duration:
        end = min(pos + window_seconds, duration)
        start_sample = int(pos * sr)
        end_sample = int(end * sr)
        segment = audio[start_sample:end_sample]

        if len(segment) < sr * 0.5:  # skip tiny segments
            pos += hop_seconds
            continue

        # Pad to window size
        if len(segment) < int(window_seconds * sr):
            segment = np.pad(segment, (0, int(window_seconds * sr) - len(segment)))

        # Predict
        clipwise_output, _ = tagger.inference(segment[np.newaxis, :])
        probs = clipwise_output[0]

        # Map AudioSet labels to our categories
        event_mappings = {
            "Speech": (0, 0.5),
            "Laughter": (16, 0.3),
            "Applause": (25, 0.3),
            "Music": (137, 0.4),
            "Crowd": (36, 0.3),
            "Silence": None,  # detected by absence of other events
        }

        for event_type, mapping in event_mappings.items():
            if mapping is None:
                continue
            idx, threshold = mapping
            if idx < len(probs) and probs[idx] > threshold:
                events.append({
                    "start": round(pos + chunk_start, 3),
                    "end": round(end + chunk_start, 3),
                    "event_type": event_type.lower(),
                    "confidence": round(float(probs[idx]), 3),
                })

        pos += hop_seconds

    logger.info("PANNs: detected %d audio events", len(events))
    return events


def _analyze_with_energy(
    audio_path: Path,
    chunk_start: float,
) -> list[dict]:
    """Fallback: simple RMS energy analysis."""
    import librosa

    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Compute RMS energy in 1-second windows
    hop_length = sr  # 1 second
    frame_length = sr
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    events = []
    mean_rms = np.mean(rms) if len(rms) > 0 else 0
    std_rms = np.std(rms) if len(rms) > 0 else 1

    for i, energy in enumerate(rms):
        timestamp = i + chunk_start
        # High energy = potential highlight
        if energy > mean_rms + 1.5 * std_rms:
            events.append({
                "start": round(timestamp, 3),
                "end": round(timestamp + 1.0, 3),
                "event_type": "high_energy",
                "confidence": round(float(min(1.0, (energy - mean_rms) / (std_rms + 1e-8))), 3),
            })

    logger.info("Energy analysis: %d high-energy events", len(events))
    return events


def merge_chunk_audio_events(
    chunk_events: list[list[dict]],
    chunks_meta: list[dict],
) -> list[dict]:
    """Merge audio events from overlapping chunks."""
    all_events = []
    for events, chunk_meta in zip(chunk_events, chunks_meta):
        overlap = config.CHUNK_OVERLAP
        chunk_start = chunk_meta["start"]

        # Trust region (same logic as transcript merge)
        trust_start = chunk_start + overlap / 2 if chunk_meta["index"] > 0 else chunk_start
        for event in events:
            if event["start"] >= trust_start:
                all_events.append(event)

    all_events.sort(key=lambda e: e["start"])
    return all_events


def save_audio_events(events: list[dict], output_path: Path) -> Path:
    """Save audio events to JSON."""
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)
    logger.info("Saved %d audio events: %s", len(events), output_path)
    return output_path
