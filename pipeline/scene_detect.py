"""
Shot boundary detection using AutoShot (primary) with
PySceneDetect/TransNetV2 fallback.
"""

import json
import logging
import sys
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_AUTOSHOT_PATH = Path("/opt/autoshot")
if _AUTOSHOT_PATH.exists():
    sys.path.insert(0, str(_AUTOSHOT_PATH))


def detect_scenes(
    video_path: Path,
    chunk_start: float = 0.0,
) -> list[dict]:
    """
    Detect shot boundaries in a video chunk.

    Returns list of scene boundaries:
      [{start, end, transition_type}]
    """
    try:
        return _detect_with_autoshot(video_path, chunk_start)
    except (ImportError, Exception) as e:
        logger.warning("AutoShot unavailable (%s), using PySceneDetect fallback", e)
        return _detect_with_pyscenedetect(video_path, chunk_start)


def _detect_with_autoshot(video_path: Path, chunk_start: float) -> list[dict]:
    """Shot detection using AutoShot."""
    # AutoShot inference — adapted from their inference script
    import torch
    from model import AutoShotModel  # from /opt/autoshot

    model = AutoShotModel.load_pretrained()
    model.eval()

    predictions = model.predict(str(video_path))

    scenes = []
    for i, boundary_frame in enumerate(predictions):
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        timestamp = boundary_frame / fps + chunk_start
        scenes.append({
            "boundary": round(timestamp, 3),
            "transition_type": "cut",
        })

    logger.info("AutoShot: detected %d scene boundaries", len(scenes))
    return scenes


def _detect_with_pyscenedetect(video_path: Path, chunk_start: float) -> list[dict]:
    """Fallback using PySceneDetect ContentDetector."""
    from scenedetect import detect, ContentDetector

    scene_list = detect(str(video_path), ContentDetector(threshold=27.0))

    scenes = []
    for scene in scene_list:
        start_time = scene[0].get_seconds() + chunk_start
        end_time = scene[1].get_seconds() + chunk_start
        scenes.append({
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "boundary": round(start_time, 3),
            "transition_type": "cut",
        })

    logger.info("PySceneDetect: detected %d scenes", len(scenes))
    return scenes


def merge_chunk_scenes(
    chunk_scenes: list[list[dict]],
    chunks_meta: list[dict],
) -> list[dict]:
    """Merge scene boundaries from overlapping chunks, deduplicate."""
    all_boundaries = []

    for scenes, chunk_meta in zip(chunk_scenes, chunks_meta):
        for scene in scenes:
            all_boundaries.append(scene["boundary"])

    # Deduplicate: boundaries within 0.5s of each other are the same cut
    all_boundaries.sort()
    deduped = []
    for b in all_boundaries:
        if not deduped or abs(b - deduped[-1]) > 0.5:
            deduped.append(b)

    return [{"boundary": round(b, 3), "transition_type": "cut"} for b in deduped]


def save_scenes(scenes: list[dict], output_path: Path) -> Path:
    """Save scene boundaries to JSON."""
    with open(output_path, "w") as f:
        json.dump(scenes, f, indent=2)
    logger.info("Saved %d scene boundaries: %s", len(scenes), output_path)
    return output_path
