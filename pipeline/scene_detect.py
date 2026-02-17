"""
Pipeline Step 3 — Scene Detection
PySceneDetect ContentDetector for scene boundary timestamps.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCENE_THRESHOLD, SCENE_MIN_SCENE_LEN


def detect_scenes(video_path: str) -> list:
    """
    Detect scene boundaries in the video.

    Returns:
        [
            {"start": 0.0, "end": 15.3, "duration": 15.3},
            {"start": 15.3, "end": 42.7, "duration": 27.4},
            ...
        ]
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    print(f"  🎬 Detecting scenes...")

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=SCENE_THRESHOLD,
            min_scene_len=SCENE_MIN_SCENE_LEN * video.frame_rate,
        )
    )

    # Process video (downscale for speed)
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for start_time, end_time in scene_list:
        start_s = start_time.get_seconds()
        end_s = end_time.get_seconds()
        scenes.append({
            "start": round(start_s, 2),
            "end": round(end_s, 2),
            "duration": round(end_s - start_s, 2),
        })

    print(f"  🎬 Found {len(scenes)} scenes")
    return scenes
