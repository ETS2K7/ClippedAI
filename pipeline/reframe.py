"""
Pipeline Step 6 — Reframe to 9:16
Face detection + smoothed crop for vertical video.
"""

import os
import sys
import subprocess
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FACE_DETECTION_FPS, SMOOTHING_ALPHA, TARGET_ASPECT_RATIO


def detect_faces(video_path: str, start: float, end: float) -> list:
    """
    Detect faces in the clip region at FACE_DETECTION_FPS.

    Returns:
        [
            {"time": 45.2, "x_center": 520, "y_center": 310, "w": 210, "h": 280},
            ...
        ]
    """
    from ultralytics import YOLO

    # Extract frames at low FPS for face detection
    frames_dir = "/tmp/clipped/faces"
    os.makedirs(frames_dir, exist_ok=True)

    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-t", str(duration),
        "-i", video_path,
        "-vf", f"fps={FACE_DETECTION_FPS}",
        "-q:v", "3",
        "-y",
        os.path.join(frames_dir, "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # Load YOLOv8 once (cached after first call)
    global _yolo_model
    if "_yolo_model" not in globals() or _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    model = _yolo_model

    # Detect faces in each frame
    faces = []
    frame_files = sorted([
        f for f in os.listdir(frames_dir) if f.startswith("frame_")
    ])

    for i, fname in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, fname)
        frame_time = start + (i / FACE_DETECTION_FPS)

        results = model(frame_path, verbose=False, conf=0.3)

        # Get person detections (class 0)
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Find largest person detection (most prominent)
                person_boxes = []
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        person_boxes.append({
                            "x_center": (x1 + x2) / 2,
                            "y_center": (y1 + y2) / 2,
                            "w": x2 - x1,
                            "h": y2 - y1,
                            "area": area,
                        })

                if person_boxes:
                    # Use the largest person
                    best = max(person_boxes, key=lambda b: b["area"])
                    faces.append({
                        "time": round(frame_time, 2),
                        "x_center": round(best["x_center"]),
                        "y_center": round(best["y_center"]),
                        "w": round(best["w"]),
                        "h": round(best["h"]),
                    })

    # Cleanup frames
    for f in frame_files:
        try:
            os.remove(os.path.join(frames_dir, f))
        except OSError:
            pass

    return faces


def compute_crop_x(faces: list, source_w: int, source_h: int) -> int:
    """
    Compute the horizontal crop position for 9:16 reframing.
    Uses exponential smoothing on face positions.

    Returns: x offset for FFmpeg crop filter
    """
    # Target crop dimensions
    crop_w = int(source_h * TARGET_ASPECT_RATIO)

    # Clamp crop width to source
    if crop_w >= source_w:
        return 0  # Source is already narrow enough

    if not faces:
        # No faces detected — center crop
        return (source_w - crop_w) // 2

    # Extract x_center positions
    positions = [f["x_center"] for f in faces]

    # Apply exponential smoothing
    smoothed = [positions[0]]
    for i in range(1, len(positions)):
        s = SMOOTHING_ALPHA * positions[i] + (1 - SMOOTHING_ALPHA) * smoothed[-1]
        smoothed.append(s)

    # Use the average of smoothed positions
    avg_x = sum(smoothed) / len(smoothed)

    # Compute crop offset (center the crop on the face)
    crop_x = int(avg_x - crop_w / 2)

    # Clamp to valid range
    crop_x = max(0, min(crop_x, source_w - crop_w))

    return crop_x
