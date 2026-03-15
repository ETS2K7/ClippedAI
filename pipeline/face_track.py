"""
ClippedAI — Face Detection + Tracking + Active Speaker Detection

YOLO26 face detection → BoT-FaceSORT multi-face tracking → LoCoNet ASD.
Global re-ID via InsightFace embeddings + DBSCAN clustering.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# YOLO26 FACE DETECTION
# ============================================================

def detect_faces_yolo26(
    video_path: str, conf: float = 0.5, nms: float = 0.7,
) -> list[dict]:
    """
    Run YOLO26n detection on every frame, filter for person class.
    Returns list of per-frame detections.
    """
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, iou=nms, verbose=False, classes=[0])

        faces = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                faces.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "conf": round(float(box.conf[0]), 3),
                })

        all_detections.append({
            "frame": frame_idx,
            "faces": faces,
        })
        frame_idx += 1

    cap.release()
    return all_detections


# ============================================================
# BoT-FaceSORT MULTI-FACE TRACKING
# ============================================================

def track_faces_botfacesort(
    video_path: str,
    detections: list[dict],
    device: str = "0",
) -> list[dict]:
    """
    Run BoT-FaceSORT tracking on YOLO26-detected faces.
    Uses create_tracker API with botfacesort configuration.
    """
    # Set up path for BoT-FaceSORT imports
    botfacesort_root = "/opt/bot-facesort"
    sys.path.insert(0, botfacesort_root)
    os.chdir(botfacesort_root)

    try:
        import torch
        from tracker.tracker_zoo import create_tracker

        tracker_device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        # Create tracker with botfacesort config
        config_path = os.path.join(botfacesort_root, "tracker", "configs", "botfacesort.yaml")

        # Reid weights — use adaface if available
        reid_weights_path = os.path.join(botfacesort_root, "tracker", "appearance", "adaface_ir18_webface4m.onnx")
        if not os.path.exists(reid_weights_path):
            reid_weights_path = None

        tracker = create_tracker(
            "botfacesort",
            config_path,
            reid_weights_path,
            tracker_device,
            half=False,
            sc=True,   # shot change detection
            sm=True,    # shared feature memory
        )

        cap = cv2.VideoCapture(video_path)
        track_history = {}

        for frame_data in detections:
            frame_idx = frame_data["frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Format: [x1, y1, x2, y2, conf, label, det_index]
            dets = []
            for i, face in enumerate(frame_data["faces"]):
                x1, y1, x2, y2 = face["bbox"]
                dets.append([x1, y1, x2, y2, face["conf"], 1.0, i])

            if dets:
                dets_array = np.array(dets)
            else:
                dets_array = np.empty((0, 7))

            try:
                tracks = tracker.update(dets_array, frame)
            except Exception:
                continue

            if tracks is not None and len(tracks) > 0:
                for track in tracks:
                    if len(track) >= 5:
                        x1, y1, x2, y2, track_id = track[:5]
                        track_id = int(track_id)
                        if track_id not in track_history:
                            track_history[track_id] = []
                        track_history[track_id].append({
                            "frame": frame_idx,
                            "bbox": [round(float(x1), 1), round(float(y1), 1),
                                     round(float(x2), 1), round(float(y2), 1)],
                        })

        cap.release()

    except ImportError as e:
        # Fallback: simple IoU-based tracking if BoT-FaceSORT import fails
        print(f"BoT-FaceSORT import failed ({e}), using simple IoU tracking fallback")
        track_history = _simple_iou_tracking(detections)

    finally:
        os.chdir("/root")

    tracks_list = []
    for track_id, frames in track_history.items():
        tracks_list.append({
            "track_id": track_id,
            "frames": sorted(frames, key=lambda x: x["frame"]),
            "frame_count": len(frames),
        })

    return tracks_list


def _simple_iou_tracking(detections: list[dict], iou_thresh: float = 0.3) -> dict:
    """Simple IoU-based tracking fallback."""
    track_history = {}
    next_id = 0
    active_tracks = {}  # track_id -> last bbox

    for frame_data in detections:
        frame_idx = frame_data["frame"]
        faces = frame_data["faces"]

        used_tracks = set()
        for face in faces:
            bbox = face["bbox"]
            best_id = None
            best_iou = iou_thresh

            for tid, last_bbox in active_tracks.items():
                if tid in used_tracks:
                    continue
                iou = _compute_iou(bbox, last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is None:
                best_id = next_id
                next_id += 1
                track_history[best_id] = []

            used_tracks.add(best_id)
            active_tracks[best_id] = bbox
            track_history[best_id].append({
                "frame": frame_idx,
                "bbox": bbox,
            })

    return track_history


def _compute_iou(box1, box2):
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# ============================================================
# LoCoNet ACTIVE SPEAKER DETECTION
# ============================================================

def run_loconet_asd(
    video_path: str,
    tracks: list[dict],
    audio_path: str | None = None,
) -> list[dict]:
    """
    Run LoCoNet active speaker detection on tracked faces.
    Falls back to face-size heuristic if LoCoNet model unavailable.
    """
    import torch

    # Extract audio
    if audio_path is None:
        audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1",
             "-f", "wav", audio_path],
            capture_output=True, check=True,
        )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Try loading LoCoNet
    loconet_model = None
    try:
        sys.path.insert(0, "/opt/loconet")
        loconet_weight = "/opt/loconet/pretrained/loconet_ava.pth"
        if os.path.exists(loconet_weight):
            from loconet import LoCoNet
            device = "cuda" if torch.cuda.is_available() else "cpu"
            loconet_model = LoCoNet()
            loconet_model.load_state_dict(torch.load(loconet_weight, map_location=device))
            loconet_model.to(device)
            loconet_model.eval()
            print("LoCoNet ASD model loaded successfully")
        else:
            print(f"LoCoNet weights not found at {loconet_weight}, using face-size heuristic")
    except Exception as e:
        print(f"LoCoNet load failed ({e}), using face-size heuristic fallback")

    for track in tracks:
        asd_scores = []
        for frame_data in track["frames"]:
            frame_idx = frame_data["frame"]
            bbox = frame_data["bbox"]

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                asd_scores.append(0.0)
                frame_data["asd_score"] = 0.0
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if loconet_model is not None:
                # Use LoCoNet for ASD
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_crop_resized = cv2.resize(face_crop, (112, 112))
                    face_tensor = torch.from_numpy(face_crop_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    face_tensor = face_tensor.to(device)
                    try:
                        with torch.no_grad():
                            score = loconet_model.forward_single(face_tensor, frame_idx / fps, audio_path)
                            score = round(float(score), 3)
                    except Exception:
                        score = _face_size_heuristic(x1, y1, x2, y2, w, h)
                else:
                    score = 0.0
            else:
                # Heuristic: larger faces in frame are more likely to be the speaker
                score = _face_size_heuristic(x1, y1, x2, y2, w, h)

            asd_scores.append(score)
            frame_data["asd_score"] = score

        track["mean_asd_score"] = round(float(np.mean(asd_scores)), 3) if asd_scores else 0.0

    cap.release()
    return tracks


def _face_size_heuristic(x1, y1, x2, y2, frame_w, frame_h):
    """Face-size heuristic for ASD when LoCoNet is unavailable."""
    face_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    # Larger faces closer to center are more likely speakers
    face_cx = (x1 + x2) / 2
    face_cy = (y1 + y2) / 2
    center_dist = ((face_cx - frame_w / 2) ** 2 + (face_cy - frame_h / 2) ** 2) ** 0.5
    max_dist = (frame_w ** 2 + frame_h ** 2) ** 0.5 / 2
    center_score = 1.0 - (center_dist / max_dist)
    size_score = min(face_area / frame_area * 10, 1.0)
    return round(min(0.5 * size_score + 0.5 * center_score, 1.0), 3)


# ============================================================
# GLOBAL RE-ID (InsightFace + DBSCAN)
# ============================================================

def global_reid_dbscan(
    tracks: list[dict],
    video_path: str,
    cosine_threshold: float = 0.62,
) -> list[dict]:
    """
    Global re-ID: 1 InsightFace embedding per track per second,
    cluster with DBSCAN (cosine threshold).
    """
    try:
        from insightface.app import FaceAnalysis
        from sklearn.cluster import DBSCAN
    except ImportError:
        # Assign sequential global IDs if InsightFace unavailable
        for i, track in enumerate(tracks):
            track["global_id"] = i
        return tracks

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    track_embeddings = {}
    for track in tracks:
        embeddings = []
        frames = track["frames"]
        sample_interval = max(1, int(fps))

        for i in range(0, len(frames), sample_interval):
            frame_data = frames[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data["frame"])
            ret, frame = cap.read()
            if not ret:
                continue

            x1, y1, x2, y2 = [int(v) for v in frame_data["bbox"]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            faces = app.get(face_crop)
            if faces:
                embeddings.append(faces[0].embedding)

        if embeddings:
            track_embeddings[track["track_id"]] = np.mean(embeddings, axis=0)

    cap.release()

    if len(track_embeddings) < 2:
        for i, track in enumerate(tracks):
            track["global_id"] = track.get("global_id", i)
        return tracks

    track_ids = list(track_embeddings.keys())
    embedding_matrix = np.array([track_embeddings[tid] for tid in track_ids])
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = embedding_matrix / (norms + 1e-8)

    clustering = DBSCAN(eps=1 - cosine_threshold, min_samples=1, metric="cosine")
    labels = clustering.fit_predict(embedding_matrix)

    cluster_map = {}
    for tid, label in zip(track_ids, labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(tid)

    for track in tracks:
        for label, tids in cluster_map.items():
            if track["track_id"] in tids:
                track["global_id"] = int(label)
                break

    return tracks


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_chunk(
    chunk_path: str,
    output_dir: str,
    conf: float = 0.5,
    nms: float = 0.7,
    cosine_threshold: float = 0.62,
) -> dict:
    """
    Full face analysis pipeline for one chunk:
    1. YOLO26 face detection
    2. BoT-FaceSORT tracking (or IoU fallback)
    3. LoCoNet ASD (or face-size heuristic fallback)
    4. InsightFace re-ID with DBSCAN
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: YOLO26 face detection...")
    detections = detect_faces_yolo26(chunk_path, conf=conf, nms=nms)
    total_faces = sum(len(d["faces"]) for d in detections)
    print(f"  Detected {total_faces} faces across {len(detections)} frames")

    print("Step 2: BoT-FaceSORT tracking...")
    tracks = track_faces_botfacesort(chunk_path, detections)
    print(f"  Created {len(tracks)} tracks")

    print("Step 3: LoCoNet ASD...")
    tracks = run_loconet_asd(chunk_path, tracks)

    print("Step 4: Global re-ID...")
    tracks = global_reid_dbscan(tracks, chunk_path, cosine_threshold=cosine_threshold)

    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    analysis = {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "tracks": tracks,
        "detection_count": total_faces,
    }

    output_path = Path(output_dir) / "face_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"Face analysis saved to {output_path}")
    return analysis
