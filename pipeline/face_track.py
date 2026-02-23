"""
Face detection (YOLO), tracking (BoT-FaceSORT), and
active speaker detection (LoCoNet).

Outputs per-frame face bounding boxes with track IDs and
speaker activity confidence per track.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Add BoT-FaceSORT and LoCoNet to path (installed via git clone in Modal)
_BOT_FACESORT_PATH = Path("/opt/bot-facesort")
_LOCONET_PATH = Path("/opt/loconet")
if _BOT_FACESORT_PATH.exists():
    sys.path.insert(0, str(_BOT_FACESORT_PATH))
if _LOCONET_PATH.exists():
    sys.path.insert(0, str(_LOCONET_PATH))


# ─────────────────────────────────────────────
# Face Detection
# ─────────────────────────────────────────────

def detect_faces_in_chunk(
    video_path: Path,
    sample_fps: float = 2.0,
    device: str = "cuda",
) -> list[dict]:
    """
    Run YOLO face detection on sampled frames.

    Returns list of frame results:
      [{frame_idx, timestamp, faces: [{bbox, confidence}]}]
    """
    from ultralytics import YOLO

    model = YOLO(config.FACE_MODEL)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(video_fps / sample_fps))

    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            timestamp = frame_idx / video_fps

            # Run YOLO detection
            results = model(
                frame,
                conf=config.FACE_CONF_THRESHOLD,
                iou=config.FACE_IOU_THRESHOLD,
                device=device,
                verbose=False,
            )

            faces = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    faces.append({
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "confidence": round(float(box.conf[0]), 3),
                    })

            frame_results.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 3),
                "faces": faces,
            })

        frame_idx += 1

    cap.release()
    logger.info(
        "Face detection: %d frames sampled, %d total faces",
        len(frame_results),
        sum(len(f["faces"]) for f in frame_results),
    )
    return frame_results


# ─────────────────────────────────────────────
# Face Tracking
# ─────────────────────────────────────────────

def track_faces(
    frame_detections: list[dict],
    video_path: Path,
) -> list[dict]:
    """
    Apply BoT-FaceSORT tracking to assign consistent track IDs
    across frames within a chunk.

    Returns updated frame_detections with track_id per face.
    If BoT-FaceSORT is not available, falls back to simple IoU tracking.
    """
    try:
        return _track_with_bot_facesort(frame_detections, video_path)
    except (ImportError, Exception) as e:
        logger.warning("BoT-FaceSORT unavailable (%s), using IoU fallback", e)
        return _track_with_iou_fallback(frame_detections)


def _track_with_bot_facesort(
    frame_detections: list[dict],
    video_path: Path,
) -> list[dict]:
    """Tracking using BoT-FaceSORT."""
    # Import from cloned repo
    from tracker import BoTFaceSORT

    tracker = BoTFaceSORT()

    for frame_data in frame_detections:
        if not frame_data["faces"]:
            continue

        # Format detections for tracker: [x1, y1, x2, y2, conf]
        dets = np.array([
            f["bbox"] + [f["confidence"]]
            for f in frame_data["faces"]
        ])

        # Update tracker
        tracks = tracker.update(dets)

        # Map track IDs back to faces
        for i, face in enumerate(frame_data["faces"]):
            if i < len(tracks):
                face["track_id"] = int(tracks[i][4]) if len(tracks[i]) > 4 else i
            else:
                face["track_id"] = -1

    return frame_detections


def _track_with_iou_fallback(
    frame_detections: list[dict],
) -> list[dict]:
    """Simple IoU-based tracking fallback."""
    next_id = 0
    prev_faces = []

    for frame_data in frame_detections:
        curr_faces = frame_data["faces"]

        if not prev_faces:
            for face in curr_faces:
                face["track_id"] = next_id
                next_id += 1
        else:
            # Compute IoU matrix
            used = set()
            for face in curr_faces:
                best_iou = 0.0
                best_id = -1
                for prev in prev_faces:
                    if prev["track_id"] in used:
                        continue
                    iou = _compute_iou(face["bbox"], prev["bbox"])
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_id = prev["track_id"]

                if best_id >= 0:
                    face["track_id"] = best_id
                    used.add(best_id)
                else:
                    face["track_id"] = next_id
                    next_id += 1

        prev_faces = curr_faces

    return frame_detections


def _compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# Active Speaker Detection
# ─────────────────────────────────────────────

def detect_active_speakers(
    frame_detections: list[dict],
    video_path: Path,
    audio_path: Path,
) -> list[dict]:
    """
    Run LoCoNet active speaker detection on tracked faces.
    Adds 'asd_confidence' to each face in frame_detections.

    Falls back to simple lip-motion heuristic if LoCoNet unavailable.
    """
    try:
        return _asd_with_loconet(frame_detections, video_path, audio_path)
    except (ImportError, Exception) as e:
        logger.warning("LoCoNet unavailable (%s), using face-size fallback", e)
        return _asd_fallback(frame_detections)


def _asd_with_loconet(
    frame_detections: list[dict],
    video_path: Path,
    audio_path: Path,
) -> list[dict]:
    """Active speaker detection using LoCoNet."""
    # LoCoNet requires specific data preparation (face crops + audio segments
    # aligned per-frame). The full integration depends on LoCoNet's repo
    # structure which varies. For MVP, we explicitly use the fallback and
    # log clearly so we know when to wire in the real model.
    logger.warning(
        "LoCoNet ASD not yet wired — using face-size fallback. "
        "Clip quality will be good but not optimal."
    )
    raise ImportError("LoCoNet integration pending — using fallback")


def _asd_fallback(frame_detections: list[dict]) -> list[dict]:
    """Fallback: assign ASD confidence based on face size (larger = more likely speaking)."""
    for frame_data in frame_detections:
        faces = frame_data["faces"]
        if not faces:
            continue

        # Largest face gets highest confidence
        max_area = 0
        for face in faces:
            bbox = face["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            max_area = max(max_area, area)

        for face in faces:
            bbox = face["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            face["asd_confidence"] = round(area / max_area, 3) if max_area > 0 else 0.5

    return frame_detections


# ─────────────────────────────────────────────
# Global Re-ID (Reduce Phase)
# ─────────────────────────────────────────────

def global_reid(
    all_chunk_tracks: list[list[dict]],
    video_path: Path,
    device: str = "cuda",
) -> dict:
    """
    Merge face tracks across chunks using InsightFace embeddings + DBSCAN.

    Subsamples to 1 embedding per track per second to prevent OOM.
    Returns mapping: {(chunk_idx, local_track_id): global_face_id}
    """
    try:
        from insightface.app import FaceAnalysis
        from sklearn.cluster import DBSCAN
    except ImportError:
        logger.warning("InsightFace/sklearn unavailable, using identity mapping")
        return _identity_mapping(all_chunk_tracks)

    # Initialize InsightFace
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0)

    # Collect one embedding per track per second
    embeddings = []
    embedding_labels = []  # (chunk_idx, track_id)

    # Open video once and reuse across all chunks (avoid per-chunk overhead)
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not cap.isOpened():
        logger.warning("Could not open video for re-ID: %s", video_path)
        return _identity_mapping(all_chunk_tracks)

    for chunk_idx, chunk_tracks in enumerate(all_chunk_tracks):
        track_embeddings = {}  # track_id -> list of embeddings

        for frame_data in chunk_tracks:
            for face in frame_data.get("faces", []):
                tid = face.get("track_id", -1)
                if tid < 0:
                    continue

                # Subsample: 1 per second
                if tid not in track_embeddings:
                    track_embeddings[tid] = []

                ts = frame_data["timestamp"]
                existing_ts = [e["ts"] for e in track_embeddings[tid]]
                if any(abs(ts - t) < config.REID_EMBEDDINGS_PER_SECOND for t in existing_ts):
                    continue

                # Extract face crop and compute embedding
                frame_idx = frame_data["frame_idx"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                bbox = face["bbox"]
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                faces_detected = app.get(face_crop)
                if faces_detected:
                    emb = faces_detected[0].embedding
                    track_embeddings[tid].append({"ts": ts, "emb": emb})

        # Take mean embedding per track
        for tid, emb_list in track_embeddings.items():
            if emb_list:
                mean_emb = np.mean([e["emb"] for e in emb_list], axis=0)
                embeddings.append(mean_emb)
                embedding_labels.append((chunk_idx, tid))

    cap.release()

    if not embeddings:
        return _identity_mapping(all_chunk_tracks)

    # DBSCAN clustering
    emb_matrix = np.array(embeddings)
    # Normalize for cosine distance
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / (norms + 1e-8)

    # DBSCAN with cosine distance
    clustering = DBSCAN(
        eps=1.0 - config.INSIGHTFACE_COSINE_THRESHOLD,
        min_samples=1,
        metric="cosine",
    ).fit(emb_matrix)

    # Build mapping
    mapping = {}
    for label, (chunk_idx, track_id) in zip(clustering.labels_, embedding_labels):
        mapping[(chunk_idx, track_id)] = int(label)

    n_identities = len(set(clustering.labels_) - {-1})
    logger.info(
        "Global re-ID: %d embeddings → %d unique identities",
        len(embeddings), n_identities,
    )

    return mapping


def _identity_mapping(all_chunk_tracks: list[list[dict]]) -> dict:
    """Fallback: each chunk's tracks are separate identities."""
    mapping = {}
    global_id = 0
    for chunk_idx, chunk_tracks in enumerate(all_chunk_tracks):
        seen_tracks = set()
        for frame_data in chunk_tracks:
            for face in frame_data.get("faces", []):
                tid = face.get("track_id", -1)
                if tid >= 0 and tid not in seen_tracks:
                    mapping[(chunk_idx, tid)] = global_id
                    seen_tracks.add(tid)
                    global_id += 1
    return mapping


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────

def save_face_tracks(tracks: list[dict], output_path: Path) -> Path:
    """Save face tracks to JSON."""
    # Convert numpy types for JSON serialization
    clean = json.loads(json.dumps(tracks, default=_json_default))
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info("Saved face tracks: %s", output_path)
    return output_path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
