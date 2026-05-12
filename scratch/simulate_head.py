import json
import numpy as np
import modal
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from src.video_processing import _robust_median, _prominent_distinct_faces, _face_area, _face_cx, _stabilize_bool_state
from src.signal_helpers import smooth_segment, stabilize_segment

app = modal.App("simulate-head")

@app.local_entrypoint()
def main():
    remote_cache = modal.Dict.from_name("clippedai-asd-cache")
    # Grab the clip with the most frames
    target_key = "699388d603b1a41a60a7e611194c6c7491d19a991e44596b3b5e32a8febbb1f4"
    if target_key not in remote_cache:
        print("Key not found")
        return
        
    data = remote_cache[target_key]
    frames_count = len(data)
    frame_faces = {item["frame_number"]: item["faces"] for item in data}
    
    w = 1280
    h = 720
    
    def _norm_x(f): return ((f["x1"] + f["x2"]) / 2.0) / w
    def _norm_y(f): return ((f["y1"] + f["y2"]) / 2.0) / h

    # Create speaker array (let's assume SPEAKER_00 for first half, SPEAKER_01 for second half)
    speaker_array = ["SPEAKER_00"] * (frames_count // 2) + ["SPEAKER_01"] * (frames_count - frames_count // 2)

    clip_spk_xs = {}
    for fi in range(frames_count):
        faces = frame_faces.get(fi, [])
        speaking = [f for f in faces if f.get("speaking", False)]
        spk = speaker_array[fi]
        if len(speaking) == 1 and spk is not None:
            clip_spk_xs.setdefault(spk, []).append(_face_cx(speaking[0]))
            
    clip_side_map = {
        spk: (1 if _robust_median(xs) > w / 2 else 0)
        for spk, xs in clip_spk_xs.items() if len(xs) >= 10
    }

    raw_n_spk = np.zeros(frames_count, dtype=int)
    raw_spk_cx = np.full((frames_count, 4), -1.0)
    
    for fi in range(frames_count):
        faces = frame_faces.get(fi, [])
        speaking = [f for f in faces if f.get("speaking", False)]
        spk = speaker_array[fi]
        
        # Path A
        if len(speaking) == 1:
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _norm_x(speaking[0])
            continue
            
        # Skip B
        
        # Path C
        if len(speaking) >= 2:
            by_x = sorted(speaking, key=lambda f: f["x1"])[:4]
            raw_n_spk[fi] = len(by_x)
            for i, f in enumerate(by_x):
                raw_spk_cx[fi, i] = _norm_x(f)
            continue
            
        # Path C.5
        if faces and spk is not None:
            if spk in clip_side_map:
                faces_by_x = sorted(faces, key=lambda f: f["x1"])
                best = faces_by_x[min(clip_side_map[spk], len(faces_by_x) - 1)]
            else:
                best = max(faces, key=_face_area)
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _norm_x(best)
            continue
            
        # Path D
        if faces:
            best = max(faces, key=_face_area)
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _norm_x(best)
            continue
            
        # Path E
        if spk is not None and spk in clip_spk_xs:
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _robust_median(clip_spk_xs[spk]) / w

    col = raw_spk_cx[:, 0]
    valid_mask = col != -1
    valid_idx = np.where(valid_mask)[0]
    all_idx = np.arange(len(col))
    
    def _hold_fill(arr, vidx, all_i):
        out = np.interp(all_i, vidx, arr[vidx])
        first_v = vidx[0]
        if first_v > 0: out[:first_v] = arr[first_v]
        return out
        
    if len(valid_idx) > 0:
        raw_spk_cx[:, 0] = _hold_fill(col, valid_idx, all_idx)
    
    out = smooth_segment(raw_spk_cx[:, 0], 0.5, 12)
    
    center_frames = np.where((out > 0.40) & (out < 0.60))[0]
    print(f"Frames where camera is centered (0.40 to 0.60): {len(center_frames)}")

