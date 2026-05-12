import json
import numpy as np
import modal
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from src.video_processing import _robust_median, _prominent_distinct_faces, _face_area, _face_cx, _stabilize_bool_state
from src.signal_helpers import smooth_segment, stabilize_segment

app = modal.App("simulate-math")

@app.local_entrypoint()
def main():
    remote_cache = modal.Dict.from_name("clippedai-asd-cache")
    target_key = "699388d603b1a41a60a7e611194c6c7491d19a991e44596b3b5e32a8febbb1f4"
    target_data = remote_cache[target_key]
    
    print(f"Simulating math on {target_key}")
    
    frames_count = len(target_data)
    frame_faces = {item["frame_number"]: item["faces"] for item in target_data}
    
    w = 1280
    h = 720
    
    def _norm_x(f): return ((f["x1"] + f["x2"]) / 2.0) / w
    def _norm_y(f): return ((f["y1"] + f["y2"]) / 2.0) / h

    raw_n_spk = np.zeros(frames_count, dtype=int)
    raw_spk_cx = np.full((frames_count, 4), -1.0)
    
    for fi in range(frames_count):
        faces = frame_faces.get(fi, [])
        speaking = [f for f in faces if f.get("speaking", False)]
        
        if len(speaking) == 1:
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _norm_x(speaking[0])
        elif len(speaking) >= 2:
            by_x = sorted(speaking, key=lambda f: f["x1"])[:4]
            raw_n_spk[fi] = len(by_x)
            for i, f in enumerate(by_x):
                raw_spk_cx[fi, i] = _norm_x(f)
        elif faces:
            best = max(faces, key=_face_area)
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _norm_x(best)
            
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
    
    # We want to see what 'out' is during the two-face segment
    center_frames = np.where((out > 0.40) & (out < 0.60))[0]
    print(f"Frames where camera is centered (0.40 to 0.60): {len(center_frames)}")
    if len(center_frames) > 0:
        print("Sample centered frames:", center_frames[:10])
        for cf in center_frames[:5]:
            print(f"Frame {cf}: out={out[cf]:.3f}, raw={raw_spk_cx[cf, 0]:.3f}, faces={len(frame_faces.get(cf, []))}")
            
    print(f"Min cx: {np.min(out):.3f}, Max cx: {np.max(out):.3f}")

