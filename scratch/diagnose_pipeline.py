import modal
import numpy as np

app = modal.App("diagnose-pipeline")

@app.local_entrypoint()
def main():
    try:
        remote_cache = modal.Dict.from_name("clippedai-asd-cache")
        key = "3e56f782efaad2b2fbc21efc0e18689cffc8037a3975dddb13acc6a4d126f48e"
        
        if key not in remote_cache:
            print("Key not found")
            return
            
        data = remote_cache[key]
        frames_count = len(data)
        
        # Simulate video_processing.py phase 6
        raw_spk_cx = np.full((frames_count, 4), -1.0)
        
        for i in range(frames_count):
            faces = data[i]["faces"]
            speaking = [f for f in faces if f.get("speaking", False)]
            
            if len(speaking) == 1:
                f = speaking[0]
                raw_spk_cx[i, 0] = ((f["x1"] + f["x2"]) / 2.0) / 1280.0
            elif len(faces) >= 1:
                f = max(faces, key=lambda x: (x["x2"]-x["x1"])*(x["y2"]-x["y1"]))
                raw_spk_cx[i, 0] = ((f["x1"] + f["x2"]) / 2.0) / 1280.0
                
        # We simulate _hold_fill
        def _hold_fill(arr, vidx, aidx):
            vmask = arr != -1
            fwd = np.where(vmask, aidx, 0)
            np.maximum.accumulate(fwd, out=fwd)
            out = arr[fwd]
            first_v = vidx[0]
            if first_v > 0:
                out[:first_v] = arr[first_v]
            return out
            
        col_x = raw_spk_cx[:, 0]
        valid_x = col_x != -1
        if np.any(valid_x):
            valid_idx = np.where(valid_x)[0]
            all_idx = np.arange(len(col_x))
            col_x = _hold_fill(col_x, valid_idx, all_idx)
            
        # We simulate smooth_segment
        raw = col_x
        seg_len = len(raw)
        valid_mask = raw != -1
        valid_points = raw[valid_mask]
        
        sorted_pts = np.sort(valid_points)
        gaps = np.diff(sorted_pts)
        gap_indices = np.where(gaps > 0.078)[0]
        
        boundaries = np.concatenate([[-1], gap_indices, [len(sorted_pts) - 1]])
        clusters = []
        for i in range(len(boundaries) - 1):
            start = int(boundaries[i]) + 1
            end = int(boundaries[i + 1]) + 1
            clusters.append(sorted_pts[start:end])
            
        largest_cluster = max(clusters, key=len)
        std_largest = float(np.std(largest_cluster))
        
        print(f"Largest cluster std: {std_largest:.4f}")
        
        if std_largest < 0.15:
            print("LOCKED MODE")
            cluster_centers = [float(np.median(c)) for c in clusters]
            print(f"Cluster centers: {cluster_centers}")
            out = np.full(seg_len, 0.5)
            last_pos = cluster_centers[0]
            for i in range(seg_len):
                if raw[i] != -1:
                    target = float(raw[i])
                    best_center = min(cluster_centers, key=lambda c: abs(c - target))
                    last_pos = best_center
                out[i] = last_pos
        else:
            print("TRACKING MODE")
            out = raw.copy()
            
        # Print output mapping
        print("\nOutput segments:")
        last_val = out[0]
        start_idx = 0
        for i in range(1, len(out)):
            if out[i] != last_val:
                print(f"Frames {start_idx:4d} to {i-1:4d}: cx = {last_val:.3f}")
                last_val = out[i]
                start_idx = i
        print(f"Frames {start_idx:4d} to {len(out)-1:4d}: cx = {last_val:.3f}")

    except Exception as e:
        print("Error:", e)

