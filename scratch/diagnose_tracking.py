import modal
import numpy as np

app = modal.App("diagnose-track")

@app.local_entrypoint()
def main():
    remote_cache = modal.Dict.from_name("clippedai-asd-cache")
    key = "bc0014b35a7b7bb59f669e7e1d8ff4fbf41fe89b9c06fe2f816b6c8cbb5c5066"
    data = remote_cache[key]
    
    xs = np.full(len(data), -1.0)
    for i in range(len(data)):
        faces = data[i]["faces"]
        speaking = [f for f in faces if f.get("speaking", False)]
        if len(speaking) == 1:
            xs[i] = ((speaking[0]["x1"] + speaking[0]["x2"]) / 2.0) / 1280.0
            
    # Mock smooth_segment
    seg_len = len(xs)
    valid_mask = xs != -1
    valid_points = xs[valid_mask]
    
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
    std = float(np.std(largest_cluster))
    is_stationary = std < 0.094

    print(f"Clusters: {len(clusters)}")
    for i, c in enumerate(clusters):
        print(f"  Cluster {i}: len={len(c)}, median={np.median(c):.3f}, std={np.std(c):.3f}")
        
    print(f"Largest cluster std: {std:.3f}")
    print(f"Is stationary: {is_stationary}")

