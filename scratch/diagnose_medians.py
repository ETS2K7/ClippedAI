import modal
import numpy as np

app = modal.App("diagnose-med")

@app.local_entrypoint()
def main():
    remote_cache = modal.Dict.from_name("clippedai-asd-cache")
    key = "bc0014b35a7b7bb59f669e7e1d8ff4fbf41fe89b9c06fe2f816b6c8cbb5c5066"
    data = remote_cache[key]
    
    xs = []
    for i in range(len(data)):
        faces = data[i]["faces"]
        speaking = [f for f in faces if f.get("speaking", False)]
        if len(speaking) == 1:
            xs.append((speaking[0]["x1"] + speaking[0]["x2"]) / 2.0)
            
    if xs:
        print(f"Total speaking frames: {len(xs)}")
        print(f"Median: {np.median(xs):.1f}")
        print(f"Min: {np.min(xs):.1f}, Max: {np.max(xs):.1f}")
    else:
        print("No speaking frames found")

