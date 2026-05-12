import modal
import json
import numpy as np

app = modal.App("diagnose-cache")

@app.local_entrypoint()
def main():
    try:
        remote_cache = modal.Dict.from_name("clippedai-asd-cache")
        # Try a few keys we saw in logs
        for key in ["3e56f782", "03248a94", "bc0014b3"]:
            if key in remote_cache:
                data = remote_cache[key]
                print(f"\n--- Cache Key: {key} ---")
                print(f"Total frames tracked: {len(data)}")
                
                # Let's see what the faces actually look like in the first 20 frames
                for i in range(min(20, len(data))):
                    item = data[i]
                    frame_num = item["frame_number"]
                    faces = item["faces"]
                    speaking_faces = [f for f in faces if f.get("speaking", False)]
                    
                    print(f"F{frame_num}: {len(faces)} faces, {len(speaking_faces)} speaking.")
                    for f in faces:
                        cx = (f["x1"] + f["x2"]) / 2.0
                        speaking = "YES" if f.get("speaking") else "NO"
                        print(f"   cx={cx:.1f}, speak={speaking}")
                break
    except Exception as e:
        print("Error:", e)

