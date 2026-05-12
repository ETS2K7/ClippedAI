import modal
import numpy as np

app = modal.App("diagnose-maps")

@app.local_entrypoint()
def main():
    try:
        remote_cache = modal.Dict.from_name("clippedai-asd-cache")
        key = "bc0014b35a7b7bb59f669e7e1d8ff4fbf41fe89b9c06fe2f816b6c8cbb5c5066" # Clip 2 (no scene cuts)
        
        if key not in remote_cache:
            print("Key not found")
            return
            
        data = remote_cache[key]
        frames_count = len(data)
        
        # We don't have speaker_array here, but we can simulate it
        # Let's just print how many frames have len(speaking) == 1
        
        speaking_1_count = 0
        faces_1_count = 0
        
        for i in range(frames_count):
            faces = data[i]["faces"]
            speaking = [f for f in faces if f.get("speaking", False)]
            if len(speaking) == 1:
                speaking_1_count += 1
            if len(faces) == 1:
                faces_1_count += 1
                
        print(f"Frames with exactly 1 speaking face: {speaking_1_count} / {frames_count}")
        print(f"Frames with exactly 1 detected face: {faces_1_count} / {frames_count}")

    except Exception as e:
        print("Error:", e)

