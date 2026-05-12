import modal

app = modal.App("diagnose-faces")

@app.local_entrypoint()
def main():
    remote_cache = modal.Dict.from_name("clippedai-asd-cache")
    key = "bc0014b35a7b7bb59f669e7e1d8ff4fbf41fe89b9c06fe2f816b6c8cbb5c5066"
    data = remote_cache[key]
    
    # Print the coordinates of all detected faces in a sample of frames
    for i in range(0, min(100, len(data)), 10):
        faces = data[i]["faces"]
        coords = [f"{((f['x1']+f['x2'])/2.0):.1f}" for f in faces]
        print(f"Frame {i:4d}: faces at {coords}")

