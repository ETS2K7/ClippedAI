import modal

app = modal.App("diagnose-faces-2")

@app.local_entrypoint()
def main():
    remote_cache = modal.Dict.from_name("clippedai-asd-cache")
    # Find a clip that has many frames with >= 2 faces
    for k in remote_cache.keys():
        data = remote_cache[k]
        two_faces_count = sum(1 for item in data if len(item["faces"]) >= 2)
        if two_faces_count > len(data) * 0.5: # More than 50% of frames have >= 2 faces
            print(f"Found clip {k} with {two_faces_count}/{len(data)} frames having >= 2 faces")
            
            # Print some coordinates
            for i in range(0, min(100, len(data)), 10):
                faces = data[i]["faces"]
                coords = [f"{((f['x1']+f['x2'])/2.0)/1280.0:.3f}" for f in faces]
                print(f"Frame {i:4d}: faces at {coords}")
            break
