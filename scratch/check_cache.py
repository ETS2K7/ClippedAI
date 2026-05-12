import modal

app = modal.App("check-cache")

@app.local_entrypoint()
def main():
    try:
        remote_cache = modal.Dict.from_name("clippedai-asd-cache")
        keys = list(remote_cache.keys())
        print(f"Found {len(keys)} keys in cache")
        for k in keys:
            data = remote_cache[k]
            print(f"Key {k[:8]}: {len(data)} frames")
            if len(data) > 0:
                print(f"  Sample face count in frame 0: {len(data[0].get('faces', []))}")
                print(f"  Frame 0 exact data: {data[0]}")
                break
    except Exception as e:
        print("Error:", e)

