import modal

app = modal.App("list-cache")

@app.local_entrypoint()
def main():
    try:
        remote_cache = modal.Dict.from_name("clippedai-asd-cache")
        keys = list(remote_cache.keys())
        print(f"Found {len(keys)} keys in cache")
        for k in keys:
            print(k)
    except Exception as e:
        print("Error:", e)

