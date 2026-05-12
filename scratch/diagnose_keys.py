import modal

app = modal.App("diagnose-keys")

@app.local_entrypoint()
def main():
    remote_cache = modal.Dict.from_name("clippedai-asd-cache")
    keys = list(remote_cache.keys())
    for k in keys:
        print(f"Key: {k}, frames: {len(remote_cache[k])}")
