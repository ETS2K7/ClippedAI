import modal
import os

app = modal.App("tun-test")

@app.function()
def check_tun():
    exists = os.path.exists("/dev/net/tun")
    print(f"/dev/net/tun exists: {exists}")
    return exists

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            check_tun.remote()
