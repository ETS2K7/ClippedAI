import modal
import subprocess
import time

image = (modal.Image.debian_slim()
    .apt_install(["curl", "gnupg", "ca-certificates"])
    .run_commands([
        "curl -fsSL https://pkg.cloudflareclient.com/pubkey.gpg | gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg",
        "echo \"deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ bookworm main\" | tee /etc/apt/sources.list.d/cloudflare-client.list",
        "apt-get update && apt-get install -y cloudflare-warp"
    ])
    .pip_install("yt-dlp")
)

app = modal.App("warp-test", image=image)

@app.function()
def test_warp():
    print("Starting warp-svc...")
    svc = subprocess.Popen(["warp-svc"])
    time.sleep(2)
    
    print("Registering WARP...")
    subprocess.run(["warp-cli", "--accept-tos", "register"], check=True)
    subprocess.run(["warp-cli", "--accept-tos", "set-mode", "proxy"], check=True)
    subprocess.run(["warp-cli", "--accept-tos", "connect"], check=True)
    time.sleep(3)
    
    print("Testing curl through proxy...")
    subprocess.run(["curl", "-x", "socks5://127.0.0.1:40000", "https://api.ipify.org"])
    
    print("\nTesting yt-dlp...")
    try:
        output = subprocess.check_output(
            ["yt-dlp", "--proxy", "socks5://127.0.0.1:40000", "--dump-json", "https://www.youtube.com/watch?v=PqGehRgKTLo"],
            stderr=subprocess.STDOUT, text=True
        )
        print("Success!")
        return "SUCCESS"
    except subprocess.CalledProcessError as e:
        print("Failed!")
        print(e.output)
        return "FAILED"
    finally:
        svc.terminate()

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            test_warp.remote()
