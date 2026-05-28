import modal
import subprocess
import time

image = (modal.Image.debian_slim()
    .apt_install(["curl", "wget"])
    .run_commands([
        "wget -O /usr/local/bin/wgcf https://github.com/ViRb3/wgcf/releases/download/v2.2.22/wgcf_2.2.22_linux_amd64",
        "chmod +x /usr/local/bin/wgcf",
        "wget -O wireproxy.tar.gz https://github.com/octeep/wireproxy/releases/download/v1.0.8/wireproxy_linux_amd64.tar.gz",
        "tar -xzf wireproxy.tar.gz -C /usr/local/bin/",
        "rm wireproxy.tar.gz"
    ])
    .pip_install("yt-dlp")
)

app = modal.App("wireproxy-test2", image=image)

@app.function()
def test_wireproxy():
    print("Generating wgcf profile...")
    subprocess.run(["wgcf", "register", "--accept-tos"], check=True)
    subprocess.run(["wgcf", "generate"], check=True)
    
    print("Appending Socks5 config...")
    with open("wgcf-profile.conf", "a") as f:
        f.write("\n[Socks5]\nBindAddress = 127.0.0.1:40000\n")
        
    print("Starting wireproxy...")
    svc = subprocess.Popen(["wireproxy", "-c", "wgcf-profile.conf"])
    time.sleep(3)
    
    print("Testing curl through SOCKS5 proxy...")
    try:
        ip = subprocess.check_output(["curl", "-s", "-x", "socks5h://127.0.0.1:40000", "https://api.ipify.org"], text=True)
        print(f"Proxy IP: {ip}")
    except Exception as e:
        print(f"Curl failed: {e}")
        
    print("\nTesting yt-dlp...")
    try:
        output = subprocess.check_output(
            ["yt-dlp", "--proxy", "socks5://127.0.0.1:40000", "--dump-json", "https://www.youtube.com/watch?v=PqGehRgKTLo"],
            stderr=subprocess.STDOUT, text=True
        )
        print("Success! Download metadata retrieved via Cloudflare WARP IP!")
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
            test_wireproxy.remote()
