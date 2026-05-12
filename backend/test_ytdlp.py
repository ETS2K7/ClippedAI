import modal
import subprocess

image = modal.Image.debian_slim().pip_install("yt-dlp")
app = modal.App("yt-dlp-test", image=image)

@app.function()
def test_download():
    print("Testing yt-dlp...")
    try:
        output = subprocess.check_output(
            ["yt-dlp", "--dump-json", "https://www.youtube.com/watch?v=PqGehRgKTLo"],
            stderr=subprocess.STDOUT, text=True
        )
        print("Success!")
        return "SUCCESS"
    except subprocess.CalledProcessError as e:
        print("Failed!")
        print(e.output)
        return "FAILED"

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            test_download.remote()
