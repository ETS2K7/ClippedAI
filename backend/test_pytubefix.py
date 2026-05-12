import modal

image = modal.Image.debian_slim().pip_install("pytubefix")
app = modal.App("pytubefix-test", image=image)

@app.function()
def test_download():
    from pytubefix import YouTube
    from pytubefix.cli import on_progress
    print("Testing pytubefix...")
    try:
        yt = YouTube("https://www.youtube.com/watch?v=PqGehRgKTLo", use_oauth=True, allow_oauth_cache=True)
        # Using use_oauth requires user to authorize... actually we want non-interactive!
        yt = YouTube("https://www.youtube.com/watch?v=PqGehRgKTLo", use_po_token=True)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        print(f"Success! Title: {yt.title}")
        return "SUCCESS"
    except Exception as e:
        print("Failed!")
        print(e)
        return "FAILED"

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            test_download.remote()
