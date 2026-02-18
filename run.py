"""
ClippedAI — Local CLI
Downloads video locally (yt-dlp has browser cookies), uploads to Modal for processing.
"""

import json
import os
import sys
import re
import subprocess
import tempfile


def sanitize_filename(name: str) -> str:
    """Remove unsafe characters from filename."""
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:50] if name else "untitled"


def download_locally(url: str) -> tuple:
    """
    Download video on this machine using yt-dlp.
    Works because local machine has browser cookie access (no bot detection).

    Returns: (video_bytes, title)
    """
    from config import YTDLP_FORMAT, MAX_VIDEO_DURATION

    tmp_dir = tempfile.mkdtemp(prefix="clipped_")
    video_path = os.path.join(tmp_dir, "source.mp4")

    # Get metadata
    meta_cmd = ["yt-dlp", "--dump-json", "--no-download", url]
    print("  📥 Fetching video metadata...")
    result = subprocess.run(meta_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {result.stderr[:500]}")

    meta = json.loads(result.stdout)
    duration = meta.get("duration", 0)
    title = meta.get("title", "Untitled")

    if duration > MAX_VIDEO_DURATION:
        raise ValueError(f"Video too long: {duration/3600:.1f}h (max 4h)")

    print(f"  📥 Downloading: {title} ({duration:.0f}s)")

    # Download
    dl_cmd = [
        "yt-dlp",
        "-f", YTDLP_FORMAT,
        "--merge-output-format", "mp4",
        "-o", video_path,
        "--no-playlist",
        "--retries", "3",
        url,
    ]
    result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr[:500]}")

    if not os.path.exists(video_path):
        raise FileNotFoundError("Downloaded file not found")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    # Cleanup
    try:
        os.remove(video_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    size_mb = len(video_bytes) / (1024 * 1024)
    print(f"  ✅ Downloaded: {size_mb:.1f} MB")
    return video_bytes, title


def load_local_file(file_path: str) -> tuple:
    """Load a local video file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        video_bytes = f.read()

    title = os.path.splitext(os.path.basename(file_path))[0]
    size_mb = len(video_bytes) / (1024 * 1024)
    print(f"  📂 Loaded local file: {title} ({size_mb:.1f} MB)")
    return video_bytes, title


def main():
    print()
    print("🎬 ClippedAI — Core Clipping Pipeline")
    print("=" * 45)

    # Get input
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = input("\nYouTube URL or local file path: ").strip()

    if not source:
        print("❌ No source provided")
        sys.exit(1)

    # Parse optional args
    max_clips = 5
    min_duration = 15
    max_duration = 90

    if len(sys.argv) > 2:
        max_clips = int(sys.argv[2])
    else:
        val = input("Max clips (default 5): ").strip()
        if val:
            max_clips = int(val)

    if len(sys.argv) > 3:
        min_duration = int(sys.argv[3])
    if len(sys.argv) > 4:
        max_duration = int(sys.argv[4])

    # Determine source type
    is_url = source.startswith("http://") or source.startswith("https://")

    if is_url:
        print(f"\n📥 Downloading video locally (bypasses YouTube bot detection)...")
        video_bytes, title = download_locally(source)
    else:
        video_bytes, title = load_local_file(source)

    size_mb = len(video_bytes) / (1024 * 1024)
    print(f"\n🚀 Uploading {size_mb:.1f} MB to Modal (A10G GPU)...")
    print(f"   Title: {title}")
    print(f"   Max clips: {max_clips}")
    print(f"   Duration range: {min_duration}–{max_duration}s")
    print(f"   Processing may take 2-5 minutes...\n")

    # Call Modal function
    try:
        import modal
        process_video = modal.Function.from_name("clipped-ai", "process_video")
        result = process_video.remote(
            video_bytes=video_bytes,
            video_title=title,
            max_clips=max_clips,
            min_clip_duration=min_duration,
            max_clip_duration=max_duration,
        )
    except Exception as e:
        print(f"\n❌ Modal execution failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: modal deploy modal_app.py")
        print("  2. Check: modal secret list (cerebras-api-key should exist)")
        print("  3. Check: modal app list (clipped-ai should be deployed)")
        sys.exit(1)

    if not result.get("clips"):
        print("\n⚠️  No clips were generated. The video may be:")
        print("  - Too short for the minimum clip duration")
        print("  - Lacking interesting moments")
        sys.exit(0)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Clear previous output
    for f in os.listdir("output"):
        if f.endswith(".mp4") or f == "metadata.json":
            os.remove(os.path.join("output", f))

    # Save clips
    print(f"\n📥 Saving {len(result['clips'])} clips...\n")
    for i, clip in enumerate(result["clips"]):
        safe_title = sanitize_filename(clip["title"])
        filename = f"clip_{i+1:02d}_{safe_title}.mp4"
        filepath = os.path.join("output", filename)

        with open(filepath, "wb") as f:
            f.write(clip["video_bytes"])

        size_mb = len(clip["video_bytes"]) / (1024 * 1024)
        score = clip.get("virality_score", 0)
        duration = clip.get("duration", 0)
        print(f"  💾 {filename}")
        print(f"     {duration:.0f}s | score: {score:.2f} | {size_mb:.1f} MB")
        if clip.get("reasoning"):
            reason = clip["reasoning"][:80]
            print(f"     💡 {reason}...")

    # Save metadata (without video bytes)
    meta = {
        "video_title": result["video_title"],
        "video_duration": result["video_duration"],
        "timings": result["timings"],
        "clips": [
            {k: v for k, v in c.items() if k != "video_bytes"}
            for c in result["clips"]
        ],
    }
    with open("output/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 45}")
    print(f"✅ {len(result['clips'])} clips saved to output/")
    print(f"📋 Metadata: output/metadata.json")
    print(f"\n⏱️  Pipeline timings:")
    for step, secs in result["timings"].items():
        print(f"   {step}: {secs}s")
    print(f"{'=' * 45}")


if __name__ == "__main__":
    main()
