"""
ClippedAI — Local CLI
Invokes the Modal pipeline and downloads clips to output/
"""

import modal
import json
import os
import sys
import re


def sanitize_filename(name: str) -> str:
    """Remove unsafe characters from filename."""
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:50] if name else "untitled"


def main():
    print()
    print("🎬 ClippedAI — Core Clipping Pipeline")
    print("=" * 45)

    # Get input
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("\nYouTube URL: ").strip()

    if not url:
        print("❌ No URL provided")
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

    print(f"\n🚀 Processing on Modal (A10G GPU)...")
    print(f"   URL: {url}")
    print(f"   Max clips: {max_clips}")
    print(f"   Duration range: {min_duration}–{max_duration}s")
    print(f"   This may take 2-5 minutes...\n")

    # Call Modal function
    try:
        process_video = modal.Function.from_name("clipped-ai", "process_video")
        result = process_video.remote(
            youtube_url=url,
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
        print("\n⚠️ No clips were generated. The video may be:")
        print("  - Too short")
        print("  - Geo-blocked or private")
        print("  - Lacking interesting moments")
        sys.exit(0)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Clear previous output
    for f in os.listdir("output"):
        if f.endswith(".mp4") or f == "metadata.json":
            os.remove(os.path.join("output", f))

    # Save clips
    print(f"\n📥 Downloading {len(result['clips'])} clips...\n")
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
            print(f"     💡 {clip['reasoning'][:80]}...")

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
