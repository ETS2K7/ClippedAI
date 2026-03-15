#!/usr/bin/env python3
"""
ClippedAI — CLI

Usage:
    python run.py "https://youtube.com/watch?v=VIDEO_ID"
    python run.py "https://youtube.com/watch?v=VIDEO_ID" --max-clips 3
    python run.py "https://youtube.com/watch?v=VIDEO_ID" --dry-run

Downloads video locally with yt-dlp, uploads to Modal volume,
then triggers the pipeline on Modal.
"""
import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import modal


def parse_args():
    parser = argparse.ArgumentParser(
        description="ClippedAI — AI-powered video clipping pipeline"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--max-clips", type=int, default=1, help="Max clips to generate")
    parser.add_argument("--duration", type=str, default="15-60", help="Clip duration range (e.g., '15-60')")
    parser.add_argument("--dry-run", action="store_true", help="Download and analyze only, don't render")
    parser.add_argument("--force-reanalyze", action="store_true", help="Re-run all analysis")
    parser.add_argument("--reselect-clips", action="store_true", help="Re-run clip selection only")
    parser.add_argument("--rerender", action="store_true", help="Re-render with updated settings")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for clips")
    return parser.parse_args()


def download_video(url: str, download_dir: str) -> str:
    """Download video with yt-dlp to a local directory. Returns path to downloaded file."""
    print(f"\n📥 Downloading video...")
    print(f"   URL: {url}")

    output_template = os.path.join(download_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--progress",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n❌ yt-dlp failed:")
        print(result.stderr[-2000:])
        sys.exit(1)

    # Find the downloaded file
    mp4_files = list(Path(download_dir).glob("*.mp4"))
    if not mp4_files:
        print("❌ No MP4 file found after download")
        sys.exit(1)

    video_path = str(mp4_files[0])
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"   ✅ Downloaded: {Path(video_path).name} ({file_size:.1f} MB)")
    return video_path


def compute_video_hash(video_path: str) -> str:
    """Compute collision-resistant video hash (same as pipeline/ingest.py)."""
    chunk_size = 10 * 1024 * 1024

    with open(video_path, "rb") as f:
        head = f.read(chunk_size)
        f.seek(0, 2)
        file_size = f.tell()
        if file_size > chunk_size:
            f.seek(-chunk_size, 2)
            tail = f.read(chunk_size)
        else:
            tail = head

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())

    h = hashlib.sha256()
    h.update(head)
    h.update(tail)
    h.update(struct.pack("d", duration))
    return h.hexdigest()[:16]


def upload_to_volume(video_path: str, video_hash: str):
    """Upload the local video to the Modal volume."""
    print(f"\n☁️  Uploading to Modal volume...")

    vol = modal.Volume.from_name("clippedai-vol")
    dest_path = f"{video_hash}/source.mp4"

    # Check if already uploaded
    try:
        for entry in vol.listdir(video_hash):
            if entry.path == dest_path:
                print(f"   ✅ Already uploaded (cached)")
                return
    except Exception:
        pass  # Directory doesn't exist yet

    # Upload using batch_upload
    with vol.batch_upload(force=True) as batch:
        batch.put_file(video_path, f"/{dest_path}")

    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"   ✅ Uploaded to /vol/{video_hash}/source.mp4 ({file_size:.1f} MB)")


def download_clips(video_hash: str, clip_paths: list[str], output_dir: str):
    """Download rendered clips from Modal volume to local output directory."""
    print(f"\n📦 Downloading clips to {output_dir}/...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name("clippedai-vol")

    for clip_path in clip_paths:
        filename = Path(clip_path).name
        local_path = os.path.join(output_dir, filename)

        # Volume paths are relative to mount — strip /vol/
        vol_relative = clip_path.replace("/vol/", "")
        data = b""
        for chunk in vol.read_file(vol_relative):
            data += chunk

        with open(local_path, "wb") as f:
            f.write(data)

        file_size = len(data) / (1024 * 1024)
        print(f"   ✅ {filename} ({file_size:.1f} MB)")


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("  ClippedAI — Core Clipping Pipeline")
    print("=" * 60)

    # Step 1: Download locally
    download_dir = os.path.join("/tmp", "clippedai_downloads")
    os.makedirs(download_dir, exist_ok=True)

    video_path = download_video(args.url, download_dir)

    # Step 2: Compute hash
    print(f"\n🔑 Computing video hash...")
    video_hash = compute_video_hash(video_path)
    print(f"   Hash: {video_hash}")

    # Step 3: Upload to Modal volume
    upload_to_volume(video_path, video_hash)

    if args.dry_run:
        print(f"\n🏁 Dry run complete. Video hash: {video_hash}")
        return

    # Step 4: Run pipeline on Modal (calls the deployed app)
    print(f"\n🚀 Running pipeline on Modal...")
    print(f"   Max clips: {args.max_clips}")

    process_video = modal.Function.from_name(f"clippedai", "process_video")
    clip_paths = process_video.remote(video_hash, max_clips=args.max_clips)

    if not clip_paths:
        print("\n❌ No clips were generated")
        sys.exit(1)

    print(f"\n   ✅ {len(clip_paths)} clip(s) rendered on Modal")

    # Step 5: Download clips
    download_clips(video_hash, clip_paths, args.output_dir)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  ✅ Done! {len(clip_paths)} clip(s) saved to {args.output_dir}/")
    print(f"  ⏱  Total time: {elapsed:.1f}s")
    print(f"  🔑 Video hash: {video_hash}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
