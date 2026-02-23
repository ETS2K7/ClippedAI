#!/usr/bin/env python3
"""
ClippedAI — CLI Entry Point

Usage:
    python run.py "https://youtube.com/watch?v=VIDEO_ID"
    python run.py "URL" --max-clips 3 --duration 15-45
    python run.py "URL" --dry-run
    python run.py "URL" --force-reanalyze
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import config


def main():
    parser = argparse.ArgumentParser(
        description="ClippedAI — Generate viral short-form clips from YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py "https://youtube.com/watch?v=VIDEO_ID"
    python run.py "URL" --max-clips 3
    python run.py "URL" --duration 15-45
    python run.py "URL" --dry-run
    python run.py "URL" --force-reanalyze
        """,
    )

    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--max-clips", type=int, default=config.MAX_CLIPS,
                       help=f"Maximum number of clips (default: {config.MAX_CLIPS})")
    parser.add_argument("--duration", type=str, default=f"{config.MIN_CLIP_DURATION}-{config.MAX_CLIP_DURATION}",
                       help=f"Clip duration range MIN-MAX in seconds (default: {config.MIN_CLIP_DURATION}-{config.MAX_CLIP_DURATION})")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory (default: output/)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run analysis only, show clip candidates without rendering")
    parser.add_argument("--force-reanalyze", action="store_true",
                       help="Force re-analysis even if cache exists")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse duration range
    try:
        min_dur, max_dur = map(int, args.duration.split("-"))
    except ValueError:
        print(f"Error: Invalid duration format '{args.duration}'. Use MIN-MAX (e.g., 15-60)")
        sys.exit(1)

    # Build settings dict
    settings = {
        "max_clips": args.max_clips,
        "min_duration": min_dur,
        "max_duration": max_dur,
        "ideal_duration": config.IDEAL_CLIP_DURATION,
        "dry_run": args.dry_run,
        "force_reanalyze": args.force_reanalyze,
    }

    # Print banner
    _print_banner(args.url, settings)

    # Run on Modal
    start_time = time.time()

    try:
        import modal
        from modal_app import process_video

        print("\n⏳ Running pipeline on Modal...")
        print("   This may take 2-5 minutes on first run.\n")

        with modal.enable_output():
            clip_paths = process_video.remote(args.url, settings)

        elapsed = time.time() - start_time

        if not clip_paths:
            print("\n❌ No clips were generated. Check logs for details.")
            sys.exit(1)

        # Download clips to local output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📥 Downloading {len(clip_paths)} clips to {output_dir}/...\n")

        downloaded = []
        for remote_path in clip_paths:
            filename = Path(remote_path).name
            local_path = output_dir / filename

            # In Modal context, volume files are accessible directly
            # For local CLI, we'd use modal.Volume.read_file
            try:
                from modal_app import volume as vol
                # Strip mount prefix: read_file expects relative path within volume
                mount_prefix = config.MODAL_VOLUME_MOUNT + "/"
                relative_path = remote_path.replace(mount_prefix, "")
                data = vol.read_file(relative_path)
                with open(local_path, "wb") as f:
                    for chunk in data:
                        f.write(chunk)
                downloaded.append(local_path)
            except Exception as e:
                # Fallback: try direct copy if running in same context
                src = Path(remote_path)
                if src.exists():
                    shutil.copy2(src, local_path)
                    downloaded.append(local_path)
                else:
                    print(f"   ⚠️  Could not download {filename}: {e}")

        # Print summary
        _print_summary(downloaded, elapsed)

    except ImportError:
        print("\n❌ Modal is not installed. Run: pip install modal")
        print("   Then authenticate: modal token new")
        sys.exit(1)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Pipeline failed after {elapsed:.0f}s: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _print_banner(url: str, settings: dict):
    """Print startup banner."""
    print("""
╔══════════════════════════════════════╗
║         🎬  ClippedAI  🎬           ║
║   Viral Clips from YouTube Videos    ║
╚══════════════════════════════════════╝
    """)
    print(f"  URL:       {url}")
    print(f"  Max clips: {settings['max_clips']}")
    print(f"  Duration:  {settings['min_duration']}-{settings['max_duration']}s")
    print(f"  Ideal:     {settings['ideal_duration'][0]}-{settings['ideal_duration'][1]}s")
    if settings.get("dry_run"):
        print("  Mode:      DRY RUN (no rendering)")
    if settings.get("force_reanalyze"):
        print("  Cache:     FORCE REANALYZE")


def _print_summary(clips: list[Path], elapsed: float):
    """Print completion summary table."""
    print(f"\n{'═' * 50}")
    print(f"  ✅ Pipeline Complete — {elapsed:.1f}s")
    print(f"{'═' * 50}")
    print()

    if clips:
        print(f"  {'#':<4} {'Filename':<25} {'Size':>8}")
        print(f"  {'─' * 4} {'─' * 25} {'─' * 8}")
        for i, clip in enumerate(clips, 1):
            size = clip.stat().st_size / 1e6
            print(f"  {i:<4} {clip.name:<25} {size:>6.1f}MB")
        print()
        print(f"  📂 Output: {clips[0].parent}/")
    else:
        print("  ⚠️  No clips downloaded.")

    print()


if __name__ == "__main__":
    main()
