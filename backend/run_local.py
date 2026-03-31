"""
ClippedAI Local Pipeline Runner
Runs the full pipeline locally: Download → Transcribe → LLM → Fast-ASD (Modal) → Subtitles → Merge
All generated clips are saved to the /output folder.
"""
import os
import sys
import shutil
import pathlib

# Run from the backend directory so imports resolve cleanly
os.chdir(pathlib.Path(__file__).parent)
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from config import get_logger
from src.downloader import download_video
from src.transcriber import transcribe
from src.llm import select_clips
from src.video_processing import extract_segment, track_speaker_and_frame, merge_and_cleanup
from src.subtitles import generate_subtitles

logger = get_logger("local_runner")

def run(url: str):
    output_dir = pathlib.Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a local output folder override so merge_and_cleanup saves here
    os.makedirs("output", exist_ok=True)

    logger.info(f"Starting ClippedAI local pipeline for: {url}")

    # Phase 1: Download
    video_path = download_video(url)
    logger.info(f"Downloaded to: {video_path}")

    # Phase 2: Transcribe
    words = transcribe(video_path, url)
    logger.info(f"Transcription complete. {len(words)} words.")

    # Phase 3: LLM clip selection
    clips = select_clips(words)
    logger.info(f"Selected {len(clips)} clips.")

    for index, clip in enumerate(clips):
        logger.info(f"\n--- Clip {index + 1}/{len(clips)}: {clip['start_time']}s → {clip['end_time']}s ---")

        # Phase 4: Extract segment
        ext_vid = extract_segment(video_path, clip, index)

        # Phase 5: Fast-ASD speaker tracking (runs on Modal GPU)
        trk_vid, chunk_meta = track_speaker_and_frame(ext_vid, index, clip, words)

        # Phase 6: Generate subtitles
        sub_file = generate_subtitles(words, clip, index, chunk_meta)

        # Phase 7: Merge and write to output/
        merge_and_cleanup(trk_vid, ext_vid, sub_file, index)

        final_path = f"output/clip_{index + 1}.mp4"
        if os.path.exists(final_path):
            # Copy to the root /output folder for easy access
            dest = output_dir / f"clip_{index + 1}.mp4"
            shutil.copy2(final_path, dest)
            logger.info(f"✅ Saved: {dest.resolve()}")
        else:
            logger.warning(f"⚠️  Expected clip not found at {final_path}")

    logger.info(f"\n🎉 Pipeline complete. Clips saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_local.py <youtube_url>")
        sys.exit(1)
    run(sys.argv[1])
