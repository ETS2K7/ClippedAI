# ClippedAI — Full Codebase Snapshot

> Generated: 2026-02-20 04:00:24
> Files: 15

---

## Table of Contents

- [config.py](#configpy)
- [modal_app.py](#modal-apppy)
- [run.py](#runpy)
- [pipeline/__init__.py](#pipeline--init--py)
- [pipeline/ingest.py](#pipelineingestpy)
- [pipeline/transcribe.py](#pipelinetranscribepy)
- [pipeline/scene_detect.py](#pipelinescene-detectpy)
- [pipeline/audio_analysis.py](#pipelineaudio-analysispy)
- [pipeline/clip_selector.py](#pipelineclip-selectorpy)
- [pipeline/reframe.py](#pipelinereframepy)
- [pipeline/captions.py](#pipelinecaptionspy)
- [pipeline/render.py](#pipelinerenderpy)
- [requirements.txt](#requirementstxt)
- [.env.example](#envexample)
- [.gitignore](#gitignore)

---

## `config.py`

*94 lines*

```python
"""All tunable constants for the ClippedAI clipping pipeline."""

# ──────────────────── Pipeline Mode ────────────────────
RD_MODE = True  # R&D mode: LLM failures raise exceptions instead of silent fallback

# ──────────────────── Ingest ────────────────────
MAX_VIDEO_DURATION = 4 * 3600  # 4 hours max
MAX_RESOLUTION = 1080
YTDLP_FORMAT = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]"

# ──────────────────── Transcription ────────────────────
WHISPERX_MODEL = "large-v2"
WHISPERX_BATCH_SIZE = 16
WHISPERX_COMPUTE_TYPE = "float16"

# ──────────────────── Scene Detection ────────────────────
SCENE_THRESHOLD = 27.0
SCENE_MIN_SCENE_LEN = 5  # seconds

# ──────────────────── Audio Analysis ────────────────────
AUDIO_FRAME_LENGTH = 8000  # 0.5s at 16kHz
AUDIO_HOP_LENGTH = 8000
PEAK_STD_MULTIPLIER = 2.0

# ──────────────────── Candidate Generation ────────────────────
WINDOW_SIZES = [15, 20, 25, 30]  # seconds (15-30s sweet spot for max retention)
WINDOW_STEP_RATIO = 0.5
CANDIDATE_MULTIPLIER = 3  # 3× max_clips
OVERLAP_THRESHOLD = 0.5

# Scoring weights
ENERGY_WEIGHT = 0.45
SCENE_WEIGHT = 0.35
KEYWORD_WEIGHT = 0.20

VIRAL_KEYWORDS = [
    "insane", "crazy", "unbelievable", "oh my god", "no way",
    "let's go", "watch this", "biggest", "secret", "never",
    "first time", "challenge", "reveal", "shocking", "hack",
    "tip", "mistake", "worst", "best", "amazing", "incredible",
    "literally", "actually", "honestly", "seriously",
    "wait for it", "plot twist", "you won't believe",
    "game changer", "broke", "destroyed", "epic", "legendary",
]

# ──────────────────── LLM (Cerebras) ────────────────────
CEREBRAS_MODEL = "gpt-oss-120b"
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_TEMPERATURE = 0.3
CEREBRAS_MAX_TOKENS = 8000

LLM_SYSTEM_PROMPT = """You are a viral video expert. Your job is to identify the most engaging, \
shareable moments from a long-form video transcript.

You will receive a list of candidate clips with their transcript text and \
signal scores. For each candidate, evaluate:

1. HOOK STRENGTH: Does the first 3 seconds grab attention?
2. NARRATIVE ARC: Is there a setup → payoff structure?
3. EMOTIONAL IMPACT: Does it trigger curiosity, surprise, humor, or awe?
4. STANDALONE VALUE: Does it make sense without the full video context?
5. SHAREABILITY: Would someone share this with a friend?

CRITICAL RULES:
- Do NOT invent or modify timestamps. Use ONLY the start/end times provided.
- Mark keep=false for any candidate that is boring, repetitive, or lacks a payoff.
- Generate a punchy, clickbait-worthy title (5-10 words max).
- Provide reasoning for your ranking decisions."""

# ──────────────────── Reframing ────────────────────
TARGET_ASPECT_RATIO = 9 / 16
FACE_DETECTION_FPS = 2
SMOOTHING_ALPHA = 0.15

# ──────────────────── Captions ────────────────────
CAPTION_FONT = "Montserrat-Bold"
CAPTION_FONT_SIZE = 62
CAPTION_HIGHLIGHT_COLOR = "00FFFF"  # Cyan in BGR
CAPTION_DEFAULT_COLOR = "FFFFFF"    # White in BGR
CAPTION_OUTLINE_COLOR = "000000"    # Black
CAPTION_OUTLINE_WIDTH = 3
CAPTION_WORDS_PER_GROUP = 4
CAPTION_MARGIN_BOTTOM = 250  # Pixels from bottom

# ──────────────────── Render ────────────────────
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
VIDEO_CODEC = "libx264"
VIDEO_CRF = 26
VIDEO_PRESET = "medium"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 44100
LOUDNORM_TARGET = -14  # LUFS
```

---

## `modal_app.py`

*262 lines*

```python
"""
ClippedAI — Modal App
Single Modal function that runs the full clipping pipeline on A10G GPU.

Architecture:
  - Video is downloaded LOCALLY on user's Mac (yt-dlp has browser cookie access)
  - Video bytes are uploaded to this Modal function
  - All processing (transcribe, analyze, select, render) happens on Modal GPU
  - Rendered clip bytes are returned to the local machine
"""

import modal

app = modal.App("clipped-ai")

# ──────────────────────────────────────
# Container image with all dependencies
# ──────────────────────────────────────
clipping_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        "libsndfile1",
        "fonts-liberation",   # Fallback fonts
    )
    .pip_install(
        # Core ML — versions must match WhisperX requirements
        "torch>=2.8.0",
        "torchaudio>=2.8.0",
        "whisperx @ git+https://github.com/m-bain/whisperX.git",

        # Scene detection
        "scenedetect[opencv]==0.6.4",

        # Audio analysis
        "numpy>=1.24",
        "scipy>=1.11",
        "librosa>=0.10",

        # Face detection
        "ultralytics>=8.0",

        # Captions
        "pysubs2>=1.7",

        # LLM API
        "httpx>=0.25",
    )
    # Bundle pipeline code into the image
    .add_local_dir("pipeline", remote_path="/app/pipeline")
    .add_local_file("config.py", remote_path="/app/config.py")
)


@app.function(
    image=clipping_image,
    timeout=60,
    secrets=[modal.Secret.from_name("cerebras-api-key")],
)
def test_cerebras():
    """Quick test: verify Cerebras API works from inside Modal container."""
    import os
    import httpx

    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    print(f"CEREBRAS_API_KEY set: {bool(api_key)} (len={len(api_key)})")

    if not api_key:
        return {"error": "CEREBRAS_API_KEY not found in environment"}

    try:
        response = httpx.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3.1-8b",
                "messages": [
                    {"role": "user", "content": "Say hello in JSON"},
                ],
                "temperature": 0.1,
                "max_tokens": 50,
                "response_format": {"type": "json_object"},
            },
            timeout=30.0,
        )
        return {
            "status_code": response.status_code,
            "body": response.text[:500],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.function(
    image=clipping_image,
    gpu="A10G",
    timeout=1800,  # 30 min max
    secrets=[modal.Secret.from_name("cerebras-api-key")],
)
def process_video(
    video_bytes: bytes,
    video_title: str = "Untitled",
    max_clips: int = 5,
    min_clip_duration: int = 15,
    max_clip_duration: int = 60,
) -> dict:
    """
    Full pipeline: video bytes → rendered clips with metadata.

    Video is pre-downloaded on the local machine and uploaded as bytes.
    All processing happens in this single container:
    - GPU used for WhisperX transcription
    - CPU used for everything else (FFmpeg, scene detect, etc.)
    """
    import os
    import sys
    import time

    # Add bundled code to path
    sys.path.insert(0, "/app")

    from pipeline import ingest, transcribe, scene_detect, audio_analysis
    from pipeline import clip_selector, reframe, captions, render

    os.makedirs("/tmp/clipped/segments", exist_ok=True)
    os.makedirs("/tmp/clipped/output", exist_ok=True)

    timings = {}

    print("=" * 60)
    print("🎬 ClippedAI Pipeline Starting")
    print("=" * 60)

    # ── Step 1: Save uploaded video + extract audio ──
    t0 = time.time()
    video_info = ingest.save_uploaded_video(video_bytes, title=video_title)
    audio_path = ingest.extract_audio(video_info["video_path"])
    timings["ingest"] = round(time.time() - t0, 1)
    print(f"✅ Ingest ({timings['ingest']}s) — {video_info['title']} "
          f"({video_info['duration']:.0f}s, {video_info['width']}x{video_info['height']})")

    # ── Step 2: Transcribe (GPU) ──
    t0 = time.time()
    transcript = transcribe.transcribe(audio_path)
    timings["transcribe"] = round(time.time() - t0, 1)
    print(f"✅ Transcribed ({timings['transcribe']}s) — "
          f"{len(transcript['segments'])} segments, lang={transcript['language']}")

    # ── Step 3: Scene Detection ──
    t0 = time.time()
    scenes = scene_detect.detect_scenes(video_info["video_path"])
    timings["scene_detect"] = round(time.time() - t0, 1)
    print(f"✅ Scenes ({timings['scene_detect']}s) — {len(scenes)} scenes")

    # ── Step 4: Audio Analysis ──
    t0 = time.time()
    audio = audio_analysis.analyze_audio(audio_path)
    timings["audio_analysis"] = round(time.time() - t0, 1)
    print(f"✅ Audio ({timings['audio_analysis']}s) — {len(audio['peaks'])} peaks")

    # ── Step 5: Clip Selection ──
    t0 = time.time()
    settings = {
        "max_clips": max_clips,
        "min_duration": min_clip_duration,
        "max_duration": max_clip_duration,
    }
    selected = clip_selector.select_clips(
        transcript, scenes, audio, video_info["title"], settings
    )
    timings["clip_selection"] = round(time.time() - t0, 1)
    print(f"✅ Selected ({timings['clip_selection']}s) — {len(selected)} clips")

    if not selected:
        print("⚠️ No clips selected — returning empty result")
        return {
            "video_title": video_info["title"],
            "video_duration": video_info["duration"],
            "clips": [],
            "timings": timings,
        }

    # ── Steps 6-8: Render Each Clip ──
    t0 = time.time()
    clips = []

    for i, clip_info in enumerate(selected):
        print(f"\n🎬 Rendering clip {i+1}/{len(selected)}: \"{clip_info['title']}\"")
        print(f"   {clip_info['start']:.1f}s – {clip_info['end']:.1f}s "
              f"({clip_info['duration']:.0f}s)")

        # Cut raw segment
        seg_path = render.cut_segment(
            video_info["video_path"],
            clip_info["start"],
            clip_info["end"],
            i,
        )

        # Detect faces + compute crop position
        faces = reframe.detect_faces(
            video_info["video_path"],
            clip_info["start"],
            clip_info["end"],
        )
        crop_x = reframe.compute_crop_x(
            faces,
            source_w=video_info["width"],
            source_h=video_info["height"],
        )
        print(f"   👤 {len(faces)} face detections, crop_x={crop_x}")

        # Generate captions
        clip_words = transcribe.get_words_in_range(
            transcript,
            clip_info["start"],
            clip_info["end"],
        )
        ass_path = captions.generate_ass(clip_words, clip_index=i)
        print(f"   📝 {len(clip_words)} words for captions")

        # Final render
        output_path = render.final_render(
            seg_path, crop_x,
            video_info["width"], video_info["height"],
            ass_path, i,
        )

        # Read rendered file bytes
        with open(output_path, "rb") as f:
            rendered_bytes = f.read()

        clips.append({
            "video_bytes": rendered_bytes,
            "title": clip_info["title"],
            "description": clip_info.get("description", ""),
            "hashtags": clip_info.get("hashtags", []),
            "start": clip_info["start"],
            "end": clip_info["end"],
            "duration": clip_info["duration"],
            "virality_score": clip_info.get("virality_score", 0),
            "reasoning": clip_info.get("reasoning", ""),
        })

    timings["render"] = round(time.time() - t0, 1)
    timings["total"] = round(sum(timings.values()), 1)

    print(f"\n{'=' * 60}")
    print(f"✅ Pipeline complete — {len(clips)} clips rendered")
    print(f"⏱️  Total: {timings['total']}s")
    print(f"{'=' * 60}")

    return {
        "video_title": video_info["title"],
        "video_duration": video_info["duration"],
        "clips": clips,
        "timings": timings,
    }
```

---

## `run.py`

*218 lines*

```python
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
    max_duration = 60

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
```

---

## `pipeline/__init__.py`

*1 lines*

```python
# ClippedAI pipeline modules
```

---

## `pipeline/ingest.py`

*189 lines*

```python
"""
Pipeline Step 1 — Ingest
Handles both local video files and remote URLs.
On Modal: receives pre-downloaded video bytes, writes to disk, probes metadata.
Locally: downloads via yt-dlp (has browser cookie access).
"""

import subprocess
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YTDLP_FORMAT, MAX_VIDEO_DURATION, MAX_RESOLUTION


WORK_DIR = "/tmp/clipped"


def save_uploaded_video(video_bytes: bytes, title: str = "Untitled") -> dict:
    """
    Save pre-downloaded video bytes to disk and probe metadata.
    Used when video is downloaded locally and uploaded to Modal.

    Returns:
        {
            "video_path": str,
            "title": str,
            "duration": float,
            "width": int,
            "height": int,
            "fps": float,
        }
    """
    os.makedirs(WORK_DIR, exist_ok=True)
    video_path = os.path.join(WORK_DIR, "source.mp4")

    print(f"  📥 Writing uploaded video ({len(video_bytes) / 1024 / 1024:.1f} MB)...")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # Probe for metadata
    probe = _ffprobe_full(video_path)

    if probe["duration"] > MAX_VIDEO_DURATION:
        raise ValueError(
            f"Video too long: {probe['duration']/3600:.1f}h "
            f"(max {MAX_VIDEO_DURATION/3600:.0f}h)"
        )

    return {
        "video_path": video_path,
        "title": title,
        "duration": probe["duration"],
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["fps"],
    }


def download_video_locally(url: str) -> tuple:
    """
    Download video on the LOCAL machine using yt-dlp.
    This works because the local machine has browser cookie access.

    Returns: (video_bytes, title)
    """
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="clipped_")
    video_path = os.path.join(tmp_dir, "source.mp4")

    # Get metadata first
    meta_cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        url,
    ]
    print(f"  📥 Fetching video metadata...")
    result = subprocess.run(meta_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {result.stderr[:500]}")

    meta = json.loads(result.stdout)
    duration = meta.get("duration", 0)
    title = meta.get("title", "Untitled")

    if duration > MAX_VIDEO_DURATION:
        raise ValueError(
            f"Video too long: {duration/3600:.1f}h "
            f"(max {MAX_VIDEO_DURATION/3600:.0f}h)"
        )

    print(f"  📥 Downloading: {title} ({duration:.0f}s)")

    # Download video
    dl_cmd = [
        "yt-dlp",
        "-f", YTDLP_FORMAT,
        "--merge-output-format", "mp4",
        "-o", video_path,
        "--no-playlist",
        "--retries", "3",
        "--no-check-certificates",
        url,
    ]
    result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr[:500]}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Downloaded file not found at {video_path}")

    # Read bytes
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


def extract_audio(video_path: str) -> str:
    """
    Extract mono 16kHz WAV audio for WhisperX.

    Returns: path to audio.wav
    """
    audio_path = os.path.join(WORK_DIR, "audio.wav")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz
        "-ac", "1",               # Mono
        "-y",                     # Overwrite
        audio_path,
    ]
    print(f"  🔊 Extracting audio...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    return audio_path


def _ffprobe_full(video_path: str) -> dict:
    """Get video dimensions, fps, and duration via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "v:0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]
    fmt = data.get("format", {})

    # Parse fps from r_frame_rate (e.g., "30000/1001")
    fps_parts = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    # Duration from format (more reliable) or stream
    duration = float(fmt.get("duration", stream.get("duration", 0)))

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(fps, 2),
        "duration": round(duration, 2),
    }
```

---

## `pipeline/transcribe.py`

*121 lines*

```python
"""
Pipeline Step 2 — Transcribe (GPU)
WhisperX large-v2 with word-level timestamp alignment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WHISPERX_MODEL, WHISPERX_BATCH_SIZE, WHISPERX_COMPUTE_TYPE


def transcribe(audio_path: str) -> dict:
    """
    Transcribe audio using WhisperX on GPU.

    Returns:
        {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.52,
                    "text": "Hey guys welcome back",
                    "words": [
                        {"word": "Hey", "start": 0.0, "end": 0.28},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  🎙️ Loading WhisperX {WHISPERX_MODEL} on {device}...")

    # Load model
    model = whisperx.load_model(
        WHISPERX_MODEL,
        device,
        compute_type=WHISPERX_COMPUTE_TYPE,
    )

    # Transcribe
    print(f"  🎙️ Transcribing...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=WHISPERX_BATCH_SIZE)

    detected_language = result.get("language", "en")
    print(f"  🎙️ Detected language: {detected_language}")

    # Align for word-level timestamps
    print(f"  🎙️ Aligning word timestamps...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Clean up GPU memory
    del model, align_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Build clean output
    segments = []
    for seg in aligned["segments"]:
        words = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })

        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
            "words": words,
        })

    print(f"  🎙️ Transcription complete: {len(segments)} segments")
    return {
        "language": detected_language,
        "segments": segments,
    }


def get_words_in_range(transcript: dict, start: float, end: float) -> list:
    """
    Extract word-level timestamps for a specific time range.
    Used to generate captions for a clip.

    Returns: [{"word": "Hey", "start": 0.0, "end": 0.28}, ...]
    """
    words = []
    for seg in transcript["segments"]:
        # Skip segments entirely outside range
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in seg.get("words", []):
            if w["start"] >= start and w["end"] <= end:
                # Adjust timestamps relative to clip start
                words.append({
                    "word": w["word"],
                    "start": round(w["start"] - start, 3),
                    "end": round(w["end"] - start, 3),
                })
    return words
```

---

## `pipeline/scene_detect.py`

*53 lines*

```python
"""
Pipeline Step 3 — Scene Detection
PySceneDetect ContentDetector for scene boundary timestamps.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCENE_THRESHOLD, SCENE_MIN_SCENE_LEN


def detect_scenes(video_path: str) -> list:
    """
    Detect scene boundaries in the video.

    Returns:
        [
            {"start": 0.0, "end": 15.3, "duration": 15.3},
            {"start": 15.3, "end": 42.7, "duration": 27.4},
            ...
        ]
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    print(f"  🎬 Detecting scenes...")

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=SCENE_THRESHOLD,
            min_scene_len=SCENE_MIN_SCENE_LEN * video.frame_rate,
        )
    )

    # Process video (downscale for speed)
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for start_time, end_time in scene_list:
        start_s = start_time.get_seconds()
        end_s = end_time.get_seconds()
        scenes.append({
            "start": round(start_s, 2),
            "end": round(end_s, 2),
            "duration": round(end_s - start_s, 2),
        })

    print(f"  🎬 Found {len(scenes)} scenes")
    return scenes
```

---

## `pipeline/audio_analysis.py`

*88 lines*

```python
"""
Pipeline Step 4 — Audio Analysis
RMS energy curves + peak detection for finding high-energy moments.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUDIO_FRAME_LENGTH, AUDIO_HOP_LENGTH, PEAK_STD_MULTIPLIER


def analyze_audio(audio_path: str) -> dict:
    """
    Analyze audio for energy levels and peaks.

    Returns:
        {
            "energy_curve": [0.02, 0.03, ...],  # RMS per 0.5s frame
            "peaks": [
                {"time": 124.5, "energy": 0.92},
                ...
            ],
            "avg_energy": 0.12,
            "max_energy": 0.95,
        }
    """
    import numpy as np
    import librosa

    print(f"  📊 Analyzing audio energy...")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Compute RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=AUDIO_FRAME_LENGTH,
        hop_length=AUDIO_HOP_LENGTH,
    )[0]

    # Normalize to 0-1
    max_rms = float(np.max(rms)) if np.max(rms) > 0 else 1.0
    energy_curve = (rms / max_rms).tolist()

    # Compute stats
    avg_energy = float(np.mean(energy_curve))
    peak_threshold = avg_energy + PEAK_STD_MULTIPLIER * float(np.std(energy_curve))

    # Find peaks (frames where energy significantly exceeds average)
    peaks = []
    frame_duration = AUDIO_HOP_LENGTH / sr  # 0.5s per frame
    for i, energy in enumerate(energy_curve):
        if energy > peak_threshold:
            peaks.append({
                "time": round(i * frame_duration, 2),
                "energy": round(energy, 4),
            })

    # Merge nearby peaks (within 2 seconds)
    merged_peaks = _merge_nearby_peaks(peaks, min_gap=2.0)

    print(f"  📊 Found {len(merged_peaks)} energy peaks (threshold: {peak_threshold:.3f})")
    return {
        "energy_curve": [round(e, 4) for e in energy_curve],
        "peaks": merged_peaks,
        "avg_energy": round(avg_energy, 4),
        "max_energy": round(float(max(energy_curve)) if energy_curve else 0, 4),
        "frame_duration": round(frame_duration, 4),
    }


def _merge_nearby_peaks(peaks: list, min_gap: float = 2.0) -> list:
    """Merge peaks that are within min_gap seconds of each other."""
    if not peaks:
        return []

    merged = [peaks[0].copy()]
    for peak in peaks[1:]:
        if peak["time"] - merged[-1]["time"] < min_gap:
            # Keep the higher energy peak
            if peak["energy"] > merged[-1]["energy"]:
                merged[-1] = peak.copy()
        else:
            merged.append(peak.copy())

    return merged
```

---

## `pipeline/clip_selector.py`

*457 lines*

```python
"""
Pipeline Step 5 — Clip Selection
Candidate generation via sliding window + Cerebras LLM ranking.
This is the CORE IP — clip quality lives or dies here.
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WINDOW_SIZES, WINDOW_STEP_RATIO, CANDIDATE_MULTIPLIER, OVERLAP_THRESHOLD,
    ENERGY_WEIGHT, SCENE_WEIGHT, KEYWORD_WEIGHT, VIRAL_KEYWORDS,
    CEREBRAS_MODEL, CEREBRAS_API_URL, CEREBRAS_TEMPERATURE, CEREBRAS_MAX_TOKENS,
    LLM_SYSTEM_PROMPT, RD_MODE,
)


def select_clips(
    transcript: dict,
    scenes: list,
    audio: dict,
    video_title: str,
    settings: dict,
) -> list:
    """
    Generate candidates from signals, then rank with LLM.

    Returns: List of selected clips with titles, scores, and metadata.
    """
    max_clips = settings.get("max_clips", 5)
    min_duration = settings.get("min_duration", 15)
    max_duration = settings.get("max_duration", 60)

    # Phase A: Generate candidates
    candidates = generate_candidates(
        transcript, scenes, audio, min_duration, max_duration, max_clips
    )

    if not candidates:
        print("  ⚠️ No candidates generated — video may be too short")
        return []

    print(f"  🧠 Generated {len(candidates)} candidates, sending to LLM...")

    # Phase B: Rank with LLM
    ranked = rank_with_llm(candidates, video_title, max_clips)

    return ranked


def generate_candidates(
    transcript: dict,
    scenes: list,
    audio: dict,
    min_duration: float,
    max_duration: float,
    max_clips: int,
) -> list:
    """
    Generate candidate clip regions using TWO strategies:
    1. Sliding windows (coverage-based)
    2. Event-centered windows (around peaks, scene changes)

    Scores each window with energy, scene density, and keyword signals.
    """
    segments = transcript["segments"]
    if not segments:
        return []

    all_windows = []
    target_count = max_clips * CANDIDATE_MULTIPLIER

    # Start from first speech, not t=0 (avoids awkward silence openings)
    first_speech_start = segments[0]["start"]
    video_duration = segments[-1]["end"]

    # ── Strategy 1: Sliding windows (coverage) ──
    for window_size in WINDOW_SIZES:
        step = window_size * WINDOW_STEP_RATIO
        t = first_speech_start
        while t + min_duration <= video_duration:
            window = _make_window(segments, audio, scenes, t, t + window_size,
                                  min_duration, max_duration)
            if window:
                all_windows.append(window)
            t += step

    # ── Strategy 2: Event-centered windows ──
    # Center windows around energy peaks
    for peak in audio.get("peaks", []):
        center = peak["time"]
        for radius in [10, 15]:  # Try 20s and 30s clips centered on peak
            win_start = max(first_speech_start, center - radius)
            win_end = min(video_duration, center + radius)
            window = _make_window(segments, audio, scenes, win_start, win_end,
                                  min_duration, max_duration)
            if window:
                all_windows.append(window)

    # Center windows around scene boundaries
    for scene in scenes:
        scene_time = scene["start"]
        # Clip starting at scene change (hook potential)
        for length in [15, 25]:
            win_start = scene_time
            win_end = min(video_duration, scene_time + length)
            window = _make_window(segments, audio, scenes, win_start, win_end,
                                  min_duration, max_duration)
            if window:
                all_windows.append(window)

    # Deduplicate overlapping windows
    all_windows.sort(key=lambda w: w["composite_score"], reverse=True)
    deduped = _deduplicate(all_windows)

    # Return top N
    result = deduped[:target_count]
    for i, c in enumerate(result):
        c["id"] = i

    print(f"    (sliding: {len(all_windows)} raw, {len(deduped)} deduped, {len(result)} kept)")
    return result


def _make_window(segments, audio, scenes, start, end, min_dur, max_dur):
    """Score and validate a single candidate window. Returns dict or None."""
    actual_duration = end - start
    if actual_duration < min_dur:
        return None
    if actual_duration > max_dur:
        end = start + max_dur

    # Snap to sentence boundaries
    start_snapped, end_snapped = _snap_to_sentences(segments, start, end)
    if end_snapped - start_snapped < min_dur:
        return None

    text = _get_text_in_range(segments, start_snapped, end_snapped)
    if len(text.split()) < 10:
        return None

    energy_score = _score_energy(audio, start_snapped, end_snapped)
    scene_score = _score_scenes(scenes, start_snapped, end_snapped)
    keyword_score = _score_keywords(text)

    composite = (
        energy_score * ENERGY_WEIGHT +
        scene_score * SCENE_WEIGHT +
        keyword_score * KEYWORD_WEIGHT
    )

    # Extract hook text (first ~3 seconds of transcript)
    hook_text = _get_text_in_range(segments, start_snapped, start_snapped + 3.0)

    return {
        "start": round(start_snapped, 2),
        "end": round(end_snapped, 2),
        "duration": round(end_snapped - start_snapped, 2),
        "transcript_text": text,
        "hook_text": hook_text.strip() if hook_text else "",
        "energy_score": round(energy_score, 4),
        "scene_score": round(scene_score, 4),
        "keyword_score": round(keyword_score, 4),
        "composite_score": round(composite, 4),
    }


def rank_with_llm(candidates: list, video_title: str, max_clips: int) -> list:
    """
    Send candidates to Cerebras LLM for viral ranking.
    In RD_MODE: raises exception on failure (no silent fallback).
    In production: falls back to composite scoring.
    """
    import httpx

    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    if not api_key:
        msg = "No CEREBRAS_API_KEY — cannot rank clips"
        if RD_MODE:
            raise RuntimeError(f"🛑 LLM FAILED: {msg}")
        print(f"  ⚠️ {msg}")
        return _fallback_ranking(candidates, max_clips, msg)

    # Build user prompt — structurally isolate hook text from full transcript
    candidate_blocks = []
    for c in candidates:
        block = (
            f"---\n"
            f"CANDIDATE {c['id']}\n"
            f"Time: {c['start']:.1f}s – {c['end']:.1f}s ({c['duration']:.0f}s)\n"
            f"Signals: energy={c['energy_score']:.2f}, "
            f"scenes={c['scene_score']:.2f}, keywords={c['keyword_score']:.2f}\n"
            f"HOOK (first 3 seconds): \"{c.get('hook_text', '')}\"\n"
            f"FULL TRANSCRIPT:\n\"{c['transcript_text'][:1500]}\"\n"
            f"---"
        )
        candidate_blocks.append(block)

    user_prompt = (
        f'Video title: "{video_title}"\n\n'
        f"Here are {len(candidates)} candidate clips. Rank them by virality potential.\n\n"
        f"{''.join(candidate_blocks)}\n\n"
        f"CRITICAL: Evaluate the FIRST 3 SECONDS of each candidate's transcript. "
        f"If the opening is weak, generic, or lacks a hook, mark keep=false.\n\n"
        f"Return a JSON array with exactly {len(candidates)} objects, one per candidate:\n"
        f"[\n"
        f'  {{\n'
        f'    "candidate_id": <int>,\n'
        f'    "rank": <int 1..N>,\n'
        f'    "virality_score": <float 0-1>,\n'
        f'    "keep": <bool>,\n'
        f'    "title": "<string, 5-10 words>",\n'
        f'    "description": "<string, 1-2 sentences>",\n'
        f'    "hashtags": ["<string>", ...],\n'
        f'    "hook_strength": "<weak|medium|strong>",\n'
        f'    "reasoning": "<string, why this is/isn\'t engaging>"\n'
        f"  }}\n"
        f"]\n\n"
        f"Return ONLY the JSON array. No other text."
    )

    # Call Cerebras API
    print(f"  🔗 Calling Cerebras API (model={CEREBRAS_MODEL}, {len(candidates)} candidates)...")
    try:
        response = httpx.post(
            CEREBRAS_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": CEREBRAS_MODEL,
                "messages": [
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": CEREBRAS_TEMPERATURE,
                "max_tokens": CEREBRAS_MAX_TOKENS,
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            err = f"HTTP {response.status_code}: {response.text[:300]}"
            print(f"  ❌ Cerebras: {err}")
            if RD_MODE:
                raise RuntimeError(f"🛑 LLM FAILED: {err}")
            return _fallback_ranking(candidates, max_clips, err)
    except httpx.TimeoutException as e:
        err = f"Timeout after 120s: {e}"
        print(f"  ❌ Cerebras: {err}")
        if RD_MODE:
            raise RuntimeError(f"🛑 LLM FAILED: {err}")
        return _fallback_ranking(candidates, max_clips, err)

    # Parse response
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"  📝 LLM response ({len(content)} chars)")

        # Try parsing as direct JSON
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                rankings = parsed
            elif isinstance(parsed, dict):
                # LLM may wrap array in an object
                for key in ("clips", "rankings", "candidates", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        rankings = parsed[key]
                        break
                else:
                    for v in parsed.values():
                        if isinstance(v, list) and len(v) > 0:
                            rankings = v
                            break
                    else:
                        raise ValueError(f"No array in JSON object. Keys: {list(parsed.keys())}")
            else:
                raise ValueError(f"Unexpected JSON type: {type(parsed)}")
        except json.JSONDecodeError:
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON array in response")
            rankings = json.loads(json_match.group())

        print(f"  ✅ Parsed {len(rankings)} rankings from LLM")
    except Exception as e:
        err = f"{type(e).__name__}: {e} | content: {content[:200] if 'content' in dir() else 'N/A'}"
        print(f"  ❌ Parse failed: {err}")
        if RD_MODE:
            raise RuntimeError(f"🛑 LLM PARSE FAILED: {err}")
        return _fallback_ranking(candidates, max_clips, err)

    # Merge LLM rankings with candidate data
    ranked_map = {r["candidate_id"]: r for r in rankings}
    results = []
    for c in candidates:
        r = ranked_map.get(c["id"])
        if r and r.get("keep", False):
            results.append({
                "start": c["start"],
                "end": c["end"],
                "duration": c["duration"],
                "transcript_text": c["transcript_text"],
                "virality_score": r.get("virality_score", 0),
                "title": r.get("title", f"Clip {c['id']}"),
                "description": r.get("description", ""),
                "hashtags": r.get("hashtags", []),
                "hook_strength": r.get("hook_strength", "unknown"),
                "reasoning": r.get("reasoning", ""),
                "energy_score": c["energy_score"],
                "scene_score": c["scene_score"],
                "keyword_score": c["keyword_score"],
            })

    # Sort by virality, take top max_clips
    results.sort(key=lambda x: x["virality_score"], reverse=True)
    final = results[:max_clips]
    # Sort final by timestamp for natural order
    final.sort(key=lambda x: x["start"])

    if not final and RD_MODE:
        raise RuntimeError(
            f"🛑 LLM marked ALL candidates as keep=false. "
            f"Rankings: {json.dumps(rankings, indent=2)[:1000]}"
        )

    print(f"  🧠 LLM selected {len(final)} clips (from {len(results)} kept)")
    return final


def _fallback_ranking(candidates: list, max_clips: int, reason: str = "unknown") -> list:
    """Use composite scores when LLM is unavailable. Only used when RD_MODE=False."""
    print(f"  ⚠️ Fallback ranking — reason: {reason}")
    sorted_c = sorted(candidates, key=lambda x: x["composite_score"], reverse=True)
    results = []
    for c in sorted_c[:max_clips]:
        results.append({
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "transcript_text": c["transcript_text"],
            "virality_score": c["composite_score"],
            "title": f"Highlight at {c['start']:.0f}s",
            "description": c["transcript_text"][:100],
            "hashtags": [],
            "hook_strength": "unknown",
            "reasoning": f"Fallback (composite score). LLM error: {reason}",
            "energy_score": c["energy_score"],
            "scene_score": c["scene_score"],
            "keyword_score": c["keyword_score"],
        })
    results.sort(key=lambda x: x["start"])
    return results


# ──────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────

def _snap_to_sentences(segments: list, start: float, end: float) -> tuple:
    """
    Snap start/end to nearest segment boundaries.
    Tight snapping: only snap forward for start (max 1.5s),
    only snap backward for end (max 1.5s).
    """
    best_start = start
    best_end = end

    # Snap start FORWARD to next segment start (don't snap backward — avoids chopping hooks)
    for seg in segments:
        if seg["start"] >= start and seg["start"] <= start + 1.5:
            best_start = seg["start"]
            break

    # Snap end BACKWARD to closest segment end (don't extend past window)
    for seg in reversed(segments):
        if seg["end"] <= end and seg["end"] >= end - 1.5:
            best_end = seg["end"]
            break

    return best_start, best_end


def _get_text_in_range(segments: list, start: float, end: float) -> str:
    """Get concatenated transcript text within a time range."""
    texts = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        texts.append(seg["text"])
    return " ".join(texts)


def _score_energy(audio: dict, start: float, end: float) -> float:
    """
    Score energy peaks per-window-duration (not global).
    Higher density of peaks within the window = higher score.
    """
    peaks_in_window = [p for p in audio["peaks"] if start <= p["time"] <= end]
    duration = end - start
    if duration <= 0:
        return 0.0

    # Peaks per 10 seconds — normalized so 2+ peaks/10s = 1.0
    peak_density = len(peaks_in_window) / (duration / 10.0)
    return min(peak_density / 2.0, 1.0)


def _score_scenes(scenes: list, start: float, end: float) -> float:
    """Score based on scene change density in the window."""
    changes = 0
    for scene in scenes:
        if start < scene["start"] < end:
            changes += 1

    duration_min = (end - start) / 60.0
    if duration_min == 0:
        return 0.0
    density = changes / duration_min
    # Normalize: 4+ changes per minute = 1.0
    return min(density / 4.0, 1.0)


def _score_keywords(text: str) -> float:
    """Score based on viral keyword density."""
    text_lower = text.lower()
    word_count = len(text_lower.split())
    if word_count == 0:
        return 0.0

    matches = sum(1 for kw in VIRAL_KEYWORDS if kw in text_lower)
    # Normalize: 5+ matches in a segment = 1.0
    return min(matches / 5.0, 1.0)


def _deduplicate(windows: list) -> list:
    """Remove windows that overlap >50% with a higher-scored window."""
    kept = []
    for w in windows:
        overlaps = False
        for k in kept:
            overlap_start = max(w["start"], k["start"])
            overlap_end = min(w["end"], k["end"])
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                min_duration = min(w["duration"], k["duration"])
                if min_duration > 0 and (overlap_duration / min_duration) > OVERLAP_THRESHOLD:
                    overlaps = True
                    break
        if not overlaps:
            kept.append(w)
    return kept
```

---

## `pipeline/reframe.py`

*138 lines*

```python
"""
Pipeline Step 6 — Reframe to 9:16
Face detection + smoothed crop for vertical video.
"""

import os
import sys
import subprocess
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FACE_DETECTION_FPS, SMOOTHING_ALPHA, TARGET_ASPECT_RATIO


def detect_faces(video_path: str, start: float, end: float) -> list:
    """
    Detect faces in the clip region at FACE_DETECTION_FPS.

    Returns:
        [
            {"time": 45.2, "x_center": 520, "y_center": 310, "w": 210, "h": 280},
            ...
        ]
    """
    from ultralytics import YOLO

    # Extract frames at low FPS for face detection
    frames_dir = "/tmp/clipped/faces"
    os.makedirs(frames_dir, exist_ok=True)

    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-t", str(duration),
        "-i", video_path,
        "-vf", f"fps={FACE_DETECTION_FPS}",
        "-q:v", "3",
        "-y",
        os.path.join(frames_dir, "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # Load YOLOv8 once (cached after first call)
    global _yolo_model
    if "_yolo_model" not in globals() or _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    model = _yolo_model

    # Detect faces in each frame
    faces = []
    frame_files = sorted([
        f for f in os.listdir(frames_dir) if f.startswith("frame_")
    ])

    for i, fname in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, fname)
        frame_time = start + (i / FACE_DETECTION_FPS)

        results = model(frame_path, verbose=False, conf=0.3)

        # Get person detections (class 0)
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Find largest person detection (most prominent)
                person_boxes = []
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        person_boxes.append({
                            "x_center": (x1 + x2) / 2,
                            "y_center": (y1 + y2) / 2,
                            "w": x2 - x1,
                            "h": y2 - y1,
                            "area": area,
                        })

                if person_boxes:
                    # Use the largest person
                    best = max(person_boxes, key=lambda b: b["area"])
                    faces.append({
                        "time": round(frame_time, 2),
                        "x_center": round(best["x_center"]),
                        "y_center": round(best["y_center"]),
                        "w": round(best["w"]),
                        "h": round(best["h"]),
                    })

    # Cleanup frames
    for f in frame_files:
        try:
            os.remove(os.path.join(frames_dir, f))
        except OSError:
            pass

    return faces


def compute_crop_x(faces: list, source_w: int, source_h: int) -> int:
    """
    Compute the horizontal crop position for 9:16 reframing.
    Uses exponential smoothing on face positions.

    Returns: x offset for FFmpeg crop filter
    """
    # Target crop dimensions
    crop_w = int(source_h * TARGET_ASPECT_RATIO)

    # Clamp crop width to source
    if crop_w >= source_w:
        return 0  # Source is already narrow enough

    if not faces:
        # No faces detected — center crop
        return (source_w - crop_w) // 2

    # Extract x_center positions
    positions = [f["x_center"] for f in faces]

    # Apply exponential smoothing
    smoothed = [positions[0]]
    for i in range(1, len(positions)):
        s = SMOOTHING_ALPHA * positions[i] + (1 - SMOOTHING_ALPHA) * smoothed[-1]
        smoothed.append(s)

    # Use the average of smoothed positions
    avg_x = sum(smoothed) / len(smoothed)

    # Compute crop offset (center the crop on the face)
    crop_x = int(avg_x - crop_w / 2)

    # Clamp to valid range
    crop_x = max(0, min(crop_x, source_w - crop_w))

    return crop_x
```

---

## `pipeline/captions.py`

*100 lines*

```python
"""
Pipeline Step 7 — Captions
Generate .ass subtitle files with strict 1-to-1 word-level timing.

Each word appears ONLY when spoken and disappears IMMEDIATELY when it ends.
No grouping. No merging. No lookahead. One word at a time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAPTION_FONT, CAPTION_FONT_SIZE, CAPTION_HIGHLIGHT_COLOR,
    CAPTION_DEFAULT_COLOR, CAPTION_OUTLINE_COLOR, CAPTION_OUTLINE_WIDTH,
    CAPTION_MARGIN_BOTTOM,
    OUTPUT_WIDTH, OUTPUT_HEIGHT,
)

WORK_DIR = "/tmp/clipped"


def generate_ass(words: list, clip_index: int) -> str:
    """
    Generate an .ass subtitle file with strict word-level timing.

    One word on screen at a time. Appears when spoken. Disappears when
    the word ends. No previous words, no upcoming words.

    Args:
        words: [{"word": "Hey", "start": 0.0, "end": 0.28}, ...]
               Timestamps are relative to clip start (0-based).
        clip_index: For unique filename.

    Returns: Path to .ass file
    """
    if not words:
        ass_path = os.path.join(WORK_DIR, f"clip_{clip_index:02d}.ass")
        with open(ass_path, "w") as f:
            f.write(_ass_header())
        return ass_path

    events = []
    for word in words:
        t_start = word["start"]
        t_end = word["end"]

        # Prevent 0-duration flickering
        if t_end - t_start < 0.05:
            t_end = t_start + 0.05

        # Only the current word is shown — bold + highlighted
        text = (
            f"{{\\b1\\c&H{CAPTION_HIGHLIGHT_COLOR}&}}"
            f"{word['word']}"
        )

        events.append({"start": t_start, "end": t_end, "text": text})

    # Write .ass file
    ass_path = os.path.join(WORK_DIR, f"clip_{clip_index:02d}.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(_ass_header())
        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for event in events:
            start_str = _format_ass_time(event["start"])
            end_str = _format_ass_time(event["end"])
            f.write(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{event['text']}\n"
            )

    return ass_path


def _ass_header() -> str:
    """Generate ASS file header with style definition."""
    return f"""[Script Info]
Title: ClippedAI Captions
ScriptType: v4.00+
PlayResX: {OUTPUT_WIDTH}
PlayResY: {OUTPUT_HEIGHT}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{CAPTION_FONT},{CAPTION_FONT_SIZE},&H00{CAPTION_DEFAULT_COLOR},&H000000FF,&H00{CAPTION_OUTLINE_COLOR},&H80000000,-1,0,0,0,100,100,0,0,1,{CAPTION_OUTLINE_WIDTH},0,2,40,40,{CAPTION_MARGIN_BOTTOM},1
"""


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time format: H:MM:SS.CC"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
```

---

## `pipeline/render.py`

*125 lines*

```python
"""
Pipeline Step 8 — Final Render
FFmpeg segment cutting + vertical reframe + caption burn + audio normalize.
"""

import subprocess
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, VIDEO_CODEC, VIDEO_CRF, VIDEO_PRESET,
    AUDIO_CODEC, AUDIO_BITRATE, AUDIO_SAMPLE_RATE, LOUDNORM_TARGET,
    TARGET_ASPECT_RATIO,
)

WORK_DIR = "/tmp/clipped"
SEGMENTS_DIR = os.path.join(WORK_DIR, "segments")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")


def cut_segment(video_path: str, start: float, end: float, clip_index: int) -> str:
    """
    Cut a raw segment from the source video without re-encoding.

    Returns: path to raw segment file
    """
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    segment_path = os.path.join(SEGMENTS_DIR, f"clip_{clip_index:02d}_raw.mp4")

    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-y",
        segment_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg segment cut failed: {result.stderr}")

    return segment_path


def final_render(
    segment_path: str,
    crop_x: int,
    source_w: int,
    source_h: int,
    ass_path: str,
    clip_index: int,
) -> str:
    """
    Single-pass FFmpeg render:
    1. Crop to 9:16 region (face-centered)
    2. Scale to 1080x1920
    3. Burn captions (.ass)
    4. Normalize audio (loudnorm -14 LUFS)
    5. Encode H.264 + AAC

    Returns: path to final rendered clip
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"clip_{clip_index:02d}.mp4")

    # Compute crop dimensions
    crop_w = int(source_h * TARGET_ASPECT_RATIO)
    if crop_w > source_w:
        # Source is narrower than 9:16 — use full width, crop height
        crop_w = source_w
        crop_h = int(source_w / TARGET_ASPECT_RATIO)
        crop_y = max(0, (source_h - crop_h) // 2)
        crop_filter = f"crop={crop_w}:{crop_h}:0:{crop_y}"
    else:
        crop_filter = f"crop={crop_w}:{source_h}:{crop_x}:0"

    # Build video filter chain
    vf_parts = [
        crop_filter,
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos",
    ]

    # Add captions if .ass file has content
    if ass_path and os.path.exists(ass_path):
        # Escape special characters in path for FFmpeg filter
        ass_escaped = ass_path.replace("\\", "/").replace(":", "\\\\:").replace("'", "\\\\'")
        vf_parts.append(f"ass='{ass_escaped}'")

    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-i", segment_path,
        "-vf", vf,
        "-af", f"loudnorm=I={LOUDNORM_TARGET}:TP=-1.5:LRA=11",
        "-c:v", VIDEO_CODEC,
        "-preset", VIDEO_PRESET,
        "-crf", str(VIDEO_CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-movflags", "+faststart",
        "-y",
        output_path,
    ]

    print(f"    🔧 Rendering clip {clip_index + 1}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg render failed for clip {clip_index}: {result.stderr[-500:]}"
        )

    # Verify output exists and has content
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Rendered file not found: {output_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    ✅ Clip {clip_index + 1} rendered ({size_mb:.1f} MB)")

    return output_path
```

---

## `requirements.txt`

*1 lines*

```text
modal>=0.68.0
```

---

## `.env.example`

*2 lines*

```bash
# Modal secrets (set via `modal secret create`)
CEREBRAS_API_KEY=csk-your-key-here
```

---

## `.gitignore`

*31 lines*

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
*.egg

# Environment
.env
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Output
output/

# Modal
.modal/

# Logs
*.log
```

---

**Total: 15 files, 1880 lines**