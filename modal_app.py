"""
ClippedAI — Modal App

Architecture:
  - Video is downloaded LOCALLY on user's Mac (yt-dlp has browser cookies)
  - Video bytes are uploaded to the CPU orchestrator
  - Transcription runs on a SEPARATE A10G GPU function (pay only for GPU time)
  - All other processing (scene detect, analysis, render) runs on CPU
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
    # Bake YOLO weights into image (avoids cold-start download per container)
    .run_commands(
        "mkdir -p /app && python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt', "
        "'/app/yolov8n.pt')\""
    )
    # Bundle pipeline code into the image
    .add_local_dir("pipeline", remote_path="/app/pipeline")
    .add_local_file("config.py", remote_path="/app/config.py")
)


# ──────────────────────────────────────
# GPU-only: Transcription worker
# ──────────────────────────────────────
@app.function(
    image=clipping_image,
    gpu="A10G",
    timeout=900,  # 15 min max for transcription
)
def transcribe_gpu(audio_bytes: bytes) -> dict:
    """
    Dedicated GPU function for WhisperX transcription.
    A10G spins up, transcribes, returns result, spins down.
    No GPU billing during scene detect, rendering, etc.
    """
    import sys
    sys.path.insert(0, "/app")
    from pipeline import transcribe

    # Write audio to temp file
    audio_path = "/tmp/clipped/audio.wav"
    import os
    os.makedirs("/tmp/clipped", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    return transcribe.transcribe(audio_path)


# ──────────────────────────────────────
# Debug: Test Cerebras API from container
# ──────────────────────────────────────
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
                "model": "gpt-oss-120b",
                "messages": [
                    {"role": "user", "content": "Say hello in JSON"},
                ],
                "temperature": 0.1,
                "max_tokens": 50,
            },
            timeout=30.0,
        )
        return {
            "status_code": response.status_code,
            "body": response.text[:500],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ──────────────────────────────────────
# CPU orchestrator: Full pipeline
# ──────────────────────────────────────
@app.function(
    image=clipping_image,
    timeout=1800,  # 30 min max
    memory=4096,   # 4GB RAM for large video files
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

    GPU is used ONLY for transcription (via transcribe_gpu.remote()).
    Everything else runs on this cheap CPU container.
    """
    import os
    import sys
    import time

    # Add bundled code to path
    sys.path.insert(0, "/app")

    from pipeline import ingest, scene_detect, audio_analysis
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

    # ── Step 2: Transcribe (GPU — separate A10G function) ──
    t0 = time.time()
    with open(audio_path, "rb") as f:
        audio_bytes_data = f.read()
    print("📡 Sending audio to GPU for transcription...")
    transcript = transcribe_gpu.remote(audio_bytes_data)
    timings["transcribe"] = round(time.time() - t0, 1)
    print(f"✅ Transcribed ({timings['transcribe']}s) — "
          f"{len(transcript['segments'])} segments, lang={transcript['language']}")

    # ── Step 3: Scene Detection (CPU) ──
    t0 = time.time()
    scenes = scene_detect.detect_scenes(video_info["video_path"])
    timings["scene_detect"] = round(time.time() - t0, 1)
    print(f"✅ Scenes ({timings['scene_detect']}s) — {len(scenes)} scenes")

    # ── Step 4: Audio Analysis (CPU) ──
    t0 = time.time()
    audio = audio_analysis.analyze_audio(audio_path)
    timings["audio_analysis"] = round(time.time() - t0, 1)
    print(f"✅ Audio ({timings['audio_analysis']}s) — {len(audio['peaks'])} peaks")

    # ── Step 5: Clip Selection + LLM Ranking (CPU + Cerebras API) ──
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

    # ── Steps 6-8: Render Each Clip (CPU + FFmpeg) ──
    t0 = time.time()
    clips = []

    # Import transcribe here for word extraction (no GPU needed for this)
    from pipeline import transcribe as transcribe_mod

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

        # Detect faces + compute crop position (YOLO uses pre-baked weights)
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
        clip_words = transcribe_mod.get_words_in_range(
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
            "hook_strength": clip_info.get("hook_strength", "unknown"),
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
