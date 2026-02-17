# ClippedAI — Core Clipping Pipeline Implementation Plan

> **Objective**: YouTube URL in → captioned short-form clips out. Everything runs on **Modal** (GPU cloud). Your Mac just invokes and downloads results to `output/`.
>
> Nothing else gets built until clip quality is 100% satisfactory.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Environment & Prerequisites](#3-environment--prerequisites)
4. [Modal App Definition](#4-modal-app-definition)
5. [Pipeline Step 1 — Ingest](#5-pipeline-step-1--ingest)
6. [Pipeline Step 2 — Transcribe (GPU)](#6-pipeline-step-2--transcribe-gpu)
7. [Pipeline Step 3 — Scene Detection](#7-pipeline-step-3--scene-detection)
8. [Pipeline Step 4 — Audio Analysis](#8-pipeline-step-4--audio-analysis)
9. [Pipeline Step 5 — Clip Selection (LLM)](#9-pipeline-step-5--clip-selection-llm)
10. [Pipeline Step 6 — Reframe to 9:16](#10-pipeline-step-6--reframe-to-916)
11. [Pipeline Step 7 — Captions](#11-pipeline-step-7--captions)
12. [Pipeline Step 8 — Final Render](#12-pipeline-step-8--final-render)
13. [Orchestrator Function](#13-orchestrator-function)
14. [Local CLI](#14-local-cli)
15. [Configuration & Constants](#15-configuration--constants)
16. [Verification Plan](#16-verification-plan)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  LOCAL MAC                                                          │
│                                                                     │
│  run.py (CLI)                                                       │
│    └─ modal.Function.from_name("clipped-ai", "process_video")       │
│         │                                                           │
│         ▼  (sends YouTube URL + settings over HTTPS)                │
│                                                                     │
│  output/                                                            │
│    ├── clip_01_amazing_moment.mp4                                   │
│    ├── clip_02_funny_reaction.mp4                                   │
│    ├── clip_03_key_insight.mp4                                      │
│    └── metadata.json                                                │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MODAL (A10G GPU Container)                                         │
│                                                                     │
│  process_video(url, max_clips, settings)                            │
│    │                                                                │
│    ├─ 1. ingest.download_video()          yt-dlp → /tmp/video.mp4  │
│    ├─ 2. ingest.extract_audio()           FFmpeg → /tmp/audio.wav  │
│    ├─ 3. transcribe.transcribe()          WhisperX large-v2 (GPU)  │
│    ├─ 4. scene_detect.detect_scenes()     PySceneDetect (CPU)      │
│    ├─ 5. audio_analysis.analyze()         RMS + peak detect (CPU)  │
│    ├─ 6. clip_selector.select_clips()     Candidates + Cerebras    │
│    │                                                                │
│    │  For each selected clip:                                       │
│    ├─ 7. render.cut_segment()             FFmpeg segment cut        │
│    ├─ 8. reframe.reframe_vertical()       9:16 face crop            │
│    ├─ 9. captions.generate_ass()          .ass subtitle file        │
│    └─10. render.final_render()            Burn captions + normalize │
│                                                                     │
│    Returns: { clips: [{ video_bytes, title, metadata }] }           │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│  Cerebras API     │
│  (Llama-70B)      │
│  Clip ranking     │
└───────────────────┘
```

> **Key principle**: Single Modal function, single container. The A10G GPU handles WhisperX transcription, then the same container does all CPU work (FFmpeg, scene detection, etc.) sequentially. No inter-container data transfers. Simple to debug, fast to iterate.

---

## 2. Project Structure

```
ClippedAI/
├── modal_app.py                 # Modal app + image definition + orchestrator
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py                # yt-dlp download + FFmpeg audio extraction
│   ├── transcribe.py            # WhisperX GPU transcription
│   ├── scene_detect.py          # PySceneDetect scene boundaries
│   ├── audio_analysis.py        # RMS energy + peak detection
│   ├── clip_selector.py         # Candidate generation + Cerebras LLM ranking
│   ├── reframe.py               # 9:16 vertical reframing with face tracking
│   ├── captions.py              # .ass subtitle generation
│   └── render.py                # FFmpeg segment cutting + final assembly
├── config.py                    # All constants, thresholds, prompts
├── run.py                       # Local CLI — invoke Modal + download clips
├── requirements.txt             # Local deps (just `modal`)
└── .env.example                 # Required env vars / Modal secrets
```

---

## 3. Environment & Prerequisites

### Modal Account + CLI

```bash
# Install Modal client
pip install modal

# Authenticate (opens browser)
modal token new
```

### Cerebras API Key

```bash
# Create Modal secret for Cerebras
modal secret create cerebras-api-key CEREBRAS_API_KEY=csk-xxxxxxxxxxxxxxxx
```

> Get a key at https://cloud.cerebras.ai — free tier gives enough for testing.

### Local `requirements.txt`

```
modal>=0.68.0
```

That's it locally. Everything else runs inside the Modal container.

### `.env.example`

```bash
# Modal secrets (set via `modal secret create`)
CEREBRAS_API_KEY=csk-your-key-here
```

---

## 4. Modal App Definition

#### [NEW] `modal_app.py`

This is the heart of the system. It defines the Modal app, the container image with all dependencies, and the main `process_video` function.

### Container Image

```python
import modal

app = modal.App("clipped-ai")

# Container image with ALL pipeline dependencies pre-installed
clipping_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        "libsndfile1",       # Required by librosa/soundfile
    )
    .pip_install(
        # Transcription
        "whisperx @ git+https://github.com/m-bain/whisperX.git",
        "torch==2.1.2",
        "torchaudio==2.1.2",

        # Scene detection
        "scenedetect[opencv]==0.6.4",

        # Audio analysis
        "numpy>=1.24",
        "scipy>=1.11",
        "librosa>=0.10",

        # Face detection for reframing
        "ultralytics>=8.0",  # YOLOv8

        # Captions
        "pysubs2>=1.7",

        # LLM API calls
        "httpx>=0.25",

        # Video download
        "yt-dlp>=2024.1",
    )
)
```

### Why These Dependencies

| Package | Purpose | Why This One |
|---------|---------|-------------|
| `whisperx` | Transcription + word alignment | Best word-level timestamp accuracy |
| `torch 2.1.2` | GPU backend for WhisperX | Stable CUDA 12 support on Modal A10G |
| `scenedetect` | Scene boundary detection | Battle-tested, CPU-only, fast |
| `librosa` | Audio RMS + onset detection | Industry standard for audio analysis |
| `ultralytics` | YOLOv8 face detection | Fast, accurate, single-line API |
| `pysubs2` | .ass subtitle generation | Clean API for ASS format |
| `httpx` | Cerebras API calls | Async-capable, modern HTTP client |
| `yt-dlp` | YouTube downloading | Most maintained fork, handles auth |
| `ffmpeg` (apt) | Video/audio processing | Does everything — cut, encode, burn subs |

---

## 5. Pipeline Step 1 — Ingest

#### [NEW] `pipeline/ingest.py`

### `download_video(url: str) → dict`

```
Input:  YouTube URL (e.g. "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
Output: {
    "video_path": "/tmp/clipped/source.mp4",
    "title": "Video Title",
    "duration": 1847.3,          # seconds
    "resolution": "1920x1080",
    "fps": 30.0
}
```

**Implementation details:**
- Use `yt-dlp` with format selection: `bestvideo[height<=1080]+bestaudio/best[height<=1080]`
  - Cap at 1080p — higher res is wasted since final output is 1080x1920 (9:16)
- Download to `/tmp/clipped/source.mp4` (Modal ephemeral storage)
- Extract metadata via `yt-dlp --dump-json` (title, duration, resolution)
- **Error handling**: Retry up to 3 times on network errors. Raise clear error on geo-blocked / private videos.

### `extract_audio(video_path: str) → str`

```
Input:  "/tmp/clipped/source.mp4"
Output: "/tmp/clipped/audio.wav"
```

**FFmpeg command:**
```bash
ffmpeg -i /tmp/clipped/source.mp4 \
       -vn \                          # No video
       -acodec pcm_s16le \            # 16-bit PCM
       -ar 16000 \                    # 16kHz sample rate (WhisperX requirement)
       -ac 1 \                        # Mono
       /tmp/clipped/audio.wav
```

---

## 6. Pipeline Step 2 — Transcribe (GPU)

#### [NEW] `pipeline/transcribe.py`

### `transcribe(audio_path: str) → dict`

```
Input:  "/tmp/clipped/audio.wav"
Output: {
    "language": "en",
    "segments": [
        {
            "start": 0.0,
            "end": 4.52,
            "text": "Hey guys welcome back to the channel",
            "words": [
                {"word": "Hey",     "start": 0.0,  "end": 0.28},
                {"word": "guys",    "start": 0.32, "end": 0.58},
                {"word": "welcome", "start": 0.62, "end": 1.04},
                {"word": "back",    "start": 1.08, "end": 1.32},
                {"word": "to",      "start": 1.36, "end": 1.44},
                {"word": "the",     "start": 1.48, "end": 1.56},
                {"word": "channel", "start": 1.60, "end": 2.04}
            ]
        },
        ...
    ]
}
```

**Implementation details:**
1. Load WhisperX model: `whisperx.load_model("large-v2", "cuda", compute_type="float16")`
2. Transcribe: `model.transcribe(audio_path, batch_size=16)`
3. Align for word timestamps: `whisperx.align(result["segments"], align_model, audio)`
4. Return structured segments with word-level timing

**Why WhisperX over plain Whisper:**
- Native word-level timestamps via forced alignment (not approximated)
- 70x faster with batched inference on GPU
- `large-v2` has the best accuracy for English content

**Performance**: ~60 seconds for a 30-minute video on A10G.

---

## 7. Pipeline Step 3 — Scene Detection

#### [NEW] `pipeline/scene_detect.py`

### `detect_scenes(video_path: str) → list[dict]`

```
Input:  "/tmp/clipped/source.mp4"
Output: [
    {"start": 0.0,    "end": 15.3,   "duration": 15.3},
    {"start": 15.3,   "end": 42.7,   "duration": 27.4},
    {"start": 42.7,   "end": 58.1,   "duration": 15.4},
    ...
]
```

**Implementation details:**
- Use `scenedetect.ContentDetector(threshold=27.0)`
  - `threshold=27.0` is the sweet spot — catches real scene changes without over-segmenting
- Process at 2 FPS (skip frames for speed): `video_manager.set_downscale_factor(2)`
- Return list of scene segments with start/end times

**Why scene detection matters for clip selection:**
- High scene density in a time window = visually interesting content
- Scene boundaries make natural clip start/end points
- Prevents cutting mid-visual-transition

---

## 8. Pipeline Step 4 — Audio Analysis

#### [NEW] `pipeline/audio_analysis.py`

### `analyze_audio(audio_path: str) → dict`

```
Input:  "/tmp/clipped/audio.wav"
Output: {
    "energy_curve": [0.02, 0.03, 0.15, 0.82, ...],    # RMS per 0.5s frame
    "peaks": [
        {"time": 124.5, "energy": 0.92, "type": "spike"},
        {"time": 267.0, "energy": 0.85, "type": "spike"},
        ...
    ],
    "avg_energy": 0.12,
    "max_energy": 0.95
}
```

**Implementation details:**
1. Load audio with `librosa.load(audio_path, sr=16000)`
2. Compute RMS energy: `librosa.feature.rms(y=audio, frame_length=8000, hop_length=8000)`
   - 8000 samples at 16kHz = 0.5s window
3. Detect peaks: find frames where `energy > mean + 2 * std`
   - These correspond to laughter, applause, shouts, music drops
4. Return energy curve + peak timestamps

**Why audio analysis matters:**
- Energy spikes strongly correlate with "viral moments"
- Crowd reactions, emotional outbursts, punchlines = high-energy
- Used as a signal in the candidate scoring formula

---

## 9. Pipeline Step 5 — Clip Selection (LLM)

#### [NEW] `pipeline/clip_selector.py`

This is the **most critical module** — clip quality lives or dies here.

### Phase A: Candidate Generation

### `generate_candidates(transcript, scenes, audio, settings) → list[dict]`

```
Input:
  - transcript: {segments: [...]}
  - scenes: [{start, end, duration}, ...]
  - audio: {energy_curve, peaks, ...}
  - settings: {max_clips: 5, min_duration: 15, max_duration: 90}

Output: [
    {
        "id": 0,
        "start": 45.2,
        "end": 102.8,
        "duration": 57.6,
        "transcript_text": "So then I went to this restaurant and...",
        "energy_score": 0.73,
        "scene_score": 0.45,
        "keyword_score": 0.20,
        "composite_score": 0.52
    },
    ...  # 3× max_clips candidates (e.g. 15 candidates for 5 clips)
]
```

**Algorithm:**
1. **Sliding window** over transcript segments:
   - Window sizes: try 30s, 45s, 60s, 90s
   - Start each window at a sentence boundary (segment start)
   - End each window at a sentence boundary (segment end)
   - Step: advance by 50% of window size (overlapping windows)

2. **Score each window** with 3 signals:

   ```
   energy_score = count(peaks in window) / max_peaks_in_any_window
   
   scene_score  = count(scene_changes in window) / window_duration_minutes
                  (normalized 0–1)
   
   keyword_score = count(viral_keywords in transcript_text) / len(words)
                   viral_keywords: ["insane", "crazy", "unbelievable", "oh my god",
                                    "no way", "let's go", "watch this", "biggest",
                                    "secret", "never", "first time", "challenge",
                                    "reveal", "shocking", "hack", "tip", ...]
   
   composite_score = (energy_score × 0.40) +
                     (scene_score  × 0.30) +
                     (keyword_score × 0.30)
   ```

3. **Deduplicate** overlapping windows: if two windows overlap by >50%, keep the higher-scored one
4. **Return top N** candidates (N = 3 × `max_clips`)

### Phase B: LLM Ranking

### `rank_with_llm(candidates: list, full_transcript: str, video_title: str) → list[dict]`

```
Input:  List of candidates + full transcript context + video title
Output: [
    {
        "candidate_id": 3,
        "rank": 1,
        "virality_score": 0.92,
        "keep": true,
        "title": "The Restaurant That Changed Everything",
        "description": "When he walked into the restaurant...",
        "hashtags": ["#storytime", "#restaurant", "#reaction"],
        "reasoning": "Strong narrative arc with emotional payoff..."
    },
    ...
]
```

**Cerebras API call:**
- Model: `llama-3.3-70b`
- Endpoint: `https://api.cerebras.ai/v1/chat/completions`
- Temperature: `0.3` (low for consistent ranking)

**System prompt:**
```
You are a viral video expert. Your job is to identify the most engaging,
shareable moments from a long-form video transcript.

You will receive a list of candidate clips with their transcript text and
signal scores. For each candidate, evaluate:

1. HOOK STRENGTH: Does the first 3 seconds grab attention?
2. NARRATIVE ARC: Is there a setup → payoff structure?
3. EMOTIONAL IMPACT: Does it trigger curiosity, surprise, humor, or awe?
4. STANDALONE VALUE: Does it make sense without the full video context?
5. SHAREABILITY: Would someone share this with a friend?

CRITICAL RULES:
- Do NOT invent timestamps. Use only the start/end provided.
- Mark keep=false for any candidate that is boring, repetitive, or lacks a payoff.
- Generate a punchy, clickbait-worthy title (5-10 words max).
- Provide reasoning for your ranking.
```

**User prompt template:**
```
Video title: "{video_title}"

Here are {n} candidate clips. Rank them by virality potential.

{for each candidate}
---
CANDIDATE {id}
Time: {start:.1f}s – {end:.1f}s ({duration:.0f}s)
Signal scores: energy={energy_score:.2f}, scenes={scene_score:.2f}, keywords={keyword_score:.2f}
Transcript:
"{transcript_text}"
---

{end for}

Return a JSON array with exactly {n} objects, one per candidate:
[
  {
    "candidate_id": <int>,
    "rank": <int 1..N>,
    "virality_score": <float 0-1>,
    "keep": <bool>,
    "title": "<string, 5-10 words>",
    "description": "<string, 1-2 sentences>",
    "hashtags": ["<string>", ...],
    "reasoning": "<string, why this is/isn't engaging>"
  }
]

Return ONLY the JSON array. No other text.
```

**Response parsing:**
- Parse JSON from response
- Filter to `keep=true` only
- Sort by `virality_score` descending
- Take top `max_clips`

**Fallback**: If Cerebras fails (timeout, rate limit), fall back to composite_score ranking without LLM.

---

## 10. Pipeline Step 6 — Reframe to 9:16

#### [NEW] `pipeline/reframe.py`

### `detect_faces(video_path: str, start: float, end: float) → list[dict]`

```
Input:  Video path + clip time range
Output: [
    {"time": 45.2, "x": 420, "y": 180, "w": 210, "h": 280},
    {"time": 45.7, "x": 425, "y": 178, "w": 212, "h": 282},
    ...
]
```

**Implementation:**
- Sample frames at 2 FPS within the clip range
- Run YOLOv8-face (`yolov8n-face`) on each frame
- Return bounding boxes per frame
- If multiple faces: use the largest (most prominent speaker)

### `compute_crop_positions(faces: list, source_w: int, source_h: int, target_ratio: float = 9/16) → list[dict]`

**Algorithm:**
1. Target crop width = `source_h × (9/16)` (maximizes vertical space)
2. For each frame, center crop on face X position
3. **Exponential smoothing** with `α = 0.15`:
   ```
   smoothed_x[t] = α × face_x[t] + (1 - α) × smoothed_x[t-1]
   ```
   This prevents jarring jumps when face moves
4. Clamp crop to frame boundaries
5. If no face detected in a frame: hold last known position
6. If no faces at all in clip: center crop (landscape → vertical center)

### `generate_crop_filter(crop_positions, fps) → str`

Returns an FFmpeg `-vf` filter string with per-frame crop keyframes:
```
crop=608:1080:x='if(lt(t,0.5),420,if(lt(t,1.0),425,...))':0
```

For simplicity in MVP, we use a single averaged crop position (not keyframed):
```
crop=608:1080:AVERAGED_X:0,scale=1080:1920
```

---

## 11. Pipeline Step 7 — Captions

#### [NEW] `pipeline/captions.py`

### `generate_ass(words: list, style: str = "highlight") → str`

```
Input:
  - words: [{"word": "Hey", "start": 0.0, "end": 0.28}, ...]
  - style: "highlight" (default MVP style)

Output: "/tmp/clipped/clip_01.ass"  (ASS subtitle file path)
```

**ASS file structure:**
```
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Style: Default,Montserrat Bold,62,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,
       -1,0,0,0,100,100,0,0,1,3,0,2,10,10,80,1

[Events]
; Word-by-word with active highlight
Dialogue: 0,0:00:00.00,0:00:00.28,Default,,0,0,0,,{\b1\c&H00FFFF&}Hey
Dialogue: 0,0:00:00.32,0:00:00.58,Default,,0,0,0,,{\b1\c&H00FFFF&}guys
...
```

**Caption style — "highlight" (MVP default):**
- Font: **Montserrat Bold**, 62px
- Position: **bottom center** (10% from bottom)
- Active word: **bold + cyan** (`\c&H00FFFF&`)
- Previous words: **dimmed** (`\alpha&H80&`)
- Max 3 words visible at a time (rolling window)
- Black semi-transparent background box behind text

**Word grouping logic:**
1. Group words into lines of ~3-5 words (fits 1080px width)
2. Show each word at its timestamp
3. Current word gets the highlight color
4. Previous words on same line are dimmed
5. New line clears previous words

---

## 12. Pipeline Step 8 — Final Render

#### [NEW] `pipeline/render.py`

### `cut_segment(video_path: str, start: float, end: float) → str`

```bash
ffmpeg -ss {start} -to {end} -i source.mp4 \
       -c copy \                      # No re-encode for speed
       -avoid_negative_ts make_zero \
       /tmp/clipped/segments/clip_01_raw.mp4
```

Returns path to raw segment.

### `final_render(segment_path, crop_filter, ass_path, output_path) → str`

This is the single-pass FFmpeg command that does everything:

```bash
ffmpeg -i /tmp/clipped/segments/clip_01_raw.mp4 \
       -vf "crop=608:1080:{crop_x}:0, \
            scale=1080:1920:flags=lanczos, \
            ass=/tmp/clipped/clip_01.ass" \
       -af "loudnorm=I=-14:TP=-1.5:LRA=11" \
       -c:v libx264 \
       -preset medium \
       -crf 18 \
       -pix_fmt yuv420p \
       -c:a aac \
       -b:a 192k \
       -ar 44100 \
       -movflags +faststart \
       -y \
       /tmp/clipped/output/clip_01.mp4
```

**Filter breakdown:**
1. `crop=608:1080:{x}:0` — Crop landscape to 9:16 region (face-centered)
2. `scale=1080:1920:flags=lanczos` — Scale up to 1080×1920 (full vertical HD)
3. `ass=clip_01.ass` — Burn captions (single pass, no re-encode needed separately)
4. `loudnorm=I=-14` — Normalize audio to -14 LUFS (YouTube/TikTok standard)
5. `libx264 crf 18` — High quality encode (visually lossless)
6. `movflags +faststart` — Enables streaming playback

**Performance**: ~15 seconds per 60-second clip on Modal CPU.

---

## 13. Orchestrator Function

#### Inside `modal_app.py`

### `process_video(youtube_url, max_clips, settings) → dict`

This is the main Modal function that chains all steps:

```python
@app.function(
    image=clipping_image,
    gpu="A10G",
    timeout=1800,                                  # 30 min max
    secrets=[modal.Secret.from_name("cerebras-api-key")],
)
def process_video(
    youtube_url: str,
    max_clips: int = 5,
    min_clip_duration: int = 15,
    max_clip_duration: int = 90,
) -> dict:
    """Full pipeline: YouTube URL → rendered clips with metadata."""

    import os, json, time
    from pipeline import ingest, transcribe, scene_detect, audio_analysis
    from pipeline import clip_selector, reframe, captions, render

    os.makedirs("/tmp/clipped/segments", exist_ok=True)
    os.makedirs("/tmp/clipped/output", exist_ok=True)

    timings = {}

    # Step 1 — Ingest
    t0 = time.time()
    video_info = ingest.download_video(youtube_url)
    audio_path = ingest.extract_audio(video_info["video_path"])
    timings["ingest"] = round(time.time() - t0, 1)
    print(f"✅ Ingest complete ({timings['ingest']}s) — {video_info['title']}")

    # Step 2 — Transcribe (GPU)
    t0 = time.time()
    transcript = transcribe.transcribe(audio_path)
    timings["transcribe"] = round(time.time() - t0, 1)
    print(f"✅ Transcribed ({timings['transcribe']}s) — {len(transcript['segments'])} segments")

    # Step 3 — Scene detection (CPU)
    t0 = time.time()
    scenes = scene_detect.detect_scenes(video_info["video_path"])
    timings["scene_detect"] = round(time.time() - t0, 1)
    print(f"✅ Scenes detected ({timings['scene_detect']}s) — {len(scenes)} scenes")

    # Step 4 — Audio analysis (CPU)
    t0 = time.time()
    audio = audio_analysis.analyze_audio(audio_path)
    timings["audio_analysis"] = round(time.time() - t0, 1)
    print(f"✅ Audio analyzed ({timings['audio_analysis']}s) — {len(audio['peaks'])} peaks")

    # Step 5 — Clip selection (CPU + Cerebras API)
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
    print(f"✅ Selected {len(selected)} clips ({timings['clip_selection']}s)")

    # Steps 6-8 — Render each clip
    t0 = time.time()
    clips = []
    for i, clip_info in enumerate(selected):
        print(f"  🎬 Rendering clip {i+1}/{len(selected)}: {clip_info['title']}")

        # Cut raw segment
        seg_path = render.cut_segment(
            video_info["video_path"],
            clip_info["start"],
            clip_info["end"],
            i
        )

        # Detect faces + compute crop
        faces = reframe.detect_faces(
            video_info["video_path"],
            clip_info["start"],
            clip_info["end"]
        )
        crop_x = reframe.compute_crop_x(
            faces,
            source_w=video_info["width"],
            source_h=video_info["height"]
        )

        # Generate captions
        clip_words = transcribe.get_words_in_range(
            transcript, clip_info["start"], clip_info["end"]
        )
        ass_path = captions.generate_ass(clip_words, clip_index=i)

        # Final render
        output_path = render.final_render(
            seg_path, crop_x,
            video_info["width"], video_info["height"],
            ass_path, i
        )

        # Read rendered file
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        clips.append({
            "video_bytes": video_bytes,
            "title": clip_info["title"],
            "description": clip_info.get("description", ""),
            "hashtags": clip_info.get("hashtags", []),
            "start": clip_info["start"],
            "end": clip_info["end"],
            "duration": clip_info["end"] - clip_info["start"],
            "virality_score": clip_info.get("virality_score", 0),
            "reasoning": clip_info.get("reasoning", ""),
        })

    timings["render"] = round(time.time() - t0, 1)
    timings["total"] = sum(timings.values())
    print(f"✅ All clips rendered ({timings['render']}s)")
    print(f"⏱️  Total: {timings['total']}s")

    return {
        "video_title": video_info["title"],
        "video_duration": video_info["duration"],
        "clips": clips,
        "timings": timings,
    }
```

---

## 14. Local CLI

#### [NEW] `run.py`

```python
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
    return name[:50]


def main():
    print("🎬 ClippedAI — Core Clipping Pipeline")
    print("=" * 45)

    # Get input
    url = input("\nYouTube URL: ").strip()
    if not url:
        print("❌ No URL provided")
        sys.exit(1)

    max_clips_input = input("Max clips (default 5): ").strip()
    max_clips = int(max_clips_input) if max_clips_input else 5

    min_dur_input = input("Min clip duration in seconds (default 15): ").strip()
    min_duration = int(min_dur_input) if min_dur_input else 15

    max_dur_input = input("Max clip duration in seconds (default 90): ").strip()
    max_duration = int(max_dur_input) if max_dur_input else 90

    print(f"\n🚀 Processing on Modal (A10G GPU)...")
    print(f"   URL: {url}")
    print(f"   Max clips: {max_clips}")
    print(f"   Duration range: {min_duration}–{max_duration}s")
    print(f"   This may take 2-5 minutes...\n")

    # Call Modal
    process_video = modal.Function.from_name("clipped-ai", "process_video")
    result = process_video.remote(
        youtube_url=url,
        max_clips=max_clips,
        min_clip_duration=min_duration,
        max_clip_duration=max_duration,
    )

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Save clips
    for i, clip in enumerate(result["clips"]):
        safe_title = sanitize_filename(clip["title"])
        filename = f"clip_{i+1:02d}_{safe_title}.mp4"
        filepath = os.path.join("output", filename)

        with open(filepath, "wb") as f:
            f.write(clip["video_bytes"])

        score = clip.get("virality_score", 0)
        duration = clip.get("duration", 0)
        print(f"  💾 {filename} ({duration:.0f}s, score: {score:.2f})")

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

    print(f"\n✅ {len(result['clips'])} clips saved to output/")
    print(f"📋 Metadata saved to output/metadata.json")
    print(f"⏱️  Pipeline timings: {json.dumps(result['timings'], indent=2)}")


if __name__ == "__main__":
    main()
```

---

## 15. Configuration & Constants

#### [NEW] `config.py`

```python
"""All tunable constants for the clipping pipeline."""

# ──────────────────── Ingest ────────────────────
MAX_VIDEO_DURATION = 4 * 3600     # 4 hours max
MAX_RESOLUTION = 1080             # Cap download at 1080p
YTDLP_FORMAT = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"

# ──────────────────── Transcription ────────────────────
WHISPERX_MODEL = "large-v2"
WHISPERX_BATCH_SIZE = 16
WHISPERX_COMPUTE_TYPE = "float16"

# ──────────────────── Scene Detection ────────────────────
SCENE_THRESHOLD = 27.0            # ContentDetector threshold
SCENE_MIN_SCENE_LEN = 5           # Minimum 5 seconds per scene

# ──────────────────── Audio Analysis ────────────────────
AUDIO_FRAME_LENGTH = 8000         # 0.5s at 16kHz
AUDIO_HOP_LENGTH = 8000
PEAK_STD_MULTIPLIER = 2.0         # peak = mean + N * std

# ──────────────────── Candidate Generation ────────────────────
WINDOW_SIZES = [30, 45, 60, 90]   # seconds
WINDOW_STEP_RATIO = 0.5           # 50% overlap
CANDIDATE_MULTIPLIER = 3          # 3× max_clips candidates
OVERLAP_THRESHOLD = 0.5           # Dedupe if >50% overlap

# Scoring weights
ENERGY_WEIGHT = 0.40
SCENE_WEIGHT = 0.30
KEYWORD_WEIGHT = 0.30

# Viral keyword list
VIRAL_KEYWORDS = [
    "insane", "crazy", "unbelievable", "oh my god", "no way",
    "let's go", "watch this", "biggest", "secret", "never",
    "first time", "challenge", "reveal", "shocking", "hack",
    "tip", "mistake", "worst", "best", "amazing", "incredible",
    "literally", "actually", "honestly", "seriously",
    "wait for it", "plot twist", "you won't believe",
]

# ──────────────────── LLM (Cerebras) ────────────────────
CEREBRAS_MODEL = "llama-3.3-70b"
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_TEMPERATURE = 0.3
CEREBRAS_MAX_TOKENS = 4000

# ──────────────────── Reframing ────────────────────
TARGET_ASPECT_RATIO = 9 / 16
FACE_DETECTION_FPS = 2            # Sample frames for face detection
SMOOTHING_ALPHA = 0.15            # Exponential smoothing for crop movement

# ──────────────────── Captions ────────────────────
CAPTION_FONT = "Montserrat Bold"
CAPTION_FONT_SIZE = 62
CAPTION_COLOR = "&H00FFFFFF"      # White (ASS BGR format)
CAPTION_HIGHLIGHT_COLOR = "&H00FFFF&"  # Cyan
CAPTION_BG_ALPHA = "&H80000000"   # Semi-transparent black
CAPTION_WORDS_PER_LINE = 4
CAPTION_POSITION_Y = 80           # % from top (bottom placement)

# ──────────────────── Render ────────────────────
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
VIDEO_CODEC = "libx264"
VIDEO_CRF = 18
VIDEO_PRESET = "medium"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 44100
LOUDNORM_TARGET = -14             # LUFS
```

---

## 16. Verification Plan

### Test Videos

| # | Type | Example URL | What to Evaluate |
|---|------|-------------|-----------------|
| 1 | **Podcast** | Any 30+ min talking head | Caption accuracy, reframing keeps speaker centered, clip starts at interesting points |
| 2 | **MrBeast-style** | High-energy, fast cuts | Multiple good moments found, energy peaks detected, titles are catchy |
| 3 | **Tutorial** | Coding or cooking tutorial | Clips are self-contained, make sense standalone, no mid-sentence cuts |

### Quality Checklist (per clip)

- [ ] Clip starts at a natural moment (not mid-word or mid-sentence)
- [ ] Clip ends at a natural moment (not abruptly cut off)
- [ ] Transcript text matches spoken words (>95% accuracy)
- [ ] Word timestamps align with spoken words (word-level sync)
- [ ] Caption highlight lands on the correct word at the correct time
- [ ] 9:16 reframe keeps the primary face/speaker visible
- [ ] Crop position doesn't jump erratically
- [ ] Audio is clear and properly normalized
- [ ] No encoding artifacts or visual glitches
- [ ] The clip is actually an interesting, shareable moment
- [ ] Title is relevant and catchy (not generic)

### How to Run

```bash
# Deploy to Modal
modal deploy modal_app.py

# Run the pipeline
python run.py

# Enter:
#   URL: https://www.youtube.com/watch?v=XXXX
#   Max clips: 5
#   Min duration: 15
#   Max duration: 90

# Check output/
ls -la output/
open output/clip_01_*.mp4   # Preview in QuickTime
cat output/metadata.json    # Review scores + reasoning
```

### Iteration Loop

```
1. Run pipeline on test video
2. Watch every clip
3. Identify issues:
   - Bad clip boundaries → tune candidate generation / window sizes
   - Boring clips selected → improve LLM prompt / scoring weights
   - Caption timing off → debug WhisperX alignment
   - Reframing jittery → adjust smoothing alpha
   - Audio too quiet/loud → adjust loudnorm params
4. Tweak config.py + pipeline code
5. Re-deploy: `modal deploy modal_app.py`
6. Repeat until every clip slaps
```
