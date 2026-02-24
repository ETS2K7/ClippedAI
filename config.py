"""
ClippedAI — Core Pipeline Configuration
All tunable parameters in one place.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
LLM_PROVIDER_PRIORITY = ["groq", "cerebras"]
LLM_MODEL_RANKING = "llama-3.3-70b-versatile"  # Groq production model
LLM_MODEL_FALLBACK = "llama-3.3-70b"  # Cerebras format
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# ─────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────
CHUNK_DURATION = 300        # 5-minute chunks
CHUNK_OVERLAP = 30          # seconds of overlap between chunks
ASPECT_RATIO = "9:16"
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
VIDEO_CRF = 20              # H.264 quality (lower = better, 18-23 typical)
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "medium"

# ─────────────────────────────────────────────
# Clip Selection
# ─────────────────────────────────────────────
MAX_CLIPS = 5
MIN_CLIP_DURATION = 15      # seconds
MAX_CLIP_DURATION = 60      # hard ceiling: retention drops sharply after 60s
IDEAL_CLIP_DURATION = (15, 45)  # target range for highest retention
NO_TEMPORAL_OVERLAP = True  # clips must NOT share any source frames
SEMANTIC_SIMILARITY_PENALTY = 0.75  # penalize clips with >0.75 cosine sim

# ─────────────────────────────────────────────
# ASR (WhisperX)
# ─────────────────────────────────────────────
WHISPERX_MODEL = "large-v3-turbo"
WHISPERX_COMPUTE_TYPE = "float16"
WHISPERX_BATCH_SIZE = 16
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─────────────────────────────────────────────
# Face Detection & Tracking
# ─────────────────────────────────────────────
FACE_MODEL = "yolov8n.pt"  # Ultralytics YOLO model (auto-downloads)
FACE_CONF_THRESHOLD = 0.5
FACE_IOU_THRESHOLD = 0.45
INSIGHTFACE_COSINE_THRESHOLD = 0.62  # for global re-ID clustering
REID_EMBEDDINGS_PER_SECOND = 1  # subsample to prevent OOM

# ─────────────────────────────────────────────
# Active Speaker Detection (LoCoNet)
# ─────────────────────────────────────────────
LOCONET_CONFIDENCE_HIGH = 0.85   # above = definitely active speaker
LOCONET_CONFIDENCE_LOW = 0.60    # below = uncertain, use fallback

# ─────────────────────────────────────────────
# Reframing
# ─────────────────────────────────────────────
REFRAME_SAFETY_MARGIN = 0.15     # 15% padding around face crop
REFRAME_MIN_FACE_HEIGHT = 0.55   # face must be >= 55% of frame height
REFRAME_HYSTERESIS = (1.5, 3.0)  # seconds — dynamic based on content
REFRAME_SMOOTHING_WINDOW = 15    # frames for Kalman smoother
CROP_SEGMENT_DURATION = 0.5      # seconds per pre-computed crop segment
GOOD_FRAME_THRESHOLD = 0.92      # minimum ratio for validation pass

# ─────────────────────────────────────────────
# Captions
# ─────────────────────────────────────────────
CAPTION_STYLE = "highlight_word"
CAPTION_FONT = "Arial"
CAPTION_FONT_SIZE_RATIO = 25     # output_height / this = font size
CAPTION_MAX_LINES = 2
CAPTION_MAX_WIDTH_RATIO = 0.80   # 80% of frame width
CAPTION_SAFE_ZONE_Y = 0.80       # captions placed at 80% down
CAPTION_GAP_THRESHOLD = 0.3      # seconds — gap > this = blank screen
CAPTION_MIN_WORD_DURATION = 0.2  # seconds — minimum display time
CAPTION_CONFIDENCE_THRESHOLD = 0.4  # drop words below this confidence

# ─────────────────────────────────────────────
# YouTube Download
# ─────────────────────────────────────────────
YT_PLAYER_CLIENTS = ["android", "ios", "web"]
YT_COOKIE_TTL = 21600       # 6 hours in seconds
YT_MAX_DOWNLOADS_PER_HOUR = 15
YT_MAX_DOWNLOADS_PER_DAY = 50
YT_COOLDOWN_AFTER_FAILURES = 3
YT_COOLDOWN_DURATION = 3600  # 1 hour
YT_MAX_RETRIES = 3
YT_RETRY_BACKOFF = (2, 10)  # exponential backoff range in seconds

# ─────────────────────────────────────────────
# Video Hash
# ─────────────────────────────────────────────
HASH_HEAD_BYTES = 10 * 1024 * 1024   # first 10MB
HASH_TAIL_BYTES = 10 * 1024 * 1024   # last 10MB

# ─────────────────────────────────────────────
# Modal
# ─────────────────────────────────────────────
MODAL_VOLUME_NAME = "clippedai-vol"
MODAL_VOLUME_MOUNT = "/vol"
MODAL_APP_NAME = "clippedai"

# ─────────────────────────────────────────────
# Paths (on Modal Volume)
# ─────────────────────────────────────────────
def video_dir(video_hash: str) -> Path:
    """Return path to video artifacts on Modal Volume."""
    return Path(MODAL_VOLUME_MOUNT) / video_hash

def artifact_path(video_hash: str, filename: str) -> Path:
    """Return path to a specific artifact file."""
    return video_dir(video_hash) / filename
