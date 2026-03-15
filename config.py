"""
ClippedAI Configuration
"""
import os

# --- LLM ---
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-4-scout-17b"

# --- Face Detection ---
FACE_MODEL = "yolo26n"

# --- Chunking ---
CHUNK_DURATION = 300  # 5 minutes in seconds
CHUNK_OVERLAP = 30    # seconds

# --- Output ---
ASPECT_RATIO = "9:16"
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
CAPTION_STYLE = "karaoke"

# --- Clip Selection ---
MAX_CLIPS = 5
MIN_CLIP_DURATION = 15
IDEAL_CLIP_DURATION = (15, 45)
MAX_CLIP_DURATION = 60

# --- YouTube Rate Guard ---
YT_MAX_DOWNLOADS_PER_HOUR = 15
YT_COOLDOWN_AFTER_FAILURES = 3

# --- Rendering ---
VIDEO_CODEC = "libx264"
VIDEO_CRF = 20
VIDEO_PRESET = "medium"

# --- Volume ---
VOLUME_NAME = "clippedai-vol"
VOLUME_MOUNT = "/vol"

# --- Secrets ---
GROQ_SECRET_NAME = "groq-secret"
HF_SECRET_NAME = "huggingface-secret"

# --- Modal ---
APP_NAME = "clippedai"
GPU_TYPE_ASR = "A10G"
GPU_TYPE_VISION = "T4"
