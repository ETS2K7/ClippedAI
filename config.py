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

# --- Face Tracking ---
FACE_CONF_THRESHOLD = 0.5
FACE_NMS_THRESHOLD = 0.7
ASD_CONFIDENCE_THRESHOLD = 0.85   # LoCoNet: above this → center on active face
ASD_OVERLAP_THRESHOLD = 0.60      # Below this → safe wide shot

# --- Reframing ---
REFRAME_SEGMENT_DURATION = 0.5    # seconds per crop segment
HYSTERESIS_MIN = 1.5              # seconds — minimum speaker hold
HYSTERESIS_MAX = 3.0              # seconds — maximum speaker hold
SAFETY_MARGIN = 0.15              # 15% padding around face
MIN_FACE_HEIGHT_RATIO = 0.55      # minimum 55% face height in frame
GOOD_FRAME_THRESHOLD = 0.92       # pre-render validation
REID_COSINE_THRESHOLD = 0.62      # DBSCAN cosine distance for re-ID

# --- Pinned Commits ---
BOTFACESORT_COMMIT = "3d597ec3f8cc98c8bebc4b390a8d9ee619da4efb"
LOCONET_COMMIT = "68d90c83fde956d60245d0715bbd18e9e5fd3bae"
