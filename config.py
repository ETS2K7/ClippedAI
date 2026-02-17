"""All tunable constants for the ClippedAI clipping pipeline."""

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
WINDOW_SIZES = [30, 45, 60, 90]  # seconds
WINDOW_STEP_RATIO = 0.5
CANDIDATE_MULTIPLIER = 3  # 3× max_clips
OVERLAP_THRESHOLD = 0.5

# Scoring weights
ENERGY_WEIGHT = 0.40
SCENE_WEIGHT = 0.30
KEYWORD_WEIGHT = 0.30

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
CEREBRAS_MODEL = "llama-3.3-70b"
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_TEMPERATURE = 0.3
CEREBRAS_MAX_TOKENS = 4000

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
VIDEO_CRF = 18
VIDEO_PRESET = "medium"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 44100
LOUDNORM_TARGET = -14  # LUFS
