import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ─── Lazy API key access ─────────────────────────────────────────────────
# Defer validation to first use so containerised startup doesn't crash
# before the app has a chance to log diagnostic information.
_assemblyai_key: str | None = None
_gemini_key: str | None = None


def _get_assemblyai_key() -> str:
    global _assemblyai_key
    if _assemblyai_key is None:
        _assemblyai_key = os.getenv("ASSEMBLYAI_KEY")
        if not _assemblyai_key:
            raise ValueError("ASSEMBLYAI_KEY must be set in the environment.")
    return _assemblyai_key


def _get_gemini_key() -> str:
    global _gemini_key
    if _gemini_key is None:
        _gemini_key = os.getenv("GEMINI_KEY")
        if not _gemini_key:
            raise ValueError("GEMINI_KEY must be set in the environment.")
    return _gemini_key


# Public accessors used by other modules — lazy-validated on first call
ASSEMBLYAI_KEY = _get_assemblyai_key  # Call as ASSEMBLYAI_KEY()
GEMINI_KEY = _get_gemini_key          # Call as GEMINI_KEY()


# Local pipeline video paths (used by downloader.py and run_local.py)
MASTER_VIDEO_TMPL = "master_video.%(ext)s"
MASTER_VIDEO_FILE = "master_video.mp4"

