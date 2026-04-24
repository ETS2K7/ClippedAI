import os
import logging
from dotenv import load_dotenv
import functools
from typing import Callable, Any

# Configure logging level based on environment
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO" if os.environ.get("NODE_ENV") == "production" else "DEBUG").upper()

load_dotenv()

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger


# Module for loading environment variables and providing lazy accessors for API keys.

def validate_required_env_vars():
    """
    Validates that all required environment variables are set at startup.
    Raises RuntimeError if any required variables are missing.
    Call this explicitly in main.py startup phase.
    """
    required_vars = [
        "ASSEMBLYAI_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_BUCKET_NAME",
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            "Please set these in your environment or Modal secrets."
        )
    
    logger = get_logger(__name__)
    logger.info("All required environment variables validated successfully.")

# ─── Lazy API key access ─────────────────────────────────────────────────
# Defer validation to first use so containerised startup doesn't crash
# before the app has a chance to log diagnostic information.
_assemblyai_key: str | None = None


def _get_assemblyai_key() -> str:
    global _assemblyai_key
    if _assemblyai_key is None:
        _assemblyai_key = os.getenv("ASSEMBLYAI_KEY")
        if not _assemblyai_key:
            raise ValueError("ASSEMBLYAI_KEY must be set in the environment.")
    return _assemblyai_key

# Public accessors used by other modules — lazy-validated on first call
ASSEMBLYAI_KEY = _get_assemblyai_key  # Call as ASSEMBLYAI_KEY()

GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "clippedai-493912")


# Local pipeline video paths (used by downloader.py and run_local.py)
MASTER_VIDEO_TMPL = "master_video.%(ext)s"
MASTER_VIDEO_FILE = "master_video.mp4"

