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


# Keys
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY")
if not ASSEMBLYAI_KEY:
    raise ValueError("ASSEMBLYAI_KEY must be set in the .env file.")

GEMINI_KEY = os.getenv("GEMINI_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_KEY must be set in the .env file.")

# Global configurations
OUTPUT_DIR = "output"
MASTER_VIDEO_TMPL = "master_video.%(ext)s"
MASTER_VIDEO_FILE = "master_video.mp4"
