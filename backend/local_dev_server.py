"""
local_dev_server.py — Local FastAPI server that mirrors the Modal pipeline.

Replaces:
  - Apify YouTube download  → yt-dlp
  - AWS S3 storage          → frontend/public/local-clips/
  - Modal container         → local Python process
  - NVENC GPU encoding      → libx264 CPU (M4 Apple Silicon)

Usage:
    ./run_local.sh
    # or directly:
    source local_venv/bin/activate && uvicorn local_dev_server:app --port 8000 --reload
"""

import hashlib
import hmac
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import uuid

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ─── Bootstrap ────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / "local.env")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("local_dev")

# Ensure the backend src/ directory is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# ─── Config ────────────────────────────────────────────────────────────────────
FRONTEND_PUBLIC_DIR = pathlib.Path(
    os.environ.get(
        "FRONTEND_PUBLIC_DIR",
        str(pathlib.Path(__file__).parent.parent / "frontend" / "public"),
    )
)
LOCAL_CLIPS_DIR = FRONTEND_PUBLIC_DIR / "local-clips"
LOCAL_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

AUTH_TOKEN = os.environ.get("LOCAL_DEV_AUTH_TOKEN", "local-dev-secret")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "http://localhost:3000/api/webhooks/modal")

# Absolute path to bundled Komika Axis font (used by ASS subtitle engine)
FONT_PATH = str(pathlib.Path(__file__).parent / "fonts" / "Komika_Axis.ttf")

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="ClippedAI Local Dev Server")
auth_scheme = HTTPBearer()


class ProcessVideoRequest(BaseModel):
    s3_key: str
    youtube_url: str | None = None
    uploaded_file_id: str | None = None
    user_id: str | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None
    font_family: str | None = None
    font_color: str | None = None
    font_size: int | None = None


# ─── Auth ──────────────────────────────────────────────────────────────────────
def verify_auth(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not token.credentials or not hmac.compare_digest(token.credentials, AUTH_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── YouTube Download (yt-dlp) ─────────────────────────────────────────────────
def _download_youtube_ytdlp(youtube_url: str, video_path: pathlib.Path) -> None:
    """Download a YouTube video using yt-dlp (free, works locally, no IP bans)."""
    logger.info(f"[yt-dlp] Downloading {youtube_url} → {video_path}")
    ytdlp_bin = shutil.which("yt-dlp") or "/opt/homebrew/bin/yt-dlp"
    cmd = [
        ytdlp_bin,
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", "mp4",
        "--output", str(video_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[-1000:]}")
    if not video_path.exists():
        raise RuntimeError(f"yt-dlp completed but output file not found: {video_path}")
    logger.info(f"[yt-dlp] Downloaded {video_path.stat().st_size / 1024 / 1024:.1f} MB")


# ─── Webhook ───────────────────────────────────────────────────────────────────
def _send_webhook(
    webhook_url: str,
    webhook_secret: str,
    uploaded_file_id: str,
    user_id: str,
    result: dict,
) -> None:
    payload = {
        "uploaded_file_id": uploaded_file_id,
        "user_id": user_id,
        "status": result.get("status", "failed"),
        "clips": result.get("clips", []),
    }
    payload_str = json.dumps(payload, separators=(",", ":"))
    signature = hmac.new(
        webhook_secret.encode("utf-8"),
        payload_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Secret": webhook_secret,
        "X-Signature": signature,
    }
    try:
        resp = requests.post(webhook_url, data=payload_str, headers=headers, timeout=30)
        resp.raise_for_status()
        logger.info(f"[webhook] Delivered → HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"[webhook] Delivery failed: {e}")


# ─── Core Pipeline ─────────────────────────────────────────────────────────────
def _run_pipeline(request: ProcessVideoRequest) -> None:
    """
    Full local pipeline — mirrors _process_video_pipeline in main.py.
    Runs in a background thread so the HTTP endpoint returns immediately.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.llm import select_clips
    from src.subtitles import generate_subtitles
    from src.transcriber import transcribe
    from src.video_processing import (
        extract_segment,
        merge_and_cleanup,
        track_speaker_and_frame,
    )

    run_id = str(uuid.uuid4())
    base_dir = pathlib.Path("/tmp") / f"clippedai_local_{run_id}"
    base_dir.mkdir(parents=True, exist_ok=True)
    video_path = base_dir / "input.mp4"

    webhook_url = request.webhook_url or WEBHOOK_URL
    webhook_secret = request.webhook_secret or AUTH_TOKEN

    try:
        # ── Phase 1: Ingest ──────────────────────────────────────────────────
        if request.youtube_url:
            logger.info("[pipeline] Phase 1: Downloading YouTube video via yt-dlp")
            _download_youtube_ytdlp(request.youtube_url, video_path)
        else:
            raise RuntimeError(
                "Local dev server only supports YouTube URLs currently. "
                "File upload support coming soon."
            )

        # ── Phase 2: Transcribe ──────────────────────────────────────────────
        logger.info("[pipeline] Phase 2: Transcribing with AssemblyAI")
        words = transcribe(str(video_path))

        # ── Phase 3: Select clips ────────────────────────────────────────────
        logger.info("[pipeline] Phase 3: Selecting clips with Groq LLM")
        clips = select_clips(words)
        logger.info(f"[pipeline] {len(clips)} clips selected")

        # ── Phases 4-7: Process each clip ────────────────────────────────────
        logger.info(f"[pipeline] Phase 4-7: Processing {len(clips)} clips (parallel, CPU)")
        output_clips: list[dict | None] = [None] * len(clips)
        clip_errors: list[str] = []
        run_output_dir = LOCAL_CLIPS_DIR / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        def _process_one(index: int, clip: dict) -> dict:
            ext_vid = extract_segment(str(video_path), clip, index, str(base_dir), use_gpu=False)
            trk_vid, chunk_meta = track_speaker_and_frame(ext_vid, index, clip, words, str(base_dir))
            sub_file = generate_subtitles(
                words, clip, index, chunk_meta,
                font_family=request.font_family,
                font_size=request.font_size,
                font_color=request.font_color,
                work_dir=str(base_dir),
            )
            merge_and_cleanup(trk_vid, ext_vid, sub_file, index, str(base_dir), use_gpu=False)

            # Copy finished clip to frontend public dir
            clip_filename = f"clip_{index}.mp4"
            src_path = base_dir / clip_filename
            dst_path = run_output_dir / clip_filename
            shutil.copy2(str(src_path), str(dst_path))
            os.remove(str(src_path))

            # s3Key is a path relative to /public — frontend resolves it as /local-clips/<run_id>/clip_N.mp4
            local_key = f"local-clips/{run_id}/{clip_filename}"
            return {
                "s3Key": local_key,
                "thumbnailKey": None,
                "thumbnailKeys": {},
                "clipTitle": clip.get("title", f"Clip {index + 1}"),
                "viralityScore": float(clip.get("virality_score") or 0.0),
            }

        max_workers = min(len(clips), 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_one, idx, clip): idx
                for idx, clip in enumerate(clips)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    output_clips[idx] = future.result()
                    logger.info(f"[pipeline] Clip {idx + 1} done")
                except Exception as exc:
                    import traceback
                    logger.error(f"[pipeline] Clip {idx + 1} failed: {exc}\n{traceback.format_exc()}")
                    clip_errors.append(f"Clip {idx + 1}: {exc}")

        final_clips = [c for c in output_clips if c is not None]
        if not final_clips:
            raise RuntimeError(f"All clips failed: {clip_errors}")

        result = {"status": "success", "clips": final_clips}
        logger.info(f"[pipeline] ✓ Complete — {len(final_clips)} clip(s) ready")

    except Exception as exc:
        import traceback
        logger.error(f"[pipeline] ✗ Pipeline failed: {exc}\n{traceback.format_exc()}")
        result = {"status": "failed", "clips": [], "error": str(exc)}
    finally:
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

    if request.uploaded_file_id and request.user_id:
        _send_webhook(webhook_url, webhook_secret, request.uploaded_file_id, request.user_id, result)


# ─── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/process_video", status_code=202)
def process_video(
    request: ProcessVideoRequest,
    _: HTTPAuthorizationCredentials = Depends(verify_auth),
):
    """
    Drop-in replacement for the Modal process_video endpoint.
    Returns 202 immediately and runs the pipeline in a background thread.
    """
    logger.info(
        f"[server] Job received — file={request.uploaded_file_id} "
        f"youtube={request.youtube_url or '(upload)'}"
    )
    thread = threading.Thread(target=_run_pipeline, args=(request,), daemon=True)
    thread.start()
    return {"status": "processing_started"}


@app.get("/health")
def health():
    return {"status": "ok", "mode": "local_dev"}
