import os
import hmac
import shutil
import pathlib
import uuid
import subprocess
import boto3
import modal
import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config import get_logger
from src.transcriber import transcribe
from src.llm import select_clips
from src.video_processing import extract_segment, track_speaker_and_frame, merge_and_cleanup
from src.subtitles import generate_subtitles

logger = get_logger(__name__)

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "clippedai-7137")

class ProcessVideoRequest(BaseModel):
    s3_key: str
    youtube_url: str | None = None
    # Webhook callback fields — Modal calls back when done
    uploaded_file_id: str | None = None
    user_id: str | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None

image = (modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "libsm6", "libxext6", "wget"])
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_file("config.py", remote_path="/root/config.py", copy=True)
)

app = modal.App("clippedai", image=image)

auth_scheme = HTTPBearer()

def _download_youtube(youtube_url: str, video_path: pathlib.Path) -> None:
    """Download YouTube video using yt-dlp with optional cookies for IP bypass."""
    cookie_path = "/run/secrets/youtube-cookies/cookies.txt"

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", str(video_path),
    ]

    if os.path.exists(cookie_path):
        logger.info("Using YouTube cookies for authenticated download")
        cmd += ["--cookies", cookie_path]
    else:
        logger.warning("No YouTube cookies found — download may fail on data center IPs")

    cmd.append(youtube_url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"yt-dlp stderr: {result.stderr[-2000:]}")
        raise RuntimeError(f"yt-dlp failed (exit {result.returncode}): {result.stderr[-500:]}")


def _process_video_pipeline(
    s3_key: str,
    youtube_url: str | None = None,
) -> dict:
    """
    Shared video processing pipeline used by both CLI and HTTP endpoints.
    Downloads/resolves the video, transcribes, selects clips, and processes them.
    Returns a dict with status and list of output clip S3 keys.
    """
    run_id = str(uuid.uuid4())
    base_dir = pathlib.Path("/tmp") / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    video_path = base_dir / "input.mp4"
    s3_client = boto3.client("s3")
    bucket = os.environ.get("S3_BUCKET_NAME", S3_BUCKET)

    logger.info(f"Resolving input source for s3_key={s3_key}")
    if youtube_url:
        logger.info("Downloading YouTube video")
        _download_youtube(youtube_url, video_path)
        logger.info("Uploading downloaded video to S3")
        s3_client.upload_file(str(video_path), bucket, s3_key)
    else:
        logger.info("Downloading from S3")
        s3_client.download_file(bucket, s3_key, str(video_path))

    try:
        words = transcribe(str(video_path), s3_key)
        clips = select_clips(words)
        output_clips = []

        for index, clip in enumerate(clips):
            logger.info(f"--- Processing Clip {index + 1} ---")
            ext_vid = extract_segment(str(video_path), clip, index)
            trk_vid, chunk_meta = track_speaker_and_frame(ext_vid, index, clip, words)
            sub_file = generate_subtitles(words, clip, index, chunk_meta)
            merge_and_cleanup(trk_vid, ext_vid, sub_file, index)

            clip_out_path = f"output/clip_{index + 1}.mp4"
            s3_key_dir = os.path.dirname(s3_key)
            output_s3_key = f"{s3_key_dir}/clip_{index}.mp4"

            logger.info(f"Uploading clip {index + 1} to S3")
            s3_client.upload_file(clip_out_path, bucket, output_s3_key)
            output_clips.append(output_s3_key)
            os.remove(clip_out_path)

        return {"status": "success", "clips": output_clips}

    finally:
        if base_dir.exists():
            logger.info("Cleaning up temp directory")
            shutil.rmtree(base_dir, ignore_errors=True)


def _send_webhook(
    webhook_url: str,
    webhook_secret: str,
    uploaded_file_id: str,
    user_id: str,
    result: dict,
) -> None:
    """POST processing results back to the Next.js webhook endpoint."""
    payload = {
        "uploaded_file_id": uploaded_file_id,
        "user_id": user_id,
        "status": result.get("status", "failed"),
        "clips": result.get("clips", []),
        "secret": webhook_secret,
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info(f"Webhook delivered: {resp.status_code}")
    except Exception as e:
        logger.error(f"Webhook delivery failed: {e}")


@app.cls(
    gpu="any",
    timeout=1200,
    secrets=[
        modal.Secret.from_name("clippedai-secret"),
    ]
)
class ClippedAI:

    @modal.method()
    def process_video_cli(self, s3_key: str, youtube_url: str = None):
        return _process_video_pipeline(s3_key, youtube_url)

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        auth_token = os.environ.get("AUTH_TOKEN")
        if not auth_token or not hmac.compare_digest(token.credentials, auth_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Run the pipeline
        try:
            result = _process_video_pipeline(request.s3_key, request.youtube_url)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result = {"status": "failed", "clips": [], "error": str(e)}

        # If webhook fields are provided, call back to Next.js
        if request.webhook_url and request.webhook_secret and request.uploaded_file_id and request.user_id:
            _send_webhook(
                request.webhook_url,
                request.webhook_secret,
                request.uploaded_file_id,
                request.user_id,
                result,
            )

        return result


@app.local_entrypoint()
def run_cli_job(s3_key: str, youtube_url: str = None):
    print(f"Submitting job to Modal for s3_key: {s3_key}")
    ClippedAI().process_video_cli.remote(s3_key, youtube_url)
