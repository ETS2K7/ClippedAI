import os
import shutil
import pathlib
import uuid
import subprocess
import boto3
import modal
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
    subprocess.run(cmd, check=True)


@app.cls(
    gpu="any",
    timeout=1200,
    secrets=[
        modal.Secret.from_name("clippedai-secret"),
        modal.Secret.from_name("youtube-cookies", required=False),
    ]
)
class ClippedAI:

    @modal.method()
    def process_video_cli(self, s3_key: str, youtube_url: str = None):
        import traceback
        logger = get_logger(__name__)

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs("output", exist_ok=True)

        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        bucket = os.environ.get("S3_BUCKET_NAME", S3_BUCKET)

        logger.info(f"Resolving input source: {youtube_url or s3_key}")
        if youtube_url:
            logger.info(f"Downloading YouTube video: {youtube_url}")
            _download_youtube(youtube_url, video_path)
            logger.info(f"Uploading downloaded video to S3: {s3_key}")
            s3_client.upload_file(str(video_path), bucket, s3_key)
        else:
            logger.info(f"Downloading from S3: {s3_key}")
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

                logger.info(f"Uploading {clip_out_path} to S3: {output_s3_key}")
                s3_client.upload_file(clip_out_path, bucket, output_s3_key)
                output_clips.append(output_s3_key)
                os.remove(clip_out_path)

            return {"status": "success", "clips": output_clips}

        finally:
            if base_dir.exists():
                logger.info("Cleaning up /tmp directory")
                shutil.rmtree(base_dir, ignore_errors=True)

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key
        youtube_url = request.youtube_url

        auth_token = os.environ.get("AUTH_TOKEN")
        if not auth_token or token.credentials != auth_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs("output", exist_ok=True)

        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        bucket = os.environ.get("S3_BUCKET_NAME", S3_BUCKET)

        logger.info(f"Resolving input source: {youtube_url or s3_key}")
        if youtube_url:
            logger.info(f"Downloading YouTube video: {youtube_url}")
            _download_youtube(youtube_url, video_path)
            logger.info(f"Uploading downloaded video to S3: {s3_key}")
            s3_client.upload_file(str(video_path), bucket, s3_key)
        else:
            logger.info(f"Downloading from S3: {s3_key}")
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

                logger.info(f"Uploading {clip_out_path} to S3: {output_s3_key}")
                s3_client.upload_file(clip_out_path, bucket, output_s3_key)
                output_clips.append(output_s3_key)
                os.remove(clip_out_path)

            return {"status": "success", "clips": output_clips}

        finally:
            if base_dir.exists():
                logger.info("Cleaning up /tmp directory")
                shutil.rmtree(base_dir, ignore_errors=True)


@app.local_entrypoint()
def run_cli_job(s3_key: str, youtube_url: str = None):
    print(f"Submitting job to Modal for s3_key: {s3_key}")
    ClippedAI().process_video_cli.remote(s3_key, youtube_url)
