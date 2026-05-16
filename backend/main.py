import os
import subprocess
import json
import shutil
import uuid
import hmac
import hashlib
import time
import pathlib
import logging
import boto3
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import modal
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ─── Inline Logging Logic (Self-Contained) ────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
for _noisy in ("hpack", "httpx", "httpcore", "botocore", "s3transfer", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger

logger = get_logger(__name__)

# ─── Modal App Configuration ──────────────────────────────────────────────────
app = modal.App("clippedai")

# Define project base path for local file inclusion
PROJECT_DIR = pathlib.Path(__file__).parent.resolve()

# Shared GPU Image with all required system dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "wget")
    .pip_install(
        "fastapi[standard]", "openai", "anthropic", "google-cloud-aiplatform", "google-genai",
        "boto3", "pydantic", "requests", "opencv-python", "numpy", 
        "scenedetect", "apify-client", "python-dotenv"
    )
    # Modern Modal 1.0+ File Inclusion
    .add_local_dir(PROJECT_DIR / "src", remote_path="/root/src")
)

# ─── Data Models ──────────────────────────────────────────────────────────────
class ProcessVideoRequest(BaseModel):
    youtube_url: Optional[str] = None
    s3_key: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    uploaded_file_id: Optional[str] = None
    user_id: Optional[str] = None
    font_family: str = "TheBoldFont"
    font_color: str = "yellow"
    font_size: int = 70
    add_subtitles: bool = True
    caption_template: str = "karaoke"
    timeframe_start: Optional[float] = None
    timeframe_end: Optional[float] = None
    specific_moments: Optional[str] = None

# ─── Utility Functions ────────────────────────────────────────────────────────
def validate_required_env_vars():
    """Fail fast if critical environment variables are missing."""
    required = ["S3_BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "GOOGLE_CLOUD_PROJECT", "GCP_SERVICE_ACCOUNT_JSON"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

def _create_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )

def _download_youtube(url: str, dest_path: pathlib.Path) -> str:
    """Download YouTube video using Apify client."""
    from apify_client import ApifyClient
    client = ApifyClient(os.environ.get("APIFY_TOKEN"))
    
    run_input = {
        "downloadTimeoutMins": 10,
        "format": "mp4",
        "urls": [url],
        "videoQuality": "highest",
    }
    
    logger.info(f"Triggering Apify download for: {url}")
    run = client.actor("p_v_v_p~youtube-video-downloader").call(run_input=run_input)
    
    # Wait for the run to finish and get the download URL
    dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
    if not dataset_items or "downloadUrl" not in dataset_items[0]:
        raise RuntimeError("Apify failed to provide a download URL.")
    
    download_url = dataset_items[0]["downloadUrl"]
    video_title = dataset_items[0].get("title", "Untitled Video")
    
    # Download the file to local storage
    import requests
    response = requests.get(download_url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded: {video_title} to {dest_path}")
    return video_title

def _send_webhook(url: str, secret: str, file_id: str, user_id: str, data: dict, video_title: str = ""):
    """Deliver result to the frontend via secure webhook."""
    import requests
    import hmac
    import hashlib
    
    payload = json.dumps({
        "uploadedFileId": file_id,
        "userId": user_id,
        "videoTitle": video_title,
        **data
    })
    
    headers = {
        "Content-Type": "application/json",
    }
    
    if secret:
        signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        headers["X-Webhook-Signature"] = signature
        
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=30)
        logger.info(f"Webhook delivered: {response.status_code}")
    except Exception as e:
        logger.error(f"Webhook failed: {e}")

class PipelineTimer:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.steps = {}
        self.current_step = None
        self.start_time = None

    def begin(self, step_name: str):
        if self.current_step:
            self._flush()
        self.current_step = step_name
        self.start_time = time.time()
        logger.info(f"[{self.run_id}] ▶ {step_name}")

    def _flush(self):
        if self.current_step:
            duration = time.time() - self.start_time
            self.steps[self.current_step] = duration
            logger.info(f"[{self.run_id}] ✓ {self.current_step} ({duration:.1fs})")
            self.current_step = None

# ─── Modal App Classes ───────────────────────────────────────────────────────
auth_scheme = HTTPBearer()

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    scaledown_window=15,
    max_containers=5,
    retries=0,
    secrets=[
        modal.Secret.from_name("clippedai-secret"),
        modal.Secret.from_name("my-gcp-secret"),
    ]
)
class ClippedAI:
    @modal.enter()
    def startup(self):
        """Pre-warm resources during container startup."""
        logger.info("Container starting — pre-warming resources...")
        validate_required_env_vars()
        
        # Pre-resolve worker classes for instant access
        self.whisperx_cls = modal.Cls.from_name("whisperx-worker", "WhisperXWorker")
        self.asd_cls = modal.Cls.from_name("fast-asd-tracker", "FastASDTracker")
        
        logger.info("Container warm and ready.")

    @modal.method()
    def run_pipeline(self, request_dict: dict):
        """
        Merged GPU pipeline: download -> Trim -> Parallel Spawn (ASD, Proxy, Transcription) -> Gemini select -> render -> webhook.
        """
        import shutil
        import pathlib
        import uuid
        import json
        import subprocess
        import os
        import boto3
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from src.llm import select_clips

        # Ensure worker classes are resolved
        if not hasattr(self, "asd_cls"):
            self.asd_cls = modal.Cls.from_name("fast-asd-tracker", "FastASDTracker")
        if not hasattr(self, "whisperx_cls"):
            self.whisperx_cls = modal.Cls.from_name("whisperx-worker", "WhisperXWorker")
        
        request = ProcessVideoRequest(**request_dict)
        run_id = str(uuid.uuid4())
        timer = PipelineTimer(run_id)
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = base_dir / "input.mp4"
        s3_client = _create_s3_client()
        bucket = os.environ.get("S3_BUCKET_NAME", "clippedai-7137")
        video_title = None

        try:
            # ── Phase 1: Ingest ───────────────────────────────────────────────
            timer.begin("ingestion")
            if request.youtube_url:
                video_title = _download_youtube(request.youtube_url, video_path)
            elif request.s3_key:
                logger.info(f"Ingesting from S3: {request.s3_key}")
                s3_client.download_file(bucket, request.s3_key, str(video_path))
            
            # ── Phase 1.5: Frame-Accurate Trim (Sync Anchor) ──────────────────
            if request.timeframe_start is not None and request.timeframe_end is not None:
                if request.timeframe_end > request.timeframe_start:
                    logger.info(f"Trimming input video: {request.timeframe_start}s to {request.timeframe_end}s")
                    trimmed_path = base_dir / "trimmed_input.mp4"
                    duration = request.timeframe_end - request.timeframe_start
                    
                    subprocess.run([
                        "/usr/bin/ffmpeg", "-y",
                        "-ss", str(request.timeframe_start),
                        "-i", str(video_path),
                        "-t", str(duration),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k",
                        "-avoid_negative_ts", "make_zero",
                        str(trimmed_path)
                    ], check=True)
                    video_path = trimmed_path

            # ── Phase 1.7: Spawn Parallel Workers ─────────────────────────────
            timer.begin("parallel_spawn")
            logger.info("Spawning parallel workers on synchronized timebase...")
            
            # 1. ASD Tracking Spawn
            lowres_video_path = base_dir / "input_360p.mp4"
            subprocess.run([
                "/usr/bin/ffmpeg", "-y", "-i", str(video_path),
                "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                "-an", str(lowres_video_path)
            ], check=True)
            with open(lowres_video_path, "rb") as f:
                video_bytes_lowres = f.read()
            asd_call = self.asd_cls().process_video.spawn(video_bytes_lowres)

            # 2. H.264 Proxy Generation
            proxy_path = base_dir / "proxy_1080p.mp4"
            rendering_audio_path = base_dir / "global_audio.aac"
            proxy_proc = subprocess.Popen([
                "/usr/bin/ffmpeg", "-y", "-i", str(video_path),
                "-c:v", "h264_nvenc", "-preset", "p1", "-qp", "19",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "0:a:0?",
                "-avoid_negative_ts", "make_zero",
                str(proxy_path)
            ])

            # 3. Transcription Prep
            transcription_audio_path = base_dir / "input_audio.wav"
            subprocess.run([
                "/usr/bin/ffmpeg", "-y", "-i", str(video_path), 
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", 
                str(transcription_audio_path)
            ], check=True)
            with open(transcription_audio_path, "rb") as f:
                audio_bytes = f.read()
            
            # ── Phase 2: Transcription & Selection ──────
            timer.begin("transcription")
            hf_token = os.environ.get("HF_TOKEN", "")
            words_json = self.whisperx_cls().transcribe.remote(audio_bytes, hf_token)
            words = json.loads(words_json)
            
            timer.begin("clip_selection")
            clips = select_clips(words, request.specific_moments)
            timer._flush()

            # ── Phase 3.7: Sync Parallel Tasks ───────────────────────────────
            timer.begin("parallel_wait")
            logger.info("Awaiting tracking and proxy generation...")
            tracking_data_json = asd_call.get()
            tracking_data = json.loads(tracking_data_json)
            proxy_proc.wait()
            if proxy_proc.returncode != 0:
                raise RuntimeError("Proxy generation failed")
            
            # ── Phase 4-7: Final Rendering ────────────────────────────────────
            if not clips:
                logger.info("No clips to process. Skipping rendering.")
                if request.webhook_url:
                    _send_webhook(
                        request.webhook_url, request.webhook_secret,
                        request.uploaded_file_id, request.user_id,
                        {"status": "success", "clips": [], "message": "No viral clips found."},
                        video_title
                    )
                return {"status": "success", "clips": []}

            s3_key_dir = os.path.dirname(request.s3_key) if request.s3_key else f"outputs/{run_id}"
            timer.begin(f"parallel_rendering_{len(clips)}_clips")
            logger.info(f"Processing {len(clips)} clips in parallel...")
            
            output_clips = [None] * len(clips)
            has_nvenc = True # Always True on A10G

            with ProcessPoolExecutor(max_workers=len(clips)) as executor:
                future_to_idx = {
                    executor.submit(
                        _process_single_clip,
                        str(proxy_path), clip, index, words,
                        bucket, s3_key_dir, tracking_data, str(rendering_audio_path),
                        request.font_family, request.font_color, request.font_size, request.add_subtitles,
                        str(base_dir), has_nvenc, request.caption_template,
                    ): index for index, clip in enumerate(clips)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    output_clips[idx] = future.result()
            
            final_result = {"status": "success", "clips": [c for c in output_clips if c]}
            
            if request.webhook_url:
                _send_webhook(
                    request.webhook_url, request.webhook_secret,
                    request.uploaded_file_id, request.user_id,
                    final_result, video_title
                )
            return final_result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if request.webhook_url:
                _send_webhook(
                    request.webhook_url, request.webhook_secret,
                    request.uploaded_file_id, request.user_id,
                    {"status": "failed", "clips": [], "error": str(e)},
                    video_title
                )
            raise e
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    @modal.method()
    def _download_and_trigger(self, request_dict: dict):
        """CPU task to handle external downloads before waking up the GPU."""
        import pathlib
        import uuid
        
        request = ProcessVideoRequest(**request_dict)
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            video_path = base_dir / "input.mp4"
            video_title = "Untitled Video"
            
            if request.youtube_url:
                video_title = _download_youtube(request.youtube_url, video_path)
            
            s3_client = _create_s3_client()
            bucket = os.environ.get("S3_BUCKET_NAME", "clippedai-7137")
            s3_key = f"uploads/{run_id}/input.mp4"
            s3_client.upload_file(str(video_path), bucket, s3_key)
            
            new_request = request_dict.copy()
            new_request["s3_key"] = s3_key
            new_request["youtube_url"] = None
            
            self.run_pipeline.spawn(new_request)
            logger.info(f"CPU download complete. GPU pipeline triggered for {run_id}")
            
        except Exception as e:
            logger.error(f"CPU download phase failed: {e}")
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        auth_token = os.environ.get("AUTH_TOKEN")
        if not auth_token or not token.credentials or not hmac.compare_digest(token.credentials, auth_token):
            raise HTTPException(status_code=401, detail="Invalid token")

        if request.youtube_url:
            self._download_and_trigger.spawn(request.dict())
        else:
            self.run_pipeline.spawn(request.dict())
        return {"status": "processing_started"}

def _process_single_clip(
    video_path, clip, index, words,
    bucket, s3_key_dir, tracking_data, audio_path,
    font_family, font_color, font_size, add_subtitles,
    work_dir, has_nvenc, caption_template
):
    """Worker process for rendering and uploading a single clip."""
    import os
    import pathlib
    import boto3
    from src.video_processing import track_speaker_and_frame
    
    out_path = os.path.join(work_dir, f"clip_{index}.mp4")
    
    try:
        # 1. Track and Render (Single Pass)
        rendered_path, meta = track_speaker_and_frame(
            video_path, index, clip, words, work_dir,
            tracking_data=tracking_data,
            audio_file=audio_path,
            streaming_output_path=out_path,
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            add_subtitles=add_subtitles,
            use_gpu=has_nvenc,
            caption_template=caption_template
        )
        
        # 2. Upload to S3
        s3_client = boto3.client("s3")
        s3_key = f"{s3_key_dir}/clips/clip_{index}.mp4"
        s3_client.upload_file(rendered_path, bucket, s3_key, ExtraArgs={'ContentType': 'video/mp4'})
        
        return {
            "index": index,
            "s3Key": s3_key,
            "title": clip.get("title", f"Clip {index}"),
            "duration": clip.get("end_time", 0) - clip.get("start_time", 0),
            "viral_reason": clip.get("viral_reason", "")
        }
    except Exception as e:
        print(f"Error processing clip {index}: {e}")
        return None
