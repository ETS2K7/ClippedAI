import os
import hmac
import json
import shutil
import pathlib
import uuid
import time
import subprocess
import boto3
from botocore.config import Config as BotoConfig
import modal
import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config import get_logger, validate_required_env_vars
import functools
from src.transcriber import transcribe
from src.llm import select_clips
from src.video_processing import extract_segment, track_speaker_and_frame, merge_and_cleanup
from src.subtitles import generate_subtitles

logger = get_logger(__name__)

# HTTP session with connection pooling for Modal API calls
_http_session = None

def get_http_session():
    """Get or create a requests.Session with connection pooling."""
    global _http_session
    if _http_session is None:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        _http_session = requests.Session()
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
            pool_block=False,
        )
        
        _http_session.mount("http://", adapter)
        _http_session.mount("https://", adapter)
    
    return _http_session


@functools.lru_cache(maxsize=1)
def _create_s3_client():
    """Creates a standard S3 client with caching."""
    try:
        return boto3.client("s3", config=BotoConfig(
            max_pool_connections=20,
        ))
    except Exception:
        return boto3.client("s3")


class PipelineTimer:
    """Structured pipeline tracing for full observability.
    Tracks per-phase wall-clock time and logs a summary at the end."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.phases: list[tuple[str, float]] = []
        self._phase_start: float | None = None
        self._phase_name: str = ""
        self.start_time = time.monotonic()

    def begin(self, name: str):
        self._flush()
        self._phase_name = name
        self._phase_start = time.monotonic()
        logger.info(f"[{self.run_id[:8]}] ▶ {name}")

    def _flush(self):
        if self._phase_start is not None:
            elapsed = time.monotonic() - self._phase_start
            self.phases.append((self._phase_name, elapsed))
            logger.info(f"[{self.run_id[:8]}] ✓ {self._phase_name} ({elapsed:.1f}s)")
            self._phase_start = None

    def summary(self):
        self._flush()
        total = time.monotonic() - self.start_time
        parts = " | ".join(f"{n}: {t:.1f}s" for n, t in self.phases)
        logger.info(f"[{self.run_id[:8]}] Pipeline complete in {total:.1f}s — {parts}")
        return {"total_seconds": round(total, 1), "phases": {n: round(t, 1) for n, t in self.phases}}

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "clippedai-ap-south-1")
S3_REGION = os.environ.get("AWS_REGION", "ap-south-1")

class ProcessVideoRequest(BaseModel):
    s3_key: str
    youtube_url: str | None = None
    # Webhook callback fields — Modal calls back when done
    uploaded_file_id: str | None = None
    user_id: str | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None
    # Formatting Configuration (Architecture Bridge)
    font_family: str | None = None
    font_color: str | None = None
    font_size: int | None = None
    caption_template: str | None = None
    add_subtitles: bool = True
    output_format: str = "vertical"

image = (modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "libsm6", "libxext6", "wget", "git", "fontconfig"])
    .pip_install_from_requirements("requirements.txt")
    .pip_install("apify-client", "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2", "ffmpeg-python", "gdown", "python_speech_features", "pandas")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("fonts", remote_path="/usr/share/fonts/truetype/custom", copy=True)
    .run_commands([
        "fc-cache -fv",
        "git clone https://github.com/sieve-community/fast-asd.git /fast-asd",
        "mkdir -p /root/.cache/models",
        "mkdir -p /root/model/faceDetector/s3fd",
        "gdown 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea -O /root/.cache/models/pretrain_TalkSet.model",
        "wget -O /root/model/faceDetector/s3fd/sfd_face.pth https://storage.googleapis.com/mango-public-models/sfd_face.pth",
        "ln -s /fast-asd/talknet/model /root/model_symlink",
        # Auto-patch legacy Numpy references to fix version collision
        "find /fast-asd -type f -name '*.py' -exec sed -i 's/np\\.float/float/g' {} +",
        "find /fast-asd -type f -name '*.py' -exec sed -i 's/np\\.bool/bool/g' {} +",
        "find /fast-asd -type f -name '*.py' -exec sed -i 's/np\\.int/int/g' {} +",
        
        # Patch relative paths to absolute paths for thread-safety (no os.chdir needed)
        "find /fast-asd -type f -name '*.py' -exec sed -i \"s|'./model|'/fast-asd/talknet/model|g\" {} +",
        "find /fast-asd -type f -name '*.py' -exec sed -i 's|\"./model|\"/fast-asd/talknet/model|g' {} +"
    ])
    .add_local_file("config.py", remote_path="/root/config.py", copy=True)
)

app = modal.App("clippedai", image=image)

auth_scheme = HTTPBearer()

def _download_youtube(youtube_url: str, video_path: pathlib.Path) -> str | None:
    """Download YouTube video using Apify API. Returns the video title if found."""
    from apify_client import ApifyClient

    apify_token = os.environ.get("APIFY_TOKEN")
    if not apify_token:
        raise RuntimeError("APIFY_TOKEN is missing from environment secrets.")

    logger.info(f"Starting Apify youtube downloader for {youtube_url}")
    client = ApifyClient(apify_token)

    run_input = {
        "videos": [{"url": youtube_url}],
        "preferQuality": "720p",
        "preferFormat": "mp4"
    }

    try:
        run = client.actor("streamers/youtube-video-downloader").call(run_input=run_input)
        logger.info(f"Apify run finished with status: {run.get('status')}")

        dataset = client.dataset(run["defaultDatasetId"])
        items = dataset.list_items().items
        if not items:
            raise RuntimeError("Apify actor did not return any dataset items (download failed)")

        item = items[0]
        download_url = item.get("downloadedFileUrl") or item.get("downloadUrl")
        
        if not download_url:
            raise RuntimeError(f"No download URL found in Apify output (Ensure the actor is configured to use KVS natively): {item}")

        logger.info(f"Streaming video directly from Apify KVS to Modal: {download_url}")

        # Download the file directly from Apify KeyValueStore bypassing any AWS transmission
        with requests.get(download_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(str(video_path), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192*4):
                    f.write(chunk)
            
        logger.info("Successfully fetched YouTube video via Apify")
        return item.get("title")
        
    except Exception as e:
        logger.error(f"Apify download failed: {str(e)}")
        return None

def _process_single_clip(
    video_path: str,
    clip: dict,
    index: int,
    words: list,
    s3_client,
    bucket: str,
    s3_key_dir: str,
    font_family: str | None = None,
    font_color: str | None = None,
    font_size: int | None = None,
    add_subtitles: bool = True,
    work_dir: str = "",
    use_gpu: bool = False,
    tracker=None,
) -> dict:
    """
    Processes a single clip through the full sub-pipeline:
    extract -> track -> subtitle -> merge -> S3 upload.
    Returns a dict with {"s3Key": ..., "thumbnailKey": ...}.

    Designed to run concurrently with other clip pipelines via ThreadPoolExecutor.
    """
    logger.info(f"--- Processing Clip {index + 1} (parallel) ---")
    ext_vid = extract_segment(video_path, clip, index, work_dir, use_gpu=use_gpu)
    trk_vid, chunk_meta = track_speaker_and_frame(ext_vid, index, clip, words, work_dir, tracker=tracker)
    sub_file = None
    if add_subtitles:
        sub_file = generate_subtitles(
            words, clip, index, chunk_meta,
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            work_dir=work_dir,
        )
    fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
    merge_and_cleanup(trk_vid, ext_vid, sub_file, index, work_dir, use_gpu=use_gpu, fonts_dir=fonts_dir)

    clip_out_path = f"{work_dir}/clip_{index}.mp4"
    output_s3_key = f"{s3_key_dir}/clip_{index}.mp4"

    # Upload original video with caching headers
    logger.info(f"Uploading clip {index + 1} to S3")
    s3_client.upload_file(
        clip_out_path, bucket, output_s3_key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=31536000, immutable",
        },
    )

    os.remove(clip_out_path)

    return {
        "s3Key": output_s3_key,
        "thumbnailKey": None,
        "thumbnailKeys": {},
        "clipTitle": clip.get("title", f"Clip {index + 1}"),
        "viralityScore": float(clip.get("virality_score") or 0.0),
    }


def _render_clips_pipeline(
    s3_key: str,
    words: list,
    clips: list,
    previous_phases: list,
    font_family: str | None = None,
    font_color: str | None = None,
    font_size: int | None = None,
    add_subtitles: bool = True,
    tracker=None,
) -> dict:
    """
    Shared video processing pipeline used by both CLI and HTTP endpoints.
    Downloads/resolves the video, transcribes, selects clips, and processes them.
    Returns a dict with status and list of output clip S3 keys.

    Optimizations applied:
      - Audio-only AssemblyAI upload (P0)
      - Copy-mode segment extraction (P0)
      - Parallel clip processing via ThreadPoolExecutor (P0)
      - NVENC GPU encoding with CPU fallback (P1)
      - S3 Transfer Acceleration (P2)
      - Structured pipeline tracing (P2)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    run_id = str(uuid.uuid4())
    timer = PipelineTimer(run_id)
    base_dir = pathlib.Path("/tmp") / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    # Detect NVENC at pipeline scope so it's available for clip workers
    try:
        nvenc_result = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
        )
        has_nvenc = "h264_nvenc" in nvenc_result.stdout
    except Exception:
        has_nvenc = False
    logger.info(f"NVENC detection in pipeline: {has_nvenc}")

    video_path = base_dir / "input.mp4"
    s3_client = _create_s3_client()
    bucket = os.environ.get("S3_BUCKET_NAME", S3_BUCKET)

    timer.phases = previous_phases
    
    # Phase 3.5: GPU S3 Download
    timer.begin("gpu_s3_download")
    logger.info("Downloading from S3 directly to GPU container")
    s3_client.download_file(bucket, s3_key, str(video_path))

    try:
        s3_key_dir = os.path.dirname(s3_key)

        # Phase 4-7: Parallel Clip Processing
        timer.begin(f"parallel_processing_{len(clips)}_clips")
        logger.info(f"Processing {len(clips)} clips in parallel...")
        # Initialise with None to maintain order without sorting later
        output_clips = [None] * len(clips)
        clip_errors = []

        # Remove max_workers cap to process all clips concurrently for speed
        max_workers = len(clips)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _process_single_clip,
                    str(video_path), clip, index, words,
                    s3_client, bucket, s3_key_dir,
                    font_family, font_color, font_size, add_subtitles,
                    str(base_dir), has_nvenc, tracker,
                ): index
                for index, clip in enumerate(clips)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    clip_data = future.result()
                    output_clips[idx] = clip_data
                    logger.info(f"Clip {idx + 1} completed successfully")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    logger.error(f"Clip {idx + 1} failed: {e}\n{tb}")
                    clip_errors.append(f"Clip {idx + 1}: {e}")
                    continue

        # Filter out failed clips while maintaining order
        final_clips = [c for c in output_clips if c is not None]

        if not final_clips:
            raise RuntimeError(f"All clip processing pipelines failed. Details: {clip_errors}")

        timing = timer.summary()
        return {"status": "success", "clips": final_clips, "timing": timing}

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
    video_title: str | None = None,
) -> None:
    """POST processing results back to the Next.js webhook endpoint with HMAC signature."""
    import hashlib

    payload = {
        "uploaded_file_id": uploaded_file_id,
        "user_id": user_id,
        "status": result.get("status", "failed"),
        "clips": result.get("clips", []),
        "video_title": video_title,
    }
    if "timing" in result:
        payload["timing"] = result["timing"]
    
    # Serialize tightly so both server and client calculate matching digests
    payload_str = json.dumps(payload, separators=(',', ':'))
    
    signature = hmac.new(
        webhook_secret.encode('utf-8'),
        payload_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Secret": webhook_secret,
        "X-Signature": signature,
    }
    
    session = get_http_session()
    
    try:
        resp = session.post(webhook_url, data=payload_str, headers=headers, timeout=30)
        resp.raise_for_status()
        logger.info(f"Webhook delivered: {resp.status_code}")
    except Exception as e:
        logger.error(f"Webhook delivery failed: {e}")


@app.cls(
    gpu="T4",
    timeout=1200,
    scaledown_window=15,          # Aggressive idle shutdown to prevent credit leakage
    max_containers=10,            # Absolute upper bound on simultaneous tasks
    retries=0,                    # Nullify automatic retries on failure (zero wastage)
    secrets=[
        modal.Secret.from_name("clippedai-secret"),
        modal.Secret.from_name("my-gcp-secret"),
    ]
)
class ClippedAI:
    @modal.enter()
    def startup(self):
        """Pre-warm resources during container startup, not first request.
        This runs once when the container boots, before any requests arrive."""
        logger.info("Container starting — pre-warming resources...")
        validate_required_env_vars()
        
        # Verify GPU availability for NVENC
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
            )
            self._has_nvenc = "h264_nvenc" in result.stdout
            logger.info(f"NVENC available: {self._has_nvenc}")
        except Exception:
            self._has_nvenc = False

        # Initialize TalkNet model for local tracking (avoids remote Modal calls)
        import sys
        if "/fast-asd/talknet" not in sys.path:
            sys.path.append("/fast-asd/talknet")
        try:
            import demoTalkNet
            self.talknet_s, self.talknet_DET = demoTalkNet.setup()
            self.demoTalkNet = demoTalkNet
            logger.info("TalkNet models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TalkNet: {e}")
            self.demoTalkNet = None

        logger.info("Container warm and ready.")

    @modal.method()
    def process_clips_gpu(self, request_dict: dict, words: list, clips: list, previous_phases: list):
        request = ProcessVideoRequest(**request_dict)
        # Run the pipeline
        try:
            result = _render_clips_pipeline(
                request.s3_key,
                words,
                clips,
                previous_phases,
                font_family=request.font_family,
                font_color=request.font_color,
                font_size=request.font_size,
                add_subtitles=request.add_subtitles,
                tracker=(self.demoTalkNet, self.talknet_s, self.talknet_DET) if getattr(self, 'demoTalkNet', None) else None,
            )
        except RuntimeError as e:
            logger.error(f"Pipeline failed (RuntimeError): {e}")
            result = {"status": "failed", "clips": [], "error": f"Runtime error: {str(e)}"}
        except ValueError as e:
            logger.error(f"Pipeline failed (ValueError): {e}")
            result = {"status": "failed", "clips": [], "error": f"Invalid input: {str(e)}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Pipeline failed (Network error): {e}")
            result = {"status": "failed", "clips": [], "error": f"Network error: {str(e)}"}
        except Exception as e:
            logger.error(f"Pipeline failed (Unexpected): {e}")
            result = {"status": "failed", "clips": [], "error": f"Unexpected error: {str(e)}"}

        return result

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        auth_token = os.environ.get("AUTH_TOKEN")
        # Validate token format before comparison
        if not auth_token or not token.credentials or len(token.credentials) < 16:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not hmac.compare_digest(token.credentials, auth_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Dispatch execution to CPU wrapper to prevent timeout and GPU locking
        process_video_cpu_wrapper.spawn(request.dict())
        return {"status": "processing_started"}


@app.function(
    timeout=1200, 
    max_containers=20, 
    retries=0, 
    secrets=[
        modal.Secret.from_name("clippedai-secret"),
        modal.Secret.from_name("my-gcp-secret"),
    ]
)
def process_video_cpu_wrapper(request_dict: dict):
    """CPU-only ingestion wrapper. Performs Video Download, Transcription, and LLM selection locally before waking GPU."""
    request = ProcessVideoRequest(**request_dict)
    
    import uuid
    import shutil
    import pathlib
    
    run_id = str(uuid.uuid4())
    timer = PipelineTimer(run_id)
    base_dir = pathlib.Path("/tmp") / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    video_path = base_dir / "input_ingestion.mp4"
    
    s3_client = _create_s3_client()
    bucket = os.environ.get("S3_BUCKET_NAME", S3_BUCKET)
    
    video_title = None
    try:
        timer.begin("cpu_ingestion")
        if request.youtube_url:
            # ── S3 deduplication cache check ──────────────────────────────────
            # Key is now youtube-downloads/{videoId}/original.mp4 (shared across users).
            # If a previous run already downloaded this video, skip Apify entirely.
            _already_in_s3 = False
            try:
                s3_client.head_object(Bucket=bucket, Key=request.s3_key)
                _already_in_s3 = True
                logger.info(
                    f"[S3 Cache] 🟢 Hit — '{request.s3_key}' already exists. "
                    "Skipping Apify download."
                )
            except Exception as _he:
                # 404 / NoSuchKey means not cached yet — proceed with download
                logger.info(f"[S3 Cache] 🔴 Miss — downloading via Apify.")

            if _already_in_s3:
                logger.info("Downloading cached video from S3 to CPU container...")
                s3_client.download_file(bucket, request.s3_key, str(video_path))
            else:
                logger.info("Executing CPU-bound YouTube Apify download...")
                video_title = _download_youtube(request.youtube_url, video_path)
                if not video_title:
                    logger.warning("Could not extract video title from Apify metadata.")
                logger.info("Uploading ingestion artifact to S3...")
                s3_client.upload_file(str(video_path), bucket, request.s3_key)

            request_dict["youtube_url"] = None
        else:
            logger.info("Downloading from S3 to CPU container for transcription")
            try:
                s3_client.download_file(bucket, request.s3_key, str(video_path))
            except Exception as e:
                err_str = str(e)
                if "404" in err_str or "Not Found" in err_str or "NoSuchKey" in err_str:
                    parts = request.s3_key.split("/")
                    # New flat key format: youtube-downloads/{videoId}/original.mp4 → parts[1] is videoId
                    if parts[0] == "youtube-downloads" and len(parts) >= 2:
                        video_id = parts[1]
                        reconstructed_url = f"https://www.youtube.com/watch?v={video_id}"
                        logger.warning(
                            f"S3 key {request.s3_key} returned 404. "
                            f"Re-ingesting from YouTube: {reconstructed_url}"
                        )
                        _download_youtube(reconstructed_url, video_path)
                        logger.info("Re-uploading re-downloaded video to S3")
                        s3_client.upload_file(str(video_path), bucket, request.s3_key)
                    else:
                        raise RuntimeError(
                            f"S3 object not found ({request.s3_key}) and no YouTube URL to recover from. "
                            "Please re-upload the file."
                        ) from e
                else:
                    raise
            
        timer.begin("transcription")
        words = transcribe(str(video_path), request.s3_key)
        
        timer.begin("clip_selection")
        clips = select_clips(words)
        
        # Flush timer to get accurate timings for CPU phases
        timer._flush()
        
    except Exception as e:
        logger.error(f"CPU Wrapper failed: {e}")
        if request.webhook_url:
            _send_webhook(
                request.webhook_url, request.webhook_secret,
                request.uploaded_file_id, request.user_id,
                {"status": "failed", "clips": [], "error": f"CPU Pipeline error: {str(e)}"},
                video_title
            )
        return
    finally:
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

    # Trigger GPU pipeline
    try:
        logger.info(f"Delegating to GPU Pipeline (passing {len(words)} words, {len(clips)} clips)...")
        result = ClippedAI().process_clips_gpu.remote(request_dict, words, clips, timer.phases)
    except Exception as e:
        logger.error(f"GPU pipeline delegation failed: {e}")
        result = {"status": "failed", "clips": [], "error": f"GPU Wrapper error: {str(e)}"}

    if request.webhook_url and request.webhook_secret and request.uploaded_file_id and request.user_id:
        _send_webhook(
            request.webhook_url,
            request.webhook_secret,
            request.uploaded_file_id,
            request.user_id,
            result,
            video_title,
        )


@app.local_entrypoint()
def run_cli_job(s3_key: str, youtube_url: str = None):
    print(f"Submitting job to CPU wrapper for s3_key: {s3_key}")
    request_dict = ProcessVideoRequest(
        s3_key=s3_key, 
        youtube_url=youtube_url, 
        output_format="vertical"
    ).model_dump()
    process_video_cpu_wrapper.remote(request_dict)

@app.local_entrypoint()
def main():
    pass
