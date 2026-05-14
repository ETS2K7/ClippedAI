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

# Distributed Caching
transcript_cache = modal.Dict.from_name("clippedai-transcript-cache", create_if_missing=True)
asd_cache = modal.Dict.from_name("clippedai-asd-cache", create_if_missing=True)
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

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "clippedai-7137")
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")

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
    specific_moments: str | None = None
    timeframe_start: float | None = None
    timeframe_end: float | None = None

image = (modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "libsm6", "libxext6", "wget", "git", "fontconfig"])
    .pip_install_from_requirements("requirements.txt")
    .pip_install("apify-client")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("fonts", remote_path="/usr/share/fonts/truetype/custom", copy=True)
    .run_commands(["fc-cache -fv"])
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
        "preferQuality": "1080p",
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
    bucket: str,
    s3_key_dir: str,
    tracking_data: list,
    audio_path: str,
    font_family: str | None = None,
    font_color: str | None = None,
    font_size: int | None = None,
    add_subtitles: bool = True,
    work_dir: str = "",
    use_gpu: bool = False,
    caption_template: str | None = None,
) -> dict:
    import boto3
    s3_client = boto3.client("s3")
    """
    Processes a single clip through the full sub-pipeline:
    extract -> track -> subtitle -> merge -> S3 upload.
    Returns a dict with {"s3Key": ..., "thumbnailKey": ...}.
    """

    clip_out_path = f"{work_dir}/clip_{index}.mp4"
    logger.info(f"--- Processing Clip {index + 1} (parallel streaming) | caption_template={caption_template} ---")
    ext_vid = extract_segment(video_path, clip, index, work_dir, use_gpu=use_gpu)
    
    # Unified single-pass hardware streaming export bypasses CPU frame compression and disk bottlenecks
    trk_vid, chunk_meta = track_speaker_and_frame(
        ext_vid, index, clip, words, work_dir,
        tracking_data=tracking_data,
        audio_file=audio_path,
        remote_cache=asd_cache,
        streaming_output_path=clip_out_path,
        font_family=font_family,
        font_size=font_size,
        font_color=font_color,
        add_subtitles=add_subtitles,
        use_gpu=use_gpu,
    )
    
    # Clean up intermediate segment extraction file
    try: os.remove(ext_vid)
    except: pass
    # Append a short timestamp to bust the browser's immutable cache on Force Retry
    cache_buster = int(time.time())
    output_s3_key = f"{s3_key_dir}/clip_{index}_{cache_buster}.mp4"

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
    run_id: str,
    base_dir: pathlib.Path,
    font_family: str | None = None,
    font_color: str | None = None,
    font_size: int | None = None,
    add_subtitles: bool = True,
    caption_template: str | None = None,
    tracking_data: list | None = None,
    audio_path: str | None = None,
    video_path_override: str | None = None,
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
    from concurrent.futures import ProcessPoolExecutor, as_completed

    run_id = str(uuid.uuid4())
    timer = PipelineTimer(run_id)
    base_dir = pathlib.Path("/tmp") / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    # Detect NVENC at pipeline scope so it's available for clip workers
    try:
        nvenc_result = subprocess.run(
            ["/usr/bin/ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
        )
        has_nvenc = "h264_nvenc" in nvenc_result.stdout
    except Exception:
        has_nvenc = False
    logger.info(f"NVENC detection in pipeline: {has_nvenc}")

    video_path = pathlib.Path(video_path_override) if video_path_override else base_dir / "input.mp4"
    s3_client = _create_s3_client()
    bucket = os.environ.get("S3_BUCKET_NAME", S3_BUCKET)

    timer.phases = previous_phases
    
    # Phase 3.5: GPU S3 Download (Skip if we have a local proxy)
    if not video_path_override:
        timer.begin("gpu_s3_download")
        logger.info("Downloading from S3 directly to GPU container")
        s3_client.download_file(bucket, s3_key, str(video_path))
    else:
        logger.info("Using local proxy for rendering, skipping S3 download.")
    
    # Phase 4-7: Parallel Clip Processing
    if not clips:
        logger.info("No clips to process. Skipping rendering.")
        return {"status": "success", "clips": [], "message": "No viral clips found."}

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
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _process_single_clip,
                    str(video_path), clip, index, words,
                    bucket, s3_key_dir, tracking_data, str(audio_path),
                    font_family, font_color, font_size, add_subtitles,
                    str(base_dir), has_nvenc, caption_template,
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
    gpu="A10G",
    timeout=600,
    scaledown_window=15,    # Kill GPU instantly if idle
    max_containers=5,             # Conservative bound to protect credits
    retries=0,
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
                ["/usr/bin/ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
            )
            self._has_nvenc = "h264_nvenc" in result.stdout
            logger.info(f"NVENC available: {self._has_nvenc}")
        except Exception:
            self._has_nvenc = False
        
        # Pre-resolve worker classes for instant access in run_pipeline
        self.whisperx_cls = modal.Cls.from_name("whisperx-worker", "WhisperXWorker")
        self.asd_cls = modal.Cls.from_name("fast-asd-tracker", "FastASDTracker")
        self.this_cls = modal.Cls.from_name("clippedai", "ClippedAI")
        
        logger.info("Container warm and ready.")

    @modal.method()
    def run_pipeline(self, request_dict: dict):
        """
        Merged GPU pipeline: download -> WhisperX transcribe -> Gemini select -> render -> webhook.
        Runs entirely on the warm GPU container. No inter-container handoff.
        """
        import shutil
        import pathlib
        
        # Ensure worker classes are resolved (fallback if startup failed)
        if not hasattr(self, "asd_cls"):
            self.asd_cls = modal.Cls.from_name("fast-asd-tracker", "FastASDTracker")
        if not hasattr(self, "whisperx_cls"):
            self.whisperx_cls = modal.Cls.from_name("whisperx-worker", "WhisperXWorker")
        
        request = ProcessVideoRequest(**request_dict)
        run_id = str(uuid.uuid4())
        timer = PipelineTimer(run_id)
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            video_path = base_dir / "input.mp4"
            video_title = "Untitled Video"
            s3_client = _create_s3_client()
            bucket = os.environ.get("S3_BUCKET", "clipped-ai-storage")
            
            # Phase 1: Ingestion
            if request.youtube_url:
                timer.begin("ingestion")
                logger.info(f"Phase 1: Downloading from YouTube: {request.youtube_url}")
                video_title = _download_youtube(request.youtube_url, video_path)
            elif request.s3_key:
                timer.begin("ingestion")
                logger.info(f"Phase 1: Ingesting from S3: {request.s3_key}")
                s3_client.download_file(bucket, request.s3_key, str(video_path))
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found at {video_path}")
            
            # Check for existing proxy
            proxy_path = base_dir / "proxy_1080p.mp4"
            video_path_override = None
            if request.s3_key:
                s3_key_dir = os.path.dirname(request.s3_key)
                potential_proxy_key = os.path.join(s3_key_dir, "proxy_1080p.mp4")
                try:
                    s3_client.head_object(Bucket=bucket, Key=potential_proxy_key)
                    logger.info(f"Found existing proxy at {potential_proxy_key}, downloading...")
                    s3_client.download_file(bucket, potential_proxy_key, str(proxy_path))
                    video_path_override = str(proxy_path)
                except: pass

            # ── Phase 1.5: Triple Parallel Spawn ──────────────────────────────
            timer.begin("parallel_spawn")
            logger.info("Spawning parallel ASD, Transcription, and Proxy Generation...")
            
            lowres_video_path = base_dir / "input_360p.mp4"
            subprocess.run([
                "/usr/bin/ffmpeg", "-y", "-i", str(video_path),
                "-vf", "scale=-2:360",
                "-c:v", "h264_nvenc", "-preset", "p1", "-qp", "28",
                "-an", str(lowres_video_path)
            ], check=True)

            with open(lowres_video_path, "rb") as f:
                video_bytes_lowres = f.read()
            
            asd_call = self.asd_cls().process_video.spawn(video_bytes_lowres)

            rendering_audio_path = base_dir / "global_audio.aac"
            if not video_path_override:
                proxy_proc = subprocess.Popen([
                    "/usr/bin/ffmpeg", "-y", "-i", str(video_path),
                    "-c:v", "h264_nvenc", "-preset", "p1", "-qp", "19",
                    "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "0:a:0?",
                    str(proxy_path)
                ])
                video_path_override = str(proxy_path)
            else:
                proxy_proc = None

            # ── Phase 2: Transcription ───────────────────────────────────────
            timer.begin("transcription")
            transcription_audio_path = base_dir / "input_audio.wav"
            subprocess.run([
                "/usr/bin/ffmpeg", "-y", "-i", str(video_path), 
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", 
                str(transcription_audio_path)
            ], check=True)
            
            with open(transcription_audio_path, "rb") as f:
                audio_bytes = f.read()
            
            words_json = self.whisperx_cls().transcribe.remote(audio_bytes, os.environ.get("HF_TOKEN", ""))
            words = json.loads(words_json)

            # ── Phase 3: Selection ───────────────────────────────────────────
            timer.begin("clip_selection")
            clips = select_clips(words, request.specific_moments)
            
            # ── Phase 3.7: Parallel Wait ─────────────────────────────────────
            timer.begin("parallel_wait")
            tracking_data_json = asd_call.get()
            tracking_data = json.loads(tracking_data_json)
            
            if proxy_proc:
                proxy_proc.wait()
            
            subprocess.run([
                "/usr/bin/ffmpeg", "-y", "-i", str(proxy_path),
                "-vn", "-acodec", "copy", str(rendering_audio_path)
            ], check=True)

            # ── Phase 4-7: GPU Rendering ─────────────────────────────────────
            logger.info(f"Starting GPU rendering ({len(words)} words, {len(clips)} clips)...")
            result = _render_clips_pipeline(
                request.s3_key or f"uploads/{run_id}/input.mp4", words, clips, timer.phases,
                run_id=run_id, base_dir=base_dir,
                video_path_override=video_path_override,
                tracking_data=tracking_data,
                audio_path=str(rendering_audio_path),
                font_family=request.font_family,
                font_color=request.font_color,
                font_size=request.font_size,
                add_subtitles=request.add_subtitles,
                caption_template=request.caption_template,
            )
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result = {"status": "failed", "clips": [], "error": f"Pipeline error: {str(e)}"}
        finally:
            if base_dir.exists():
                shutil.rmtree(base_dir, ignore_errors=True)

        if request.webhook_url and request.webhook_secret and request.uploaded_file_id and request.user_id:
            _send_webhook(
                request.webhook_url, request.webhook_secret,
                request.uploaded_file_id, request.user_id,
                result, video_title,
            )

@app.cls(
    secrets=[modal.Secret.from_name("clippedai-secret")]
)
class ClippedAIDownloader:
    @modal.method()
    def _download_and_trigger(self, request_dict: dict):
        """CPU task to handle external downloads before waking up the GPU."""
        import pathlib
        import uuid
        import shutil
        
        request = ProcessVideoRequest(**request_dict)
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            video_path = base_dir / "input.mp4"
            video_title = "Untitled Video"
            
            # Download on CPU
            if request.youtube_url:
                logger.info(f"CPU Worker: Downloading from YouTube: {request.youtube_url}")
                video_title = _download_youtube(request.youtube_url, video_path)
            
            # Upload to S3 so GPU can pick it up
            s3_client = _create_s3_client()
            bucket = os.environ.get("S3_BUCKET", "clipped-ai-storage")
            s3_key = f"uploads/{run_id}/input.mp4"
            logger.info(f"CPU Worker: Uploading to S3: {s3_key}")
            s3_client.upload_file(str(video_path), bucket, s3_key)
            
            # Now trigger the GPU pipeline
            new_request = request_dict.copy()
            new_request["s3_key"] = s3_key # Override with the S3 key we just uploaded
            new_request["youtube_url"] = None # Avoid re-downloading
            
            gpu_cls = modal.Cls.from_name("clippedai", "ClippedAI")()
            gpu_cls.run_pipeline.spawn(new_request)
            logger.info(f"CPU download complete. GPU pipeline triggered for {run_id}")
            
        except Exception as e:
            logger.error(f"CPU download phase failed: {e}")
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

@app.cls(
    secrets=[modal.Secret.from_name("clippedai-secret")]
)
class ClippedAIWeb:
    @modal.fastapi_endpoint(method="POST")
    def warmup(self, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        """Wakes up the GPU container and all worker containers to eliminate cold starts."""
        auth_token = os.environ.get("AUTH_TOKEN")
        if not auth_token or not token.credentials or not hmac.compare_digest(token.credentials, auth_token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        logger.info("Warmup request received. Triggering all workers...")
        
        # Non-blocking triggers to speed up response
        try: modal.Cls.from_name("clippedai", "ClippedAI")().startup.spawn()
        except: pass

        try: modal.Cls.from_name("whisperx-worker", "WhisperXWorker")().transcribe.spawn(b"", "")
        except: pass

        try: modal.Cls.from_name("fast-asd-tracker", "FastASDTracker")().process_video.spawn(b"")
        except: pass

        return {"status": "warming_up"}

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        auth_token = os.environ.get("AUTH_TOKEN")
        if not auth_token or not token.credentials or not hmac.compare_digest(token.credentials, auth_token):
            raise HTTPException(status_code=401, detail="Invalid token")

        logger.info(f"API request received for URL: {request.youtube_url or 'S3/Local'}")
        
        if request.youtube_url:
            # Save credits: Use CPU for the download phase
            downloader = modal.Cls.from_name("clippedai", "ClippedAIDownloader")()
            downloader._download_and_trigger.spawn(request.dict())
            logger.info("Spawned CPU download task via ClippedAIDownloader.")
        else:
            # Local file or S3 key: Go straight to GPU
            gpu_cls = modal.Cls.from_name("clippedai", "ClippedAI")()
            gpu_cls.run_pipeline.spawn(request.dict())
            logger.info("Spawned GPU pipeline task via ClippedAI.run_pipeline")
            
        return {"status": "processing_started"}

