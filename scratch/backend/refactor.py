import re

with open('main.py', 'r') as f:
    content = f.read()

# 1. Update GPU
content = content.replace('gpu="any",', 'gpu="T4",')

# 2. Refactor _process_video_pipeline signature and logic
# It currently has: def _process_video_pipeline(\n    s3_key: str,\n    youtube_url: str | None = None,\n    font_family: str | None = None,\n    font_color: str | None = None,\n    font_size: int | None = None,\n    add_subtitles: bool = True,\n) -> dict:
# We'll replace it with def _render_clips_pipeline(...):
old_pipeline_def = """def _process_video_pipeline(
    s3_key: str,
    youtube_url: str | None = None,
    font_family: str | None = None,
    font_color: str | None = None,
    font_size: int | None = None,
    add_subtitles: bool = True,
) -> dict:"""

new_pipeline_def = """def _render_clips_pipeline(
    s3_key: str,
    words: list,
    clips: list,
    previous_phases: list,
    font_family: str | None = None,
    font_color: str | None = None,
    font_size: int | None = None,
    add_subtitles: bool = True,
) -> dict:"""

content = content.replace(old_pipeline_def, new_pipeline_def)

# Inside _render_clips_pipeline, replace the ingestion and transcription phases:
# The old code has:
old_ingestion = """    # Phase 1: Video Ingestion
    timer.begin("ingestion")
    logger.info(f"Resolving input source for s3_key={s3_key}")
    if youtube_url:
        logger.info("Downloading YouTube video")
        _download_youtube(youtube_url, video_path)
        logger.info("Uploading downloaded video to S3")
        s3_client.upload_file(str(video_path), bucket, s3_key)
    else:
        # Try downloading from S3; if missing (e.g. prior upload crash), attempt re-ingestion
        logger.info("Downloading from S3 (Transfer Acceleration)")
        try:
            s3_client.download_file(bucket, s3_key, str(video_path))
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "Not Found" in err_str or "NoSuchKey" in err_str:
                # The S3 object doesn't exist — likely a prior failed upload.
                # If this is a YouTube key, reconstruct the URL and re-ingest.
                parts = s3_key.split("/")
                # youtube-downloads/<userId>-<ts>/<videoId>/original.mp4
                if parts[0] == "youtube-downloads" and len(parts) >= 3:
                    video_id = parts[2]
                    reconstructed_url = f"https://www.youtube.com/watch?v={video_id}"
                    logger.warning(
                        f"S3 key {s3_key} returned 404. "
                        f"Re-ingesting from YouTube: {reconstructed_url}"
                    )
                    _download_youtube(reconstructed_url, video_path)
                    logger.info("Re-uploading re-downloaded video to S3")
                    s3_client.upload_file(str(video_path), bucket, s3_key)
                else:
                    raise RuntimeError(
                        f"S3 object not found ({s3_key}) and no YouTube URL to recover from. "
                        "Please re-upload the file."
                    ) from e
            else:
                raise

    try:
        # Phase 2: Transcription
        timer.begin("transcription")
        words = transcribe(str(video_path), s3_key)

        # Phase 3: LLM Clip Selection
        timer.begin("clip_selection")
        clips = select_clips(words)
        s3_key_dir = os.path.dirname(s3_key)"""

new_ingestion = """    timer.phases = previous_phases
    
    # Phase 3.5: GPU S3 Download
    timer.begin("gpu_s3_download")
    logger.info("Downloading from S3 directly to GPU container")
    s3_client.download_file(bucket, s3_key, str(video_path))

    try:
        s3_key_dir = os.path.dirname(s3_key)"""

content = content.replace(old_ingestion, new_ingestion)

# 3. Update ClippedAI class process_video_cli and process_clips_gpu
old_methods = """    @modal.method()
    def process_video_cli(self, s3_key: str, youtube_url: str = None):
        return _process_video_pipeline(s3_key, youtube_url)

    @modal.method()
    def process_clips_gpu(self, request_dict: dict):
        request = ProcessVideoRequest(**request_dict)
        # Run the pipeline
        try:
            result = _process_video_pipeline(
                request.s3_key,
                request.youtube_url,
                font_family=request.font_family,
                font_color=request.font_color,
                font_size=request.font_size,
                add_subtitles=request.add_subtitles,
            )"""

new_methods = """    @modal.method()
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
            )"""

content = content.replace(old_methods, new_methods)

# 4. Refactor process_video_cpu_wrapper
old_cpu_wrapper = """def process_video_cpu_wrapper(request_dict: dict):
    \"\"\"CPU-only ingestion wrapper. Downloads YouTube natively without holding a GPU hostage.\"\"\"
    request = ProcessVideoRequest(**request_dict)
    
    if request.youtube_url:
        logger.info("Executing CPU-bound YouTube Apify download...")
        import uuid
        import shutil
        import pathlib
        
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        video_path = base_dir / "input_ingestion.mp4"
        
        try:
            _download_youtube(request.youtube_url, video_path)
            
            s3_client = _create_s3_client()
            bucket = os.environ.get("S3_BUCKET_NAME", S3_BUCKET)
            logger.info("Uploading ingestion artifact to S3...")
            s3_client.upload_file(str(video_path), bucket, request.s3_key)
            
            # Nullify so GPU just downloads directly from S3 natively
            request_dict["youtube_url"] = None
        except Exception as e:
            logger.error(f"Ingestion wrapper failed: {e}")
            if request.webhook_url:
                _send_webhook(
                    request.webhook_url, request.webhook_secret,
                    request.uploaded_file_id, request.user_id,
                    {"status": "failed", "clips": [], "error": f"CPU Ingestion error: {str(e)}"}
                )
            return
        finally:
            if base_dir.exists():
                shutil.rmtree(base_dir, ignore_errors=True)

    # Trigger GPU pipeline
    try:
        logger.info("Delegating to GPU Pipeline...")
        result = ClippedAI().process_clips_gpu.remote(request_dict)
    except Exception as e:"""

new_cpu_wrapper = """def process_video_cpu_wrapper(request_dict: dict):
    \"\"\"CPU-only ingestion wrapper. Performs Video Download, Transcription, and LLM selection locally before waking GPU.\"\"\"
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
    
    try:
        timer.begin("cpu_ingestion")
        if request.youtube_url:
            logger.info("Executing CPU-bound YouTube Apify download...")
            _download_youtube(request.youtube_url, video_path)
            
            logger.info("Uploading ingestion artifact to S3...")
            s3_client.upload_file(str(video_path), bucket, request.s3_key)
            request_dict["youtube_url"] = None
        else:
            logger.info("Downloading from S3 to CPU container for transcription")
            s3_client.download_file(bucket, request.s3_key, str(video_path))
            
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
                {"status": "failed", "clips": [], "error": f"CPU Pipeline error: {str(e)}"}
            )
        return
    finally:
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

    # Trigger GPU pipeline
    try:
        logger.info(f"Delegating to GPU Pipeline (passing {len(words)} words, {len(clips)} clips)...")
        result = ClippedAI().process_clips_gpu.remote(request_dict, words, clips, timer.phases)
    except Exception as e:"""

content = content.replace(old_cpu_wrapper, new_cpu_wrapper)

# 5. Update run_cli_job
old_cli = """@app.local_entrypoint()
def run_cli_job(s3_key: str, youtube_url: str = None):
    print(f"Submitting job to Modal for s3_key: {s3_key}")
    ClippedAI().process_video_cli.remote(s3_key, youtube_url)"""

new_cli = """@app.local_entrypoint()
def run_cli_job(s3_key: str, youtube_url: str = None):
    print(f"Submitting job to CPU wrapper for s3_key: {s3_key}")
    request_dict = ProcessVideoRequest(
        s3_key=s3_key, 
        youtube_url=youtube_url, 
        output_format="vertical"
    ).model_dump()
    process_video_cpu_wrapper.remote(request_dict)"""

content = content.replace(old_cli, new_cli)

with open('main_new.py', 'w') as f:
    f.write(content)

print("Done generating main_new.py")
