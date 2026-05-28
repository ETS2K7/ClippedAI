"""
Integration tests for the backend video processing pipeline glue logic.
These tests verify that the orchestration functions call the right components in the right order.
"""

import os
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Ensure API keys are set to dummy values for tests to pass lazy validation."""
    os.environ["AWS_ACCESS_KEY_ID"] = "dummy"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "dummy"
    os.environ["ASSEMBLYAI_KEY"] = "dummy"
    os.environ["GEMINI_KEY"] = "dummy"
    yield


@patch("main._create_s3_client")
@patch("main.extract_segment")
@patch("main.track_speaker_and_frame")
@patch("subprocess.run")
@patch("os.remove")
def test_render_clips_pipeline_success(
    mock_os_remove,
    mock_sub_run,
    mock_track,
    mock_extract,
    mock_create_s3,
):
    from main import _render_clips_pipeline

    # Mock the S3 client
    mock_s3 = MagicMock()
    mock_create_s3.return_value = mock_s3

    # Mock FFmpeg detection
    mock_ffmpeg_res = MagicMock()
    mock_ffmpeg_res.stdout = "h264_nvenc"
    mock_sub_run.return_value = mock_ffmpeg_res

    # Mock intermediate video processing steps
    mock_extract.side_effect = ["extracted_1.mp4", "extracted_2.mp4"]
    mock_track.side_effect = [("tracked_1.mp4", []), ("tracked_2.mp4", [])]

    # Run the pipeline
    s3_key = "user-123/original.mp4"
    words = [{"start": 0, "end": 1000, "text": "hello", "speaker": "A"}]
    clips = [
        {"start_time": 0, "end_time": 30, "title": "Clip 1", "virality_score": 9.5},
        {"start_time": 30, "end_time": 60, "title": "Clip 2", "virality_score": 8.0},
    ]
    previous_phases = []

    result = _render_clips_pipeline(
        s3_key=s3_key,
        words=words,
        clips=clips,
        previous_phases=previous_phases,
        add_subtitles=True,
    )

    # Assertions
    assert result["status"] == "success"
    assert len(result["clips"]) == 2

    # Check S3 downloads
    mock_s3.download_file.assert_called_once()
    assert mock_s3.upload_file.call_count == 2


@patch("main._create_s3_client")
@patch("main._download_youtube")
@patch("main.transcribe")
@patch("main.select_clips")
@patch("main.ClippedAI")
@patch("main._send_webhook")
def test_process_video_cpu_wrapper_success(
    mock_send_webhook,
    mock_clipped_ai,
    mock_select_clips,
    mock_transcribe,
    mock_download_yt,
    mock_create_s3,
):
    from main import process_video_cpu_wrapper

    # Mock the S3 client to simulate a Cache Miss
    mock_s3 = MagicMock()
    mock_s3.head_object.side_effect = Exception("NoSuchKey")
    mock_create_s3.return_value = mock_s3

    # Mock transcribe and select_clips
    mock_transcribe.return_value = [{"start": 0, "end": 1000, "text": "hello"}]
    mock_select_clips.return_value = [{"start_time": 0, "end_time": 30}]

    # Mock ClippedAI remote call
    mock_gpu_instance = MagicMock()
    mock_clipped_ai.return_value = mock_gpu_instance
    mock_gpu_instance.process_clips_gpu.remote.return_value = {"status": "success", "clips": []}

    request_dict = {
        "s3_key": "user-123/original.mp4",
        "youtube_url": "https://www.youtube.com/watch?v=123",
        "uploaded_file_id": "file-123",
        "user_id": "user-123",
        "webhook_url": "http://localhost/webhook",
        "webhook_secret": "secret",
        "add_subtitles": True,
        "output_format": "vertical",
    }

    # Execute Modal function locally using .local()
    process_video_cpu_wrapper.local(request_dict)

    # Verify Youtube download and transcription were triggered
    mock_download_yt.assert_called_once()
    mock_transcribe.assert_called_once()
    mock_select_clips.assert_called_once()
    mock_gpu_instance.process_clips_gpu.remote.assert_called_once()
    mock_send_webhook.assert_called_once()
