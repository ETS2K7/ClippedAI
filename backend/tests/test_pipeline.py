"""
Integration tests for the backend video processing pipeline glue logic.
These tests mock external HTTP requests, APIs, and heavy ML models, but verify
that the orchestration functions call the right components in the right order.
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


@patch("boto3.client")
@patch("main._download_youtube")
@patch("main.transcribe")
@patch("main.select_clips")
@patch("main.extract_segment")
@patch("main.track_speaker_and_frame")
@patch("main.generate_subtitles")
@patch("main.merge_and_cleanup")
@patch("os.remove")
def test_process_video_pipeline_success(
    mock_remove,
    mock_merge,
    mock_generate_subs,
    mock_track,
    mock_extract,
    mock_select_clips,
    mock_transcribe,
    mock_download_yt,
    mock_boto3,
):
    """
    Test the main _process_video_pipeline orchestration logic.
    Ensures that for N returned clips, the video processing functions are called N times.
    """
    from main import _process_video_pipeline

    # Mock the S3 client
    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3

    # Mock Transcriber output
    mock_transcribe.return_value = [
        {"start": 0, "end": 1000, "text": "hello", "speaker": "A"}
    ]

    # Mock LLM output (returning 2 clips)
    mock_select_clips.return_value = [
        {"start_time": 0, "end_time": 30},
        {"start_time": 30, "end_time": 60},
    ]

    # Mock intermediate video processing steps
    mock_extract.side_effect = ["extracted_1.mp4", "extracted_2.mp4"]
    mock_track.side_effect = [("tracked_1.mp4", []), ("tracked_2.mp4", [])]
    mock_generate_subs.side_effect = ["sub_1.ass", "sub_2.ass"]

    # Run the pipeline
    s3_key = "user-123/original.mp4"
    result = _process_video_pipeline(s3_key=s3_key, youtube_url="http://youtube.com/watch?v=123")

    # Assertions
    assert result["status"] == "success"
    assert len(result["clips"]) == 2
    assert result["clips"][0]["s3Key"] == "user-123/clip_0.mp4"
    assert result["clips"][1]["s3Key"] == "user-123/clip_1.mp4"

    # Verify Youtube download was triggered since youtube_url was provided
    mock_download_yt.assert_called_once()

    # Verify Transcriber and LLM were called once
    mock_transcribe.assert_called_once()
    mock_select_clips.assert_called_once()

    # Verify video processing functions were called exactly twice (once per clip)
    assert mock_extract.call_count == 2
    assert mock_track.call_count == 2
    assert mock_generate_subs.call_count == 2
    assert mock_merge.call_count == 2

    # Verify S3 upload was called for each clip + 1 for the original youtube video
    assert mock_s3.upload_file.call_count == 3


@patch("boto3.client")
@patch("main.transcribe")
def test_process_video_pipeline_transcription_failure(
    mock_transcribe,
    mock_boto3,
):
    """
    Test that if transcription fails, the pipeline raises an exception and doesn't
    attempt to process clips.
    """
    from main import _process_video_pipeline

    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3

    # Simulate transcription failing with an error
    mock_transcribe.side_effect = Exception("AssemblyAI Error")

    with pytest.raises(Exception, match="AssemblyAI Error"):
        _process_video_pipeline("test/key.mp4", None)

    # Transcription was attempted
    mock_transcribe.assert_called_once()
