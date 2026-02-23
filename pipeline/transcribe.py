"""
WhisperX transcription + Pyannote 3.1 diarization.

Outputs word-level timestamps with speaker labels per chunk.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


def transcribe_chunk(
    audio_path: Path,
    chunk_start: float = 0.0,
    device: str = "cuda",
    num_speakers: Optional[int] = None,
) -> dict:
    """
    Transcribe an audio chunk with WhisperX + Pyannote diarization.

    Returns dict with:
      - segments: list of {start, end, text, speaker, words: [{word, start, end, score}]}
      - language: detected language code
    """
    import whisperx
    import torch

    # Step 1: Load model + transcribe
    logger.info("Transcribing: %s", audio_path.name)
    model = whisperx.load_model(
        config.WHISPERX_MODEL,
        device=device,
        compute_type=config.WHISPERX_COMPUTE_TYPE,
    )

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(
        audio,
        batch_size=config.WHISPERX_BATCH_SIZE,
    )
    language = result.get("language", "en")
    logger.info("Detected language: %s", language)

    # Step 2: Forced alignment for word-level timestamps
    logger.info("Aligning words...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device=device,
        return_char_alignments=False,
    )

    # Free alignment model
    del align_model
    torch.cuda.empty_cache()

    # Step 3: Speaker diarization
    logger.info("Diarizing speakers...")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=config.HF_TOKEN,
        device=device,
    )
    diarize_segments = diarize_model(
        audio,
        min_speakers=1,
        max_speakers=num_speakers or 10,
    )

    # Assign speakers to word-level segments
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Free diarize model
    del diarize_model
    torch.cuda.empty_cache()

    # Step 4: Offset timestamps to global time (chunk_start)
    segments = []
    for seg in result.get("segments", []):
        segment = {
            "start": round(seg["start"] + chunk_start, 3),
            "end": round(seg["end"] + chunk_start, 3),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", "UNKNOWN"),
            "words": [],
        }

        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            word_entry = {
                "word": w["word"].strip(),
                "start": round(w["start"] + chunk_start, 3),
                "end": round(w["end"] + chunk_start, 3),
                "score": round(w.get("score", 0.0), 3),
            }
            segment["words"].append(word_entry)

        if segment["text"]:
            segments.append(segment)

    logger.info(
        "Transcription complete: %d segments, %d words",
        len(segments),
        sum(len(s["words"]) for s in segments),
    )

    return {
        "language": language,
        "segments": segments,
    }


def merge_chunk_transcripts(
    chunk_transcripts: list[dict],
    chunks_meta: list[dict],
) -> dict:
    """
    Merge transcripts from overlapping chunks.
    In overlap regions, keep the transcript from the chunk where
    the words fall in the non-overlap middle section (higher confidence).
    """
    if not chunk_transcripts:
        return {"language": "en", "segments": []}

    language = chunk_transcripts[0].get("language", "en")
    all_segments = []

    for i, (transcript, chunk_meta) in enumerate(
        zip(chunk_transcripts, chunks_meta)
    ):
        chunk_start = chunk_meta["start"]
        chunk_end = chunk_meta["end"]

        # Determine the "trusted" region for this chunk:
        # First chunk: trust from start to (end - overlap/2)
        # Last chunk: trust from (start + overlap/2) to end
        # Middle chunks: trust from (start + overlap/2) to (end - overlap/2)
        overlap = config.CHUNK_OVERLAP

        if i == 0:
            trust_start = chunk_start
        else:
            trust_start = chunk_start + overlap / 2

        if i == len(chunk_transcripts) - 1:
            trust_end = chunk_end
        else:
            trust_end = chunk_end - overlap / 2

        for seg in transcript.get("segments", []):
            # Keep segment if its midpoint falls in the trusted region
            seg_mid = (seg["start"] + seg["end"]) / 2
            if trust_start <= seg_mid <= trust_end:
                all_segments.append(seg)

    # Sort by start time
    all_segments.sort(key=lambda s: s["start"])

    return {
        "language": language,
        "segments": all_segments,
    }


def save_transcript(transcript: dict, output_path: Path) -> Path:
    """Save transcript to JSON."""
    with open(output_path, "w") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info("Saved transcript: %s", output_path)
    return output_path
