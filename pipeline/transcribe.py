"""
ClippedAI — Speech Transcription + Diarization

WhisperX large-v3-turbo + Pyannote 4.0 (community-1 model).
Outputs word-level timestamps + speaker labels.
"""
import json
import os
from pathlib import Path


def transcribe_chunk(chunk_path: str, output_dir: str, hf_token: str) -> dict:
    """
    Transcribe a video chunk with WhisperX + Pyannote 4.0.

    Returns dict with:
        - segments: list of segments with speaker labels
        - word_segments: list of words with start/end/speaker
    """
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Step 1: Transcribe with WhisperX large-v3-turbo
    model = whisperx.load_model(
        "large-v3-turbo",
        device=device,
        compute_type=compute_type,
    )
    audio = whisperx.load_audio(chunk_path)
    result = model.transcribe(audio)

    # Step 2: Forced alignment for word-level timestamps
    align_model, align_metadata = whisperx.load_align_model(
        language_code=result["language"],
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

    # Step 3: Speaker diarization with Pyannote
    # Pre-extract audio to WAV to bypass Pyannote's internal AudioDecoder
    # (avoids torchcodec/torchaudio version compatibility issues)
    import subprocess
    import torchaudio
    from pyannote.audio import Pipeline as PyannotePipeline

    wav_path = chunk_path.rsplit(".", 1)[0] + "_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", chunk_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, check=True,
    )

    diarize_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    if device == "cuda":
        diarize_pipeline.to(torch.device("cuda"))

    # Load audio as waveform and pass as dict to bypass AudioDecoder
    waveform, sample_rate = torchaudio.load(wav_path)
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
    diarize_result = diarize_pipeline(audio_dict)

    # Pyannote 4.0 community-1 returns DiarizeOutput with .speaker_diarization
    # Extract the Annotation and convert to DataFrame for WhisperX
    import pandas as pd

    annotation = diarize_result.speaker_diarization
    diarize_rows = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        diarize_rows.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
        })
    diarize_segments = pd.DataFrame(diarize_rows)

    # Assign speakers to WhisperX segments
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Step 4: Filter hallucinations
    filtered_words = []
    for seg in result.get("segments", []):
        for word in seg.get("words", []):
            # Drop low-confidence words
            if word.get("score", 1.0) < 0.4:
                continue
            # Ensure required fields
            if "start" not in word or "end" not in word or "word" not in word:
                continue
            filtered_words.append({
                "word": word["word"].strip(),
                "start": round(word["start"], 3),
                "end": round(word["end"], 3),
                "score": round(word.get("score", 1.0), 3),
                "speaker": word.get("speaker", "SPEAKER_00"),
            })

    # Drop repeated words within 0.5s
    deduped_words = []
    for w in filtered_words:
        if deduped_words and w["word"] == deduped_words[-1]["word"]:
            if w["start"] - deduped_words[-1]["end"] < 0.5:
                continue
        deduped_words.append(w)

    # Build output
    transcript = {
        "language": result.get("language", "en"),
        "segments": result.get("segments", []),
        "word_segments": deduped_words,
    }

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "transcript.json"
    with open(output_path, "w") as f:
        json.dump(transcript, f, indent=2)

    return transcript
