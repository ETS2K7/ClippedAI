"""
Pipeline Step 2 — Transcribe (GPU)
WhisperX large-v2 with word-level timestamp alignment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WHISPERX_MODEL, WHISPERX_BATCH_SIZE, WHISPERX_COMPUTE_TYPE


def transcribe(audio_path: str) -> dict:
    """
    Transcribe audio using WhisperX on GPU.

    Returns:
        {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.52,
                    "text": "Hey guys welcome back",
                    "words": [
                        {"word": "Hey", "start": 0.0, "end": 0.28},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  🎙️ Loading WhisperX {WHISPERX_MODEL} on {device}...")

    # Load model
    model = whisperx.load_model(
        WHISPERX_MODEL,
        device,
        compute_type=WHISPERX_COMPUTE_TYPE,
    )

    # Transcribe
    print(f"  🎙️ Transcribing...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=WHISPERX_BATCH_SIZE)

    detected_language = result.get("language", "en")
    print(f"  🎙️ Detected language: {detected_language}")

    # Align for word-level timestamps
    print(f"  🎙️ Aligning word timestamps...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Clean up GPU memory
    del model, align_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Build clean output
    segments = []
    for seg in aligned["segments"]:
        words = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })

        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
            "words": words,
        })

    print(f"  🎙️ Transcription complete: {len(segments)} segments")
    return {
        "language": detected_language,
        "segments": segments,
    }


def get_words_in_range(transcript: dict, start: float, end: float) -> list:
    """
    Extract word-level timestamps for a specific time range.
    Used to generate captions for a clip.

    Returns: [{"word": "Hey", "start": 0.0, "end": 0.28}, ...]
    """
    words = []
    for seg in transcript["segments"]:
        # Skip segments entirely outside range
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in seg.get("words", []):
            if w["start"] >= start and w["end"] <= end:
                # Adjust timestamps relative to clip start
                words.append({
                    "word": w["word"],
                    "start": round(w["start"] - start, 3),
                    "end": round(w["end"] - start, 3),
                })
    return words
