import whisperx
import gc
import torch
from typing import List, Dict, Any
from config import get_logger

logger = get_logger(__name__)

def load_whisperx_model(device: str = "cuda", compute_type: str = "float16"):
    """
    Loads the WhisperX model. Call once at container startup and cache the result.
    """
    logger.info(f"Loading WhisperX large-v3 model on {device} ({compute_type})...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    logger.info("WhisperX model loaded.")
    return model

def transcribe_whisperx(
    audio_path: str, 
    whisperx_model: Any, 
    hf_token: str,
    device: str = "cuda", 
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Transcribes a video file using WhisperX with speaker diarization.
    """
    logger.info("==================== PHASE 2: TRANSCRIPTION (WhisperX) ====================")
    
    # 1. Transcribe with original whisper
    audio = whisperx.load_audio(audio_path)
    result = whisperx_model.transcribe(audio, batch_size=batch_size)
    
    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Delete align model to save VRAM
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Diarization
    logger.info("Running speaker diarization...")
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # 4. Flatten into standard word-level JSON format
    words = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", "SPEAKER_00")
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "text": w["word"],
                    "start": int(w["start"] * 1000),
                    "end": int(w["end"] * 1000),
                    "speaker": speaker,
                    "confidence": w.get("score", 1.0)
                })
                
    logger.info(f"WhisperX transcription complete: {len(words)} words extracted.")
    return words
