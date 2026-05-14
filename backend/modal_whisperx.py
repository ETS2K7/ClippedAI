import modal
import os
import json

# Isolated image for WhisperX to avoid dependency conflicts with the main app
whisperx_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "wget")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
    )
    .pip_install(
        "transformers",
        "ffmpeg-python",
        "pandas",
        "scipy",
        "setuptools",
        "python-dotenv",
    )
    .run_commands("pip install git+https://github.com/m-bain/whisperX.git")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("config.py", remote_path="/root/config.py")
)

app = modal.App("whisperx-worker", image=whisperx_image)

@app.cls(gpu="A10G", timeout=600, scaledown_window=60)
class WhisperXWorker:
    @modal.enter()
    def setup(self):
        from src.transcriber_whisperx import load_whisperx_model
        self.device = "cuda"
        self.model = load_whisperx_model(self.device)
        print(f"WhisperXWorker ready on {self.device}")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, hf_token: str) -> str:
        from src.transcriber_whisperx import transcribe_whisperx
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(audio_bytes)
            audio_path = tf.name
            
        try:
            words = transcribe_whisperx(audio_path, self.model, hf_token, self.device)
            return json.dumps(words)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
