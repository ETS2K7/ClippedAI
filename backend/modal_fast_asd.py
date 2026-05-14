import modal
import os
import json
import tempfile
import sys

app = modal.App("fast-asd-tracker")

fast_asd_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6", "wget")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        "opencv-python",
        "scipy",
        "ffmpeg-python",
        "numpy==1.23.5",
        "gdown",
        "python_speech_features",
        "scenedetect[opencv]<0.6.0",
        "tqdm",
        "pandas",
    )
    .run_commands(
        "git clone https://github.com/sieve-community/fast-asd.git /fast-asd",
        "mkdir -p /root/.cache/models",
        "mkdir -p /root/model/faceDetector/s3fd",
        "gdown 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea -O /root/.cache/models/pretrain_TalkSet.model",
        "wget -O /root/model/faceDetector/s3fd/sfd_face.pth https://storage.googleapis.com/mango-public-models/sfd_face.pth",
        "ln -s /fast-asd/talknet/model /root/model_symlink",  # Just in case any other relative paths are needed
    )
)


@app.cls(
    image=fast_asd_image, 
    gpu="A10G", 
    timeout=1200,
    scaledown_window=15,
    max_containers=10,
    retries=0
)
class FastASDTracker:
    @modal.enter()
    def setup(self):
        os.chdir("/fast-asd/talknet")
        if "/fast-asd/talknet" not in sys.path:
            sys.path.append("/fast-asd/talknet")
        import demoTalkNet

        self.demoTalkNet = demoTalkNet
        self.s, self.DET = demoTalkNet.setup()

    @modal.method()
    def process_video(self, video_bytes: bytes) -> str:
        import shutil
        import uuid
        
        request_id = str(uuid.uuid4())[:8]
        req_cwd = f"/tmp/asd_cwd_{request_id}"
        
        # Create isolated structure for this specific request
        os.makedirs(os.path.join(req_cwd, "save", "pycrop"), exist_ok=True)
        
        # Symlink the model folder needed by demoTalkNet instead of copying it
        if os.path.exists("/fast-asd/talknet/model"):
            os.symlink("/fast-asd/talknet/model", os.path.join(req_cwd, "model"))
        
        # Write input video to a path inside the isolated request directory
        tf_path = os.path.join(req_cwd, "input.mp4")
        with open(tf_path, "wb") as f:
            f.write(video_bytes)
        
        prev_cwd = os.getcwd()
        os.chdir(req_cwd)

        try:
            results = self.demoTalkNet.main(
                s=self.s,
                DET=self.DET,
                video_path="input.mp4",
                start_seconds=0,
                end_seconds=-1,
                return_visualization=False,
                in_memory_threshold=0,
            )
            return json.dumps(results)
        except Exception as e:
            # Handle edge cases where no speakers are found gracefully
            if "broadcast" in str(e) or "pycrop" in str(e) or "negative values" in str(e):
                print(f"Warning: Fast-ASD found no speakers in this segment: {e}")
                return json.dumps([])
            raise e
        finally:
            os.chdir(prev_cwd)
            # Full cleanup of the isolated directory
            shutil.rmtree(req_cwd, ignore_errors=True)
