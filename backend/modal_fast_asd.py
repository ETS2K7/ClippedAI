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
        "scenedetect[opencv]==0.5.6.1",
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
        os.chdir("/fast-asd/talknet")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(video_bytes)
            tf_path = tf.name

        try:
            results = self.demoTalkNet.main(
                s=self.s,
                DET=self.DET,
                video_path=tf_path,
                start_seconds=0,
                end_seconds=-1,
                return_visualization=False,
                in_memory_threshold=0,  # Force disk processing for stability
            )
            return json.dumps(results)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)
