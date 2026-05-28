import modal
import os

app = modal.App("test-secrets")

@app.function(secrets=[modal.Secret.from_name("my-gcp-secret"), modal.Secret.from_name("clippedai-secret")])
def print_keys():
    print([k for k in os.environ.keys() if "GCP" in k or "GOOGLE" in k or "VERTEX" in k])

