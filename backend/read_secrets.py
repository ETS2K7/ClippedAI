"""
⚠️  DEBUG TOOL — DO NOT USE IN CI/CD OR PRODUCTION.
This script prints all Modal secrets to stdout. It exists only for local
development verification. Consider gating behind an explicit --confirm flag
if keeping in the repository.
"""
import modal
import os

app = modal.App("secret-reader")

@app.function(secrets=[modal.Secret.from_name("clippedai-secret")])
def read_secrets():
    return {
        "ASSEMBLYAI_KEY": os.environ.get("ASSEMBLYAI_KEY", ""),
        "GEMINI_KEY": os.environ.get("GEMINI_KEY", ""),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "APIFY_TOKEN": os.environ.get("APIFY_TOKEN", ""),
    }

@app.local_entrypoint()
def main():
    secrets = read_secrets.remote()
    for k, v in secrets.items():
        print(f"{k}={v}")
