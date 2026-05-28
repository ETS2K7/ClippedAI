import modal
import os
import json
from google.oauth2 import service_account
from google import genai
from google.genai import types

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "google-genai==0.3.0",
        "google-auth==2.28.1",
        "requests"
    )
)

app = modal.App("test-vertex", image=image)

@app.function(secrets=[modal.Secret.from_name("my-gcp-secret")])
def test_vertex():
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not gcp_json:
        print("NO GCP JSON")
        return
    try:
        credentials = service_account.Credentials.from_service_account_info(json.loads(gcp_json))
        client = genai.Client(vertexai=True, project="clippedai-493912", location="us-central1", credentials=credentials)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello"
        )
        print("VERTEX SUCCESS:", response.text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("VERTEX FAILED:", str(e))

