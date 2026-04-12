import requests
import json
import time
import os

url = "https://ebel-p25--clippedai-clippedai-process-video.modal.run"
headers = {"Authorization": "Bearer 123123"}
payload = {
    "s3_key": f"test-clip-{int(time.time())}",
    "youtube_url": "https://www.youtube.com/watch?v=Q0BOH_s9gSU",
    "font_family": "Impact",
    "font_color": "#FFD700",
    "font_size": 45
}

print(f"Submitting E2E test job: {payload}")
start = time.time()
try:
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Elapsed: {time.time() - start:.2f}s")
    print("Response Body:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
