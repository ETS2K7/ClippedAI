import urllib.request
import urllib.error
import json
import time

url = "https://ebel-p25--clippedai-clippedai-process-video.modal.run"
headers = {
    "Authorization": "Bearer 123123",
    "Content-Type": "application/json"
}
payload = {
    "s3_key": f"test-clip-prod-{int(time.time())}",
    "youtube_url": "https://www.youtube.com/watch?v=YGOTBpTScR0",
    "font_family": "Arial",
    "font_color": "#FFD700",
    "font_size": 45
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers=headers, method="POST")

print(f"Submitting E2E test job: {payload['s3_key']}")
start = time.time()

try:
    with urllib.request.urlopen(req, timeout=900) as response:
        print(f"Status Code: {response.status}")
        print(f"Elapsed: {time.time() - start:.2f}s")
        print("Response Body:")
        print(response.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print("Response Body:")
    print(e.read().decode("utf-8"))
except Exception as e:
    print(f"Error: {e}")
