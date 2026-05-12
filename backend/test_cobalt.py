import urllib.request
import json

req = urllib.request.Request(
    "https://co.wuk.sh/api/json",
    data=json.dumps({"url": "https://www.youtube.com/watch?v=PqGehRgKTLo", "vQuality": "720"}).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Failed: {e}")
