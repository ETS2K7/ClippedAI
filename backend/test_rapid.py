import urllib.request
import json

def test_api():
    # Let's just search duckduckgo for cobalt instances
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/?q=public+cobalt.tools+instances",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            print("Successfully fetched search results")
    except Exception as e:
        print(f"Failed: {e}")

test_api()
