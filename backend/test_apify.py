import os
from dotenv import load_dotenv
load_dotenv("../frontend/.env")
from apify_client import ApifyClient
client = ApifyClient(os.environ.get("APIFY_TOKEN"))
run = client.actor("streamers/youtube-video-downloader").call(run_input={"startUrls": [{"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}]})
dataset = client.dataset(run["defaultDatasetId"])
items = dataset.list_items().items
print(f"Run status: {run['status']}")
print(f"Items found: {len(items)}")
if items:
    print(items[0])
