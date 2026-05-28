import os

import pytest
from dotenv import load_dotenv


@pytest.mark.skipif(
    os.environ.get("APIFY_LIVE_TEST") != "1",
    reason="live Apify smoke test; set APIFY_LIVE_TEST=1 to run",
)
def test_apify_youtube_downloader_smoke():
    load_dotenv("../frontend/.env")
    from apify_client import ApifyClient

    token = os.environ.get("APIFY_TOKEN")
    assert token, "APIFY_TOKEN is required for the live Apify smoke test"

    client = ApifyClient(token)
    run = client.actor("streamers/youtube-video-downloader").call(
        run_input={
            "videos": [{"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}],
            "preferQuality": "720p",
            "preferFormat": "mp4",
        }
    )
    dataset = client.dataset(run["defaultDatasetId"])
    items = dataset.list_items().items

    assert run["status"] == "SUCCEEDED"
    assert items
    assert items[0].get("downloadedFileUrl") or items[0].get("downloadUrl")
