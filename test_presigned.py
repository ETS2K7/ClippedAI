import boto3
import os
from dotenv import load_dotenv

load_dotenv("backend/.env")

s3_client = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
)

url = s3_client.generate_presigned_url(
    "get_object",
    Params={
        "Bucket": "clippedai-7137",
        "Key": "youtube-downloads/cuid_admin_001-1777600698806/rNK4TPlEQjg/clip_0.mp4"
    },
    ExpiresIn=3600
)

print(url)
