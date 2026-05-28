#!/bin/bash
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=$(grep AWS_ACCESS_KEY_ID .env | cut -d '=' -f 2)
export AWS_SECRET_ACCESS_KEY=$(grep AWS_SECRET_ACCESS_KEY .env | cut -d '=' -f 2)

for i in 0 1 2; do
  aws s3 cp s3://clippedai-7137/youtube-downloads/cuid_admin_001-1777600698806/rNK4TPlEQjg/clip_$i.mp4 clip_$i.mp4
  ffmpeg -y -i clip_$i.mp4 -c copy -movflags +faststart clip_${i}_fast.mp4
  aws s3 cp clip_${i}_fast.mp4 s3://clippedai-7137/youtube-downloads/cuid_admin_001-1777600698806/rNK4TPlEQjg/clip_$i.mp4 --content-type video/mp4
done
