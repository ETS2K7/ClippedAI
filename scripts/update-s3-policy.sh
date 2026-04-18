#!/bin/bash

# Update S3 bucket policy to allow public read access for HLS files

BUCKET="clippedai-ap-south-1"

aws s3api put-bucket-policy --bucket $BUCKET --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::cloudfront:user/CloudFront Origin Access Identity E1FMPIJS726KE9"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::'$BUCKET'/*"
    },
    {
      "Sid": "AllowPublicHLSAccess",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::'$BUCKET'/*hls_*/*"
    }
  ]
}'

echo "S3 bucket policy updated to allow public HLS access"
