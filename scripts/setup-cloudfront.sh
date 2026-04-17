#!/bin/bash

# CloudFront CDN Setup Script for ClippedAI
# This script creates a CloudFront distribution with OAI for S3 access

set -e

# Configuration
S3_BUCKET="clippedai-ap-south-1"
DISTRIBUTION_COMMENT="ClippedAI CDN distribution"
CALLER_REFERENCE="clippedai-cdn-$(date +%s)"

echo "=== CloudFront CDN Setup for ClippedAI ==="
echo "S3 Bucket: $S3_BUCKET"
echo ""

# Step 1: Create Origin Access Identity (OAI)
echo "Step 1: Creating Origin Access Identity (OAI)..."
OAI_RESPONSE=$(aws cloudfront create-cloud-front-origin-access-identity \
  --cloud-front-origin-access-identity-config \
  "CallerReference=$CALLER_REFERENCE,Comment=$DISTRIBUTION_COMMENT OAI")

OAI_ID=$(echo "$OAI_RESPONSE" | jq -r '.CloudFrontOriginAccessIdentity.Id')
OAI_S3_CANONICAL_USER_ID=$(echo "$OAI_RESPONSE" | jq -r '.CloudFrontOriginAccessIdentity.S3CanonicalUserId')

echo "OAI created successfully!"
echo "OAI ID: $OAI_ID"
echo "OAI S3 Canonical User ID: $OAI_S3_CANONICAL_USER_ID"
echo ""

# Step 2: Get S3 bucket region
echo "Step 2: Getting S3 bucket region..."
S3_REGION=$(aws s3api get-bucket-location --bucket "$S3_BUCKET" | jq -r '.LocationConstraint' | sed 's/null/us-east-1/')
echo "S3 Bucket Region: $S3_REGION"
echo ""

# Step 3: Create CloudFront distribution
echo "Step 3: Creating CloudFront distribution..."
DISTRIBUTION_CONFIG=$(cat <<EOF
{
  "CallerReference": "$CALLER_REFERENCE",
  "Comment": "$DISTRIBUTION_COMMENT",
  "Enabled": true,
  "Origins": {
    "Quantity": 1,
    "Items": [{
      "Id": "S3-$S3_BUCKET",
      "DomainName": "$S3_BUCKET.s3.$S3_REGION.amazonaws.com",
      "S3OriginConfig": {
        "OriginAccessIdentity": "origin-access-identity/cloudfront/$OAI_ID"
      }
    }]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-$S3_BUCKET",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {
      "Quantity": 2,
      "Items": ["GET", "HEAD"],
      "CachedMethods": {
        "Quantity": 2,
        "Items": ["GET", "HEAD"]
      }
    },
    "Compress": true,
    "MinTTL": 0,
    "DefaultTTL": 86400,
    "MaxTTL": 31536000,
    "ForwardedValues": {
      "QueryString": false,
      "Cookies": {
        "Forward": "none"
      }
    }
  },
  "PriceClass": "PriceClass_All",
  "ViewerCertificate": {
    "CloudFrontDefaultCertificate": true
  }
}
EOF
)

DISTRIBUTION_RESPONSE=$(aws cloudfront create-distribution \
  --distribution-config "$DISTRIBUTION_CONFIG")

DISTRIBUTION_ID=$(echo "$DISTRIBUTION_RESPONSE" | jq -r '.Distribution.Id')
DISTRIBUTION_DOMAIN=$(echo "$DISTRIBUTION_RESPONSE" | jq -r '.Distribution.DomainName')
DISTRIBUTION_STATUS=$(echo "$DISTRIBUTION_RESPONSE" | jq -r '.Distribution.Status')

echo "CloudFront distribution created successfully!"
echo "Distribution ID: $DISTRIBUTION_ID"
echo "Distribution Domain: $DISTRIBUTION_DOMAIN"
echo "Status: $DISTRIBUTION_STATUS"
echo ""

# Step 4: Update S3 bucket policy
echo "Step 4: Updating S3 bucket policy to allow CloudFront OAI access..."
BUCKET_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::cloudfront:user/Origin Access Identity $OAI_ID"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::$S3_BUCKET/*"
    }
  ]
}
EOF
)

aws s3api put-bucket-policy \
  --bucket "$S3_BUCKET" \
  --policy "$BUCKET_POLICY"

echo "S3 bucket policy updated successfully!"
echo ""

# Step 5: Save configuration for reference
echo "Step 5: Saving configuration..."
cat > cloudfront-config.txt <<EOF
CloudFront CDN Configuration
============================

Distribution ID: $DISTRIBUTION_ID
Distribution Domain: $DISTRIBUTION_DOMAIN
OAI ID: $OAI_ID
OAI S3 Canonical User ID: $OAI_S3_CANONICAL_USER_ID

Next Steps:
1. Add the following to your frontend .env:
   NEXT_PUBLIC_CLOUDFRONT_DOMAIN=$DISTRIBUTION_DOMAIN

2. Wait for distribution to deploy (typically 10-15 minutes):
   aws cloudfront get-distribution --id $DISTRIBUTION_ID

3. Test the distribution:
   curl -I https://$DISTRIBUTION_DOMAIN/path/to/file

4. Check cache hit in response headers:
   X-Cache: Hit from cloudfront
EOF

echo "Configuration saved to cloudfront-config.txt"
echo ""

echo "=== Setup Complete ==="
echo "CloudFront distribution is being deployed."
echo "This typically takes 10-15 minutes."
echo "Monitor deployment with:"
echo "  aws cloudfront get-distribution --id $DISTRIBUTION_ID"
echo ""
echo "Once deployed, add to your .env:"
echo "  NEXT_PUBLIC_CLOUDFRONT_DOMAIN=$DISTRIBUTION_DOMAIN"
