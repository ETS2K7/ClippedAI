# CloudFront CDN Setup Guide

This guide walks through setting up CloudFront CDN for ClippedAI to achieve 50-200ms latency reduction and 30-40% bandwidth savings via edge caching.

## Prerequisites

- AWS account with appropriate permissions
- S3 bucket: `clippedai-ap-south-1` (or your configured bucket)
- AWS CLI installed and configured

## Step 1: Create CloudFront Distribution

### Option A: Using AWS Console

1. Go to CloudFront in AWS Console
2. Click "Create distribution"
3. **Origin settings:**
   - Origin domain: Select your S3 bucket
   - S3 bucket access: "Yes use OAI" (recommended) or "Legacy access identities"
   - Origin access control: Create new OAI
4. **Default cache behavior:**
   - Viewer protocol policy: "Redirect HTTP to HTTPS"
   - Allowed HTTP methods: GET, HEAD, OPTIONS
   - Cached HTTP methods: GET, HEAD
   - Cache policy: "CachingOptimized" (recommended)
   - Origin request policy: "CORS-S3Origin" (if needed)
   - Response headers policy: "CORS-and-SecurityHeaders" (recommended)
5. **Settings:**
   - Price class: "Use only US, Europe, Asia, Middle East, and Africa" (or based on your needs)
   - Alternate domain names (CNAMEs): `cdn.clippedai.app` (optional)
   - Custom SSL certificate: Request ACM certificate for your domain
   - Default root object: Leave empty
   - Standard logging: Enable if you want logs
6. Click "Create distribution"

### Option B: Using AWS CLI

```bash
aws cloudfront create-distribution \
  --distribution-config \
  '{
    "CallerReference": "clippedai-cdn-'$(date +%s)'",
    "Comment": "ClippedAI CDN distribution",
    "Enabled": true,
    "Origins": {
      "Quantity": 1,
      "Items": [{
        "Id": "S3-clippedai-ap-south-1",
        "DomainName": "clippedai-ap-south-1.s3.ap-south-1.amazonaws.com",
        "S3OriginConfig": {
          "OriginAccessIdentity": "origin-access-identity/cloudfront/XXXXXXXXXXXXX"
        }
      }]
    },
    "DefaultCacheBehavior": {
      "TargetOriginId": "S3-clippedai-ap-south-1",
      "ViewerProtocolPolicy": "redirect-to-https",
      "AllowedMethods": {
        "Quantity": 2,
        "Items": ["GET", "HEAD"],
        "CachedMethods": {
          "Quantity": 2,
          "Items": ["GET", "HEAD"]
        }
      },
      "ForwardedValues": {
        "QueryString": false,
        "Cookies": {
          "Forward": "none"
        }
      },
      "MinTTL": 0,
      "DefaultTTL": 86400,
      "MaxTTL": 31536000,
      "Compress": true
    },
    "PriceClass": "PriceClass_All",
    "ViewerCertificate": {
      "CloudFrontDefaultCertificate": true
    }
  }'
```

## Step 2: Update S3 Bucket Policy

After creating the OAI, update your S3 bucket policy to allow CloudFront access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::cloudfront:user/Origin Access Identity XXXXXXXXXXXXX"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::clippedai-ap-south-1/*"
    }
  ]
}
```

Replace `XXXXXXXXXXXXX` with your actual OAI ID.

## Step 3: Update Environment Variables

Add the CloudFront distribution URL to your environment:

```bash
# frontend/.env
NEXT_PUBLIC_CLOUDFRONT_DOMAIN=d1234567890.cloudfront.net
```

```bash
# backend/.env
CLOUDFRONT_DOMAIN=d1234567890.cloudfront.net
```

## Step 4: Update Code to Use CloudFront

### Frontend Changes

Update presigned URL generation to use CloudFront for public content:

```typescript
// frontend/src/actions/generation.ts
// Add function to determine if URL should use CloudFront
function shouldUseCloudFront(key: string): boolean {
  // Use CloudFront for public content (thumbnails, HLS segments)
  return key.includes('thumb_') || key.includes('hls_');
}

// Update getPresignedUrl to use CloudFront when appropriate
export async function getPresignedUrl(key: string): Promise<string> {
  if (shouldUseCloudFront(key) && env.CLOUDFRONT_DOMAIN) {
    return `https://${env.CLOUDFRONT_DOMAIN}/${key}`;
  }
  // Fall back to presigned S3 URL for private content
  // ... existing presigned URL logic
}
```

### Backend Changes

Update S3 uploads to set appropriate cache headers for CloudFront:

```python
# backend/main.py
# Already implemented with Cache-Control headers
s3_client.upload_file(
    thumb_out_path, bucket, thumb_key,
    ExtraArgs={
        "ContentType": "image/webp",
        "CacheControl": "public, max-age=31536000, immutable",
    },
)
```

## Step 5: Configure Cache Behavior

For optimal performance, configure cache behaviors:

### Video Files (MP4)
- Cache policy: `CachingOptimized`
- TTL: 86400 (1 day) for videos
- Compress: true

### Thumbnails
- Cache policy: `CachingOptimized`
- TTL: 31536000 (1 year) for thumbnails (immutable)
- Compress: true

### HLS Segments
- Cache policy: `CachingOptimized`
- TTL: 31536000 (1 year) for segments (immutable)
- Compress: true

## Step 6: Test CloudFront Distribution

1. Wait for distribution to deploy (typically 10-15 minutes)
2. Test accessing a file through CloudFront:
   ```bash
   curl -I https://d1234567890.cloudfront.net/path/to/file
   ```
3. Check response headers:
   - `X-Cache: Hit from cloudfront` (cache hit)
   - `X-Cache: Miss from cloudfront` (first request)
   - `Cache-Control: public, max-age=31536000, immutable`

## Step 7: Update DNS (Optional)

If you want to use a custom domain:

1. Request ACM certificate for your domain
2. Add CNAME record in your DNS:
   ```
   cdn.clippedai.app CNAME d1234567890.cloudfront.net
   ```
3. Update CloudFront distribution to use custom domain and SSL certificate

## Expected Results

- **Latency reduction**: 50-200ms for most users
- **Bandwidth savings**: 30-40% via cache hits
- **Global performance**: Edge locations worldwide
- **Reduced S3 costs**: Fewer direct S3 requests

## Monitoring

Enable CloudFront metrics and alarms:

1. Enable CloudFront metrics in AWS Console
2. Set up CloudWatch alarms for:
   - 4xx/5xx error rates
   - Cache hit ratio
   - Origin latency
3. Monitor via CloudWatch dashboards

## Cost Considerations

- CloudFront pricing: $0.085/GB for US/EU (first 10TB)
- Free tier: 1TB of data transfer out per month
- S3 transfer costs reduced significantly with CloudFront

## Troubleshooting

### 403 Forbidden errors
- Check S3 bucket policy allows OAI access
- Verify OAI is correctly configured
- Ensure CloudFront distribution is deployed

### CORS issues
- Add CORS configuration to S3 bucket
- Use appropriate origin request policy
- Add response headers policy for CORS

### Cache not working
- Check Cache-Control headers on S3 objects
- Verify cache policy configuration
- Ensure query string forwarding is disabled (if not needed)
