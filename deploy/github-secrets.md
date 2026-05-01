# GitHub Actions — Required Secrets for ClippedAI Deployment

Add all of these in your GitHub repo at:
**Settings → Secrets and variables → Actions → New repository secret**

---

## Infrastructure Secrets

| Secret Name | Value | Notes |
|-------------|-------|-------|
| `DO_SERVER_IP` | `<your DO Droplet public IP>` | From DigitalOcean Dashboard |
| `DO_SSH_PRIVATE_KEY` | `<private key content>` | Output of `bootstrap.sh` — the full key including `-----BEGIN...-----END` lines |

---

## Database

| Secret Name | Value | Notes |
|-------------|-------|-------|
| `DATABASE_URL` | `postgresql://clippedai:<POSTGRES_PASSWORD>@db:5432/clippedai` | Use the same password as below |
| `POSTGRES_PASSWORD` | `<generate with: openssl rand -hex 32>` | |

---

## Auth

| Secret Name | Value | Notes |
|-------------|-------|-------|
| `AUTH_SECRET` | `<generate with: openssl rand -hex 32>` | NextAuth secret |

---

## AWS S3

| Secret Name | Value | Notes |
|-------------|-------|-------|
| `AWS_ACCESS_KEY_ID` | `<your key>` | |
| `AWS_SECRET_ACCESS_KEY` | `<your secret>` | |
| `AWS_REGION` | `us-east-1` | |
| `S3_BUCKET_NAME` | `clippedai-7137` | |

---

## CloudFront CDN (Optional)

| Secret Name | Value | Notes |
|-------------|-------|-------|
| `NEXT_PUBLIC_CLOUDFRONT_DOMAIN` | `d1ow9b4wmhldwd.cloudfront.net` | Your CloudFront distribution domain |

---

## Modal Backend

| Secret Name | Value | Notes |
|-------------|-------|-------|
| `PROCESS_VIDEO_ENDPOINT` | `https://<workspace>--clippedai-clippedai-process-video.modal.run` | From `modal deploy main.py` (workspace = active `modal profile`, e.g. `ebelseiko`) |
| `PROCESS_VIDEO_ENDPOINT_AUTH` | `<your modal auth token>` | |

---

## Generate secrets locally

```bash
# Generate AUTH_SECRET
openssl rand -hex 32

# Generate POSTGRES_PASSWORD
openssl rand -hex 24
```

---

## After adding all secrets

Push to trigger first deploy:
```bash
git add .
git commit -m "chore: initial deployment setup"
git push origin main
```

Watch it at: **GitHub → Actions tab → Deploy to DigitalOcean**

### Google OAuth
- `GOOGLE_CLIENT_ID` - Google Cloud OAuth Client ID.
- `GOOGLE_CLIENT_SECRET` - Google Cloud OAuth Client Secret.

