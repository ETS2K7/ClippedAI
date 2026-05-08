# ClippedAI

AI-powered video clipping platform that transforms long-form videos into viral-ready short clips with face tracking, smart cropping, and word-synced subtitles.

## Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend (Next.js 15)"]
        UI[Dashboard UI]
        Auth[NextAuth v5]
        Prisma[Prisma ORM]
        S3Upload[S3 Upload]
    end

    subgraph Backend["Backend (Modal GPU)"]
        API[FastAPI Endpoint]
        Transcribe[AssemblyAI]
        LLM[Gemini 2.5 Flash]
        VideoProc[OpenCV + FFmpeg]
    end

    subgraph Infra["Infrastructure"]
        DB[(PostgreSQL)]
        S3[(AWS S3)]
        DO[DigitalOcean]
        Nginx[Nginx + SSL]
    end

    UI --> Auth
    UI --> S3Upload --> S3
    UI --> API
    API --> Transcribe
    API --> LLM
    API --> VideoProc
    VideoProc --> S3
    Prisma --> DB
    Nginx --> UI
    DO --> Nginx
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, TailwindCSS v4, shadcn/ui |
| Auth | NextAuth v5 (credentials + JWT) |
| Database | PostgreSQL + Prisma ORM |
| Payments | Dodo Payments (Subscriptions & PAYG) |
| Background Jobs | Inngest |
| Storage | AWS S3 |
| Backend | Python 3.10 on Modal (serverless GPU) |
| Transcription | AssemblyAI |
| AI Clip Selection | Gemini 2.5 Flash (via GCP Vertex AI) |
| Video Processing | OpenCV, FFmpeg, scipy, scenedetect, Fast-ASD |
| Deployment | Docker Compose on DigitalOcean, Nginx, Let's Encrypt |

## Local Development

### Prerequisites
- Node.js 18+
- Python 3.10+
- PostgreSQL
- Modal account
- AWS S3 bucket

### Frontend Setup
```bash
cd frontend
cp .env.example .env
# Fill in your environment variables
npm install
npx prisma generate
npx prisma db push
npm run dev
```

### Backend Setup
```bash
cd backend
cp .env.example .env
# Fill in your environment variables
pip install -r requirements.txt
modal deploy main.py
```

## Environment Variables

### Frontend (`frontend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `AUTH_SECRET` | ✅ (prod) | NextAuth secret key |
| `AWS_ACCESS_KEY_ID` | ✅ | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | ✅ | AWS credentials |
| `AWS_REGION` | ✅ | S3 bucket region |
| `S3_BUCKET_NAME` | ✅ | S3 bucket name |
| `PROCESS_VIDEO_ENDPOINT` | ✅ | Modal backend endpoint URL |
| `PROCESS_VIDEO_ENDPOINT_AUTH` | ✅ | Bearer token for backend |
| `BASE_URL` | ✅ | App base URL (e.g. `https://clippedai.app`) |
| `DODO_PAYMENTS_API_KEY` | ✅ | Dodo Payments Secret Key |
| `DODO_WEBHOOK_SECRET` | ✅ | Dodo Webhook Secret |
| `DODO_PLAN_*` | ✅ | Dodo Product IDs (Pro, Founding, Starter) |
| `DODO_CREDITS_*` | ✅ | Dodo Pay-As-You-Go Credit Bundle IDs |

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `ASSEMBLYAI_KEY` | ✅ | AssemblyAI API key |
| `GOOGLE_CLOUD_PROJECT` | ✅ | GCP Project ID for Vertex AI (Gemini) |
| `GCP_SERVICE_ACCOUNT_JSON` | ❌ | GCP Credentials JSON (if not using ADC) |
| `APIFY_TOKEN` | ✅ | Apify token for YouTube downloads |
| `S3_BUCKET_NAME` | ✅ | S3 bucket name |
| `AWS_ACCESS_KEY_ID` | ✅ | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | ✅ | AWS credentials |
| `AUTH_TOKEN` | ✅ | Bearer token (must match frontend) |

## Pricing & Monetization

The platform integrates with **Dodo Payments** and uses a hybrid billing architecture:
- **Subscriptions:** Starter (Free) and Pro ($12.99/mo). Includes a live-updating limited "Founding Member" tier ($7.50/mo) capped at 25 users.
- **Pay-As-You-Go (PAYG):** Pro users can purchase non-expiring credit bundles (100, 250, 500).
- **Credit System:** 1 credit = 1 minute of video processing. Capacity gating enforces platform limits dynamically.

## Deployment

Production deployment uses Docker Compose on DigitalOcean with Nginx reverse proxy and Let's Encrypt SSL.

```bash
# Build and deploy
docker compose build
docker compose up -d
```

See [deploy.yml](.github/workflows/deploy.yml) for CI/CD configuration.

## License

MIT
