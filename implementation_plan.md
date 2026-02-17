# ClippedAI — Complete Implementation Plan

End-to-end blueprint for building ClippedAI: 49 core features across 5 phases.

---

## 1. Project Structure

```
ClippedAI/
├── backend/
│   ├── main.py                         # FastAPI app entrypoint
│   ├── config.py                       # Env vars, settings
│   ├── database.py                     # SQLAlchemy engine + session
│   ├── models.py                       # ORM models
│   ├── schemas.py                      # Pydantic request/response
│   ├── middleware.py                    # Rate limiter, auth, CORS
│   ├── api/
│   │   ├── __init__.py
│   │   ├── jobs.py                     # /api/jobs endpoints
│   │   ├── clips.py                    # /api/clips endpoints
│   │   ├── streams.py                  # /api/streams endpoints
│   │   ├── publish.py                  # /api/publish endpoints
│   │   ├── search.py                   # /api/search endpoints (ClipAnything)
│   │   ├── brand.py                    # /api/brand endpoints
│   │   ├── upload.py                   # /api/upload presigned URL endpoints
│   │   ├── webhooks.py                 # /api/webhooks incoming event handler
│   │   ├── oauth.py                    # /api/auth/{platform}/login + callback
│   │   ├── auth.py                     # /api/auth JWT endpoints
│   │   ├── workspaces.py               # /api/workspaces team management
│   │   └── health.py                   # /api/health + GPU queue status
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── jwt.py                      # JWT token issuing + validation
│   │   ├── dependencies.py             # FastAPI Depends(get_current_user)
│   │   └── permissions.py              # RBAC permission checks
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── state_machine.py            # Job state transitions + validation
│   │   ├── vod_workflow.py             # VOD pipeline DAG
│   │   ├── stream_workflow.py          # Streaming pipeline DAG
│   │   └── retry_policy.py            # Per-step retry config
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingest.py                   # VOD download (yt-dlp)
│   │   ├── audio_extract.py            # FFmpeg audio extraction
│   │   ├── transcribe.py               # Tiered ASR (fast + accurate)
│   │   ├── scene_detect.py             # PySceneDetect
│   │   ├── face_track.py               # YOLOv8-face / InsightFace
│   │   ├── audio_analysis.py           # Energy curves, crowd detection
│   │   ├── clip_selector.py            # LLM clip ranking + timestamp alignment
│   │   ├── reframe.py                  # Auto-reframe + layout switching
│   │   ├── captions.py                 # .ass subtitle generator + FFmpeg burn
│   │   ├── audio_enhance.py            # Studio Sound, ducking, censor
│   │   ├── effects.py                  # Zooms, transitions, B-roll, green screen
│   │   ├── hooks.py                    # Magic Hook generator
│   │   ├── export.py                   # FFmpeg final render
│   │   ├── nle_export.py               # FCPXML / Premiere XML / EDL
│   │   └── thumbnails.py              # AI thumbnail generator
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── supervisor.py               # Stream supervisor (lifecycle manager)
│   │   ├── stream_worker.py            # Per-stream async worker process
│   │   ├── stream_ingest.py            # streamlink + FFmpeg chunking
│   │   ├── ring_buffer.py              # Disk-backed sliding window buffer
│   │   ├── realtime_transcribe.py      # Streaming ASR (faster-whisper small)
│   │   ├── event_detector.py           # Event-driven clip trigger
│   │   ├── gaming_detector.py          # Template + YOLO HUD detection
│   │   ├── camera_receiver.py          # WebRTC/RTMP/RTSP ingest
│   │   └── ws_gateway.py              # WebSocket fan-out via Redis pub/sub
│   ├── intelligence/
│   │   ├── __init__.py
│   │   ├── clip_anything.py            # CLIP embeddings + vector search
│   │   ├── summarizer.py               # Video summarization
│   │   ├── qa_chat.py                  # RAG Q&A over transcript
│   │   ├── trend_analyzer.py           # SEO/trend analytics
│   │   ├── emotion_analyzer.py         # Sentiment/emotion scoring
│   │   └── agentic_editor.py           # NL command → pipeline actions
│   ├── distribution/
│   │   ├── __init__.py
│   │   ├── oauth_manager.py            # Token refresh, storage, validation
│   │   ├── publisher.py                # Multi-platform upload
│   │   ├── scheduler.py                # Content calendar / scheduling
│   │   ├── platforms/
│   │   │   ├── youtube.py
│   │   │   ├── tiktok.py
│   │   │   ├── instagram.py
│   │   │   ├── twitter.py
│   │   │   └── linkedin.py
│   │   └── analytics.py                # Pull engagement metrics
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── celery_app.py               # Celery configuration
│   │   ├── vod_tasks.py                # VOD pipeline task chain
│   │   ├── stream_tasks.py             # Stream monitoring tasks
│   │   └── publish_tasks.py            # Async publish tasks
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Prometheus metrics
│   │   ├── tracing.py                  # OpenTelemetry spans
│   │   └── alerts.py                   # Alert rules
│   └── tests/
│       ├── test_pipeline.py
│       ├── test_api.py
│       ├── test_streaming.py
│       ├── test_state_machine.py
│       ├── test_gpu_routing.py
│       ├── conftest.py                 # Fixtures: test DB, mock Modal, test VODs
│       └── golden_vods/                # Known public VODs for integration tests
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── index.css
│   │   ├── api/                        # API client
│   │   ├── components/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── JobCreator.tsx
│   │   │   ├── ClipReview.tsx
│   │   │   ├── ClipEditor.tsx
│   │   │   ├── CaptionStyler.tsx
│   │   │   ├── BrandKit.tsx
│   │   │   ├── StreamMonitor.tsx
│   │   │   ├── PublishPanel.tsx
│   │   │   ├── Calendar.tsx
│   │   │   └── SearchBar.tsx           # ClipAnything UI
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── Jobs.tsx
│   │   │   ├── Clips.tsx
│   │   │   ├── Streams.tsx
│   │   │   ├── Publish.tsx
│   │   │   └── Settings.tsx
│   │   └── hooks/
│   └── public/
├── infra/
│   ├── docker-compose.yml              # PostgreSQL, Redis, Mediamtx, Prometheus, Grafana
│   ├── modal_deploy.py                 # Modal GPU functions (multi-queue)
│   ├── prometheus.yml                  # Metrics scrape config
│   └── nginx.conf                      # Reverse proxy + rate limiting config
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 2. Tech Stack & Versions

| Component | Technology | Version |
|-----------|-----------|---------|
| Backend framework | FastAPI | ≥0.115 |
| Python | CPython | 3.12 |
| Task queue | Celery | ≥5.4 |
| Message broker | Redis | ≥7.0 |
| Database | PostgreSQL | ≥16 |
| ORM | SQLAlchemy | ≥2.0 |
| GPU inference | Modal | Latest |
| LLM API | Cerebras (Llama-3.3-70B) | Latest |
| **LLM fallback** | **Groq / Together.ai** | Latest |
| VOD transcription | WhisperX (large-v2) | ≥3.3 |
| **Streaming ASR** | **faster-whisper (small)** | ≥1.1 |
| **ASR fallback** | **Deepgram API** | Latest |
| Scene detection | PySceneDetect | ≥0.6 |
| **Face detection** | **YOLOv8-face / InsightFace** | Latest |
| Video processing | FFmpeg | ≥7.0 |
| Caption rendering | FFmpeg libass (.ass subtitles) | — |
| Vector DB | Qdrant | ≥1.9 |
| Visual embeddings | CLIP (ViT-L/14) | via `open_clip` |
| Audio enhancement | DeepFilterNet | ≥0.5 |
| OCR | EasyOCR | ≥1.7 |
| Beat detection | librosa | ≥0.10 |
| Object storage | Cloudflare R2 | — |
| Frontend | React + Vite | React 18, Vite 6 |
| Real-time | Socket.IO | ≥4.8 |
| **WS Gateway** | **Redis pub/sub + Socket.IO** | — |
| Stream server | Mediamtx | ≥1.5 |
| **Auth** | **JWT (PyJWT) + bcrypt** | Latest |
| **Rate limiting** | **slowapi** | Latest |
| **Monitoring** | **Prometheus + Grafana** | Latest |
| **Tracing** | **OpenTelemetry** | Latest |

---

## 3. Job State Machine & Orchestration

> [!IMPORTANT]
> All jobs (VOD, livestream, camera) follow a **unified state machine**. Each step is independently retryable and auditable. This replaces raw Celery chains with a proper orchestration layer.

### State Machine

```mermaid
stateDiagram-v2
    [*] --> created
    created --> ingesting : start
    ingesting --> analyzing : ingest_complete
    analyzing --> clips_generated : analysis_complete
    clips_generated --> rendering : approve / auto
    rendering --> review : render_complete
    review --> publishing : publish
    review --> rendering : re-render
    publishing --> completed : all_published
    
    ingesting --> failed : error
    analyzing --> failed : error
    rendering --> failed : error
    publishing --> failed : error
    
    failed --> ingesting : retry_step
    failed --> analyzing : retry_step
    
    created --> cancelled : cancel
    ingesting --> cancelled : cancel
    analyzing --> cancelled : cancel
    rendering --> cancelled : cancel
```

### Step Status Table

```sql
CREATE TABLE job_steps (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID REFERENCES jobs(id) ON DELETE CASCADE,
    step_name       VARCHAR(50) NOT NULL,    -- ingest, transcribe, diarize, scene_detect, face_track, audio_analyze, clip_select, render, export
    status          VARCHAR(20) DEFAULT 'pending',  -- pending, running, completed, failed, skipped, cancelled
    worker_id       VARCHAR(255),
    modal_call_id   VARCHAR(255),             -- For cancellation
    attempt_count   INTEGER DEFAULT 0,
    max_attempts    INTEGER DEFAULT 3,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    output_path     TEXT,                     -- R2 path to step output
    duration_ms     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_steps_job ON job_steps(job_id);
CREATE INDEX idx_steps_status ON job_steps(status);
```

### Retry Policy

| Step | Max Attempts | Backoff | Timeout |
|------|:-:|---------|---------|
| ingest | 3 | 30s exponential | 10 min |
| transcribe | 2 | 60s | 20 min |
| scene_detect | 2 | 30s | 10 min |
| face_track | 2 | 30s | 15 min |
| clip_select (LLM) | 3 | 10s | 60s |
| render (per clip) | 2 | 30s | 5 min |
| publish | 3 | 60s exponential | 2 min |

### Orchestration Logic

```python
class JobOrchestrator:
    def advance(job_id: str):
        """Check current state, execute next step, handle transitions.
        Uses Redis Redlock to prevent duplicate orchestration."""
        lock = redlock.lock(f"orchestrate:{job_id}", timeout=30_000)
        if not lock:
            return  # Another worker is orchestrating this job
        try:
            job = db.get(job_id)
            match job.status:
                case "created":
                    transition(job, "ingesting")
                    dispatch_step(job, "ingest")
                case "ingesting":
                    if step_done(job, "ingest"):
                        transition(job, "analyzing")
                        dispatch_parallel([
                            ("transcribe", gpu_queue),
                            ("scene_detect", gpu_queue),
                            ("face_track", gpu_queue),
                            ("audio_analyze", gpu_queue),
                        ])
                case "analyzing":
                    if all_steps_done(job, ["transcribe", "scene_detect", "face_track", "audio_analyze"]):
                        dispatch_step(job, "clip_select")
                        transition(job, "clips_generated")
                # ... etc
        finally:
            redlock.unlock(lock)

    def on_step_failure(job_id, step_name, error):
        step = get_step(job_id, step_name)
        if step.attempt_count < step.max_attempts:
            retry_step(step)  # with backoff
        else:
            transition(job, "failed", error=error)
            notify_user_ws(job_id, "step_failed", step_name)

    def cancel(job_id):
        job = db.get(job_id)
        for step in get_running_steps(job_id):
            if step.modal_call_id:
                modal.FunctionCall.from_id(step.modal_call_id).cancel()
            celery_app.control.revoke(step.worker_id, terminate=True)
        transition(job, "cancelled")
```

### Orchestrator Recovery

> [!CAUTION]
> On API restart or worker crash, orphaned jobs must be recovered. A periodic **recovery loop** scans for stuck steps.

```python
# Runs every 60s via Celery beat
def recover_orphaned_jobs():
    stuck = db.query("""
        SELECT * FROM job_steps
        WHERE status = 'running'
        AND started_at < now() - (max_attempts * timeout_seconds || ' seconds')::interval
    """)
    for step in stuck:
        if step.attempt_count < step.max_attempts:
            step.status = 'pending'  # Will be retried
            orchestrator.advance(step.job_id)
        else:
            transition(step.job_id, 'failed', error='Step timed out after recovery')
```

---

## 4. Authentication & Authorization

### JWT Auth Flow

```python
# POST /api/auth/register → {email, password, name}
# POST /api/auth/login    → {access_token, refresh_token}
# POST /api/auth/refresh  → {access_token}

# Access token: 15 min expiry, contains {user_id, role, workspace_id}
# Refresh token: 7 day expiry, stored hashed in DB, rotated on use

class AuthMiddleware:
    async def __call__(request):
        token = request.headers["Authorization"].split("Bearer ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        request.state.user = payload
```

### RBAC Permissions

| Role | Jobs | Clips | Publish | Stream | Settings | Team |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **viewer** | read | read | — | read | — | — |
| **editor** | CRUD | CRUD | request | CRUD | own | — |
| **admin** | CRUD | CRUD | CRUD | CRUD | all | CRUD |
| **owner** | * | * | * | * | * | * |

### Storage Isolation

> [!CAUTION]
> All R2 paths are prefixed with `{workspace_id}/` to enforce per-team data isolation. Backend validates that the requesting user's workspace matches the resource's path prefix.

```
R2 path structure:
  {workspace_id}/sources/{job_id}/video.mp4
  {workspace_id}/analysis/{job_id}/transcript.json
  {workspace_id}/clips/{clip_id}/final.mp4
  {workspace_id}/clips/{clip_id}/thumbnail.jpg
```

---

## 5. Database Schema

> [!IMPORTANT]
> Heavy payloads (transcripts, face data, scenes, audio analysis) are stored in R2 as JSON files — **not** in the database. The `jobs` table only holds R2 paths to these artifacts.

```sql
-- Users & Auth
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    name            VARCHAR(255),
    password_hash   VARCHAR(255) NOT NULL,
    refresh_token_hash VARCHAR(255),          -- Hashed refresh token
    created_at      TIMESTAMPTZ DEFAULT now(),
    deleted_at      TIMESTAMPTZ               -- Soft delete
);

-- Workspaces (teams)
CREATE TABLE workspaces (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    owner_id        UUID REFERENCES users(id),
    settings        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now(),
    deleted_at      TIMESTAMPTZ
);

-- Workspace membership
CREATE TABLE workspace_members (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    role            VARCHAR(20) DEFAULT 'editor',  -- owner, admin, editor, viewer
    joined_at       TIMESTAMPTZ DEFAULT now(),
    UNIQUE(workspace_id, user_id)
);

-- API Tokens (machine access)
CREATE TABLE api_tokens (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    workspace_id    UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    name            VARCHAR(255),
    token_hash      VARCHAR(255) NOT NULL,
    scopes          TEXT[],
    last_used_at    TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Connected Platform Accounts (OAuth tokens)
CREATE TABLE connected_accounts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
    workspace_id    UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    platform        VARCHAR(20) NOT NULL,
    platform_user_id VARCHAR(255),
    access_token    TEXT NOT NULL,                    -- Fernet encrypted
    refresh_token   TEXT,                             -- Fernet encrypted
    encryption_key_version INTEGER DEFAULT 1,         -- Supports Fernet key rotation
    token_expiry    TIMESTAMPTZ,
    scopes          TEXT[],
    account_name    VARCHAR(255),
    is_active       BOOLEAN DEFAULT true,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(workspace_id, platform, platform_user_id)
);

-- Processing Jobs (lean — heavy data stored in R2)
CREATE TYPE job_mode AS ENUM ('vod', 'livestream', 'camera');
CREATE TYPE job_status AS ENUM (
    'created', 'ingesting', 'analyzing', 'clips_generated',
    'rendering', 'review', 'publishing', 'completed', 'failed', 'cancelled'
);
CREATE TYPE source_type AS ENUM ('youtube', 'twitch', 'vimeo', 'upload', 'rtmp', 'webrtc', 'rtsp');
CREATE TYPE clip_status AS ENUM ('pending', 'rendering', 'ready', 'published', 'failed');

-- Enum migration practice: use ALTER TYPE ... ADD VALUE via Alembic.
-- Keep enums small and stable. For frequently changing values, use TEXT + CHECK instead.

CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id),
    user_id         UUID REFERENCES users(id),
    stream_monitor_id UUID REFERENCES stream_monitors(id), -- NULL for VOD, set for stream clips
    mode            job_mode NOT NULL,
    status          job_status DEFAULT 'created',
    source_url      TEXT,
    source_type     source_type,
    source_path     TEXT,                            -- R2 path (prefixed workspace_id/)
    transcript_path TEXT,
    scenes_path     TEXT,
    face_data_path  TEXT,
    audio_data_path TEXT,
    llm_data_path   TEXT,
    clip_count      INTEGER DEFAULT 0,
    settings        JSONB DEFAULT '{}',
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    deleted_at      TIMESTAMPTZ
);

-- Job step tracking (see Section 3)
-- CREATE TABLE job_steps (...)  -- defined above

-- Generated Clips (versioned)
CREATE TABLE clips (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID REFERENCES jobs(id) ON DELETE CASCADE,
    workspace_id    UUID REFERENCES workspaces(id),
    version         INTEGER DEFAULT 1,               -- Increments on re-render
    title           VARCHAR(500),
    description     TEXT,
    hashtags        TEXT[],
    start_time      FLOAT NOT NULL,
    end_time        FLOAT NOT NULL,
    duration        FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
    virality_score  FLOAT,
    coherence_score FLOAT,
    aspect_ratio    VARCHAR(10) DEFAULT '9:16',
    source_path     TEXT,
    thumbnail_path  TEXT,
    nle_path        TEXT,
    caption_style   JSONB DEFAULT '{}',
    status          clip_status DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT now(),
    deleted_at      TIMESTAMPTZ
);

-- Brand Kit
CREATE TABLE brand_kits (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id),
    name            VARCHAR(255),
    logo_path       TEXT,
    colors          JSONB,
    fonts           JSONB,
    outro_path      TEXT,
    watermark       JSONB,
    created_at      TIMESTAMPTZ DEFAULT now(),
    deleted_at      TIMESTAMPTZ
);

-- Livestream Monitors (with worker locking + health)
CREATE TABLE stream_monitors (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id),
    user_id         UUID REFERENCES users(id),
    platform        VARCHAR(20),
    channel_url     TEXT NOT NULL,
    is_active       BOOLEAN DEFAULT false,
    status          VARCHAR(20) DEFAULT 'idle',     -- idle, connecting, active, error, stopped
    worker_id       VARCHAR(255),
    locked_at       TIMESTAMPTZ,
    last_seen_at    TIMESTAMPTZ,                     -- Last heartbeat
    last_error      TEXT,
    clips_generated INTEGER DEFAULT 0,
    settings        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Publishing Records
CREATE TABLE publications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clip_id         UUID REFERENCES clips(id) ON DELETE CASCADE,
    account_id      UUID REFERENCES connected_accounts(id),
    workspace_id    UUID REFERENCES workspaces(id),
    platform        VARCHAR(20) NOT NULL,
    platform_post_id VARCHAR(255),
    idempotency_key VARCHAR(255),                    -- Prevent duplicate publishes
    status          VARCHAR(20) DEFAULT 'pending',
    scheduled_at    TIMESTAMPTZ,
    published_at    TIMESTAMPTZ,
    analytics       JSONB DEFAULT '{}',
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(clip_id, account_id, status) WHERE status IN ('pending', 'scheduled')
);

-- Clip Comments (timestamp-anchored)
CREATE TABLE clip_comments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clip_id         UUID REFERENCES clips(id) ON DELETE CASCADE,
    user_id         UUID REFERENCES users(id),
    text            TEXT NOT NULL,
    timestamp       FLOAT,                           -- Video timestamp (nullable for general)
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Content Calendar
CREATE TABLE calendar_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id),
    clip_id         UUID REFERENCES clips(id),
    platform        VARCHAR(20),
    scheduled_at    TIMESTAMPTZ NOT NULL,
    status          VARCHAR(20) DEFAULT 'scheduled',
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Frame metadata (vectors stored in Qdrant, not Postgres)
CREATE TABLE frame_metadata (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID REFERENCES jobs(id) ON DELETE CASCADE,
    workspace_id    UUID REFERENCES workspaces(id),
    timestamp       FLOAT NOT NULL,
    qdrant_point_id UUID,                            -- Reference to Qdrant vector
    thumbnail_path  TEXT
);

-- Audit Log
CREATE TABLE audit_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id),
    user_id         UUID REFERENCES users(id),
    action          VARCHAR(50) NOT NULL,
    resource_type   VARCHAR(30),
    resource_id     UUID,
    metadata        JSONB DEFAULT '{}',
    ip_address      INET,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Usage/Cost Tracking (billing + infra tuning)
CREATE TABLE usage_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID REFERENCES workspaces(id),
    job_id          UUID REFERENCES jobs(id),
    resource_type   VARCHAR(30) NOT NULL,          -- gpu_asr, gpu_vision, gpu_render, llm, storage
    gpu_seconds     FLOAT DEFAULT 0,
    llm_tokens      INTEGER DEFAULT 0,
    storage_bytes   BIGINT DEFAULT 0,
    cost_estimate   FLOAT DEFAULT 0,               -- Estimated $ cost
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Feature Flags (disable features without deploy)
CREATE TABLE feature_flags (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) UNIQUE NOT NULL,  -- green_screen, dubbing, gaming_ai, youtube_publish
    enabled         BOOLEAN DEFAULT true,
    workspace_ids   UUID[],                        -- NULL = global, array = per-workspace
    metadata        JSONB DEFAULT '{}',
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Render Specs (reproducibility — version everything)
CREATE TABLE render_specs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clip_id         UUID REFERENCES clips(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL,
    pipeline_version VARCHAR(20) NOT NULL,         -- e.g. "1.2.0"
    ffmpeg_version  VARCHAR(20),
    model_hashes    JSONB DEFAULT '{}',            -- {"whisperx": "abc123", "yolov8": "def456"}
    effects_config  JSONB NOT NULL,                -- Full effects graph config
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(clip_id, version)
);

-- Indexes
CREATE INDEX idx_jobs_workspace  ON jobs(workspace_id);
CREATE INDEX idx_jobs_status     ON jobs(status);
CREATE INDEX idx_jobs_mode       ON jobs(mode);
CREATE INDEX idx_jobs_created    ON jobs(workspace_id, created_at DESC);
CREATE INDEX idx_jobs_monitor    ON jobs(stream_monitor_id) WHERE stream_monitor_id IS NOT NULL;
CREATE INDEX idx_clips_job       ON clips(job_id);
CREATE INDEX idx_clips_status    ON clips(status);
CREATE INDEX idx_clips_virality  ON clips(virality_score DESC);
CREATE INDEX idx_clips_workspace ON clips(workspace_id, created_at DESC);
CREATE INDEX idx_pubs_clip       ON publications(clip_id);
CREATE INDEX idx_pubs_account    ON publications(account_id);
CREATE INDEX idx_pubs_status     ON publications(status);
CREATE INDEX idx_pubs_schedule   ON publications(platform, status, scheduled_at);
CREATE INDEX idx_calendar_ws     ON calendar_events(workspace_id, scheduled_at);
CREATE INDEX idx_accounts_ws     ON connected_accounts(workspace_id, platform);
CREATE INDEX idx_monitors_active ON stream_monitors(is_active) WHERE is_active = true;
CREATE INDEX idx_members_ws      ON workspace_members(workspace_id);
CREATE INDEX idx_audit_ws        ON audit_logs(workspace_id, created_at DESC);
CREATE INDEX idx_steps_job       ON job_steps(job_id);
CREATE INDEX idx_frames_job_ts   ON frame_metadata(job_id, timestamp);
CREATE INDEX idx_comments_clip   ON clip_comments(clip_id);
CREATE INDEX idx_usage_ws        ON usage_events(workspace_id, created_at DESC);
CREATE INDEX idx_render_clip     ON render_specs(clip_id, version);
```

> [!IMPORTANT]
> **PgBouncer required.** With API workers + Celery workers + stream workers all connecting to Postgres, enable PgBouncer (built into Supabase) in transaction mode to multiplex connections and prevent `max_connections` exhaustion.

---

## 6. API Contract

> [!IMPORTANT]
> All endpoints require `Authorization: Bearer {jwt}`. All list endpoints enforce pagination via `?page=1&limit=20` (default `limit=20`, max `100`). Response: `{"data": [...], "total": int, "page": int, "pages": int}`. Job creation supports `Idempotency-Key` header to prevent duplicates.

### Global Error Schema

```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "Job with ID abc123 does not exist",
    "trace_id": "tr-789xyz"
  }
}
```

All endpoints return this shape on 4xx/5xx. The `code` is a machine-readable enum, `trace_id` links to OpenTelemetry spans.

### Rate Limiting Strategy

> [!NOTE]
> **slowapi (Redis-backed) is the source of truth** for all endpoint-specific rate limits. Distributed counters ensure consistency across replicas. Nginx provides only coarse global protection (e.g., total RPS cap) -- endpoint-specific policies stay in code.

### SSRF Protection

> [!CAUTION]
> The `source_url` field in `POST /api/jobs` is validated against a **strict domain allowlist**: `youtube.com`, `youtu.be`, `twitch.tv`, `vimeo.com`, `instagram.com`, `tiktok.com`, `x.com`. All private IP ranges (`10.x`, `172.16-31.x`, `192.168.x`, `169.254.x`, `localhost`) are blocked.

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Create account |
| `POST` | `/api/auth/login` | Get JWT access + refresh tokens |
| `POST` | `/api/auth/refresh` | Rotate refresh token, get new access token |
| `POST` | `/api/auth/logout` | Invalidate refresh token |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/jobs` | Create new clipping job |
| `GET` | `/api/jobs` | List jobs (paginated) |
| `GET` | `/api/jobs/{id}` | Job details + step statuses |
| `DELETE` | `/api/jobs/{id}` | Cancel running job or soft-delete |
| `POST` | `/api/jobs/{id}/retry` | Retry failed step |
| `POST` | `/api/jobs/{id}/pause` | Pause after current step |
| `POST` | `/api/jobs/{id}/resume` | Resume paused job |
| `GET` | `/api/jobs/{id}/transcript` | Fetch transcript from R2 |
| `GET` | `/api/jobs/{id}/steps` | List all step statuses with timings |
| `GET` | `/api/jobs/{id}/usage` | Cost/usage breakdown for this job |
| `GET` | `/api/jobs/{id}/export/report` | Export job report (JSON/PDF) |
| `WS` | `/ws/jobs/{id}` | Real-time progress (step transitions, % progress) |

```python
# POST /api/jobs — Request
{
  "mode": "vod",
  "source_url": "https://youtu.be/...",  # OR source_file_key
  "source_file_key": null,
  "settings": {
    "max_clips": 10,
    "clip_duration": {"min": 15, "max": 90},
    "aspect_ratio": "9:16",
    "caption_style": "highlight_word",
    "brand_kit_id": "uuid",
    "auto_publish": false
  }
}

# Response
{ "id": "uuid", "status": "created" }
```

### Uploads (Presigned URLs)

> [!NOTE]
> Backend **never** receives the file body. Frontend uploads directly to R2 via presigned URL.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/presign` | Get presigned R2 upload URL + file key |
| `POST` | `/api/upload/confirm` | Confirm upload, triggers validation |

> [!NOTE]
> Presigned upload URLs enforce `content-length-range` (max 2GB) and `content-type` (video/* only) **in the S3 signature policy**. Oversized/wrong-type uploads are rejected at the Cloudflare layer before completing.

### Clips

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/clips` | List clips (paginated, filter by job/status) |
| `GET` | `/api/clips/{id}` | Clip details + download URL |
| `PATCH` | `/api/clips/{id}` | Update metadata only (title, description) |
| `POST` | `/api/clips/{id}/rerender` | Re-render with new settings → increments version |
| `GET` | `/api/clips/{id}/download` | Presigned R2 download URL |
| `GET` | `/api/clips/{id}/nle` | Download NLE project file |
| `GET` | `/api/clips/{id}/versions` | List all render versions |
| `POST` | `/api/clips/batch/delete` | Bulk delete selected clips |
| `POST` | `/api/clips/batch/publish` | Bulk publish selected clips |

### Streams

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/streams` | Start monitoring |
| `GET` | `/api/streams` | List monitors (paginated) |
| `GET` | `/api/streams/{id}` | Monitor details + current status |
| `DELETE` | `/api/streams/{id}` | Stop monitoring |
| `POST` | `/api/streams/{id}/pause` | Pause monitoring |
| `POST` | `/api/streams/{id}/resume` | Resume monitoring |
| `GET` | `/api/streams/{id}/clips` | Get clips (paginated) |
| `GET` | `/api/streams/{id}/health` | Buffer status, latency, transcript lag |
| `WS` | `/ws/stream/{id}` | Real-time clip notifications (via WS gateway) |

### WebSocket Endpoints

> [!NOTE]
> Public WS endpoints (served by the dedicated Socket.IO gateway, not main API):
> - `wss://realtime.clippedai.com/jobs/{id}` -- job progress
> - `wss://realtime.clippedai.com/stream/{id}` -- stream clip notifications
>
> **Auth handshake**: Client sends JWT as `auth.token` in Socket.IO connection params. Gateway validates (shared JWT secret), checks workspace/job/stream access, then subscribes to Redis channels. On token expiry (15min), gateway sends `token_expired` event; client must reconnect with refreshed token.

### OAuth (Platform Accounts)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/auth/{platform}/login` | Redirect to OAuth consent |
| `GET` | `/api/auth/{platform}/callback` | Handle callback, store tokens |
| `GET` | `/api/auth/accounts` | List connected accounts |
| `DELETE` | `/api/auth/accounts/{id}` | Disconnect account |

### Publishing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/publish` | Publish via connected account |
| `POST` | `/api/publish/schedule` | Schedule publication |
| `GET` | `/api/publish/calendar` | Calendar events (paginated) |
| `GET` | `/api/publish/analytics` | Cross-platform analytics |

### Workspaces

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/workspaces` | Create workspace |
| `GET` | `/api/workspaces` | List user's workspaces |
| `POST` | `/api/workspaces/{id}/invite` | Invite member |
| `PATCH` | `/api/workspaces/{id}/members/{uid}` | Update role |
| `DELETE` | `/api/workspaces/{id}/members/{uid}` | Remove member |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health + dependency checks |
| `GET` | `/api/health/gpu` | GPU queue depth, active workers |
| `POST` | `/api/webhooks/{platform}` | Incoming platform callbacks |
| `POST` | `/api/webhooks/subscribe` | Register outbound webhook URL |
| `GET` | `/api/webhooks/subscriptions` | List outbound webhook subscriptions |

### Search + Brand

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/search` | ClipAnything across workspace (paginated results) |
| `POST` | `/api/jobs/{job_id}/search` | Search within job |
| `POST` | `/api/brand` | Create brand kit |
| `GET` | `/api/brand` | List brand kits |
| `PATCH` | `/api/brand/{id}` | Update brand kit |

### Advanced (Phase 3+)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/clips/{id}/edit` | Agentic editor command |
| `POST` | `/api/clips/{id}/dub` | AI dubbing request |
| `POST` | `/api/jobs/{id}/qa` | Video Q&A query |
| `POST` | `/api/jobs/{id}/summarize` | Video summarization |
| `GET` | `/api/clips/{id}/comments` | List clip comments |
| `POST` | `/api/clips/{id}/comments` | Add clip comment (timestamp-anchored) |

---

## 7. GPU Routing Architecture

> [!IMPORTANT]
> GPU workloads are split into **dedicated queues** on Modal to prevent contention and cold starts. All Modal functions accept **idempotent inputs** (R2 paths) so retries are safe.

| Queue | GPU | Warm Pool | Max Concurrency | Workloads |
|-------|-----|:---------:|:---:|-----------|
| `asr` | A10G | 1 warm | 10 | WhisperX (VOD), faster-whisper (streaming) |
| `vision` | A10G | 1 warm | 10 | YOLOv8 face, CLIP indexing, eye contact, SAM |
| `render` | T4 or CPU | 0 (burst) | 100 | FFmpeg rendering |

```python
# Modal function routing
@modal.function(gpu="A10G", keep_warm=1, concurrency_limit=10, timeout=1200)
def transcribe_gpu(audio_r2_path: str, model_size: str = "large-v2") -> str:
    """Idempotent: same input → same output R2 path."""
    ...

@modal.function(gpu="A10G", keep_warm=1, concurrency_limit=10, timeout=900)
def track_faces_gpu(video_r2_path: str) -> str:
    ...

@modal.function(gpu="T4", concurrency_limit=100, timeout=300)
def render_clip_gpu(clip_spec: dict) -> str:
    """Used only for GPU-heavy effects (gaze, segmentation).
    Default rendering uses CPU. Switch to GPU when:
    - clip requires gaze correction or green screen
    - queue depth exceeds CPU_RENDER_THRESHOLD
    - per-clip render time from job_steps.duration_ms > 120s on CPU"""
    ...
```

### Batching & Caching

- **Transcript caching**: Transcripts cached in R2 by content hash — re-renders skip ASR
- **Embedding caching**: CLIP embeddings persist in Qdrant — re-renders skip indexing
- **Batch CLIP indexing**: Multiple frames per GPU call (batch_size=32)
- **ASR batching**: Up to 3 VOD audio files concatenated per GPU call for utilization
- **GPU admission control**: `concurrency_limit` prevents queue explosion; excess jobs queue with backpressure

### GPU Admission Control

> [!IMPORTANT]
> Under burst load (100+ concurrent jobs), admission control prevents cold start cascades.

```python
def check_gpu_admission(queue_name: str, workspace_id: str) -> str:
    """Returns: 'admit', 'queue', or 'reject'"""
    depth = redis.get(f"gpu_queue:{queue_name}:depth")
    ws_active = redis.get(f"gpu_ws:{workspace_id}:active")

    if ws_active >= WORKSPACE_CONCURRENCY_LIMIT:
        return "queue"           # Per-workspace fairness
    if depth > QUEUE_HARD_LIMIT:
        return "reject"          # Backpressure: tell user to retry later
    if depth > QUEUE_SOFT_LIMIT:
        trigger_predictive_warm(queue_name)  # Pre-warm Modal containers
        return "queue"
    return "admit"
```

**Backpressure behavior**: Queue depth < soft limit → admit immediately. Soft → hard limit → queue + pre-warm containers. Above hard limit → HTTP 429 with retry-after header.

### Tiered LLM Strategy

| Task | Model | Cost |
|------|-------|------|
| Event pre-screening | Cerebras (Llama-8B) | Cheap |
| Clip ranking (final) | Cerebras (Llama-70B) | Standard |
| Agentic editor | Cerebras (Llama-70B) | Standard |
| Fallback | Groq → Together.ai | Varies |

### Celery Queues (orchestration only)

| Queue | Workers | Purpose |
|-------|---------|---------|
| `default` | 2 | API tasks, DB updates, orchestration, step dispatch |
| `publish` | 1 | Platform API uploads |
| `stream` | Supervisor-managed | Per-stream workers (see Section 10) |

---

## 8. Phase 1 — Core VOD Pipeline (MVP)

> **Goal**: Paste a YouTube URL → get back 5-10 short-form clips with captions.

### Module Specifications

#### [ingest.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/pipeline/ingest.py)

```
download_video(url: str, workspace_id: str) -> dict
  - Validate source_url against domain allowlist (SSRF protection)
  - yt-dlp: download best quality (up to 4K)
  - Route through rotating proxy (BrightData/Smartproxy) to avoid IP blocks
  - Retry with proxy rotation on 429/403 errors (max 3 attempts)
  - Upload to R2 at {workspace_id}/sources/{job_id}/video.mp4
  - Returns {r2_path, title, duration, resolution, fps}

validate_upload(file_key: str, workspace_id: str) -> dict
  - Verify file exists in R2 under correct workspace prefix
  - FFmpeg probe: duration, resolution, codec
  - Enforce: max duration (4hr), max resolution (4K), allowed codecs (H.264, H.265, VP9)
  - Enforce: max bitrate (50 Mbps)
  - Reject invalid / corrupt / non-video files
```

> [!WARNING]
> **Proxy rotation required.** YouTube and Twitch aggressively block data center IPs. Without a rotating residential proxy, the ingest pipeline will fail in production. Configure via `PROXY_URL` env var.

> [!NOTE]
> **No direct file uploads to backend.** Presigned R2 URL -> frontend uploads -> backend validates. Upload URLs enforce `content-length-range` and `content-type` in the S3 signature policy.

#### [transcribe.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/pipeline/transcribe.py)

> [!IMPORTANT]
> **Tiered ASR strategy.** VOD uses WhisperX large-v2 for accuracy. Streaming uses faster-whisper small for speed. Deepgram API is the managed fallback.

```
transcribe_and_diarize(audio_path: str, tier: str = "accurate") -> dict
  Tier "accurate" (VOD):
    - WhisperX large-v2, batch_size=16
    - WhisperX align() → word timestamps
    - Selective diarization: only on segments with speaker changes
      (skip monologue sections to save GPU/CPU)
  Tier "fast" (streaming):
    - faster-whisper small, beam_size=1
    - Runs LOCAL on stream container CPU (no Modal round-trip)
    - Avoids: R2 upload -> Modal cold start -> R2 download -> inference latency
    - No diarization (deferred to post-pass)
  Fallback:
    - Deepgram API (if self-hosted models fail)

  Returns {language, segments: [{start, end, text, speaker, words}]}
  Runs on Modal GPU (asr queue, A10G)
```

#### [face_track.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/pipeline/face_track.py)

```
track_faces(video_path: str, sample_fps: int = 2) -> list[dict]
  - YOLOv8-face for detection (handles low light, profile, multi-face)
  - InsightFace for re-identification across frames
  - Returns [{timestamp, faces: [{x, y, w, h, confidence, face_id}]}]
  - Runs on Modal GPU (vision queue)
```

#### [clip_selector.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/pipeline/clip_selector.py)

> [!IMPORTANT]
> **LLM ranks — it does NOT generate timestamps.** Candidate moments are derived from transcript boundaries + scene boundaries + energy spikes. The LLM selects and ranks from these candidates, preventing hallucinated timestamps.

```
select_clips(transcript, scenes, audio_analysis, settings) -> list[dict]
  Step 1 — Generate candidates:
    - Sliding window over transcript (30-90s windows at natural breaks)
    - Score each window: energy peaks × scene density × keyword triggers
    - Top 3× max_clips candidates

  Step 2 — LLM ranking:
    - Send candidate list (not full transcript) to Cerebras
    - LLM returns: ranking, virality_score, coherence_score, title, description, reasoning
    - Timestamps are LOCKED to candidate boundaries

  Fallback: if Cerebras fails → Groq → Together.ai

  Returns [{start, end, virality_score, coherence_score, title, description, hashtags}]
```

#### [captions.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/pipeline/captions.py)

> [!IMPORTANT]
> Generates `.ass` subtitle files → burns with FFmpeg `libass` in one pass. No Pillow frame rendering.

```
generate_ass(words, style_config) -> str
  - .ass file with per-word timing + ASS override tags
  - Active word: {\b1\c&H00FFFF&}, previous: {\alpha&H80&}
  - Auto-emoji insertion via dictionary
  - Returns path to .ass file

burn_captions(video_path, ass_path) -> str
  - FFmpeg: -vf "ass=filename.ass"
  - Returns path to captioned video
```

**Caption styles** (ASS animation tags): `highlight_word`, `karaoke`, `pop_in`, `typewriter`, `bounce`

#### Other Phase 1 Modules

```
audio_extract.py    — FFmpeg mono WAV 16kHz extraction
scene_detect.py     — PySceneDetect ContentDetector
audio_analysis.py   — RMS energy curves, peak detection (laughter, applause)
reframe.py          — Face-centered crop with exponential smoothing (α=0.15)
audio_enhance.py    — DeepFilterNet noise removal, -14 LUFS normalization, filler word removal
export.py           — FFmpeg H.264/265, brand kit overlay, upload to R2
nle_export.py       — FCPXML / Premiere XML / EDL generation
thumbnails.py       — Best frame + Pillow title overlay
```

### Phase 1 — Frontend

| Component | Description |
|-----------|-------------|
| `JobCreator` | URL input + presigned upload dropzone + settings |
| `Dashboard` | Job list with step-level progress bars |
| `ClipReview` | Video player + clip list, approve/reject/edit |
| `CaptionStyler` | Preview + customize caption style |
| `BrandKit` | Upload logo, set colors/fonts |

### Phase 1 — Verification

```bash
pytest backend/tests/test_pipeline.py -v
pytest backend/tests/test_state_machine.py -v
pytest backend/tests/test_api.py -v

# Integration
curl -X POST http://localhost:8000/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"mode":"vod","source_url":"https://youtu.be/dQw4w9WgXcQ","settings":{"max_clips":5}}'

# Verify:
# - Job transitions through state machine: created → ingesting → analyzing → clips_generated → rendering → review
# - Each step recorded in job_steps with timing
# - 3-5 clips generated, virality scores based on aligned candidates
# - Captions burned via .ass, NLE export works
# - Failed step retries up to max_attempts
```

---

## 9. Phase 0 — Auth + Workspaces (Build First)

> [!CAUTION]
> **Auth is Phase 0, not Phase 1.5.** Every module depends on `workspace_id` and `user_id`. Building the pipeline without auth populated means hardcoding GUIDs and a painful refactor later. Build users, workspaces, and JWT auth **first**.

> **Goal**: JWT login, workspace creation, basic RBAC — so all subsequent phases have real tenant context.

#### Auth Implementation

```
JWT flow:
  POST /api/auth/register → hash password (bcrypt) → create user + default workspace
  POST /api/auth/login    → verify password → issue access_token (15m) + refresh_token (7d)
  POST /api/auth/refresh  → verify refresh → rotate both tokens

Rate limiting:
  /api/auth/login  → 5 per minute per IP
  /api/auth/register → 3 per hour per IP
  All other endpoints → 100 per minute per user
```

#### OAuth Manager

```
class OAuthManager:
    get_auth_url(platform, user_id) → consent redirect URL
    handle_callback(platform, code, state) → store encrypted tokens
    get_valid_token(account_id) → auto-refresh if expired
    Encryption: Fernet (key from FERNET_KEY env var)
```

#### Phase 1.5 — OAuth + Publishing Verification

```bash
# Register + login
curl -X POST http://localhost:8000/api/auth/register \
  -d '{"email":"test@test.com","password":"test1234","name":"Test"}'
curl -X POST http://localhost:8000/api/auth/login \
  -d '{"email":"test@test.com","password":"test1234"}'
# Verify: returns access_token + refresh_token

# Connect YouTube -> browser OAuth flow -> verify connected_accounts row
# Publish clip -> verify appears as YouTube Short
```

> [!NOTE]
> **Frontend VOD state sync**: The `JobOrchestrator` emits events to a user-specific Redis pub/sub channel (`user:{id}:events`) on every step transition. The WebSocket gateway subscribes and pushes updates to the Dashboard for real-time progress bars.

---

## 10. Phase 2 — Livestream + Live Camera

> **Goal**: Monitor a live stream → auto-detect highlights via events → clip in near-real-time.

### Stream Supervisor Architecture

> [!IMPORTANT]
> Each region runs a **Stream Supervisor** that acts as an **orchestrator, not a host**. Instead of spawning local processes (single-node bottleneck), it uses the **Fly Machines API** to spawn an **isolated container per stream**. This ensures one heavy stream cannot starve others and enables horizontal scaling to 100+ concurrent streams.

```mermaid
graph TB
    API["FastAPI"] -->|start/stop| Q["Redis stream_control queue"]
    Q --> SUP["Stream Supervisor"]
    SUP -->|spawn/kill| W1["Worker: Stream A"]
    SUP -->|spawn/kill| W2["Worker: Stream B"]
    W1 --> |events| PS["Redis pub/sub"]
    W2 --> |events| PS
    PS --> WSG["WebSocket Gateway"]
    WSG -->|fan-out| U1["User 1"]
    WSG -->|fan-out| U2["User 2"]
```

**Per-stream container owns**: `StreamIngester` -> `DiskRingBuffer` -> `RealtimeTranscriber` -> `EventDetector`

**Supervisor responsibilities** (orchestrator pattern):
- Receive start/stop commands from `stream_control` Redis queue
- Call **Fly Machines API** to spawn/destroy per-stream containers
- Monitor heartbeats (`last_seen_at` in DB) — recreate stale containers
- Enforce max concurrent streams per workspace (admission control)
- Graceful shutdown: signal containers to drain ring buffers, flush pending clips

**Supervisor singleton + failover**:
- Deployed as a **single replica per region** (fixed replica count = 1)
- Heartbeat: updates `supervisor_lease` Redis key every 10s (TTL 30s)
- If another supervisor instance detects expired lease, it takes over
- On startup: clean stale `worker_id`/`locked_at` in `stream_monitors` table
- Stream claiming uses `SELECT ... FOR UPDATE SKIP LOCKED` to prevent duplicate workers

### WebSocket Gateway

> [!NOTE]
> WebSocket connections are handled by a **dedicated Socket.IO service** (not the main FastAPI app). It subscribes to Redis pub/sub channels for each stream/job and fans out to connected clients. Horizontal scaling uses **consistent hashing** on stream/job IDs for channel assignment + **sticky sessions** at the load balancer.
>
> **Auth**: JWT validated on handshake (shared secret). Workspace/stream access checked before channel subscription. Long-lived connections receive `token_expired` event on expiry; client must reconnect with refreshed JWT.

### Multi-Region Streaming

> [!TIP]
> For global users, deploy regional supervisors. Stream region = nearest ingest node. Workspace region affinity routes users to the closest supervisor.

```
Region routing:
  stream_region = nearest(user_ip, [us-east, eu-west, ap-south])
  supervisor = supervisors[stream_region]
  Data stays regional until clip finalized → replicated to primary R2
```

### Streaming Latency Targets

| Step | Target | Approach |
|------|--------|----------|
| Chunk ingestion | < 2s | streamlink + FFmpeg segmenting |
| Transcription | < 5s per 30s chunk | faster-whisper small, beam_size=1 |
| Event detection | < 1s | Heuristic triggers (no LLM) |
| LLM ranking | < 3s | Cerebras (if pre-gate fires) |
| Clip assembly | < 30s | Pre-cut from ring buffer |
| **Total end-to-end** | **< 45s** | — |

### Concurrency & Quotas

| Resource | Default Limit | Configurable |
|----------|:---:|:---:|
| Concurrent streams per workspace | 3 | Yes |
| Max concurrent VOD jobs per workspace | 5 | Yes |
| Minutes processed per month (free tier) | 300 | Plan-based |

#### [ring_buffer.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/streaming/ring_buffer.py) — Disk-backed

```
class DiskRingBuffer:
    def __init__(stream_id, capacity_seconds=600):
        - /data/streams/{stream_id}/ (mounted Fly Volume, NOT /tmp)
        - Deque of {filename, start_ts, end_ts} metadata (RAM only)
        - Enforce size limit: evict based on disk usage (max 2GB) AND time

    def add_chunk(audio, video, timestamp):
        - Write .ts segment to disk
        - Evict oldest when capacity exceeded (unlink)

    def get_window(start, end) -> str:
        - FFmpeg concat relevant .ts files → merged segment

    def cleanup():
        - rm -rf /tmp/streams/{stream_id}/
```

#### [event_detector.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/streaming/event_detector.py) — Event-Driven (replaces LLM polling)

> [!IMPORTANT]
> **Event-driven, not polling.** The LLM is a ranking layer, NOT the detector. Cheap heuristic triggers fire immediately; the LLM only runs for ranking when events accumulate.

```
class EventDetector:
    """Multi-signal event-driven clip detection."""

    def on_audio_chunk(audio: bytes):
        energy = compute_rms(audio)
        if energy > CROWD_THRESHOLD:
            emit_event("crowd_reaction", timestamp, energy)
        if energy > SHOUT_THRESHOLD:
            emit_event("loud_moment", timestamp, energy)

    def on_transcript_chunk(segments: list):
        for seg in segments:
            if matches_keyword_triggers(seg.text):
                emit_event("keyword_trigger", seg.start, seg.text)
            if sentiment_spike(seg.text):
                emit_event("emotional_moment", seg.start)

    def on_game_event(event: dict):   # From GamingDetector
        emit_event("game_highlight", event.timestamp, event)

    def evaluate_window(every_n_seconds=60):
        events = get_recent_events(window=300)  # last 5 min
        if len(events) < MIN_EVENT_THRESHOLD:
            return  # Skip LLM — nothing interesting
        # Score and cluster events
        candidates = cluster_events(events)
        ranked = llm_rank_candidates(candidates)
        if ranked[0].score > CLIP_THRESHOLD:
            trigger_clip_assembly(ranked[0])
```

### Event Scoring Algorithm

> [!IMPORTANT]
> This is **core IP** — the quality of clip detection depends on this scoring model.

```python
def compute_event_score(event_cluster: list[Event]) -> float:
    """Fused scoring across all signal types."""
    energy    = max(e.energy for e in cluster if e.type in ('crowd_reaction', 'loud_moment')) or 0
    keyword   = sum(KEYWORD_WEIGHTS.get(e.keyword, 0.5) for e in cluster if e.type == 'keyword_trigger')
    scene_d   = count_scene_changes(cluster.time_range) / cluster.duration
    sentiment = max_sentiment_delta(cluster)
    game      = sum(GAME_WEIGHTS[e.subtype] for e in cluster if e.type == 'game_highlight') or 0

    # Temporal decay: recent events score higher
    recency = exp(-0.01 * seconds_since(cluster.latest_event))

    score = recency * (
        (energy     * 0.30) +
        (keyword    * 0.25) +
        (scene_d    * 0.15) +
        (sentiment  * 0.15) +
        (game       * 0.15)
    )
    return min(score, 1.0)
```

| Signal | Weight | Source |
|--------|:---:|--------|
| Audio energy | 0.30 | RMS from audio chunks |
| Keyword triggers | 0.25 | Transcript keyword matching |
| Scene density | 0.15 | PySceneDetect change rate |
| Sentiment delta | 0.15 | Transcript sentiment shift |
| Game events | 0.15 | HUD detection + audio cues |
```

#### [gaming_detector.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/streaming/gaming_detector.py)

```
class GamingDetector:
    detect_events(frame, game, sample_fps=2) -> list[dict]:
      - Sample at 1-2 FPS (NOT every frame — too expensive)
      - Template matching for known HUD elements (more robust than OCR)
      - YOLOv8 HUD detector for kill feeds, scoreboards
      - EasyOCR fallback for text extraction
      - Audio cue detection: kill sounds, victory fanfares
```

> [!WARNING]
> **Experimental, limited-preview feature.** Phase 2 supports Valorant only with tuned templates. Multi-game support deferred to Phase 2.5 after baseline performance validation.

### Pre-Phase 2 Benchmark

> [!IMPORTANT]
> Before committing to streaming latency targets, run a **benchmark phase**: synthetic HLS streams (via FFmpeg) at 1, 3, 5, 10 concurrent streams on A10G. Measure real throughput for faster-whisper small + event detection. Use results to calibrate concurrency limits, workspace quotas, and latency promises.

#### Phase 2 — Verification

```bash
# Start stream monitor
curl -X POST http://localhost:8000/api/streams \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"platform":"twitch","channel_url":"https://twitch.tv/example"}'

# WebSocket for real-time
wscat -c ws://localhost:8000/ws/stream/{stream_id}

# Health check
curl http://localhost:8000/api/streams/{stream_id}/health

# Verify:
# - End-to-end latency < 45s
# - Events fire on crowd reactions / keyword triggers
# - LLM only called when events accumulate (not every 60s blindly)
# - Kill detection works on Valorant test VOD
```

---

## 11. Phase 3 — Advanced Effects + Editing

> **Goal**: Full post-production suite — visual effects, audio polish, ClipAnything, agentic editing.

### Module Specifications

#### [clip_anything.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/intelligence/clip_anything.py)

```
class ClipAnythingEngine:
    index_video(job_id, video_path):
      - Adaptive sampling: 1 FPS high-motion, 0.2 FPS low-motion
      - Encode with CLIP ViT-L/14
      - Qdrant collection per workspace (not per job)
      - Tag each point with job_id for filtering
      - TTL: embeddings cleaned up when source job deleted

    search(query, workspace_id, job_id=None):
      - CLIP text encoder → Qdrant cosine search
      - BM25 transcript search as fallback
      - Returns [{timestamp, score, thumbnail, context}]
```

#### Composable Effects Graph

> [!NOTE]
> Effects are applied as an **ordered pipeline** with clear inputs/outputs. Each step receives a video path and returns a video path. Config determines which steps are active. Re-renders use the same graph with different config.

```
Effects pipeline (composable, ordered):
  base_cut → reframe → green_screen → effects → captions → audio_suite → dub → export

Each step is a Celery task wrapping a Modal function:
  apply_effects_chain(clip_id, config: EffectsConfig) → str
    config specifies which steps to run + their params
    Skipped steps are no-ops (passthrough)

Rendering defaults to CPU-first:
  - Zooms, transitions, captions, audio → CPU (saves ~35% GPU cost)
  - Only gaze correction + segmentation → GPU
```

### Render Reproducibility

> [!IMPORTANT]
> Every render is stored with a full `render_spec` (see DB schema) containing pipeline version, FFmpeg version, model hashes, and effects config. Re-renders with the same spec produce identical output.

#### [effects.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/pipeline/effects.py)

> [!WARNING]
> Eye contact, green screen, and B-roll are **optional premium features** — GPU-heavy, off by default. Show estimated cost/time in UI before applying.

```
Magic Zooms       — FFmpeg zoompan filter (smooth_in, snap, ken_burns, pulse) [CPU]
Eye Contact       — Gaze redirection model (Modal vision queue) [GPU, ~$0.02/min]
AI Green Screen   — SAM segmentation + composite (Modal vision) [GPU, ~$0.03/min]
AI B-roll         — Pexels API stock footage + crossfade [CPU]
Transitions       — FFmpeg xfade (crossfade, wipe, zoom, glitch, swipe) [CPU]
```

#### [hooks.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/pipeline/hooks.py) + [agentic_editor.py](file:///Users/ebelthomasseiko/Desktop/ClippedAI/backend/intelligence/agentic_editor.py)

```
generate_hook(transcript, title) → LLM-generated 3-5s hook text
render_hook(hook, style) → video segment prepended with transition

AgenticEditor.execute(command: str, clip_id: str) -> ActionPlan:
  Step 1 - Retrieve relevant transcript segments via Qdrant (NOT full transcript)
    Prevents context window overflow on 3hr+ videos
  Step 2 - LLM parses command + context -> structured ActionPlan:
    {"actions": [{"type": "trim", "params": {"duration": 60}}, ...]}
  Step 3 - Validate ActionPlan against clip constraints:
    - Duration bounds, available tracks, allowed operations
    - Reject unsafe/impossible actions with error message
  Step 4 - Return preview (dry run) to user for confirmation
  Step 5 - Execute confirmed plan -> re-render via effects graph
  All agentic edits logged to audit_logs for traceability

  Allowed action types: trim, cut, add_sfx, add_zoom, change_captions,
    add_hook, change_speed, add_broll, change_layout
```

### Audio Suite

```
Audio ducking    — VAD-triggered volume automation (-12dB during speech)
Beat-sync        — librosa beat detection → FFmpeg cut-on-beat
Auto-censor      — Profanity list → bleep/silence + *** in captions
Sound effects    — Keyword → SFX mapping (whoosh, cash register, etc.)
```

---

## 12. Phase 4 — Distribution + Intelligence

> **Goal**: Multi-platform publish, scheduling, AI dubbing, summarization, analytics.

### Publisher (uses OAuth tokens from Phase 1.5)

```
PlatformPublisher.publish(clip_id, account_ids, metadata):
  1. OAuthManager.get_valid_token(account_id)
  2. Format metadata per platform
  3. Upload via API
  4. Store platform_post_id
  Platforms: YouTube, TikTok, Instagram, X, LinkedIn
```

### Other Phase 4 Modules

```
ContentScheduler  — Calendar + ETA tasks + optimal posting times
VideoSummarizer   — LLM summary + key points + chapters
VideoQA           — RAG over transcript chunks (sentence-transformers)
Team Collab       — Workspace roles, approval workflows, comment threads, audit log
```

### AI Dubbing (scoped)

> [!WARNING]
> V1 is **translation + TTS only** — no lip-sync. Lip-sync is a separate future epic requiring video generation models.

```
AI Dubbing v1:
  1. Translate transcript via LLM (preserving timing hints)
  2. Generate speech with XTTS-v2 (clone speaker voice)
  3. Time-stretch to match original pacing
  4. Mix dubbed audio with ambient track
  5. Export new clip
  6. Supported: 10 languages initially (en, es, pt, fr, de, ja, ko, zh, ar, hi)
  Future: lip-sync via video model (separate epic)
```

> [!NOTE]
> **Dubbing and complex agentic edits run as child jobs**: a new `parent_job_id` FK on `jobs` references the original clip's job. This reuses the existing job state machine, progress WebSocket, and step tracking — so the frontend shows consistent "in-progress" states for all long-running operations.

---

## 13. Security Hardening

> [!CAUTION]
> Apply across ALL modules.

### Shell Injection Prevention

```python
# ✅ ALWAYS list format
subprocess.run(["ffmpeg", "-i", input_path, "-vf", f"ass={ass_path}", output_path])
# ❌ NEVER shell=True
```

### Rules

- All filenames → UUID-based (user titles stored in DB only, never in filesystem)
- All R2 paths prefixed with `{workspace_id}/` — validated on every access
- OAuth tokens encrypted at rest (Fernet)
- Rate limiting on all endpoints (slowapi with Redis distributed counters)
- CORS restricted to frontend domain
- Input validation via Pydantic schemas on all endpoints
- MIME type validation on uploads (video/mp4, video/webm, audio/wav only)
- Presigned URLs with 1-hour TTL for downloads, IP-bound, max 3 downloads per URL
- Webhook signature verification (HMAC-SHA256) + **timestamp + nonce checking** for replay protection
- Rate-limit inbound webhooks per platform; log failures to audit_logs with `resource_type="webhook"`
- Secrets stored in platform secret managers (Railway/Fly secrets) — never committed to repo

### Fernet Key Rotation

- `connected_accounts.encryption_key_version` tracks which key encrypted each token
- Support multiple active Fernet keys in config (`FERNET_KEYS=key_v2,key_v1`)
- Decrypt attempts iterate keys newest-first; re-encrypt on next token refresh

### LLM Prompt Injection Protection

> [!CAUTION]
> User transcripts could contain adversarial text ("ignore previous instructions..."). All LLM inputs are sanitized.

```python
def sanitize_llm_input(text: str) -> str:
    """Strip known injection patterns from user-provided text."""
    text = re.sub(r'(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|above|prior)\s+(instructions|context)', '[REDACTED]', text)
    text = re.sub(r'(?i)system:\s*', '', text)
    return text[:MAX_CONTEXT_LENGTH]  # Truncate to prevent context overflow

# Applied before every LLM call that includes user content:
# - clip_selector (transcript)
# - agentic_editor (user command)
# - video_qa (user query)
```

### Per-Workspace Quotas

| Resource | Free Tier | Pro Tier |
|----------|:---------:|:--------:|
| VOD minutes/month | 300 | 3000 |
| Concurrent VOD jobs | 2 | 10 |
| Concurrent streams | 1 | 5 |
| Storage (R2) | 10 GB | 100 GB |
| Team members | 1 | 10 |

Enforced at API layer via `check_quota(workspace_id, resource)` dependency.

---

## 14. Monitoring & Observability

### Prometheus Metrics

```python
# Key metrics to expose
pipeline_step_duration     = Histogram("step_duration_seconds", labels=["step_name"])
pipeline_jobs_total        = Counter("jobs_total", labels=["mode", "status"])
gpu_queue_depth            = Gauge("gpu_queue_depth", labels=["queue_name"])
stream_latency             = Histogram("stream_e2e_latency_seconds")
llm_request_duration       = Histogram("llm_duration_seconds", labels=["provider"])
clip_render_duration       = Histogram("render_duration_seconds")
active_stream_monitors     = Gauge("active_streams")
```

### Tracing (OpenTelemetry)

- Span per pipeline step (ingest → transcribe → ... → export)
- Distributed trace across Celery → Modal → R2
- Parent span = job_id

> [!NOTE]
> **Audit logs are for user-facing actions only** (job.created, clip.published, member.added, etc.). High-volume operational events (GPU calls, stream chunks, transcription segments) go to **Prometheus metrics + OpenTelemetry traces**, not audit_logs.

| Alert | Condition | Action |
|-------|-----------|--------|
| Job stuck | Step running > 2× timeout | Page on-call |
| GPU queue depth | > 20 pending | Scale Modal concurrency |
| Stream latency | > 60s e2e | Warning notification |
| LLM failure rate | > 10% in 5min | Switch to fallback provider |
| R2 upload failure | Any | Retry with backoff |

---

## 15. Operational Details

### Data Retention

| Asset | Retention | Mechanism |
|-------|-----------|-----------|
| Source VODs | 7 days | R2 lifecycle rule |
| Analysis artifacts | 7 days | R2 lifecycle on `{ws_id}/analysis/` |
| Intermediate step outputs | 7 days | R2 lifecycle on `{ws_id}/steps/` |
| Rendered clips | Indefinite | User-managed |
| Stream buffer chunks | Immediate | `DiskRingBuffer.cleanup()` |
| Audit logs | 90 days | DB cleanup job |
| Job steps | 90 days | Archive to cold storage or compact |

> [!TIP]
> Partition `job_steps` by `created_at` month for query performance at scale.

### Frontend Real-Time

- **Jobs**: WebSocket `/ws/jobs/{id}` — step transitions, progress %, completion
- **Streams**: WebSocket `/ws/stream/{id}` — event detections, clip notifications
- **Fallback**: HTTP polling at 5s if WebSocket unavailable

### WhisperX Fallback Strategy

```
WhisperX large-v2 (primary) → faster-whisper large-v2 (no align) → Deepgram API (managed)
Diarization: WhisperX pipeline → pyannote standalone → speaker count only (no labels)
```

### Dependency License Audit

> [!CAUTION]
> Before implementation, **vet each ML dependency's license** for commercial use. Key concerns:
> - pyannote-audio: requires HuggingFace agreement
> - XTTS-v2 (Coqui): check MPL-2.0 terms for commercial TTS
> - WhisperX: MIT license, but underlying Whisper is MIT
> - Pinned versions in Docker images; upgrades tested in staging first

### Platform API Versioning

- Each platform SDK pinned to specific API version in `distribution/platforms/*`
- All platform calls behind stable internal interface (`PlatformAdapter`)
- Feature flags per platform — disable broken integrations instantly without deploy

### Model Fallback Matrix

| Capability | Primary | Fallback 1 | Fallback 2 |
|-----------|---------|-----------|-----------|
| LLM | Cerebras | Groq | Together.ai |
| ASR (VOD) | WhisperX | faster-whisper | Deepgram |
| ASR (stream) | faster-whisper | Deepgram | — |
| Face detection | YOLOv8-face | InsightFace | MediaPipe |
| Embeddings | open_clip ViT-L/14 | ViT-B/32 | — |

---

## 16. Configuration

```bash
# .env.example

# Core
DATABASE_URL=postgresql://user:pass@localhost:5432/clippedai
REDIS_URL=redis://localhost:6379/0

# Auth
JWT_SECRET_KEY=                  # For signing JWTs
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# API Keys
CEREBRAS_API_KEY=
GROQ_API_KEY=                    # LLM fallback
PEXELS_API_KEY=
HUGGINGFACE_TOKEN=
DEEPGRAM_API_KEY=                # ASR fallback

# Cloud Storage
R2_ACCOUNT_ID=
R2_ACCESS_KEY=
R2_SECRET_KEY=
R2_BUCKET=clippedai-storage
R2_PUBLIC_URL=

# Security
FERNET_KEY=                      # OAuth token encryption

# Platform APIs (OAuth credentials)
YOUTUBE_CLIENT_ID=
YOUTUBE_CLIENT_SECRET=
TIKTOK_CLIENT_KEY=
TIKTOK_CLIENT_SECRET=
INSTAGRAM_ACCESS_TOKEN=
TWITTER_BEARER_TOKEN=

# Modal
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=

# Qdrant (ClipAnything)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_PREFIX=clippedai_  # Per-workspace: clippedai_{workspace_id}

# App
MAX_UPLOAD_SIZE_MB=2000
DEFAULT_CLIP_DURATION_MIN=15
DEFAULT_CLIP_DURATION_MAX=90
WHISPERX_MODEL=large-v2
WHISPERX_BATCH_SIZE=16
STREAMING_ASR_MODEL=small        # faster-whisper model for streaming
```

---

## 17. Deployment

> [!TIP]
> For early stages, consolidate where possible: run Postgres + Redis on Fly.io to reduce vendor sprawl. Only externalize high-value services (Modal, Cerebras, R2). Add Supabase/Upstash when scaling requires it.

```mermaid
graph LR
    U["User"] --> V["Vercel (Frontend)"]
    U --> R2["Cloudflare R2 (Direct Upload)"]
    U --> WSG["WebSocket Gateway (Socket.IO)"]
    U --> A["FastAPI (Railway/Fly.io)"]
    A --> PG["PostgreSQL (Supabase)"]
    A --> R["Redis (Upstash)"]
    A --> M["Modal (GPU: ASR + Vision + Render)"]
    A --> R2
    A --> Q["Qdrant Cloud (Vector DB)"]
    A --> C["Cerebras API (LLM)"]
    A --> SUP["Stream Supervisor (Fly.io)"]
    SUP --> MM["Mediamtx"]
    A --> |pub/sub| R
    R --> WSG
    A --> P["Prometheus + Grafana"]
```

| Component | Host | Why |
|-----------|------|-----|
| Frontend | Vercel | CDN, instant deploys |
| API Server | Railway or Fly.io | Python, WebSocket support |
| PostgreSQL | Supabase | Managed, pgvector |
| Redis | Upstash | Serverless, Celery compatible |
| GPU (ASR) | Modal (A10G, warm) | WhisperX, faster-whisper |
| GPU (Vision) | Modal (A10G, warm) | YOLOv8, CLIP, gaze |
| GPU (Render) | Modal (T4, burst) | FFmpeg, auto-scales 0→100 |
| Object Storage | Cloudflare R2 | No egress, presigned uploads |
| Vector DB | Qdrant Cloud | Managed, per-workspace collections |
| LLM | Cerebras + Groq | Primary + fallback |
| Stream Supervisor | Fly.io (orchestrator) | Spawns per-stream **containers** via Fly Machines API |
| Stream Containers | Fly.io Machines (ephemeral) | Isolated per-stream: ingest + local ASR + detection |
| WebSocket Gateway | Fly.io (Socket.IO + Redis pub/sub) | Scalable WS fan-out |
| Monitoring | Grafana Cloud | Prometheus metrics + alerts |

> [!TIP]
> **Modal images are modular**: separate ASR image (WhisperX + faster-whisper), vision image (YOLOv8 + CLIP + gaze), and render image (FFmpeg). Avoid monolithic containers. Maintain a CUDA/PyTorch/driver compatibility matrix in the repo.

---

## 18. Testing Strategy

### Unit Tests
- State machine transitions + edge cases
- Orchestrator step dispatch logic
- Auth/permission checks
- Effects graph config validation

### Integration Tests
- **Golden VOD set**: 5 known public VODs (MrBeast, podcast, gaming, conference, music)
- Full pipeline end-to-end on each → verify: clip count, durations, caption presence
- Presigned upload → validate → process cycle

### Streaming Tests
- Synthetic HLS streams via FFmpeg → verify: latency < 45s, events fire, clips extracted
- Supervisor spawn/kill → verify: worker lifecycle, heartbeats, graceful shutdown

### GPU Benchmarks
- Transcribe 10min/30min/2hr VODs → measure time + cost per modal call
- Concurrent rendering: 10/50/100 clips → verify auto-scale behavior

### Load Tests
- 50 concurrent VOD jobs → verify: queue backpressure, no OOM, all complete
- 10 concurrent streams → verify: supervisor health, no resource leaks

---

## 19. Development Timeline Estimate

| Phase | Scope | Est. Duration |
|-------|-------|:---:|
| **Phase 0** | Auth + workspaces + RBAC + JWT (build first) | 1-2 weeks |
| **Phase 1** | VOD MVP (state machine + ingest -> clips -> dashboard) | 6-8 weeks |
| **Phase 1.5** | OAuth + YouTube publishing + quotas | 1 week |
| **Phase 2** | Livestream + supervisor + event detection + gaming AI | 4-6 weeks |
| **Phase 3** | Effects graph, ClipAnything, agentic editor | 4-6 weeks |
| **Phase 4** | Full distribution, dubbing (v1), collab, analytics | 3-5 weeks |
| **Total** | Full system | **19-28 weeks** |
