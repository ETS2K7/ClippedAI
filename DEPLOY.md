# ClippedAI — Deployment Guide (GitHub → DigitalOcean)

The deployment pipeline is: **push to `main` → GitHub Actions → DigitalOcean Droplet**.

---

## Architecture

```
Your Machine  →  git push  →  GitHub (main branch)
                                    ↓  (GitHub Actions triggers)
                              DigitalOcean Droplet
                                 git pull
                                 docker compose build app
                                 docker compose up -d
                              ↑
                           Nginx (80/443)
                           └─ Certbot SSL
                           └─ proxy → localhost:3000
                              └─ Next.js app container
                                 └─ PostgreSQL container
```

---

## One-Time Setup Steps

### Step 1 — Create GitHub Repo

1. Go to [github.com/new](https://github.com/new)
2. Create a **private** repo named `ClippedAI`
3. **Do not** initialize with README

Then push from your local machine:
```bash
cd /Users/ebelthomasseiko/ClippedAI
git remote add origin https://github.com/ETS2K7/ClippedAI.git
git push -u origin main
```

---

### Step 2 — Provision DigitalOcean Droplet

1. Go to your DigitalOcean Control Panel.
2. Click **Create** → **Droplets**.
3. Choose **Ubuntu 24.04 (LTS) x64**.
4. Choose a Basic Plan (e.g., $6/mo or $12/mo depending on your needs, using your $200 credit).
5. Add your SSH keys.
6. Click **Create Droplet**.
7. Once created, copy the **Public IP** of the Droplet.

---

### Step 3 — Point DNS

In your domain registrar for `clippedai.app`:

| Type | Name | Value |
|------|------|-------|
| A | @ | `YOUR_DigitalOcean_PUBLIC_IP` |
| A | www | `YOUR_DigitalOcean_PUBLIC_IP` |

Wait 5–15 minutes for propagation.

---

### Step 4 — Bootstrap the Server

```bash
./deploy/bootstrap.sh <YOUR_DO_PUBLIC_IP> https://github.com/ETS2K7/ClippedAI.git
```

This will:
- Install Docker, Nginx, Certbot
- Clone your repo to `/opt/clippedai`
- Generate a **deploy key** and print the private key

**Copy the printed private key** — you need it for GitHub Actions.

---

### Step 5 — Add GitHub Actions Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions**

Add every secret listed in [`deploy/github-secrets.md`](./github-secrets.md).

| The two infrastructure secrets: |
|---|
| `DigitalOcean_SERVER_IP` = your DigitalOcean public IP |
| `DigitalOcean_SSH_PRIVATE_KEY` = the private key printed by `bootstrap.sh` |

---

### Step 6 — Issue SSL Certificate

After DNS has propagated (verify with `dig +short clippedai.app`):

```bash
./deploy/setup-ssl.sh
```

---

### Step 7 — Trigger First Deploy

```bash
git push origin main
```

Watch it at: **GitHub → Actions tab → "Deploy to DigitalOcean"**

Then verify:
```bash
curl https://clippedai.app/api/health
# → {"status":"ok","timestamp":"..."}
```

---

## Ongoing Deployments

Every `git push origin main` automatically:
1. SSH into DigitalOcean server
2. Writes `.env.production` from GitHub secrets
3. `git pull` latest code
4. `docker compose up -d --build app` (rebuilds only the app layer)
5. Runs `prisma migrate deploy`
6. Health-checks the app

**Zero-downtime** — postgres container is never restarted unless you explicitly do so.

---

## Useful Commands (on the server)

```bash
# SSH in
ssh -i ~/.ssh/id_ed25519 root@YOUR_DigitalOcean_IP

# View live logs
cd /opt/clippedai && docker compose logs -f app

# Check container status
docker compose ps

# Manual redeploy
cd /opt/clippedai && git pull && docker compose up -d --build app

# Database backup
docker compose exec db pg_dump -U clippedai clippedai > backup_$(date +%Y%m%d).sql

# Renew SSL (runs automatically, but for manual test)
sudo certbot renew --dry-run
```

---

## Modal backend (GPU)

Use the Modal profile you want (`modal profile list` / `modal profile activate <name>`). Deploy from `backend/`:

```bash
cd backend
cp .env.example .env   # if needed — fill keys from .env.example

# One-time per workspace: create the secret Modal injects into the app
modal secret create clippedai-secret --from-dotenv .env --force

modal deploy modal_fast_asd.py   # Fast-ASD TalkNet (separate app)
modal deploy main.py             # prints PROCESS_VIDEO_ENDPOINT URL
```

Copy the `…--clippedai-clippedai-process-video.modal.run` URL into GitHub secret `PROCESS_VIDEO_ENDPOINT` and local `frontend/.env`. `AUTH_TOKEN` in `.env` must match `PROCESS_VIDEO_ENDPOINT_AUTH` on the frontend.

## YouTube Cookies (for yt-dlp)

```bash
# On your local machine — after logging into YouTube in browser
modal secret create youtube-cookies cookies.txt="$(cat /path/to/cookies.txt)"

# Redeploy Modal backend
cd ClippedAI/backend && modal deploy main.py
```
