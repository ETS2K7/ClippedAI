# ClippedAI — Deployment Guide (GitHub → OCI)

The deployment pipeline is: **push to `main` → GitHub Actions → OCI ARM server**.

---

## Architecture

```
Your Machine  →  git push  →  GitHub (main branch)
                                    ↓  (GitHub Actions triggers)
                              OCI ARM A1 Server
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

### Step 2 — Install OCI CLI

```bash
pip install oci-cli
oci --version  # verify
```

---

### Step 3 — Configure OCI CLI

```bash
oci setup config
```

You'll be prompted for:
- **Tenancy OCID** → OCI Console → Avatar → Tenancy → Copy OCID
- **User OCID** → OCI Console → Avatar → My Profile → Copy OCID
- **Region** → e.g. `ap-mumbai-1`
- **Generate new API key?** → Yes

Then paste the generated public key into:
OCI Console → Avatar → My Profile → API Keys → Add API Key → Paste public key

---

### Step 4 — Install jq (required by scripts)

```bash
brew install jq
```

---

### Step 5 — Provision OCI Instance

```bash
./deploy/oci-provision.sh
```

This takes ~3 minutes and outputs your **public IP**. Note it down.

---

### Step 6 — Point DNS

In your domain registrar for `clippedai.app`:

| Type | Name | Value |
|------|------|-------|
| A | @ | `YOUR_OCI_PUBLIC_IP` |
| A | www | `YOUR_OCI_PUBLIC_IP` |

Wait 5–15 minutes for propagation.

---

### Step 7 — Bootstrap the Server

```bash
./deploy/bootstrap.sh https://github.com/ETS2K7/ClippedAI.git
```

This will:
- Install Docker, Nginx, Certbot
- Clone your repo to `/opt/clippedai`
- Generate a **deploy key** and print the private key

**Copy the printed private key** — you need it for GitHub Actions.

---

### Step 8 — Add GitHub Actions Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions**

Add every secret listed in [`deploy/github-secrets.md`](./github-secrets.md).

| The two infrastructure secrets: |
|---|
| `OCI_SERVER_IP` = your OCI public IP |
| `OCI_SSH_PRIVATE_KEY` = the private key printed by `bootstrap.sh` |

---

### Step 9 — Issue SSL Certificate

After DNS has propagated (verify with `dig +short clippedai.app`):

```bash
./deploy/setup-ssl.sh
```

---

### Step 10 — Trigger First Deploy

```bash
git push origin main
```

Watch it at: **GitHub → Actions tab → "Deploy to Oracle Cloud"**

Then verify:
```bash
curl https://clippedai.app/api/health
# → {"status":"ok","timestamp":"..."}
```

---

## Ongoing Deployments

Every `git push origin main` automatically:
1. SSH into OCI server
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
ssh -i ~/.ssh/id_ed25519 ubuntu@YOUR_OCI_IP

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

## YouTube Cookies (for yt-dlp)

```bash
# On your local machine — after logging into YouTube in browser
modal secret create youtube-cookies cookies.txt="$(cat /path/to/cookies.txt)"

# Redeploy Modal backend
cd ClippedAI/backend && modal deploy main.py
```
