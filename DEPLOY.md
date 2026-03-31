# ClippedAI — Oracle Cloud Deployment Guide

## Prerequisites
- Oracle Cloud account (free tier)
- Domain `clippedai.app` registered (done ✓)
- SSH key pair generated

---

## Step 1 — Provision OCI ARM Instance

1. Log in to [cloud.oracle.com](https://cloud.oracle.com)
2. Go to **Compute → Instances → Create Instance**
3. Configure:
   - **Name:** `clippedai-prod`
   - **Image:** Ubuntu 22.04 (Minimal)
   - **Shape:** `VM.Standard.A1.Flex` (ARM Ampere)
   - **OCPUs:** 4
   - **Memory:** 24 GB
   - **Boot volume:** 100 GB (free tier allows 200GB total)
4. Upload your SSH public key
5. Click **Create**
6. Note the **Public IP address**

---

## Step 2 — Open Firewall Ports

### OCI Console (Security List)
1. Go to **Networking → Virtual Cloud Networks → Your VCN → Security Lists**
2. Add Ingress rules:
   | Protocol | Port | Source |
   |----------|------|--------|
   | TCP | 80 | 0.0.0.0/0 |
   | TCP | 443 | 0.0.0.0/0 |

### OS Firewall (on the VM)
```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

## Step 3 — Configure DNS

In your `name.com` dashboard for `clippedai.app`:

| Type | Name | Value |
|------|------|-------|
| A | @ | YOUR_OCI_PUBLIC_IP |
| A | www | YOUR_OCI_PUBLIC_IP |

Wait 5–15 minutes for DNS to propagate before proceeding.

---

## Step 4 — Server Setup

SSH into your instance:
```bash
ssh ubuntu@YOUR_OCI_PUBLIC_IP
```

### Install Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Install Docker Compose
```bash
sudo apt-get install -y docker-compose-plugin
docker compose version  # verify
```

### Install Nginx + Certbot
```bash
sudo apt-get update
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

---

## Step 5 — Get SSL Certificate

```bash
sudo certbot --nginx -d clippedai.app -d www.clippedai.app
```

Follow prompts. Certbot will auto-renew via cron.

---

## Step 6 — Configure Nginx

```bash
sudo cp /path/to/nginx/clippedai.conf /etc/nginx/sites-available/clippedai
sudo ln -s /etc/nginx/sites-available/clippedai /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default  # remove default site
sudo nginx -t                             # test config
sudo systemctl reload nginx
```

---

## Step 7 — Generate YouTube Cookies (for YouTube downloads)

On your **local machine** (not the server):

1. Install the browser extension **"Get cookies.txt LOCALLY"** (Chrome/Firefox)
2. Log into YouTube/Google in your browser
3. Navigate to `youtube.com`
4. Click the extension → Export cookies as `cookies.txt`
5. Create a Modal secret:

```bash
# Install Modal CLI if not already installed
pip install modal

# Create the secret (paste cookies.txt content when prompted)
modal secret create youtube-cookies cookies.txt="$(cat /path/to/cookies.txt)"
```

> **Note:** Refresh cookies every 3–6 months when they expire.

---

## Step 8 — Deploy the Application

### Clone repo on the server
```bash
git clone https://github.com/YOUR_USERNAME/ClippedAI.git ~/ClippedAI
cd ~/ClippedAI
```

### Create production env file
```bash
cp frontend/.env.production.template frontend/.env.production
nano frontend/.env.production
# Fill in all REPLACE_WITH_* values
```

Generate `AUTH_SECRET`:
```bash
openssl rand -hex 32
```

Set `POSTGRES_PASSWORD` in environment (used by docker-compose):
```bash
echo "POSTGRES_PASSWORD=YOUR_SECURE_PASSWORD" >> ~/.bashrc
source ~/.bashrc
```

### Configure Nginx (copy from repo)
```bash
sudo cp nginx/clippedai.conf /etc/nginx/sites-available/clippedai
sudo ln -sf /etc/nginx/sites-available/clippedai /etc/nginx/sites-enabled/clippedai
sudo nginx -t && sudo systemctl reload nginx
```

### Start services
```bash
cd ~/ClippedAI
docker compose up -d --build
docker compose logs -f  # watch startup logs
```

---

## Step 9 — Deploy Modal Backend

```bash
# From your local machine
cd ClippedAI/backend
pip install modal
modal deploy main.py
```

After deploying, copy the new endpoint URL and update `PROCESS_VIDEO_ENDPOINT` in `frontend/.env.production`.

Then restart the app:
```bash
# On the server
cd ~/ClippedAI
docker compose restart app
```

---

## Step 10 — Verify Deployment

```bash
# Health check
curl https://clippedai.app/api/health

# Check containers
docker compose ps

# Check logs
docker compose logs app --tail=50
docker compose logs db --tail=20
```

---

## Maintenance

### Update the app after code changes
```bash
cd ~/ClippedAI
git pull
docker compose up -d --build app
```

### Database backup
```bash
docker compose exec db pg_dump -U clippedai clippedai > backup_$(date +%Y%m%d).sql
```

### View live logs
```bash
docker compose logs -f app
```

### SSL certificate renewal (automatic, but manual test)
```bash
sudo certbot renew --dry-run
```
