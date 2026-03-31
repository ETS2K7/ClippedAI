#!/usr/bin/env bash
# ============================================================
# ClippedAI — Server Bootstrap Script
# Installs Docker, Nginx, Certbot on the OCI server
# Clones repo from GitHub and starts PostgreSQL
# Usage: ./deploy/bootstrap.sh [GITHUB_REPO_URL]
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}==>${NC} ${BOLD}$*${NC}"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
die()     { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

STATE_FILE="$(dirname "$0")/oci-state.json"
[[ -f "$STATE_FILE" ]] || die "State file not found. Run oci-provision.sh first."

PUBLIC_IP=$(jq -r '.public_ip' "$STATE_FILE")
[[ -z "$PUBLIC_IP" || "$PUBLIC_IP" == "null" ]] && die "No public IP in state file."

# GitHub repo URL — pass as arg or set here
GITHUB_REPO="${1:-}"
if [[ -z "$GITHUB_REPO" ]]; then
  echo -e "${YELLOW}Enter your GitHub repo URL (e.g. https://github.com/youruser/ClippedAI.git):${NC}"
  read -r GITHUB_REPO
fi
[[ -z "$GITHUB_REPO" ]] && die "GitHub repo URL is required."

SSH_KEY="${HOME}/.ssh/id_ed25519"
SSH_USER="ubuntu"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=30"

remote() { ssh $SSH_OPTS "$SSH_USER@$PUBLIC_IP" "$@"; }

# ── Wait for SSH to become available ──────────────────────────
info "Waiting for SSH on $PUBLIC_IP..."
for i in $(seq 1 30); do
  if ssh $SSH_OPTS "$SSH_USER@$PUBLIC_IP" "echo ok" 2>/dev/null | grep -q ok; then
    success "SSH is up"
    break
  fi
  [[ $i -eq 30 ]] && die "SSH timeout after 5 minutes"
  echo -n "."
  sleep 10
done

# ── Bootstrap on the remote server ────────────────────────────
info "Bootstrapping server at $PUBLIC_IP..."

remote bash << REMOTE
set -euo pipefail

echo "==> Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get upgrade -y -qq --no-install-recommends

echo "==> Installing Docker..."
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu
sudo systemctl enable docker
sudo systemctl start docker

echo "==> Installing Docker Compose plugin..."
sudo apt-get install -y docker-compose-plugin -qq

echo "==> Installing Nginx + Certbot..."
sudo apt-get install -y nginx certbot python3-certbot-nginx git jq -qq

echo "==> Configuring UFW firewall..."
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

echo "==> Opening OCI iptables ports..."
# OCI uses iptables by default even with UFW
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || sudo iptables-save | sudo tee /etc/iptables/rules.v4

echo "==> Cloning ClippedAI from GitHub..."
sudo mkdir -p /opt/clippedai
sudo chown ubuntu:ubuntu /opt/clippedai
if [[ -d /opt/clippedai/.git ]]; then
  cd /opt/clippedai && git pull
else
  git clone ${GITHUB_REPO} /opt/clippedai
fi

echo "==> Setting up deploy SSH key for GitHub Actions..."
# Generate a deploy key the CI runner will use
if [[ ! -f /home/ubuntu/.ssh/github_deploy ]]; then
  ssh-keygen -t ed25519 -C "github-actions-deploy" -f /home/ubuntu/.ssh/github_deploy -N ""
  echo
  echo "=== GITHUB ACTIONS DEPLOY KEY (add as repo secret OCI_SSH_PRIVATE_KEY) ==="
  cat /home/ubuntu/.ssh/github_deploy
  echo "=== END PRIVATE KEY ==="
  echo
  echo "=== AUTHORIZED PUBLIC KEY (already added to authorized_keys) ==="
  cat /home/ubuntu/.ssh/github_deploy.pub
  cat /home/ubuntu/.ssh/github_deploy.pub >> /home/ubuntu/.ssh/authorized_keys
  chmod 600 /home/ubuntu/.ssh/authorized_keys
fi

echo "==> Starting PostgreSQL container..."
cd /opt/clippedai
# Start only DB first so app can migrate on next deploy
POSTGRES_PASSWORD="\${POSTGRES_PASSWORD:-changeme}" \
  docker compose up -d db

echo "==> Server bootstrap complete!"
REMOTE

success "Bootstrap complete!"
echo
echo -e "${BOLD}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✅  Server is ready!${NC}"
echo -e "${BOLD}════════════════════════════════════════════════${NC}"
echo
echo -e "  ${YELLOW}Next steps:${NC}"
echo -e "  1. Copy the ${BOLD}OCI_SSH_PRIVATE_KEY${NC} printed above"
echo -e "     → GitHub repo → Settings → Secrets → Actions"
echo -e "  2. Add all secrets listed in ${BOLD}deploy/github-secrets.md${NC}"
echo -e "  3. Run: ${BOLD}./deploy/setup-ssl.sh${NC} (after DNS propagates)"
echo -e "  4. Push to main — GitHub Actions will auto-deploy"
echo
