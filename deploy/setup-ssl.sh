#!/usr/bin/env bash
# ============================================================
# ClippedAI — SSL + Nginx Setup Script
# Run AFTER DNS has propagated (~15 mins after pointing A-record)
# Usage: ./deploy/setup-ssl.sh
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
DOMAIN="clippedai.app"
SSH_KEY="${HOME}/.ssh/id_ed25519"
SSH_USER="ubuntu"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

info "Verifying DNS for $DOMAIN points to $PUBLIC_IP..."
RESOLVED=$(dig +short "$DOMAIN" | head -1)
if [[ "$RESOLVED" != "$PUBLIC_IP" ]]; then
  warn "DNS not yet propagated. Resolved: '$RESOLVED', expected: '$PUBLIC_IP'"
  warn "Wait a few more minutes and retry."
  echo -e "Run: ${BOLD}dig +short $DOMAIN${NC}"
  exit 1
fi
success "DNS verified: $DOMAIN → $PUBLIC_IP"

info "Copying Nginx config and issuing SSL certificate..."
# Copy Nginx config
scp $SSH_OPTS \
  "$(dirname "$0")/../nginx/clippedai.conf" \
  "$SSH_USER@$PUBLIC_IP:/tmp/clippedai.conf"

ssh $SSH_OPTS "$SSH_USER@$PUBLIC_IP" bash << REMOTE
set -euo pipefail

echo "==> Installing Nginx config..."
sudo cp /tmp/clippedai.conf /etc/nginx/sites-available/clippedai
sudo ln -sf /etc/nginx/sites-available/clippedai /etc/nginx/sites-enabled/clippedai
sudo rm -f /etc/nginx/sites-enabled/default

# Temporarily use HTTP-only config for certbot challenge
sudo sed -i 's/listen 443 ssl http2;//; /ssl_certificate/d; /ssl_protocols/d; /ssl_ciphers/d' \
  /etc/nginx/sites-available/clippedai || true

echo "==> Testing Nginx config..."
sudo nginx -t
sudo systemctl reload nginx

echo "==> Issuing Let's Encrypt certificate..."
sudo certbot --nginx \
  -d ${DOMAIN} \
  -d www.${DOMAIN} \
  --non-interactive \
  --agree-tos \
  --email admin@${DOMAIN} \
  --redirect

echo "==> Restoring full Nginx config with SSL..."
sudo cp /tmp/clippedai.conf /etc/nginx/sites-available/clippedai
sudo nginx -t && sudo systemctl reload nginx

echo "==> Setting up Certbot auto-renewal..."
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

echo "==> SSL setup complete!"
REMOTE

success "SSL certificate issued for $DOMAIN"
echo
echo -e "${BOLD}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✅  SSL configured!${NC}"
echo -e "${BOLD}════════════════════════════════════════════════${NC}"
echo
echo -e "  Now push to GitHub main branch to deploy the app:"
echo -e "  ${BOLD}git push origin main${NC}"
echo
echo -e "  Then verify:"
echo -e "  ${BOLD}curl https://clippedai.app/api/health${NC}"
echo
