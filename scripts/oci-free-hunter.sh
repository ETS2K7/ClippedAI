#!/usr/bin/env bash
# =============================================================================
# ClippedAI Instance Manager
# Keeps a paid A1.Flex alive as a bridge while continuously hunting for free
# tier capacity. When a free instance is found it updates Cloudflare DNS and
# prints migration instructions.
#
# Usage:  ./scripts/oci-free-hunter.sh [--skip-paid]
#   --skip-paid   Don't launch paid fallback (just hunt for free tier)
#
# Requires:
#   - OCI CLI configured (~/.oci/config)
#   - CF_API_TOKEN env var set (Cloudflare API token with DNS:Edit permission)
#   - SSH key at ~/.ssh/id_ed25519
# =============================================================================
set -euo pipefail
export SUPPRESS_LABEL_WARNING=True

# ── OCI constants ─────────────────────────────────────────────────────────────
TENANCY="ocid1.tenancy.oc1..aaaaaaaax6fjawzouo4vud2olkvcpls3mzwoeket3tirs2oqcm33s4uqzubq"
SUBNET="ocid1.subnet.oc1.ap-hyderabad-1.aaaaaaaaoiou3prgtz3yfjjoa425srktwikb7axs2j4xbsfm3q5gfprljy5a"
IMAGE="ocid1.image.oc1.ap-hyderabad-1.aaaaaaaa3mza2sx62iglmjxihlck45nhb3hwxnyzqckcagjlbfzeibae4kra"
AD="gXtG:AP-HYDERABAD-1-AD-1"
REGION="ap-hyderabad-1"
SSH_PUB_KEY="$HOME/.ssh/id_ed25519.pub"

# ── Cloudflare constants ───────────────────────────────────────────────────────
CF_ZONE_ID="32a0e77e54376d436fe2542b7da412a8"
CF_DOMAIN="clippedai.app"

# ── State tracking ─────────────────────────────────────────────────────────────
PAID_INSTANCE_ID_FILE="/tmp/clippedai_paid_instance_id"
FREE_INSTANCE_ID_FILE="/tmp/clippedai_free_instance_id"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helper: get current A record IP in Cloudflare ─────────────────────────────
cf_get_current_ip() {
  curl -sf "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/dns_records?type=A&name=${CF_DOMAIN}" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" | \
    python3 -c "import sys,json; r=json.load(sys.stdin); print(r['result'][0]['id'],r['result'][0]['content'])" 2>/dev/null || true
}

# ── Helper: update Cloudflare A record ────────────────────────────────────────
cf_update_dns() {
  local new_ip="$1"
  local record_info
  record_info=$(cf_get_current_ip)
  local record_id current_ip
  record_id=$(echo "$record_info" | awk '{print $1}')
  current_ip=$(echo "$record_info" | awk '{print $2}')

  if [[ -z "$record_id" ]]; then
    log "⚠️  Could not find Cloudflare A record — update DNS manually."
    return 1
  fi

  if [[ "$current_ip" == "$new_ip" ]]; then
    log "✅ Cloudflare DNS already points to $new_ip"
    return 0
  fi

  log "Updating Cloudflare DNS: $current_ip → $new_ip"
  curl -sf -X PATCH \
    "https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/dns_records/${record_id}" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"content\":\"${new_ip}\"}" | \
    python3 -c "import sys,json; r=json.load(sys.stdin); print('DNS updated ✅' if r.get('success') else 'DNS update FAILED ❌', r.get('errors',''))"
}

# ── Helper: launch OCI instance ───────────────────────────────────────────────
launch_instance() {
  local name="$1" ocpus="$2" memory_gb="$3"
  oci compute instance launch \
    --region "$REGION" \
    --availability-domain "$AD" \
    --compartment-id "$TENANCY" \
    --display-name "$name" \
    --shape "VM.Standard.A1.Flex" \
    --shape-config "{\"ocpus\":${ocpus},\"memoryInGBs\":${memory_gb}}" \
    --image-id "$IMAGE" \
    --subnet-id "$SUBNET" \
    --assign-public-ip true \
    --ssh-authorized-keys-file "$SSH_PUB_KEY" \
    --boot-volume-size-in-gbs 100 \
    2>&1
}

# ── Helper: get instance public IP (waits for RUNNING state) ──────────────────
wait_for_ip() {
  local instance_id="$1"
  local max_wait=300 elapsed=0
  log "Waiting for instance to reach RUNNING state..."
  while [[ $elapsed -lt $max_wait ]]; do
    local state ip
    state=$(oci compute instance get --instance-id "$instance_id" --region "$REGION" \
      --query 'data."lifecycle-state"' --raw-output 2>/dev/null || echo "UNKNOWN")
    if [[ "$state" == "RUNNING" ]]; then
      ip=$(oci compute instance list-vnics --instance-id "$instance_id" --region "$REGION" \
        --query 'data[0]."public-ip"' --raw-output 2>/dev/null || echo "")
      if [[ -n "$ip" && "$ip" != "null" ]]; then
        echo "$ip"
        return 0
      fi
    fi
    sleep 15
    elapsed=$((elapsed + 15))
  done
  log "⚠️  Instance did not reach RUNNING in ${max_wait}s"
  return 1
}

# ── Helper: terminate an instance by ID ───────────────────────────────────────
terminate_instance() {
  local instance_id="$1" name="$2"
  log "Terminating $name ($instance_id)..."
  oci compute instance terminate \
    --instance-id "$instance_id" \
    --region "$REGION" \
    --preserve-boot-volume false \
    --force 2>/dev/null && log "$name terminated ✅" || log "⚠️  Could not terminate $name — do it manually in OCI console"
}

# ── Step 1: Launch paid fallback instance ─────────────────────────────────────
launch_paid_instance() {
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "Launching PAID A1.Flex (2 OCPU / 12 GB) as bridge server..."
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local result
  result=$(launch_instance "clippedai-paid-bridge" 2 12)

  if echo "$result" | grep -q '"lifecycle-state"'; then
    local instance_id
    instance_id=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])")
    echo "$instance_id" > "$PAID_INSTANCE_ID_FILE"
    log "Paid instance created: $instance_id"

    local public_ip
    public_ip=$(wait_for_ip "$instance_id")
    log ""
    log "┌─────────────────────────────────────────────────────────────┐"
    log "│  ✅ PAID BRIDGE SERVER READY                                │"
    log "│  IP: $public_ip"
    log "│  SSH: ssh ubuntu@$public_ip"
    log "│  Cost: ~SGD 0.052/hr (~SGD 37/month)                       │"
    log "│                                                              │"
    log "│  NEXT STEP: Run the deployment on this server               │"
    log "│  See DEPLOY.md or run: ./scripts/deploy.sh $public_ip      │"
    log "└─────────────────────────────────────────────────────────────┘"
    log ""

    # Update Cloudflare DNS to paid instance
    if [[ -n "${CF_API_TOKEN:-}" ]]; then
      cf_update_dns "$public_ip"
    else
      log "⚠️  CF_API_TOKEN not set — update DNS manually:"
      log "    Cloudflare → DNS → A record clippedai.app → $public_ip"
    fi

    echo "$public_ip"
  else
    local error
    error=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message','Unknown error'))" 2>/dev/null || echo "$result")
    log "❌ Paid instance launch failed: $error"
    log "   Falling back to free-tier hunt only..."
    echo ""
  fi
}

# ── Step 2: Background free-tier hunter ───────────────────────────────────────
hunt_free_tier() {
  local paid_instance_id="${1:-}"
  local paid_ip="${2:-}"
  local attempt=0

  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "FREE TIER HUNTER started — polling every 5 minutes"
  log "Press Ctrl+C to stop (paid bridge will keep running)"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  while true; do
    attempt=$((attempt + 1))
    log "Attempt #${attempt}: trying free 4 OCPU / 24 GB A1.Flex..."

    local result
    result=$(launch_instance "clippedai-prod" 4 24 2>&1 || true)

    if echo "$result" | grep -q '"lifecycle-state"'; then
      local free_instance_id free_ip
      free_instance_id=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])")
      echo "$free_instance_id" > "$FREE_INSTANCE_ID_FILE"

      log ""
      log "┌─────────────────────────────────────────────────────────────┐"
      log "│  🎉 FREE TIER INSTANCE FOUND!                               │"
      log "│  Waiting for public IP...                                   │"
      log "└─────────────────────────────────────────────────────────────┘"

      free_ip=$(wait_for_ip "$free_instance_id")

      log ""
      log "┌─────────────────────────────────────────────────────────────┐"
      log "│  ✅ FREE A1.FLEX READY (4 OCPU / 24 GB)                    │"
      log "│  IP: $free_ip                                               │"
      log "│  SSH: ssh ubuntu@$free_ip                                  │"
      log "└─────────────────────────────────────────────────────────────┘"
      log ""

      # Update Cloudflare DNS to free instance
      if [[ -n "${CF_API_TOKEN:-}" ]]; then
        log "Pointing clippedai.app → free instance ($free_ip)..."
        cf_update_dns "$free_ip"
      else
        log "⚠️  Update DNS manually: clippedai.app → $free_ip"
      fi

      # Migration instructions
      log ""
      log "MIGRATION STEPS:"
      log "  1. Deploy ClippedAI to the free server:"
      log "     ssh ubuntu@$free_ip"
      log "     (run your deployment script)"
      log "  2. Verify clippedai.app is working on the free server"
      log "  3. Terminate the paid bridge (automatically in 10 min)..."

      # Wait 10 min for manual verification before terminating paid
      if [[ -n "$paid_instance_id" ]]; then
        log ""
        log "⏳ Waiting 10 minutes for you to verify the free server..."
        log "   To cancel termination, kill this script now (Ctrl+C)"
        sleep 600
        terminate_instance "$paid_instance_id" "clippedai-paid-bridge"
        rm -f "$PAID_INSTANCE_ID_FILE"
        log ""
        log "✅ Migration complete. clippedai.app → free A1.Flex @ $free_ip"
      fi

      break

    elif echo "$result" | grep -q "Out of host capacity"; then
      log "   No capacity yet. Next attempt in 5 minutes..."
    else
      local error
      error=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message',''))" 2>/dev/null || echo "Unknown error")
      log "   Unexpected error: $error — retrying in 5 min..."
    fi

    sleep 300
  done
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  local skip_paid=false
  [[ "${1:-}" == "--skip-paid" ]] && skip_paid=true

  log "ClippedAI Instance Manager starting..."

  if [[ -z "${CF_API_TOKEN:-}" ]]; then
    log ""
    log "⚠️  CF_API_TOKEN is not set."
    log "   DNS won't be updated automatically."
    log "   Get a token: Cloudflare → My Profile → API Tokens → Create Token"
    log "   Then run:  export CF_API_TOKEN=your_token_here"
    log ""
  fi

  local paid_instance_id="" paid_ip=""

  if [[ "$skip_paid" == false ]]; then
    paid_ip=$(launch_paid_instance || true)
    [[ -f "$PAID_INSTANCE_ID_FILE" ]] && paid_instance_id=$(cat "$PAID_INSTANCE_ID_FILE")
  fi

  hunt_free_tier "$paid_instance_id" "$paid_ip"
}

main "$@"
