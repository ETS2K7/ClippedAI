#!/usr/bin/env bash
# ============================================================
# ClippedAI — OCI Infrastructure Provisioning Script
# Provisions: VCN, subnet, security list, ARM A1 instance
# Usage: ./deploy/oci-provision.sh
# ============================================================
set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}==>${NC} ${BOLD}$*${NC}"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
die()     { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

# ── Prerequisites check ─────────────────────────────────────
command -v oci  >/dev/null 2>&1 || die "OCI CLI not found. Run: pip install oci-cli"
command -v jq   >/dev/null 2>&1 || die "jq not found. Run: brew install jq"
oci iam region list --output json >/dev/null 2>&1 || die "OCI CLI not authenticated. Run: oci setup config"

# ── Config ──────────────────────────────────────────────────
PROJECT_NAME="clippedai"
SSH_PUBLIC_KEY_PATH="${HOME}/.ssh/id_ed25519.pub"
SHAPE="VM.Standard.A1.Flex"
OCPUS=4
MEMORY_GB=24
BOOT_VOLUME_GB=100
STATE_FILE="$(dirname "$0")/oci-state.json"

# Detect tenancy and region from OCI config
TENANCY_OCID=$(oci iam tenancy get --output json 2>/dev/null | jq -r '.data.id' || echo "")
if [[ -z "$TENANCY_OCID" ]]; then
  TENANCY_OCID=$(grep '^tenancy' ~/.oci/config | head -1 | cut -d= -f2 | tr -d ' ')
fi
REGION=$(grep '^region' ~/.oci/config | head -1 | cut -d= -f2 | tr -d ' ')
[[ -z "$TENANCY_OCID" ]] && die "Could not detect tenancy OCID from ~/.oci/config"
[[ -z "$REGION" ]] && die "Could not detect region from ~/.oci/config"

COMPARTMENT_ID="$TENANCY_OCID"   # Use root compartment

info "Tenancy:     $TENANCY_OCID"
info "Region:      $REGION"
info "Shape:       $SHAPE ($OCPUS OCPU / ${MEMORY_GB}GB RAM)"
info "Boot volume: ${BOOT_VOLUME_GB}GB"
echo

# ── Step 1: Get availability domain ─────────────────────────
info "Fetching availability domain..."
AD=$(oci iam availability-domain list \
  --compartment-id "$COMPARTMENT_ID" \
  --output json | jq -r '.data[0].name')
success "Availability domain: $AD"

# ── Step 2: Create VCN ───────────────────────────────────────
info "Creating VCN..."
VCN_ID=$(oci network vcn create \
  --compartment-id "$COMPARTMENT_ID" \
  --display-name "${PROJECT_NAME}-vcn" \
  --cidr-block "10.0.0.0/16" \
  --dns-label "${PROJECT_NAME}vcn" \
  --wait-for-state AVAILABLE \
  --output json | jq -r '.data.id')
success "VCN created: $VCN_ID"

# ── Step 3: Internet Gateway ─────────────────────────────────
info "Creating Internet Gateway..."
IGW_ID=$(oci network internet-gateway create \
  --compartment-id "$COMPARTMENT_ID" \
  --vcn-id "$VCN_ID" \
  --is-enabled true \
  --display-name "${PROJECT_NAME}-igw" \
  --wait-for-state AVAILABLE \
  --output json | jq -r '.data.id')
success "Internet Gateway: $IGW_ID"

# ── Step 4: Route table ──────────────────────────────────────
info "Updating default route table..."
RT_ID=$(oci network route-table list \
  --compartment-id "$COMPARTMENT_ID" \
  --vcn-id "$VCN_ID" --output json | jq -r '.data[0].id')
oci network route-table update \
  --rt-id "$RT_ID" \
  --route-rules "[{\"networkEntityId\":\"$IGW_ID\",\"destination\":\"0.0.0.0/0\",\"destinationType\":\"CIDR_BLOCK\"}]" \
  --force --output json >/dev/null
success "Route table updated"

# ── Step 5: Security List ────────────────────────────────────
info "Configuring security list (SSH + HTTP + HTTPS)..."
SL_ID=$(oci network security-list list \
  --compartment-id "$COMPARTMENT_ID" \
  --vcn-id "$VCN_ID" --output json | jq -r '.data[0].id')
oci network security-list update \
  --security-list-id "$SL_ID" \
  --egress-security-rules '[{"destination":"0.0.0.0/0","protocol":"all","isStateless":false}]' \
  --ingress-security-rules '[
    {"source":"0.0.0.0/0","protocol":"6","isStateless":false,"tcpOptions":{"destinationPortRange":{"min":22,"max":22}}},
    {"source":"0.0.0.0/0","protocol":"6","isStateless":false,"tcpOptions":{"destinationPortRange":{"min":80,"max":80}}},
    {"source":"0.0.0.0/0","protocol":"6","isStateless":false,"tcpOptions":{"destinationPortRange":{"min":443,"max":443}}},
    {"source":"0.0.0.0/0","protocol":"1","isStateless":false,"icmpOptions":{"type":3,"code":4}},
    {"source":"0.0.0.0/0","protocol":"1","isStateless":false,"icmpOptions":{"type":3}}
  ]' \
  --force --output json >/dev/null
success "Security list configured"

# ── Step 6: Subnet ────────────────────────────────────────────
info "Creating public subnet..."
SUBNET_ID=$(oci network subnet create \
  --compartment-id "$COMPARTMENT_ID" \
  --vcn-id "$VCN_ID" \
  --availability-domain "$AD" \
  --display-name "${PROJECT_NAME}-subnet" \
  --cidr-block "10.0.0.0/24" \
  --dns-label "${PROJECT_NAME}sub" \
  --prohibit-public-ip-on-vnic false \
  --route-table-id "$RT_ID" \
  --security-list-ids "[\"$SL_ID\"]" \
  --wait-for-state AVAILABLE \
  --output json | jq -r '.data.id')
success "Subnet created: $SUBNET_ID"

# ── Step 7: Find Ubuntu 22.04 ARM image ───────────────────────
info "Finding Ubuntu 22.04 ARM64 image..."
IMAGE_ID=$(oci compute image list \
  --compartment-id "$COMPARTMENT_ID" \
  --operating-system "Canonical Ubuntu" \
  --operating-system-version "22.04" \
  --shape "$SHAPE" \
  --output json | jq -r \
  '[.data[] | select(.displayName | contains("aarch64") or test("ARM"; "i"))] | sort_by(."timeCreated") | last | .id')

if [[ -z "$IMAGE_ID" || "$IMAGE_ID" == "null" ]]; then
  # Fallback: get any Ubuntu 22.04 image
  IMAGE_ID=$(oci compute image list \
    --compartment-id "$COMPARTMENT_ID" \
    --operating-system "Canonical Ubuntu" \
    --operating-system-version "22.04" \
    --output json | jq -r \
    '.data | sort_by(."timeCreated") | last | .id')
fi
success "Image: $IMAGE_ID"

# ── Step 8: Read SSH public key ───────────────────────────────
[[ -f "$SSH_PUBLIC_KEY_PATH" ]] || die "SSH public key not found at $SSH_PUBLIC_KEY_PATH"
SSH_PUB_KEY=$(cat "$SSH_PUBLIC_KEY_PATH")

# ── Step 9: Create instance ────────────────────────────────────
info "Launching ARM A1 instance (this takes ~2 minutes)..."
INSTANCE_JSON=$(oci compute instance launch \
  --compartment-id "$COMPARTMENT_ID" \
  --availability-domain "$AD" \
  --shape "$SHAPE" \
  --shape-config "{\"ocpus\":$OCPUS,\"memoryInGBs\":$MEMORY_GB}" \
  --image-id "$IMAGE_ID" \
  --subnet-id "$SUBNET_ID" \
  --display-name "${PROJECT_NAME}-prod" \
  --assign-public-ip true \
  --ssh-authorized-keys-file "$SSH_PUBLIC_KEY_PATH" \
  --boot-volume-size-in-gbs "$BOOT_VOLUME_GB" \
  --wait-for-state RUNNING \
  --max-wait-seconds 300 \
  --output json)

INSTANCE_ID=$(echo "$INSTANCE_JSON" | jq -r '.data.id')
success "Instance created: $INSTANCE_ID"

# ── Step 10: Get public IP ─────────────────────────────────────
info "Fetching public IP..."
sleep 5
PUBLIC_IP=$(oci compute instance list-vnics \
  --instance-id "$INSTANCE_ID" \
  --output json | jq -r '.data[0]."publicIp"')
success "Public IP: ${GREEN}${BOLD}${PUBLIC_IP}${NC}"

# ── Save state ─────────────────────────────────────────────────
cat > "$STATE_FILE" <<EOF
{
  "instance_id": "$INSTANCE_ID",
  "public_ip": "$PUBLIC_IP",
  "vcn_id": "$VCN_ID",
  "subnet_id": "$SUBNET_ID",
  "region": "$REGION",
  "compartment_id": "$COMPARTMENT_ID"
}
EOF
success "State saved to $STATE_FILE"

# ── Final instructions ─────────────────────────────────────────
echo
echo -e "${BOLD}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✅  OCI Instance Provisioned Successfully!${NC}"
echo -e "${BOLD}════════════════════════════════════════════════${NC}"
echo
echo -e "  Public IP:  ${BOLD}${PUBLIC_IP}${NC}"
echo
echo -e "  ${YELLOW}Next steps:${NC}"
echo -e "  1. Add DNS A-record: ${BOLD}clippedai.app → ${PUBLIC_IP}${NC}"
echo -e "     and:              ${BOLD}www.clippedai.app → ${PUBLIC_IP}${NC}"
echo -e "  2. Run: ${BOLD}./deploy/bootstrap.sh${NC}"
echo
