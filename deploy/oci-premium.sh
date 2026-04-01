#!/bin/bash
# ==========================================
# OCI Premium AMD Server Launcher (Credit Usage)
# ==========================================
set -eo pipefail

LOG_FILE="$(dirname "$0")/oci-premium.log"
STATE_FILE="$(dirname "$0")/oci-state.json"
GITHUB_REPO="https://github.com/ETS2K7/ClippedAI.git"

echo "========================================================" | tee $LOG_FILE
echo "Launching Premium AMD Server (Consuming Credits) at $(date)" | tee -a $LOG_FILE
echo "Target: 4 OCPU, 24GB RAM, AP-HYDERABAD-1-AD-1 (VM.Standard.E4.Flex)" | tee -a $LOG_FILE
echo "Writing logs to $LOG_FILE" | tee -a $LOG_FILE
echo "========================================================" | tee -a $LOG_FILE

export SUPPRESS_LABEL_WARNING=True

# Known working Ubuntu 22.04 x86_64 Image OCID for Hyderabad
X86_IMAGE_ID="ocid1.image.oc1.ap-hyderabad-1.aaaaaaaay77sikrrnxr4fbgewjoigqnrpgjqr6yy6s7smpt6qzte4a226uba"

echo -n "[$(date)] Requesting Premium E4 Flex Server... " | tee -a $LOG_FILE

# Run the command and capture output
OUTPUT=$(oci compute instance launch \
  --compartment-id "ocid1.tenancy.oc1..aaaaaaaax6fjawzouo4vud2olkvcpls3mzwoeket3tirs2oqcm33s4uqzubq" \
  --availability-domain "gXtG:AP-HYDERABAD-1-AD-1" \
  --shape "VM.Standard3.Flex" \
  --shape-config '{"ocpus":4,"memoryInGBs":24}' \
  --image-id "$X86_IMAGE_ID" \
  --subnet-id "ocid1.subnet.oc1.ap-hyderabad-1.aaaaaaaaoiou3prgtz3yfjjoa425srktwikb7axs2j4xbsfm3q5gfprljy5a" \
  --display-name "clippedai-premium-temp" \
  --assign-public-ip true \
  --ssh-authorized-keys-file ~/.ssh/id_ed25519.pub \
  --boot-volume-size-in-gbs 100 \
  --output json 2>&1 || true)
  
if echo "$OUTPUT" | grep -q '"lifecycle-state": "PROVISIONING"'; then
  INSTANCE_ID=$(echo "$OUTPUT" | jq -r '.data.id')
  echo "SUCCESS!" | tee -a $LOG_FILE
  echo "==================================================" | tee -a $LOG_FILE
  echo "PREMIUM INSTANCE OFFICIALLY ALLOCATED!" | tee -a $LOG_FILE
  echo "Instance ID: $INSTANCE_ID" | tee -a $LOG_FILE
  echo "==================================================" | tee -a $LOG_FILE
  
  osascript -e 'display notification "Premium Server allocated! Bootstrapping..." with title "ClippedAI Premium Deploy"' || true
  
  echo "Waiting for instance to reach RUNNING state (takes ~60 seconds)..." | tee -a $LOG_FILE
  while true; do
    STATE=$(oci compute instance get --instance-id "$INSTANCE_ID" --output json | jq -r '.data."lifecycle-state"')
    if [ "$STATE" == "RUNNING" ]; then
      break
    fi
    sleep 5
  done
  
  echo "Gathering public IP..." | tee -a $LOG_FILE
  PUBLIC_IP=""
  while [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" == "null" ]; do
    PUBLIC_IP=$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" --output json 2>/dev/null | jq -r '.data[0]."public-ip"')
    sleep 3
  done
  
  echo "Public IP Address: $PUBLIC_IP" | tee -a $LOG_FILE
  
  echo "{\"public_ip\": \"$PUBLIC_IP\", \"instance_id\": \"$INSTANCE_ID\"}" > "$STATE_FILE"
  
  echo "Starting Server Bootstrap process directly onto the Premium Server..." | tee -a $LOG_FILE
  bash "$(dirname "$0")/bootstrap.sh" "$GITHUB_REPO" 2>&1 | tee -a $LOG_FILE
  
  osascript -e 'display notification "Premium Server fully deployed! Go to clippedai.app" with title "ClippedAI Premium Deploy"' || true
  
  echo "==================================================" | tee -a $LOG_FILE
  echo "PREMIUM DEPLOYMENT COMPLETE." | tee -a $LOG_FILE
  echo "Remember to terminate this server before your $300 trial ends!" | tee -a $LOG_FILE
  echo "==================================================" | tee -a $LOG_FILE
else
  echo "Failed to launch Premium Server." | tee -a $LOG_FILE
  echo "Raw output:" | tee -a $LOG_FILE
  echo "$OUTPUT" | tee -a $LOG_FILE
fi
