#!/bin/bash
# ==========================================
# OCI Free Tier Slot Sniper + Auto-Deploy
# ==========================================
set -eo pipefail

LOG_FILE="$(dirname "$0")/oci-retry.log"
STATE_FILE="$(dirname "$0")/oci-state.json"
GITHUB_REPO="https://github.com/ETS2K7/ClippedAI.git"

echo "========================================================" | tee -a $LOG_FILE
echo "Starting OCI Instance Sniping & Auto-Deploy at $(date)" | tee -a $LOG_FILE
echo "Target: 4 OCPU, 24GB RAM, AP-HYDERABAD-1-AD-1" | tee -a $LOG_FILE
echo "Writing logs to $LOG_FILE" | tee -a $LOG_FILE
echo "========================================================" | tee -a $LOG_FILE

while true; do
  export SUPPRESS_LABEL_WARNING=True
  
  echo -n "[$(date)] Attempting to launch... " | tee -a $LOG_FILE
  
  # Run the command and capture output
  OUTPUT=$(oci compute instance launch \
    --compartment-id "ocid1.tenancy.oc1..aaaaaaaax6fjawzouo4vud2olkvcpls3mzwoeket3tirs2oqcm33s4uqzubq" \
    --availability-domain "gXtG:AP-HYDERABAD-1-AD-1" \
    --shape "VM.Standard.A1.Flex" \
    --shape-config '{"ocpus":4,"memoryInGBs":24}' \
    --image-id "ocid1.image.oc1.ap-hyderabad-1.aaaaaaaa4pukhdpomirjcggqvg4zc4vbde3my5nxl4ccpksrgia4m5be3d2a" \
    --subnet-id "ocid1.subnet.oc1.ap-hyderabad-1.aaaaaaaaoiou3prgtz3yfjjoa425srktwikb7axs2j4xbsfm3q5gfprljy5a" \
    --display-name "clippedai-prod" \
    --assign-public-ip true \
    --ssh-authorized-keys-file ~/.ssh/id_ed25519.pub \
    --boot-volume-size-in-gbs 100 \
    --output json 2>&1 || true)
    
  # Check if CLI returned an instance ID
  if echo "$OUTPUT" | grep -q '"lifecycle-state": "PROVISIONING"'; then
    INSTANCE_ID=$(echo "$OUTPUT" | jq -r '.data.id')
    echo "SUCCESS!" | tee -a $LOG_FILE
    echo "==================================================" | tee -a $LOG_FILE
    echo "INSTANCE SUCCESSFULLY ALLOCATED!" | tee -a $LOG_FILE
    echo "Instance ID: $INSTANCE_ID" | tee -a $LOG_FILE
    echo "==================================================" | tee -a $LOG_FILE
    
    # Trigger notification
    osascript -e 'display notification "Instance allocated! Bootstrapping server..." with title "ClippedAI Deploy"' || true
    
    # Wait for instance to be RUNNING
    echo "Waiting for instance to reach RUNNING state..." | tee -a $LOG_FILE
    while true; do
      STATE=$(oci compute instance get --instance-id "$INSTANCE_ID" --output json | jq -r '.data."lifecycle-state"')
      if [ "$STATE" == "RUNNING" ]; then
        break
      fi
      sleep 5
    done
    
    # Extract Public IP
    echo "Gathering public IP..." | tee -a $LOG_FILE
    PUBLIC_IP=""
    while [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" == "null" ]; do
      PUBLIC_IP=$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" --output json 2>/dev/null | jq -r '.data[0]."public-ip"')
      sleep 3
    done
    
    echo "Public IP Address: $PUBLIC_IP" | tee -a $LOG_FILE
    
    # Write directly to oci-state.json for bootstrap.sh
    echo "{\"public_ip\": \"$PUBLIC_IP\", \"instance_id\": \"$INSTANCE_ID\"}" > "$STATE_FILE"
    
    # Pass control to bootstrap.sh
    echo "Starting Server Bootstrap..." | tee -a $LOG_FILE
    
    # We execute bootstrap.sh and tee output to both console and log
    bash "$(dirname "$0")/bootstrap.sh" "$GITHUB_REPO" 2>&1 | tee -a $LOG_FILE
    
    # Final Notification
    osascript -e 'display notification "Server is fully provisioned! Check oci-retry.log for your GitHub Key." with title "ClippedAI Deploy"' || true
    
    echo "==================================================" | tee -a $LOG_FILE
    echo "DEPLOYMENT PIPELINE COMPLETE. NO FURTHER ACTION BY THIS SCRIPT." | tee -a $LOG_FILE
    echo "==================================================" | tee -a $LOG_FILE
    exit 0
  else
    if echo "$OUTPUT" | grep -q "Out of host capacity"; then
      echo "Failed: Out of capacity. Waiting 120s." | tee -a $LOG_FILE
    elif echo "$OUTPUT" | grep -q "TooManyRequests"; then
      echo "Failed: Rate limited. Waiting 120s." | tee -a $LOG_FILE
    else
      echo "Failed: Unknown error. Waiting 120s." | tee -a $LOG_FILE
      echo "$OUTPUT" >> $LOG_FILE
    fi
  fi
  
  sleep 120
done
