#!/bin/bash
# Scripts to test backend triggering without the GUI via direct request

# Move to the root directory
cd "$(dirname "$0")/.." || exit 1

# Extract the dev PROCESS_VIDEO_ENDPOINT_AUTH from frontend/.env if it exists
if [ -f "frontend/.env" ]; then
  AUTH_TOKEN=$(grep '^PROCESS_VIDEO_ENDPOINT_AUTH=' frontend/.env | cut -d '=' -f2 | tr -d '"')
fi

# Fallback token for production if testing straight against OCI via secrets
if [ -z "$AUTH_TOKEN" ]; then
  AUTH_TOKEN="123123"
fi

# Ask the user for the instance URL if testing against production
read -p "Enter the Base URL to test (e.g. https://clippedai.app or http://localhost:3000): " BASE_URL
if [ -z "$BASE_URL" ]; then
  BASE_URL="https://clippedai.app"
fi

echo "Triggering backend E2E via Admin Direct Request to ${BASE_URL}..."

curl -X POST "${BASE_URL}/api/admin/trigger" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "source": {"url": "https://www.youtube.com/watch?v=YGOTBpTScR0"}
  }'

echo ""
echo "Done! If successful, the job has been queued to Modal."
echo "You can now log into ${BASE_URL}/admin in your browser to observe the generated clips."
