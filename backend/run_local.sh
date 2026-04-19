#!/usr/bin/env bash
# run_local.sh — Start the local ClippedAI processing server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "local.env" ]; then
  echo "❌  local.env not found."
  echo "    Copy local.env.example to local.env and fill in your API keys."
  exit 1
fi

source local_venv/bin/activate

echo "🚀  Starting ClippedAI local dev server on http://localhost:8000"
echo "    Health check: http://localhost:8000/health"
echo ""

uvicorn local_dev_server:app --host 0.0.0.0 --port 8000 --reload
