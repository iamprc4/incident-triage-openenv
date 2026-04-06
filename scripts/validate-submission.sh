#!/usr/bin/env bash
set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  echo "Error: directory not found"
  exit 1
fi

PING_URL="${PING_URL%/}"
echo "Step 1: Pinging $PING_URL/reset"
HTTP_CODE=$(curl -s -o /tmp/validate_resp.$$ -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
  echo "FAILED: /reset returned $HTTP_CODE"
  exit 1
fi

echo "Step 2: Building Docker image"
docker build "$REPO_DIR" --progress=plain >/tmp/docker_build.$$ 2>&1 || (tail -20 /tmp/docker_build.$$ && exit 1)

echo "Step 3: Running openenv validate"
cd "$REPO_DIR"
openenv validate

echo "All checks passed."
