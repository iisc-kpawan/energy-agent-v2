#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GOOGLE_API_KEY:-}" || -z "${APP_PASSWORD:-}" ]]; then
  cat <<'EOF'
Energy Agent was not started because required Codespaces secrets are missing.

Add these at https://github.com/settings/codespaces:
  GOOGLE_API_KEY  - a fresh Gemini API key
  APP_PASSWORD    - a strong password for the web login

Grant both secrets to iisc-kpawan/energy-agent-v2, then rebuild the codespace or run:
  bash deploy/start-codespace.sh
EOF
  exit 0
fi

export MCP_TOKEN="${MCP_TOKEN:-$(openssl rand -hex 32)}"
registry_token="${GHCR_TOKEN:-${GITHUB_TOKEN:-}}"
if [[ -n "$registry_token" ]]; then
  printf '%s' "$registry_token" | docker login ghcr.io -u "${GITHUB_USER:-iisc-kpawan}" --password-stdin
fi

docker compose -f compose.codespaces.yml pull
docker compose -f compose.codespaces.yml up -d
docker compose -f compose.codespaces.yml ps
echo "Energy Agent is starting on forwarded port 5000. Open it from the PORTS tab."
