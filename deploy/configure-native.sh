#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kpawan/energy-agent-v2
ENV_FILE="$ROOT/.env.native"
printf 'Gemini API key: '
IFS= read -rs GOOGLE_API_KEY
printf '\nApplication password for username energy-admin: '
IFS= read -rs APP_PASSWORD
printf '\n'
if [[ -z "$GOOGLE_API_KEY" || -z "$APP_PASSWORD" ]]; then
  echo 'API key and application password must not be empty.' >&2
  exit 1
fi
MCP_TOKEN="$(openssl rand -hex 32)"
umask 077
{
  printf 'GOOGLE_API_KEY=%s\n' "$GOOGLE_API_KEY"
  printf 'APP_USERNAME=energy-admin\n'
  printf 'APP_PASSWORD=%s\n' "$APP_PASSWORD"
  printf 'MCP_TOKEN=%s\n' "$MCP_TOKEN"
  printf 'ENERGYPLUS_MCP_TOKEN=%s\n' "$MCP_TOKEN"
  printf 'ENERGYPLUS_MCP_URL=http://127.0.0.1:8000/mcp\n'
  printf 'GEMINI_MODEL=gemini-3.1-flash-lite\n'
  printf 'MEMORY_RECENT_TURNS=12\n'
  printf 'MEMORY_SUMMARY_CHARS=10000\n'
  printf 'MAX_PARALLEL_AGENTS=4\n'
  printf 'ENERGY_AGENT_DATA_DIR=/home/kpawan/energy-agent-v2/runtime\n'
  printf 'ENERGYPLUS_OUTPUT_ROOT=/home/kpawan/energy-agent-v2/EnergyPlus-MCP/energyplus-mcp-server/outputs\n'
  printf 'PORT=5000\n'
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"
echo "Secrets saved securely to $ENV_FILE"
