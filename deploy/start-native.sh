#!/usr/bin/env bash
set -euo pipefail
ROOT=${ENERGY_AGENT_ROOT:-/home/kpawan/energy-agent-v2}
ENV_FILE=${ENERGY_AGENT_ENV_FILE:-$ROOT/.env.native}
PY=${ENERGY_AGENT_PYTHON:-/home/kpawan/.local/envs/energy-agent-v2/bin/python}
EPLUS=${ENERGYPLUS_HOME:-/home/kpawan/.local/opt/energyplus-26.1.0}
SESSION_PREFIX=${ENERGY_AGENT_SESSION_PREFIX:-energy-agent}
MCP_ROOT="$ROOT/EnergyPlus-MCP/energyplus-mcp-server"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE; run $ROOT/deploy/configure-native.sh first." >&2
  exit 1
fi
set -a
source "$ENV_FILE"
set +a
export PATH="/home/kpawan/.local/envs/energy-agent-v2/bin:$EPLUS:$PATH"
export EPLUS_IDD_PATH="$EPLUS/Energy+.idd"
export MCP_WORKSPACE_ROOT="$MCP_ROOT"
export MCP_OUTPUT_DIR="$MCP_ROOT/outputs"
export MCP_SAMPLE_FILES_PATH="$MCP_ROOT/sample_files"
export MCP_TRANSPORT=streamable-http
export MCP_HTTP_HOST=127.0.0.1
export MCP_HTTP_PORT=${MCP_HTTP_PORT:-8000}
export MCP_TOKENS="[{\"label\":\"app\",\"token\":\"$MCP_TOKEN\"}]"
mkdir -p "$ROOT/runtime" "$MCP_ROOT/outputs" "$MCP_ROOT/logs" "$ROOT/runtime/logs"
tmux has-session -t "$SESSION_PREFIX-mcp" 2>/dev/null || tmux new-session -d -s "$SESSION_PREFIX-mcp" \
  "cd '$MCP_ROOT' && exec env PATH='$PATH' EPLUS_IDD_PATH='$EPLUS_IDD_PATH' MCP_WORKSPACE_ROOT='$MCP_WORKSPACE_ROOT' MCP_OUTPUT_DIR='$MCP_OUTPUT_DIR' MCP_SAMPLE_FILES_PATH='$MCP_SAMPLE_FILES_PATH' MCP_TRANSPORT='$MCP_TRANSPORT' MCP_HTTP_HOST='$MCP_HTTP_HOST' MCP_HTTP_PORT='$MCP_HTTP_PORT' MCP_TOKENS='$MCP_TOKENS' '$PY' -m energyplus_mcp_server.server >> '$ROOT/runtime/logs/mcp.log' 2>&1"
for _ in {1..30}; do
  curl -fsS "http://127.0.0.1:$MCP_HTTP_PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$MCP_HTTP_PORT/health" >/dev/null
tmux has-session -t "$SESSION_PREFIX-app" 2>/dev/null || tmux new-session -d -s "$SESSION_PREFIX-app" \
  "cd '$ROOT' && exec env PATH='$PATH' GOOGLE_API_KEY='$GOOGLE_API_KEY' APP_USERNAME='$APP_USERNAME' APP_PASSWORD='$APP_PASSWORD' ENERGYPLUS_MCP_TOKEN='$ENERGYPLUS_MCP_TOKEN' ENERGYPLUS_MCP_URL='$ENERGYPLUS_MCP_URL' GEMINI_MODEL='$GEMINI_MODEL' MEMORY_RECENT_TURNS='$MEMORY_RECENT_TURNS' MEMORY_SUMMARY_CHARS='$MEMORY_SUMMARY_CHARS' MAX_PARALLEL_AGENTS='$MAX_PARALLEL_AGENTS' ENERGY_AGENT_DATA_DIR='$ENERGY_AGENT_DATA_DIR' ENERGYPLUS_OUTPUT_ROOT='$ENERGYPLUS_OUTPUT_ROOT' PORT='$PORT' '$PY' agent.py >> '$ROOT/runtime/logs/app.log' 2>&1"
echo "Energy Agent services started: $SESSION_PREFIX (app $PORT, MCP $MCP_HTTP_PORT)."
