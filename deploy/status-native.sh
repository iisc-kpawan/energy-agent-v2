#!/usr/bin/env bash
set -u
SESSION_PREFIX=${ENERGY_AGENT_SESSION_PREFIX:-energy-agent}
APP_PORT=${PORT:-5000}
MCP_PORT=${MCP_HTTP_PORT:-8000}
printf 'Sessions:\n'
tmux list-sessions 2>/dev/null | grep "$SESSION_PREFIX-" || true
printf '\nListeners:\n'
ss -lnt | grep -E ":($APP_PORT|$MCP_PORT) " || true
printf '\nMCP health:\n'
curl -fsS "http://127.0.0.1:$MCP_PORT/health" 2>/dev/null || echo 'not ready'
printf '\n\nApplication health:\n'
curl -fsS "http://127.0.0.1:$APP_PORT/api/health" 2>/dev/null || echo 'not ready'
printf '\n'
