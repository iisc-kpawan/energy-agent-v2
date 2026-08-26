#!/usr/bin/env bash
set -euo pipefail
SESSION_PREFIX=${ENERGY_AGENT_SESSION_PREFIX:-energy-agent}
tmux kill-session -t "$SESSION_PREFIX-app" 2>/dev/null || true
tmux kill-session -t "$SESSION_PREFIX-mcp" 2>/dev/null || true
echo "Energy Agent services stopped: $SESSION_PREFIX."
