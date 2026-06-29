#!/usr/bin/env bash
# Start the claude -p subscription shim, then the marimo editor.
# The shim is torn down automatically when marimo exits.
set -euo pipefail

PORT="${MARIMO_CLAUDE_SHIM_PORT:-8011}"

uv run python scripts/marimo_claude_shim.py &
SHIM_PID=$!
trap 'kill "$SHIM_PID" 2>/dev/null || true' EXIT

# Give the shim a moment to bind before marimo's first AI call.
for _ in $(seq 1 20); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 0.25
done

uv run marimo edit notebooks/
