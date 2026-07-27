#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT_DIR/data/ollama.pid"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill "$PID" 2>/dev/null; then
    echo "Stopped Ollama server (pid $PID)"
  else
    echo "No running process for pid $PID (already stopped?)"
  fi
  rm -f "$PIDFILE"
else
  echo "No pidfile found at $PIDFILE — is the server running in the foreground instead?"
fi
