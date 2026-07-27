#!/usr/bin/env bash
# Starts the project-scoped Ollama server: models are stored under ./data/ollama_models
# (never ~/.ollama) and the server only binds to localhost. Runs in the foreground unless
# BACKGROUND=1 is set, in which case it detaches and writes a pidfile for stop_ollama.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_MODELS="$ROOT_DIR/data/ollama_models"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
PIDFILE="$ROOT_DIR/data/ollama.pid"

if [ ! -x "$ROOT_DIR/bin/ollama" ]; then
  echo "Ollama binary not found. Run scripts/setup_ollama.sh first." >&2
  exit 1
fi

if [ "${BACKGROUND:-0}" = "1" ]; then
  nohup "$ROOT_DIR/bin/ollama" serve > "$ROOT_DIR/data/ollama.log" 2>&1 &
  echo $! > "$PIDFILE"
  echo "Ollama server started in background (pid $(cat "$PIDFILE")), logging to data/ollama.log"
  echo "Models directory: $OLLAMA_MODELS"
else
  echo "Starting Ollama server on $OLLAMA_HOST (models: $OLLAMA_MODELS)"
  exec "$ROOT_DIR/bin/ollama" serve
fi
