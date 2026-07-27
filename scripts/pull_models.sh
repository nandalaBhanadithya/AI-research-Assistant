#!/usr/bin/env bash
# Pulls the embedding + generation models into the project-scoped ./data/ollama_models
# directory. Requires the server from run_ollama.sh (BACKGROUND=1) to be running.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_MODELS="$ROOT_DIR/data/ollama_models"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"
GENERATION_MODEL="${OLLAMA_GENERATION_MODEL:-llama3.1:8b}"

echo "Pulling $EMBEDDING_MODEL (embeddings, ~274MB)..."
"$ROOT_DIR/bin/ollama" pull "$EMBEDDING_MODEL"

echo "Pulling $GENERATION_MODEL (generation)..."
"$ROOT_DIR/bin/ollama" pull "$GENERATION_MODEL"

echo "Done. Models stored under $OLLAMA_MODELS"
"$ROOT_DIR/bin/ollama" list
