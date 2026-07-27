#!/usr/bin/env bash
# Downloads a portable Ollama binary into ./bin — NOT installed system-wide, does not
# touch ~/.ollama, does not register a launchd/systemd service. Safe to delete the whole
# project folder and nothing global is left behind.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT_DIR/bin"
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.32.4}"

mkdir -p "$BIN_DIR"

if [ -x "$BIN_DIR/ollama" ]; then
  echo "Ollama binary already present at $BIN_DIR/ollama ($("$BIN_DIR/ollama" --version 2>&1 | head -1))"
  exit 0
fi

OS="$(uname -s)"
case "$OS" in
  Darwin)
    ASSET="ollama-darwin.tgz"
    ;;
  Linux)
    ARCH="$(uname -m)"
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
      ASSET="ollama-linux-arm64.tar.zst"
    else
      ASSET="ollama-linux-amd64.tar.zst"
    fi
    ;;
  *)
    echo "Unsupported OS: $OS. Install Ollama manually and place the binary at $BIN_DIR/ollama" >&2
    exit 1
    ;;
esac

URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/${ASSET}"
TMP_DIR="$(mktemp -d)"
echo "Downloading $URL ..."
curl -fL -o "$TMP_DIR/$ASSET" "$URL"

case "$ASSET" in
  *.tgz)
    tar -xzf "$TMP_DIR/$ASSET" -C "$BIN_DIR"
    ;;
  *.tar.zst)
    tar --zstd -xf "$TMP_DIR/$ASSET" -C "$BIN_DIR"
    ;;
esac

rm -rf "$TMP_DIR"
chmod +x "$BIN_DIR/ollama"
mkdir -p "$ROOT_DIR/data/ollama_models"

echo "Installed Ollama $("$BIN_DIR/ollama" --version 2>&1 | tail -1) at $BIN_DIR/ollama"
echo "Next: scripts/run_ollama.sh, then scripts/pull_models.sh"
