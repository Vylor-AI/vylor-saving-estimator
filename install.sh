#!/usr/bin/env sh
# install.sh - vylor-estimate installer and runner for macOS and Linux
# Usage: curl -fsSL https://raw.githubusercontent.com/Vylor-AI/vylor-saving-estimator/main/install.sh | sh
set -e

REPO="Vylor-AI/vylor-saving-estimator"
INSTALL_DIR="/usr/local/bin"
BINARY="vylor-estimate"

OS=$(uname -s)
case "$OS" in
  Darwin) ASSET="vylor-estimate-macos" ;;
  Linux)  ASSET="vylor-estimate-linux" ;;
  *)
    echo "[!] Unsupported OS: $OS"
    echo "    Download manually from: https://github.com/$REPO/releases/latest"
    exit 1
    ;;
esac

# Resolve latest release tag via GitHub API
LATEST=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
  | grep '"tag_name"' \
  | sed 's/.*"tag_name": *"\(.*\)".*/\1/')

if [ -z "$LATEST" ]; then
  echo "[!] Could not determine latest release. Check your internet connection."
  exit 1
fi

URL="https://github.com/$REPO/releases/download/$LATEST/$ASSET"

echo "--> Downloading vylor-estimate $LATEST for $OS..."
curl -fsSL "$URL" -o "/tmp/$BINARY"
chmod +x "/tmp/$BINARY"

INSTALLED_PATH=""
if [ -w "$INSTALL_DIR" ]; then
  mv "/tmp/$BINARY" "$INSTALL_DIR/$BINARY"
  INSTALLED_PATH="$INSTALL_DIR/$BINARY"
else
  FALLBACK="$HOME/.local/bin"
  mkdir -p "$FALLBACK"
  mv "/tmp/$BINARY" "$FALLBACK/$BINARY"
  INSTALLED_PATH="$FALLBACK/$BINARY"
fi

echo "[+] Installed to $INSTALLED_PATH"
echo ""

# Run immediately
exec "$INSTALLED_PATH" "$@"
