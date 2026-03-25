#!/usr/bin/env bash
set -euo pipefail

DISPLAY_VALUE="${DISPLAY:-:1}"
SCREEN_GEOMETRY="${XVFB_GEOMETRY:-1440x900x24}"
LOCK_FILE="/tmp/.X${DISPLAY_VALUE#:}-lock"
SOCKET_FILE="/tmp/.X11-unix/X${DISPLAY_VALUE#:}"

if [[ -S "$SOCKET_FILE" ]]; then
  echo "Display socket already exists for $DISPLAY_VALUE; reusing existing X server"
  exit 0
fi

if [[ -f "$LOCK_FILE" ]]; then
  echo "Xvfb lock already exists for $DISPLAY_VALUE"
  exit 0
fi

if ! command -v Xvfb >/dev/null 2>&1; then
  echo "Xvfb is not installed or not on PATH." >&2
  exit 1
fi

exec Xvfb "$DISPLAY_VALUE" -screen 0 "$SCREEN_GEOMETRY" -nolisten tcp -ac
