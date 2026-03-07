#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:-143.47.182.234}"
VM_USER="${VM_USER:-opc}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/oracle_insider.key}"
LOCAL_VNC_PORT="${LOCAL_VNC_PORT:-5901}"
REMOTE_VNC_PORT="${REMOTE_VNC_PORT:-5901}"
REALVNC_VIEWER_EXE="${REALVNC_VIEWER_EXE:-}"
TUNNEL_LOG="${TUNNEL_LOG:-$HOME/.ibkr_gateway_vnc_tunnel.log}"

find_realvnc() {
  if [[ -n "$REALVNC_VIEWER_EXE" && -f "$REALVNC_VIEWER_EXE" ]]; then
    printf '%s\n' "$REALVNC_VIEWER_EXE"
    return 0
  fi

  local candidate
  for candidate in \
    "/c/Program Files/RealVNC/VNC Viewer/vncviewer.exe" \
    "/c/Program Files (x86)/RealVNC/VNC Viewer/vncviewer.exe"
  do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

start_tunnel() {
  if pgrep -f "ssh .*${LOCAL_VNC_PORT}:localhost:${REMOTE_VNC_PORT}.*${VM_USER}@${VM_HOST}" >/dev/null 2>&1; then
    echo "VNC tunnel already running on localhost:${LOCAL_VNC_PORT}"
    return 0
  fi

  echo "Starting SSH tunnel localhost:${LOCAL_VNC_PORT} -> ${VM_HOST}:${REMOTE_VNC_PORT}"
  nohup ssh \
    -o ExitOnForwardFailure=yes \
    -N \
    -L "${LOCAL_VNC_PORT}:localhost:${REMOTE_VNC_PORT}" \
    -i "${SSH_KEY_PATH}" \
    "${VM_USER}@${VM_HOST}" \
    >>"${TUNNEL_LOG}" 2>&1 &
  disown
  sleep 2
}

open_viewer() {
  local viewer_posix viewer_windows
  if ! viewer_posix="$(find_realvnc)"; then
    echo "RealVNC Viewer not found automatically."
    echo "Open your VNC viewer manually and connect to localhost:${LOCAL_VNC_PORT}"
    return 0
  fi

  if command -v cygpath >/dev/null 2>&1; then
    viewer_windows="$(cygpath -w "$viewer_posix")"
  else
    viewer_windows="$viewer_posix"
  fi

  echo "Launching RealVNC Viewer for localhost:${LOCAL_VNC_PORT}"
  cmd.exe //c start "" "$viewer_windows" "localhost:${LOCAL_VNC_PORT}" >/dev/null 2>&1
}

print_next_steps() {
  cat <<EOF

Tunnel status:
  localhost:${LOCAL_VNC_PORT} -> ${VM_HOST}:${REMOTE_VNC_PORT}

Next steps in the VNC window:
  1. Log into IB Gateway manually with your paper username/password.
  2. If Gateway is not already open, start it on the VM with:
       DISPLAY=:1 ~/Jts/ibgateway/1037/ibgateway
  3. Confirm the API socket is listening on the VM:
       sudo ss -ltnp | grep 4002

Useful checks from a separate Git Bash / SSH shell:
  ssh -i "${SSH_KEY_PATH}" ${VM_USER}@${VM_HOST}
  sudo systemctl status insider-live-scoring.service --no-pager
  sudo systemctl status insider-ibkr-paper-trader.service --no-pager
  sudo journalctl -u insider-ibkr-paper-trader.service -f

To stop the tunnel later:
  pkill -f "ssh .*${LOCAL_VNC_PORT}:localhost:${REMOTE_VNC_PORT}.*${VM_USER}@${VM_HOST}"

EOF
}

start_tunnel
open_viewer
print_next_steps
