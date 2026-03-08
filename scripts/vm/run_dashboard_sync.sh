#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/opc/insider_trader}"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python3.11}"
ENV_FILE="${ENV_FILE:-/etc/insider_trades.env}"

cd "$REPO_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

exec "$VENV_PYTHON" "$REPO_ROOT/live_trading/run_dashboard_sync.py"
