#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${INSIDER_TRADES_ENV_FILE:-/etc/insider_trades.env}"
PYTHON_BIN="${TRADER_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python3.11}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

exec "$PYTHON_BIN" "$REPO_ROOT/live_trading/run_alpaca_trader.py" "$@"
