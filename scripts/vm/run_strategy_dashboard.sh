#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/opc/insider_trader}"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python3.11}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8501}"

cd "$REPO_ROOT"

exec "$VENV_PYTHON" -m streamlit run "$REPO_ROOT/live_trading/run_strategy_dashboard.py" \
  --server.headless true \
  --server.address "$HOST" \
  --server.port "$PORT"
