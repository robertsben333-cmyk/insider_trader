#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"

if [[ -n "${IBC_START_CMD:-}" ]]; then
  exec bash -lc "$IBC_START_CMD"
fi

if [[ -n "${IBGATEWAY_START_CMD:-}" ]]; then
  exec bash -lc "$IBGATEWAY_START_CMD"
fi

if [[ -n "${IBGATEWAY_HOME:-}" && -x "${IBGATEWAY_HOME}/ibgateway" ]]; then
  exec "${IBGATEWAY_HOME}/ibgateway"
fi

echo "Set IBC_START_CMD, IBGATEWAY_START_CMD, or IBGATEWAY_HOME before starting IB Gateway." >&2
exit 1
