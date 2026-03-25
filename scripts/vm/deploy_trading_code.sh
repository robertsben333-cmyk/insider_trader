#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE_REF="${1:-origin/master}"

CODE_PATHS=(
  .gitignore
  live_trading
  live_scoring.py
  scripts/systemd
  scripts/vm
  tests
)

pushd "$REPO_ROOT" >/dev/null

services=(
  insider-ibkr-paper-trader.service
  alpaca-trader.service
)

restart_services() {
  for service in "${services[@]}"; do
    if systemctl list-unit-files | grep -q "^${service}"; then
      sudo systemctl start "$service"
    fi
  done
}

trap restart_services EXIT

if ! git diff --quiet -- "${CODE_PATHS[@]}" || ! git diff --cached --quiet -- "${CODE_PATHS[@]}"; then
  echo "Refusing to deploy with local code changes in tracked deploy paths." >&2
  echo "Commit, stash, or discard changes first." >&2
  exit 1
fi

git fetch origin

for service in "${services[@]}"; do
  if systemctl list-unit-files | grep -q "^${service}"; then
    sudo systemctl stop "$service"
  fi
done

git checkout "$REMOTE_REF" -- "${CODE_PATHS[@]}"

if systemctl list-unit-files | grep -q '^alpaca-trader.service'; then
  sudo install -m 0644 scripts/systemd/alpaca-trader.service /etc/systemd/system/alpaca-trader.service
fi
sudo install -m 0644 scripts/systemd/insider-ibkr-paper-trader.service /etc/systemd/system/insider-ibkr-paper-trader.service
sudo systemctl daemon-reload

restart_services
trap - EXIT

echo "Deployed trading code from $REMOTE_REF"
git rev-parse --short "$REMOTE_REF"
sudo systemctl --no-pager --full status insider-ibkr-paper-trader.service | sed -n '1,12p'
echo "----"
sudo systemctl --no-pager --full status alpaca-trader.service | sed -n '1,12p'

popd >/dev/null
