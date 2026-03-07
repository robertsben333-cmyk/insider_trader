# Live Trading Runtime

This folder is the clean runtime boundary for the future trading app.

## What belongs here

- `strategy_settings.py`: single editable source of truth for live strategy parameters and live-script defaults.
- `models/day1_tplus2_open/manifest.json`: active strategy bundle metadata for the current `T+2 open` exit policy.
- `run_*.py`: wrapper entrypoints that point app/runtime code at the existing scripts.
- `ibkr_paper_trader.py`: separate execution service that consumes live alerts and trades them in the IBKR paper account.

## Active strategy

- Strategy id: `day1_tplus2_open`
- Entry logic: score filings as they arrive and trade when the market is open.
- Exit logic: `sell_at_open_2_trading_days_after_buy`
- Live model artifacts: `models/prod4`
- Backup model artifacts: `models_r_day1/prod4`

## Wrapper entrypoints

- `run_live_scoring.py`
- `run_rescore_live_history.py`
- `run_vm_history_backtest.py`
- `run_vm_early_exit_walkforward.py`
- `run_ibkr_paper_trader.py`

These wrappers keep the future app pointed at a stable runtime folder without forcing a risky full module move today.

## IBKR runtime

- Broker connectivity is configured through `IBKR_CONFIG`.
- Sleeve budgeting, rotation, and sizing caps are configured through `TRADING_BUDGET`.
- Entry/exit timing, stale-order replacement, and marketable-limit settings are configured through `EXECUTION_POLICY`.
- Runtime state is persisted under `live/data/trader_state/`.

## VM deployment helpers

- `scripts/vm/start_virtual_display.sh`
- `scripts/vm/start_ibgateway.sh`
- `scripts/vm/wait_for_ibgateway.py`
- `scripts/vm/run_ibkr_paper_trader.sh`
- `scripts/systemd/insider-xvfb.service`
- `scripts/systemd/insider-ibgateway.service`
- `scripts/systemd/insider-ibkr-paper-trader.service`

These templates assume an unsupported unattended setup: `Xvfb` provides the display, an external gateway launcher such as IBC can be injected via `IBC_START_CMD`, and the trader process waits for the gateway socket before starting.
