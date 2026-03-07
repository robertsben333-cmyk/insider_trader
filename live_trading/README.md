# Live Trading Runtime

This folder is the clean runtime boundary for the future trading app.

## What belongs here

- `strategy_settings.py`: single editable source of truth for live strategy parameters and live-script defaults.
- `models/day1_tplus2_open/manifest.json`: active strategy bundle metadata for the current `T+2 open` exit policy.
- `run_*.py`: wrapper entrypoints that point app/runtime code at the existing scripts.

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

These wrappers keep the future app pointed at a stable runtime folder without forcing a risky full module move today.
