# insider_trades_predictor

## Project Overview
Live insider-trades signal pipeline + IBKR paper trader. Scrapes OpenInsider, scores signals with an ML ensemble, and auto-trades via IBKR Gateway.

## Key Files
- `live_scoring.py` — live scoring pipeline (~1600 lines)
- `live_trading/ibkr_paper_trader.py` — IBKR paper trading loop (main trader logic)
- `live_trading/strategy_settings.py` — all config dataclasses + singleton instances (ACTIVE_STRATEGY, LIVE_PATHS, EXECUTION_POLICY, TRADING_BUDGET, etc.)
- `live_trading/trader_state.py` — state dataclasses (SignalCandidate, PositionLot, PendingOrder, etc.)
- `live_trading/market_calendar.py` — time helpers: `parse_time_hhmm`, `parse_iso_datetime`, `is_regular_trading_hours`, `ET`, `UTC`
- `train_models.py` — model training pipeline
- `openinsider_scraper.py` — scrapes OpenInsider; `_get_data_for_month` returns `Set[tuple]`

## Type Checking
```bash
"C:/Users/XavierFriesen/AppData/Local/Programs/Python/Python313/python.exe" -m pyright live_scoring.py
```

## Live Trader Architecture
- **Sleeves**: 2 capital sleeves rotating by trade day; each sleeve has independent cash/equity tracking
- **Signal buckets**: `entry_bucket="open"` (pre-market batch) vs `entry_bucket="intraday"` (live individual)
- **Exit policy**: `sell_after_trading_days=2` at open; `TERMINAL_CANDIDATE_STATUSES = {"expired", "rejected", "filled"}`
- **EOD urgency**: within `eod_window_minutes` of `buy_cutoff_time`, limit buffers widen by `eod_buffer_multiplier` and reprice interval shrinks 3×
- **Allocation cap**: `max_allocation_ratio=4.0` — largest position can be at most 4× smallest in a batch
- **Dedup**: `_ingest_signals` keeps only the highest-scoring candidate per (ticker, entry_date); skips tickers with an open lot on the same trade day

## Oracle Cloud Deployment
See `ORACLE_CLOUD_DEPLOYMENT.md`. Instance: Oracle Linux 9, IP 143.47.182.234, user `opc`.
