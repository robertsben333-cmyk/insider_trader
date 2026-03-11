# Alpaca Live Trader — Design Spec

**Date:** 2026-03-11
**Status:** Approved
**Approach:** A (max reuse — broker adapter + thin wiring)

---

## Overview

Add an Alpaca trading app that runs independently alongside the existing IBKR paper trader. The trading loop logic is identical — `IbkrPaperTrader` already accepts a `BrokerAdapter` Protocol and is broker-agnostic. The Alpaca app wires a new `AlpacaBrokerAdapter` into the existing class with Alpaca-specific config and separate state files.

Additionally, `live_scoring.py` is enhanced to fetch the **most recent available price** (buy_price hint) from Alpaca's market data API instead of deriving it from Polygon, because Polygon is always 15 minutes behind wall-clock time. This improves signal quality at any hour including pre-market.

---

## New Files

| File | Purpose |
|---|---|
| `live_trading/alpaca_broker.py` | `AlpacaBrokerAdapter` — implements `BrokerAdapter` Protocol via Alpaca REST API |
| `live_trading/alpaca_trader.py` | Thin wiring module: instantiates `IbkrPaperTrader` with `AlpacaBrokerAdapter` + Alpaca-specific config |
| `live_trading/run_alpaca_trader.py` | Entry point — mirrors `run_ibkr_paper_trader.py` |

## Modified Files

| File | Change |
|---|---|
| `live_trading/strategy_settings.py` | Add `AlpacaConfig` dataclass + `ALPACA_CONFIG` singleton; add Alpaca-specific paths to a new `AlpacaLivePaths` instance |
| `live_scoring.py` | Add `AlpacaMarketDataClient` for latest-quote fetching; replace Polygon-derived buy_price with Alpaca latest quote where applicable |

---

## State Files (independent from IBKR)

```
live/data/alpaca_trader_state/
  alpaca_trader_state.json
  alpaca_trader_journal.jsonl
  alpaca_signal_archive.csv
```

---

## AlpacaBrokerAdapter

Implements the existing `BrokerAdapter` Protocol from `live_trading/broker.py`. Uses the `alpaca-py` SDK (`alpaca-trade-api` as fallback).

### API Mapping

| `BrokerAdapter` method | Alpaca endpoint |
|---|---|
| `connect` / `is_connected` | REST auth check against `/v2/account` |
| `get_account_snapshot` | `GET /v2/account` → `AccountSnapshot` |
| `get_positions` | `GET /v2/positions` → `list[BrokerPositionView]` |
| `get_open_orders` | `GET /v2/orders?status=open` |
| `list_orders` | `GET /v2/orders?status=all` (with `include_closed` flag) |
| `list_fills` / `get_recent_fills` | `GET /v2/account/activities?activity_type=FILL` |
| `get_quote` | `GET /v2/stocks/{symbol}/quotes/latest` → `QuoteSnapshot` |
| `place_order` | `POST /v2/orders` (limit, DAY TIF) |
| `cancel_order` | `DELETE /v2/orders/{order_id}` |

### Paper vs Live

`AlpacaConfig.paper_trading: bool` selects the base URL:
- Paper: `https://paper-api.alpaca.markets`
- Live: `https://api.alpaca.markets`

Credentials (`ALPACA_API_KEY`, `ALPACA_API_SECRET`) loaded from the gateway env file, with CLI flag overrides.

### AlpacaConfig dataclass (strategy_settings.py)

```python
@dataclass(frozen=True)
class AlpacaConfig:
    api_key_env_var: str        # env var name for API key
    api_secret_env_var: str     # env var name for API secret
    paper_trading: bool
    connect_timeout_seconds: float
    data_feed: str              # "iex" or "sip"
```

---

## alpaca_trader.py

Thin module with a `main()` function that:
1. Loads `AlpacaConfig` from `strategy_settings.ALPACA_CONFIG`
2. Builds `AlpacaBrokerAdapter`
3. Instantiates `IbkrPaperTrader` with the Alpaca adapter + Alpaca-specific paths from `ALPACA_LIVE_PATHS`
4. Runs the same polling loop as `ibkr_paper_trader.main()`

No trading logic is duplicated — all loop behaviour (sleeves, signal ingestion, entry/exit order management, EOD urgency, dedup) lives in `IbkrPaperTrader`.

---

## AlpacaMarketDataClient (live_scoring.py)

### Purpose

Replace the Polygon-derived `buy_price` hint with a real-time quote from Alpaca. Polygon is always 15 minutes behind wall-clock time; Alpaca's `/v2/stocks/{symbol}/quotes/latest` returns the most recent available quote at any hour.

### Interface

```python
class AlpacaMarketDataClient:
    def get_latest_price(self, symbol: str) -> float | None:
        """Return latest mid/trade price for symbol. Returns None on any failure."""
```

### Integration point

In `live_scoring.py`, where `buy_price` is currently derived from the most recent Polygon bar close, call `AlpacaMarketDataClient.get_latest_price(ticker)` instead. Polygon still serves all historical bar data for feature computation — only the single "current price" lookup is replaced.

### Rate limiting

- Shared non-blocking token bucket: **200 requests/minute** cap
- If bucket is empty: skip Alpaca, fall back to Polygon-derived price — never block or queue
- Runs in the existing parallel prefetch thread pool

### Fault tolerance

- Hard timeout: **2 seconds** per request
- On timeout, HTTP error, or missing data: silently return `None` → caller uses Polygon-derived price as fallback
- `ALPACA_SUPPLEMENT_ENABLED=false` env var (and `--no-alpaca-supplement` CLI flag) disables entirely

### Speed guarantee

- At market open (09:30 ET) the live scoring run is speed-critical. Alpaca calls run **concurrently** with Polygon prefetch, not after it — they share the same thread pool
- Failures never add latency: timeout is 2 s max, non-blocking rate limiter never queues
- If `ALPACA_SUPPLEMENT_ENABLED=false`, zero overhead — pure Polygon path unchanged

---

## Unchanged

- `live_trading/ibkr_paper_trader.py` — no modifications
- `live_trading/broker.py` — no modifications (IBKR adapter untouched)
- `live_trading/trader_state.py` — no modifications
- `live_trading/signal_intake.py` — no modifications
- All strategy parameters (budget $10 000, 2 sleeves, buy cutoff 15:30, 50 bps limit buffer, etc.) are shared via the existing `TRADING_BUDGET` and `EXECUTION_POLICY` singletons

---

## Dependencies

- `alpaca-py` (preferred) or `alpaca-trade-api` — add to `requirements.txt`
- No new dependencies for the data supplement (uses `urllib` / `requests`, already present)

---

## Non-Goals

- No modifications to the IBKR trader
- No shared state between IBKR and Alpaca traders
- No Alpaca websocket streaming (REST polling is sufficient)
- No fractional shares (existing `whole_shares_only=True` applies)
