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
| `live_trading/strategy_settings.py` | Add `AlpacaConfig` dataclass + `ALPACA_CONFIG` singleton; add `AlpacaLivePaths` dataclass + `ALPACA_LIVE_PATHS` singleton |
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

## AlpacaLivePaths dataclass

A **new minimal dataclass** (does not reuse `LivePaths`) containing only the fields needed for Alpaca trading. It does not include IBKR/VM-specific fields (`gateway_env_file`, `vm_sync_dir`, etc.).

```python
@dataclass(frozen=True)
class AlpacaLivePaths:
    alert_snapshot_file: str
    trader_state_dir: str
    trader_state_file: str
    trader_journal_file: str
    trader_signal_archive_file: str
    dashboard_dir: str
    dashboard_baseline_file: str
    dashboard_executions_file: str
    dashboard_equity_history_file: str
    dashboard_latest_snapshot_file: str
```

`ALPACA_LIVE_PATHS` singleton uses `live/data/alpaca_trader_state/` and `live/data/alpaca_dashboard/` directories. It shares `alert_snapshot_file` with the IBKR trader (both read from the same live scoring output).

---

## AlpacaConfig dataclass (strategy_settings.py)

```python
@dataclass(frozen=True)
class AlpacaConfig:
    api_key_env_var: str           # env var name holding the API key, e.g. "ALPACA_API_KEY"
    api_secret_env_var: str        # env var name holding the secret, e.g. "ALPACA_API_SECRET"
    paper_trading: bool            # selects paper vs live base URL
    connect_timeout_seconds: float
    data_feed: str                 # "iex" (free) or "sip" (paid); used by both adapter and data client
    data_rate_limit_per_minute: int  # max Alpaca data API requests/min; default 200
```

Credentials are read from environment variables at runtime (not hardcoded). The env var names are stored in the config so they can be overridden in tests.

---

## AlpacaBrokerAdapter

Implements the existing `BrokerAdapter` Protocol from `live_trading/broker.py`. Uses `alpaca-py` SDK only (`alpaca-trade-api` is deprecated and not used).

**No modifications to `broker.py`** — `BrokerAdapter`, `BrokerOrderView`, `BrokerFillView`, etc. are unchanged.

### SDK

`alpaca-py` (`alpaca` package on PyPI). Added to `requirements.txt`.

### Paper vs Live base URL

`AlpacaConfig.paper_trading=True` → `https://paper-api.alpaca.markets`
`AlpacaConfig.paper_trading=False` → `https://api.alpaca.markets`

### Alpaca UUID → integer order ID mapping

Alpaca order IDs are UUIDs (strings). `BrokerAdapter` Protocol uses `int` for `broker_order_id`. `AlpacaBrokerAdapter` maintains an internal `dict[str, int]` (UUID → sequential int) assigned at order creation time. All methods that accept or return `broker_order_id: int` use this mapping. The UUIDs are stored alongside for actual API calls.

### `connect` / `is_connected` / `disconnect`

- `connect()`: makes one `GET /v2/account` call to verify credentials and sets `self._connected = True`. Raises `RuntimeError` on auth failure.
- `is_connected()`: returns the cached `self._connected` flag — **no HTTP call on every poll cycle**.
- `disconnect()`: sets `self._connected = False` and discards cached state. No-op in terms of network — Alpaca is stateless REST.

### `outside_rth` mapping

`BrokerOrderRequest.outside_rth=True` maps to `extended_hours=True` in the Alpaca order payload. This enables pre-market and post-market limit order execution, which is required for the strategy's pre-market exit orders.

### `routing_exchange` and `currency`

`ExecutionPolicy.routing_exchange` and `currency` are IBKR-specific. `AlpacaBrokerAdapter` ignores both — Alpaca routes automatically and only supports USD equities.

### `quote_wait_seconds`

Not applicable to Alpaca (stateless REST, no subscription setup needed). `AlpacaBrokerAdapter` does not accept or use this parameter. Quotes are fetched synchronously with a per-request HTTP timeout.

### API Mapping

| `BrokerAdapter` method | Alpaca endpoint | Notes |
|---|---|---|
| `connect` | `GET /v2/account` | One-time auth check |
| `disconnect` | — | No-op, clears `_connected` flag |
| `is_connected` | — | Returns cached flag, no HTTP |
| `get_account_snapshot` | `GET /v2/account` | Maps to `AccountSnapshot` |
| `get_positions` | `GET /v2/positions` | Maps to `list[BrokerPositionView]` |
| `get_open_orders` | `GET /v2/orders?status=open` | |
| `list_orders(include_closed)` | `GET /v2/orders?status=open` or `all` | |
| `list_fills` / `get_recent_fills` | `GET /v2/account/activities?activity_type=FILL` | |
| `get_quote` | `GET /v2/stocks/{symbol}/quotes/latest` | Returns empty `QuoteSnapshot` (never raises) if symbol not found |
| `place_order` | `POST /v2/orders` (limit, DAY, `extended_hours` from `outside_rth`) | |
| `cancel_order(int)` | `DELETE /v2/orders/{uuid}` | int resolved to UUID via internal mapping |

### `get_quote` error handling

If the symbol is not found or Alpaca returns an error, `get_quote` returns `QuoteSnapshot(symbol=symbol)` with all price fields as `None` — same behaviour as `DryRunBrokerAdapter.get_quote`. It never raises.

---

## alpaca_trader.py

Thin module with a `main()` function that:
1. Loads `ALPACA_CONFIG` from `strategy_settings`
2. Reads API key and secret from environment variables named by `ALPACA_CONFIG.api_key_env_var` and `ALPACA_CONFIG.api_secret_env_var`
3. Builds `AlpacaBrokerAdapter`
4. Instantiates `IbkrPaperTrader` with the Alpaca adapter + `ALPACA_LIVE_PATHS` paths + shared `TRADING_BUDGET` and `EXECUTION_POLICY`
5. Runs the same polling loop as `ibkr_paper_trader.main()` (same cycle interval, same weekend shutdown logic)

No trading logic is duplicated. All loop behaviour (sleeves, signal ingestion, entry/exit order management, EOD urgency, dedup) lives in `IbkrPaperTrader`.

---

## AlpacaMarketDataClient (live_scoring.py)

### Purpose

Replace the Polygon-derived `buy_price` hint with a real-time quote from Alpaca. Polygon is always 15 minutes behind wall-clock time; Alpaca's `/v2/stocks/{symbol}/quotes/latest` returns the most recent available quote at any hour, including pre-market and post-market.

### Interface

```python
class AlpacaMarketDataClient:
    def get_latest_price(self, symbol: str) -> float | None:
        """
        Return the latest mid or trade price for symbol via Alpaca market data API.
        Returns None on timeout, HTTP error, rate-limit skip, or missing data.
        Caller falls back to Polygon-derived price on None.
        """
```

### Configuration

`AlpacaMarketDataClient` reads from `ALPACA_CONFIG` singleton (imported from `strategy_settings`):
- `ALPACA_CONFIG.api_key_env_var` / `api_secret_env_var` → credentials
- `ALPACA_CONFIG.data_feed` → `"iex"` or `"sip"` query param
- `ALPACA_CONFIG.data_rate_limit_per_minute` → rate limiter cap

Credentials are read from OS environment variables (same pattern as `POLYGON_API_KEY`). No separate credential path — `live_scoring.py` loads them from the environment at startup, with `--alpaca-api-key` and `--alpaca-api-secret` CLI flag overrides following the existing `--polygon-api-key` pattern.

### Rate limiting

- Shared non-blocking token bucket: cap from `ALPACA_CONFIG.data_rate_limit_per_minute` (default 200/min for free IEX tier; set higher for paid SIP tier)
- If bucket is empty: skip Alpaca, return `None` → caller uses Polygon-derived price — **never block or queue**
- One shared limiter instance across the parallel prefetch thread pool

### Fault tolerance

- Hard per-request timeout: **2 seconds**
- On timeout, HTTP error, auth error, or missing symbol data: return `None` → caller uses Polygon-derived price
- `ALPACA_SUPPLEMENT_ENABLED=false` env var (and `--no-alpaca-supplement` CLI flag) disables entirely — pure Polygon path, zero overhead

### Integration point

In `live_scoring.py`, where `buy_price` is currently derived from the most recent Polygon bar close, call `AlpacaMarketDataClient.get_latest_price(ticker)` instead. If it returns `None`, fall back to the existing Polygon-derived value. Polygon still serves all historical bar data for feature computation — only the single "current price" lookup per ticker is replaced.

### Speed guarantee

- At market open (09:30 ET) the live scoring run is speed-critical
- Alpaca calls run **concurrently** with Polygon prefetch — same thread pool, not after it
- Failures never add latency: 2 s max timeout, non-blocking rate limiter never queues
- If disabled, zero overhead — pure Polygon path unchanged

---

## Unchanged

- `live_trading/ibkr_paper_trader.py` — no modifications
- `live_trading/broker.py` — no modifications (IBKR adapter untouched, Protocol unchanged)
- `live_trading/trader_state.py` — no modifications
- `live_trading/signal_intake.py` — no modifications
- All strategy parameters (budget $10 000, 2 sleeves, buy cutoff 15:30, 50 bps limit buffer, etc.) shared via existing `TRADING_BUDGET` and `EXECUTION_POLICY` singletons

---

## Dependencies

- `alpaca-py` — add to `requirements.txt` (replaces the deprecated `alpaca-trade-api`; no fallback)

---

## Non-Goals

- No modifications to the IBKR trader
- No shared state between IBKR and Alpaca traders
- No Alpaca websocket streaming (REST polling sufficient)
- No fractional shares (existing `whole_shares_only=True` applies)
- No changes to `BrokerAdapter` Protocol or `broker.py`
