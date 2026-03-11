# Alpaca Live Trader Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete Alpaca broker adapter + live trader that runs the existing `IbkrPaperTrader` loop against Alpaca, plus a real-time price supplement in `live_scoring.py` that replaces the Polygon-delayed buy_price hint with Alpaca's latest quote.

**Architecture:** `AlpacaBrokerAdapter` implements the existing `BrokerAdapter` Protocol and is wired into the existing `IbkrPaperTrader` class (which is already broker-agnostic) via a thin `alpaca_trader.py` module. `live_scoring.py` gets a stdlib-only `AlpacaMarketDataClient` with a non-blocking token bucket rate limiter; Alpaca latest-quote calls run in the existing Polygon prefetch `ThreadPoolExecutor` for zero added latency.

**Tech Stack:** Python 3.11+, `alpaca-py` (alpaca SDK), `urllib` (stdlib, for data supplement), `unittest.mock` (tests), pyright for type checking.

**Spec:** `docs/superpowers/specs/2026-03-11-alpaca-trader-design.md`

---

## Chunk 1: Config & Dependencies

### Task 1: Add alpaca-py to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependency**

Append to `requirements.txt`:
```
alpaca-py>=0.40
```

- [ ] **Step 2: Verify install**

```bash
pip install alpaca-py
python -c "from alpaca.trading.client import TradingClient; print('ok')"
```
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add alpaca-py dependency"
```

---

### Task 2: Add AlpacaConfig and AlpacaLivePaths to strategy_settings.py

**Files:**
- Modify: `live_trading/strategy_settings.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ibkr_paper_trader.py` (below existing imports/class):

```python
class AlpacaConfigTests(unittest.TestCase):
    def test_alpaca_config_singleton_exists(self) -> None:
        from live_trading.strategy_settings import ALPACA_CONFIG, AlpacaConfig
        self.assertIsInstance(ALPACA_CONFIG, AlpacaConfig)
        self.assertEqual(ALPACA_CONFIG.api_key_env_var, "ALPACA_API_KEY")
        self.assertEqual(ALPACA_CONFIG.api_secret_env_var, "ALPACA_API_SECRET")
        self.assertTrue(ALPACA_CONFIG.paper_trading)
        self.assertIn(ALPACA_CONFIG.data_feed, ("iex", "sip"))
        self.assertGreater(ALPACA_CONFIG.data_rate_limit_per_minute, 0)

    def test_alpaca_live_paths_singleton_exists(self) -> None:
        from live_trading.strategy_settings import ALPACA_LIVE_PATHS, AlpacaLivePaths
        self.assertIsInstance(ALPACA_LIVE_PATHS, AlpacaLivePaths)
        self.assertIn("alpaca", ALPACA_LIVE_PATHS.trader_state_file)
        self.assertIn("alpaca", ALPACA_LIVE_PATHS.trader_journal_file)
        # Shares alert_snapshot_file with IBKR trader
        from live_trading.strategy_settings import LIVE_PATHS
        self.assertEqual(ALPACA_LIVE_PATHS.alert_snapshot_file, LIVE_PATHS.alert_snapshot_file)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_ibkr_paper_trader.py::AlpacaConfigTests -v
```
Expected: ImportError — `ALPACA_CONFIG` not defined yet.

- [ ] **Step 3: Add dataclasses and singletons to strategy_settings.py**

Add after the existing `IbkrConfig` dataclass and before `GatewayRuntimeConfig`:

```python
@dataclass(frozen=True)
class AlpacaConfig:
    api_key_env_var: str
    api_secret_env_var: str
    paper_trading: bool
    connect_timeout_seconds: float
    data_feed: str                    # "iex" or "sip"
    data_rate_limit_per_minute: int   # token-bucket cap for Alpaca data API


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

Add singletons after the existing `IBKR_CONFIG` singleton:

```python
ALPACA_CONFIG = AlpacaConfig(
    api_key_env_var="ALPACA_API_KEY",
    api_secret_env_var="ALPACA_API_SECRET",
    paper_trading=True,
    connect_timeout_seconds=10.0,
    data_feed="iex",
    data_rate_limit_per_minute=200,
)

ALPACA_LIVE_PATHS = AlpacaLivePaths(
    alert_snapshot_file=LIVE_PATHS.alert_snapshot_file,  # shared with IBKR
    trader_state_dir="live/data/alpaca_trader_state",
    trader_state_file="live/data/alpaca_trader_state/alpaca_trader_state.json",
    trader_journal_file="live/data/alpaca_trader_state/alpaca_trader_journal.jsonl",
    trader_signal_archive_file="live/data/alpaca_trader_state/alpaca_signal_archive.csv",
    dashboard_dir="live/data/alpaca_dashboard",
    dashboard_baseline_file="live/data/alpaca_dashboard/baseline.json",
    dashboard_executions_file="live/data/alpaca_dashboard/executions.jsonl",
    dashboard_equity_history_file="live/data/alpaca_dashboard/equity_history.csv",
    dashboard_latest_snapshot_file="live/data/alpaca_dashboard/latest_snapshot.json",
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_ibkr_paper_trader.py::AlpacaConfigTests -v
```
Expected: 2 passed.

- [ ] **Step 5: Pyright check**

```bash
"C:/Users/XavierFriesen/AppData/Local/Programs/Python/Python313/python.exe" -m pyright live_trading/strategy_settings.py
```
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add live_trading/strategy_settings.py tests/test_ibkr_paper_trader.py
git commit -m "feat: add AlpacaConfig and AlpacaLivePaths to strategy_settings"
```

---

## Chunk 2: AlpacaBrokerAdapter

### Task 3: Write tests for AlpacaBrokerAdapter

**Files:**
- Create: `tests/test_alpaca_broker.py`

- [ ] **Step 1: Create test file**

```python
from __future__ import annotations

from datetime import UTC, datetime
from unittest import TestCase
from unittest.mock import MagicMock, patch, PropertyMock
import sys

from live_trading.broker import BrokerOrderRequest, QuoteSnapshot


def _make_adapter(paper: bool = True):
    """Build an AlpacaBrokerAdapter with fully mocked SDK clients."""
    with patch("live_trading.alpaca_broker.TradingClient"), \
         patch("live_trading.alpaca_broker.StockHistoricalDataClient"):
        from live_trading.alpaca_broker import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter(
            api_key="test_key",
            api_secret="test_secret",
            paper_trading=paper,
            connect_timeout_seconds=5.0,
            data_feed="iex",
        )
    return adapter


class TestAlpacaBrokerAdapterConnection(TestCase):
    def test_initial_state_not_connected(self) -> None:
        adapter = _make_adapter()
        self.assertFalse(adapter.is_connected())

    def test_connect_sets_connected_flag(self) -> None:
        adapter = _make_adapter()
        adapter._trading_client.get_account = MagicMock(return_value=MagicMock())
        adapter.connect()
        self.assertTrue(adapter.is_connected())

    def test_connect_raises_on_auth_failure(self) -> None:
        adapter = _make_adapter()
        adapter._trading_client.get_account = MagicMock(side_effect=Exception("auth error"))
        with self.assertRaises(RuntimeError):
            adapter.connect()

    def test_disconnect_clears_connected_flag(self) -> None:
        adapter = _make_adapter()
        adapter._trading_client.get_account = MagicMock(return_value=MagicMock())
        adapter.connect()
        adapter.disconnect()
        self.assertFalse(adapter.is_connected())

    def test_is_connected_does_not_make_http_call(self) -> None:
        adapter = _make_adapter()
        adapter._trading_client.get_account = MagicMock(return_value=MagicMock())
        adapter.connect()
        # Reset mock call count
        adapter._trading_client.get_account.reset_mock()
        _ = adapter.is_connected()
        adapter._trading_client.get_account.assert_not_called()


class TestAlpacaBrokerAdapterOrderIdMapping(TestCase):
    def _connected_adapter(self):
        adapter = _make_adapter()
        adapter._trading_client.get_account = MagicMock(return_value=MagicMock())
        adapter.connect()
        return adapter

    def _mock_order(self, uuid: str, symbol: str = "AAPL", filled_qty: str = "0"):
        order = MagicMock()
        order.id = uuid
        order.symbol = symbol
        order.status = MagicMock()
        order.status.value = "new"
        order.filled_qty = filled_qty
        return order

    def test_place_order_assigns_sequential_int_ids(self) -> None:
        adapter = self._connected_adapter()
        adapter._trading_client.submit_order = MagicMock(side_effect=[
            self._mock_order("uuid-001"),
            self._mock_order("uuid-002"),
        ])
        req = BrokerOrderRequest(order_ref="r1", symbol="AAPL", side="BUY", quantity=10, limit_price=150.0)
        view1 = adapter.place_order(req)
        view2 = adapter.place_order(req)
        self.assertEqual(view1.broker_order_id, 1)
        self.assertEqual(view2.broker_order_id, 2)

    def test_cancel_order_resolves_correct_uuid(self) -> None:
        adapter = self._connected_adapter()
        adapter._trading_client.submit_order = MagicMock(return_value=self._mock_order("uuid-abc"))
        adapter._trading_client.cancel_order_by_id = MagicMock()
        req = BrokerOrderRequest(order_ref="r1", symbol="AAPL", side="BUY", quantity=5, limit_price=100.0)
        view = adapter.place_order(req)
        adapter.cancel_order(view.broker_order_id)
        adapter._trading_client.cancel_order_by_id.assert_called_once_with("uuid-abc")

    def test_cancel_unknown_order_is_silent(self) -> None:
        adapter = self._connected_adapter()
        adapter._trading_client.cancel_order_by_id = MagicMock()
        adapter.cancel_order(9999)  # never registered
        adapter._trading_client.cancel_order_by_id.assert_not_called()


class TestAlpacaBrokerAdapterQuote(TestCase):
    def _connected_adapter(self):
        adapter = _make_adapter()
        adapter._trading_client.get_account = MagicMock(return_value=MagicMock())
        adapter.connect()
        return adapter

    def test_get_quote_returns_bid_ask_last(self) -> None:
        adapter = self._connected_adapter()
        mock_quote = MagicMock()
        mock_quote.bid_price = 149.5
        mock_quote.ask_price = 150.5
        mock_trade = MagicMock()
        mock_trade.price = 150.0
        adapter._data_client.get_stock_latest_quote = MagicMock(return_value={"AAPL": mock_quote})
        adapter._data_client.get_stock_latest_trade = MagicMock(return_value={"AAPL": mock_trade})
        snap = adapter.get_quote("AAPL")
        self.assertEqual(snap.symbol, "AAPL")
        self.assertAlmostEqual(snap.bid, 149.5)  # type: ignore[arg-type]
        self.assertAlmostEqual(snap.ask, 150.5)  # type: ignore[arg-type]
        self.assertAlmostEqual(snap.last, 150.0)  # type: ignore[arg-type]

    def test_get_quote_returns_empty_snapshot_on_error(self) -> None:
        adapter = self._connected_adapter()
        adapter._data_client.get_stock_latest_quote = MagicMock(side_effect=Exception("404"))
        adapter._data_client.get_stock_latest_trade = MagicMock(side_effect=Exception("404"))
        snap = adapter.get_quote("ZZZZ")
        self.assertEqual(snap.symbol, "ZZZZ")
        self.assertIsNone(snap.bid)
        self.assertIsNone(snap.ask)
        self.assertIsNone(snap.last)


class TestAlpacaBrokerAdapterPlaceOrder(TestCase):
    def _connected_adapter(self):
        adapter = _make_adapter()
        adapter._trading_client.get_account = MagicMock(return_value=MagicMock())
        adapter.connect()
        return adapter

    def _mock_order(self, uuid: str = "uuid-x", filled_qty: str = "0"):
        order = MagicMock()
        order.id = uuid
        order.symbol = "AAPL"
        order.status = MagicMock()
        order.status.value = "new"
        order.filled_qty = filled_qty
        return order

    def test_outside_rth_maps_to_extended_hours(self) -> None:
        from alpaca.trading.requests import LimitOrderRequest as AlpacaLimitOrder
        adapter = self._connected_adapter()
        captured: list = []
        def capture_order(req):
            captured.append(req)
            return self._mock_order()
        adapter._trading_client.submit_order = capture_order
        broker_req = BrokerOrderRequest(
            order_ref="r1", symbol="AAPL", side="BUY",
            quantity=5, limit_price=150.0, outside_rth=True,
        )
        adapter.place_order(broker_req)
        self.assertTrue(captured[0].extended_hours)

    def test_outside_rth_false_maps_to_no_extended_hours(self) -> None:
        adapter = self._connected_adapter()
        captured: list = []
        def capture_order(req):
            captured.append(req)
            return self._mock_order()
        adapter._trading_client.submit_order = capture_order
        broker_req = BrokerOrderRequest(
            order_ref="r1", symbol="AAPL", side="BUY",
            quantity=5, limit_price=150.0, outside_rth=False,
        )
        adapter.place_order(broker_req)
        self.assertFalse(captured[0].extended_hours)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_alpaca_broker.py -v
```
Expected: ImportError — `live_trading.alpaca_broker` not found yet.

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_alpaca_broker.py
git commit -m "test: add AlpacaBrokerAdapter tests (red)"
```

---

### Task 4: Implement AlpacaBrokerAdapter

**Files:**
- Create: `live_trading/alpaca_broker.py`

- [ ] **Step 1: Create implementation**

```python
from __future__ import annotations

from datetime import UTC, datetime
import logging

import json as _json
import urllib.request as _urllib_request

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
    )
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
except ImportError as _exc:
    raise RuntimeError(
        "alpaca-py is not installed. Run: pip install alpaca-py"
    ) from _exc

from live_trading.broker import (
    AccountSnapshot,
    BrokerFillView,
    BrokerOrderRequest,
    BrokerOrderView,
    BrokerPositionView,
    QuoteSnapshot,
)

logger = logging.getLogger(__name__)

_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_LIVE_BASE_URL = "https://api.alpaca.markets"


class AlpacaBrokerAdapter:
    """BrokerAdapter implementation backed by the Alpaca REST API (alpaca-py SDK)."""

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        paper_trading: bool,
        connect_timeout_seconds: float,
        data_feed: str,
    ) -> None:
        self._base_url = _PAPER_BASE_URL if paper_trading else _LIVE_BASE_URL
        self._api_key = api_key
        self._api_secret = api_secret
        self._trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper_trading,
            url_override=self._base_url,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )
        self._data_feed = data_feed
        self._connected = False
        # UUID ↔ sequential-int order ID mapping (Alpaca uses UUIDs; Protocol uses int)
        self._uuid_to_int: dict[str, int] = {}
        self._int_to_uuid: dict[int, str] = {}
        self._next_order_int = 1

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            self._trading_client.get_account()
            self._connected = True
        except Exception as exc:
            raise RuntimeError(f"Alpaca auth failed: {exc}") from exc

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_snapshot(self) -> AccountSnapshot:
        acct = self._trading_client.get_account()
        return AccountSnapshot(
            account_id=str(getattr(acct, "account_number", "") or ""),
            net_liquidation=float(getattr(acct, "equity", 0.0) or 0.0),
            available_funds=float(getattr(acct, "buying_power", 0.0) or 0.0),
            total_cash_value=float(getattr(acct, "cash", 0.0) or 0.0),
        )

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[BrokerPositionView]:
        positions = self._trading_client.get_all_positions()
        out: list[BrokerPositionView] = []
        for pos in positions:
            out.append(
                BrokerPositionView(
                    symbol=str(getattr(pos, "symbol", "") or ""),
                    quantity=int(float(getattr(pos, "qty", 0) or 0)),
                    avg_cost=float(getattr(pos, "avg_entry_price", 0.0) or 0.0),
                    market_price=float(getattr(pos, "current_price", None) or 0.0) or None,
                    market_value=float(getattr(pos, "market_value", None) or 0.0) or None,
                )
            )
        return out

    # ── Orders ────────────────────────────────────────────────────────────────

    def get_open_orders(self) -> list[BrokerOrderView]:
        return self.list_orders(include_closed=False)

    def list_orders(self, include_closed: bool = False) -> list[BrokerOrderView]:
        status = QueryOrderStatus.ALL if include_closed else QueryOrderStatus.OPEN
        orders = self._trading_client.get_orders(GetOrdersRequest(status=status))
        out: list[BrokerOrderView] = []
        for order in orders:
            uuid = str(getattr(order, "id", "") or "")
            order_int = self._register_order(uuid)
            total_qty = int(float(getattr(order, "qty", 0) or 0))
            filled_qty = int(float(getattr(order, "filled_qty", 0) or 0))
            side_raw = getattr(order, "side", None)
            side_str = side_raw.value.upper() if hasattr(side_raw, "value") else str(side_raw or "").upper()
            status_raw = getattr(order, "status", None)
            status_str = status_raw.value if hasattr(status_raw, "value") else str(status_raw or "")
            out.append(
                BrokerOrderView(
                    broker_order_id=order_int,
                    order_ref=str(getattr(order, "client_order_id", "") or ""),
                    symbol=str(getattr(order, "symbol", "") or ""),
                    side=side_str,
                    quantity=total_qty,
                    limit_price=float(getattr(order, "limit_price", 0.0) or 0.0),
                    filled_quantity=filled_qty,
                    remaining_quantity=max(0, total_qty - filled_qty),
                    status=status_str,
                    placed_at=str(getattr(order, "submitted_at", "") or ""),
                )
            )
        return out

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderView:
        side = OrderSide.BUY if request.side.upper() == "BUY" else OrderSide.SELL
        req = LimitOrderRequest(
            symbol=request.symbol.upper(),
            qty=request.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=request.limit_price,
            extended_hours=request.outside_rth,
            client_order_id=request.order_ref or None,
        )
        order = self._trading_client.submit_order(req)
        uuid = str(getattr(order, "id", "") or "")
        order_int = self._register_order(uuid)
        filled_qty = int(float(getattr(order, "filled_qty", 0) or 0))
        status_raw = getattr(order, "status", None)
        status_str = status_raw.value if hasattr(status_raw, "value") else str(status_raw or "")
        return BrokerOrderView(
            broker_order_id=order_int,
            order_ref=request.order_ref,
            symbol=request.symbol.upper(),
            side=request.side.upper(),
            quantity=int(request.quantity),
            limit_price=float(request.limit_price),
            filled_quantity=filled_qty,
            remaining_quantity=max(0, int(request.quantity) - filled_qty),
            status=status_str,
            placed_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )

    def cancel_order(self, broker_order_id: int) -> None:
        uuid = self._int_to_uuid.get(int(broker_order_id))
        if uuid is None:
            return
        try:
            self._trading_client.cancel_order_by_id(uuid)
        except Exception as exc:
            logger.warning("cancel_order %s (int=%d): %s", uuid, broker_order_id, exc)

    # ── Fills ─────────────────────────────────────────────────────────────────

    def get_recent_fills(self) -> list[BrokerFillView]:
        return self.list_fills()

    def list_fills(self) -> list[BrokerFillView]:
        """Fetch fill activities via direct REST call (alpaca-py SDK has no fills method on TradingClient)."""
        url = f"{self._base_url}/v2/account/activities?activity_type=FILL"
        req = _urllib_request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._api_secret,
            },
        )
        try:
            with _urllib_request.urlopen(req, timeout=10.0) as resp:
                activities = _json.loads(resp.read())
        except Exception as exc:
            logger.warning("list_fills: %s", exc)
            return []
        out: list[BrokerFillView] = []
        for act in activities:
            if not isinstance(act, dict):
                continue
            order_uuid = str(act.get("order_id", "") or "")
            order_int: int | None = self._uuid_to_int.get(order_uuid)
            side_str = str(act.get("side", "") or "").upper()
            out.append(
                BrokerFillView(
                    execution_id=str(act.get("id", "") or ""),
                    broker_order_id=order_int,
                    order_ref="",
                    symbol=str(act.get("symbol", "") or ""),
                    side=side_str,
                    quantity=int(float(act.get("qty", 0) or 0)),
                    price=float(act.get("price", 0.0) or 0.0),
                    filled_at=str(act.get("transaction_time", "") or ""),
                    commission=0.0,
                )
            )
        return out

    # ── Quotes ────────────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        key = symbol.upper()
        bid: float | None = None
        ask: float | None = None
        last: float | None = None
        try:
            quote_resp = self._data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=key, feed=self._data_feed)
            )
            quote = quote_resp.get(key)
            if quote is not None:
                raw_bid = getattr(quote, "bid_price", None)
                raw_ask = getattr(quote, "ask_price", None)
                if raw_bid:
                    bid = float(raw_bid)
                if raw_ask:
                    ask = float(raw_ask)
        except Exception as exc:
            logger.debug("get_quote(%s) quote error: %s", key, exc)
        try:
            trade_resp = self._data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=key, feed=self._data_feed)
            )
            trade = trade_resp.get(key)
            if trade is not None:
                raw_price = getattr(trade, "price", None)
                if raw_price:
                    last = float(raw_price)
        except Exception as exc:
            logger.debug("get_quote(%s) trade error: %s", key, exc)
        return QuoteSnapshot(
            symbol=key,
            bid=bid,
            ask=ask,
            last=last,
            close=None,
            captured_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _register_order(self, uuid: str) -> int:
        if uuid not in self._uuid_to_int:
            n = self._next_order_int
            self._next_order_int += 1
            self._uuid_to_int[uuid] = n
            self._int_to_uuid[n] = uuid
        return self._uuid_to_int[uuid]
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_alpaca_broker.py -v
```
Expected: all tests pass.

- [ ] **Step 3: Pyright check**

```bash
"C:/Users/XavierFriesen/AppData/Local/Programs/Python/Python313/python.exe" -m pyright live_trading/alpaca_broker.py
```
Expected: 0 errors (fix any that appear before continuing).

- [ ] **Step 4: Commit**

```bash
git add live_trading/alpaca_broker.py
git commit -m "feat: add AlpacaBrokerAdapter"
```

---

## Chunk 3: Trader Wiring & Entry Point

### Task 5: Implement alpaca_trader.py

**Files:**
- Create: `live_trading/alpaca_trader.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ibkr_paper_trader.py`:

```python
class AlpacaTraderWiringTests(unittest.TestCase):
    def test_main_dry_run_completes_one_cycle(self) -> None:
        """alpaca_trader.main(--dry-run --once) should complete without error."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "live_trading/run_alpaca_trader.py", "--dry-run", "--once"],
            capture_output=True, text=True, timeout=30,
            cwd=".",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr[-2000:])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_ibkr_paper_trader.py::AlpacaTraderWiringTests -v
```
Expected: FAIL — `run_alpaca_trader.py` not found.

- [ ] **Step 3: Implement alpaca_trader.py**

```python
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from live_trading.alpaca_broker import AlpacaBrokerAdapter
from live_trading.broker import DryRunBrokerAdapter
from live_trading.ibkr_paper_trader import IbkrPaperTrader
from live_trading.market_calendar import (
    ET,
    is_regular_trading_hours,
    is_weekend_shutdown_window,
    seconds_until_weekend_shutdown_end,
)
from live_trading.strategy_settings import (
    ALPACA_CONFIG,
    ALPACA_LIVE_PATHS,
    EXECUTION_POLICY,
    TRADING_BUDGET,
)
from live_trading.trader_state import StateStore
from datetime import datetime


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("alpaca_trader")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpaca live-trading service for the insider-trading strategy.")
    parser.add_argument("--state-file", default=ALPACA_LIVE_PATHS.trader_state_file)
    parser.add_argument("--journal-file", default=ALPACA_LIVE_PATHS.trader_journal_file)
    parser.add_argument("--signal-archive-file", default=ALPACA_LIVE_PATHS.trader_signal_archive_file)
    parser.add_argument("--alert-snapshot-file", default=ALPACA_LIVE_PATHS.alert_snapshot_file)
    parser.add_argument("--cycle-seconds", type=int, default=EXECUTION_POLICY.cycle_seconds)
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Use the in-memory DryRunBrokerAdapter.")
    return parser


def main() -> None:
    load_dotenv()
    logger = setup_logger()
    args = build_arg_parser().parse_args()

    if args.dry_run:
        broker: DryRunBrokerAdapter | AlpacaBrokerAdapter = DryRunBrokerAdapter()
        broker.connect()
    else:
        api_key = os.environ.get(ALPACA_CONFIG.api_key_env_var, "")
        api_secret = os.environ.get(ALPACA_CONFIG.api_secret_env_var, "")
        if not api_key or not api_secret:
            raise RuntimeError(
                f"Missing Alpaca credentials: set {ALPACA_CONFIG.api_key_env_var} and "
                f"{ALPACA_CONFIG.api_secret_env_var} environment variables."
            )
        broker = AlpacaBrokerAdapter(
            api_key=api_key,
            api_secret=api_secret,
            paper_trading=ALPACA_CONFIG.paper_trading,
            connect_timeout_seconds=ALPACA_CONFIG.connect_timeout_seconds,
            data_feed=ALPACA_CONFIG.data_feed,
        )

    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = Path(args.journal_file)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    store = StateStore(state_path, journal_path)
    trader = IbkrPaperTrader(
        broker=broker,
        store=store,
        alert_snapshot_path=Path(args.alert_snapshot_file),
        signal_archive_path=Path(args.signal_archive_file),
        logger=logger,
        budget_config=TRADING_BUDGET,
        execution_policy=EXECUTION_POLICY,
    )

    try:
        while True:
            start = time.time()
            now_et = datetime.now(ET)
            if is_weekend_shutdown_window(now_et):
                sleep_seconds = seconds_until_weekend_shutdown_end(now_et)
                logger.info(
                    "Weekend shutdown active. Sleeping %.1f hours until Monday 00:00 ET.",
                    sleep_seconds / 3600.0,
                )
                if args.once:
                    break
                time.sleep(sleep_seconds)
                continue

            trader.run_once(now_et)
            if args.once:
                break
            elapsed = time.time() - start
            target_cycle = (
                EXECUTION_POLICY.open_order_poll_seconds
                if is_regular_trading_hours(now_et)
                else float(args.cycle_seconds)
            )
            sleep_seconds = max(0.0, float(target_cycle) - elapsed)
            logger.info("Cycle complete. Sleeping %.1f sec", sleep_seconds)
            time.sleep(sleep_seconds)
    finally:
        if not args.dry_run:
            broker.disconnect()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement run_alpaca_trader.py**

```python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_trading import alpaca_trader

if __name__ == "__main__":
    alpaca_trader.main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_ibkr_paper_trader.py::AlpacaTraderWiringTests -v
```
Expected: 1 passed (dry-run one cycle exits 0).

- [ ] **Step 6: Pyright check**

```bash
"C:/Users/XavierFriesen/AppData/Local/Programs/Python/Python313/python.exe" -m pyright live_trading/alpaca_trader.py
```
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add live_trading/alpaca_trader.py live_trading/run_alpaca_trader.py
git commit -m "feat: add alpaca_trader wiring and run_alpaca_trader entry point"
```

---

## Chunk 4: Alpaca Data Supplement in live_scoring.py

### Task 6: Add AlpacaMarketDataClient to live_scoring.py

**Files:**
- Modify: `live_scoring.py`
- Create: `tests/test_alpaca_data_supplement.py`

The `AlpacaMarketDataClient` class and `_TokenBucket` helper live directly in `live_scoring.py` (uses stdlib `urllib` only — no new import in the scoring pipeline). They are added near the top of the file, after existing imports.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alpaca_data_supplement.py
from __future__ import annotations

import json
import os
import time
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch


class TokenBucketTests(unittest.TestCase):
    def _bucket(self, rate: int):
        from live_scoring import _TokenBucket
        return _TokenBucket(rate)

    def test_full_bucket_allows_consume(self) -> None:
        b = self._bucket(200)
        self.assertTrue(b.consume())

    def test_empty_bucket_denies_consume(self) -> None:
        b = self._bucket(1)          # 1 token/min = ~0.017 tokens/sec
        b.consume()                  # drain the initial token
        self.assertFalse(b.consume())

    def test_bucket_refills_over_time(self) -> None:
        b = self._bucket(600)        # 10 tokens/sec
        for _ in range(600):
            b.consume()              # drain all initial tokens
        time.sleep(0.15)            # wait ~1.5 tokens worth
        self.assertTrue(b.consume())


class AlpacaMarketDataClientTests(unittest.TestCase):
    def _client(self, rate: int = 200):
        from live_scoring import AlpacaMarketDataClient
        return AlpacaMarketDataClient(
            api_key="k", api_secret="s", data_feed="iex", rate_limit_per_minute=rate
        )

    def _mock_response(self, body: dict):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(body).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_returns_mid_price_from_quote(self) -> None:
        client = self._client()
        body = {"quote": {"ap": 150.5, "bp": 149.5}}
        with patch("urllib.request.urlopen", return_value=self._mock_response(body)):
            price = client.get_latest_price("AAPL")
        self.assertAlmostEqual(price, 150.0)  # type: ignore[arg-type]

    def test_returns_ask_when_no_bid(self) -> None:
        client = self._client()
        body = {"quote": {"ap": 150.0}}
        with patch("urllib.request.urlopen", return_value=self._mock_response(body)):
            price = client.get_latest_price("AAPL")
        self.assertAlmostEqual(price, 150.0)  # type: ignore[arg-type]

    def test_returns_none_on_timeout(self) -> None:
        import socket
        client = self._client()
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            price = client.get_latest_price("AAPL")
        self.assertIsNone(price)

    def test_returns_none_on_http_error(self) -> None:
        from urllib.error import HTTPError
        client = self._client()
        with patch("urllib.request.urlopen", side_effect=HTTPError(None, 500, "err", {}, None)):  # type: ignore[arg-type]
            price = client.get_latest_price("AAPL")
        self.assertIsNone(price)

    def test_returns_none_when_rate_limit_exhausted(self) -> None:
        client = self._client(rate=1)
        body = {"quote": {"ap": 100.0, "bp": 99.0}}
        with patch("urllib.request.urlopen", return_value=self._mock_response(body)):
            client.get_latest_price("AAPL")    # consume the one token
            price = client.get_latest_price("AAPL")  # bucket empty
        self.assertIsNone(price)

    def test_disabled_by_env_var_returns_none(self) -> None:
        client = self._client()
        body = {"quote": {"ap": 150.0, "bp": 149.0}}
        with patch.dict(os.environ, {"ALPACA_SUPPLEMENT_ENABLED": "false"}):
            with patch("urllib.request.urlopen", return_value=self._mock_response(body)) as mock_url:
                price = client.get_latest_price("AAPL")
                mock_url.assert_not_called()
        self.assertIsNone(price)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_alpaca_data_supplement.py -v
```
Expected: ImportError — `_TokenBucket` and `AlpacaMarketDataClient` not in `live_scoring` yet.

- [ ] **Step 3: Add _TokenBucket and AlpacaMarketDataClient to live_scoring.py**

Find the imports section near the top of `live_scoring.py` (after existing stdlib imports, before third-party). Add the following block immediately **before** the `from polygon import RESTClient` line:

```python
# ── Alpaca real-time data supplement ────────────────────────────────────────
import json as _json
import os as _os
import threading as _threading
import time as _time
import urllib.request as _urllib_request


class _TokenBucket:
    """Non-blocking token bucket for rate limiting (thread-safe)."""

    def __init__(self, rate_per_minute: int) -> None:
        self._tokens = float(rate_per_minute)
        self._max = float(rate_per_minute)
        self._refill_per_sec = float(rate_per_minute) / 60.0
        self._last = _time.monotonic()
        self._lock = _threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            now = _time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._max, self._tokens + elapsed * self._refill_per_sec)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class AlpacaMarketDataClient:
    """
    Fetches the latest quote for a symbol from Alpaca's market data API.

    Uses stdlib urllib only (no SDK dependency in live_scoring.py).
    Non-blocking: returns None on timeout, HTTP error, or rate-limit exhaustion.
    Disabled entirely when env var ALPACA_SUPPLEMENT_ENABLED=false.
    """

    _BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        data_feed: str = "iex",
        rate_limit_per_minute: int = 200,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._data_feed = data_feed
        self._bucket = _TokenBucket(rate_limit_per_minute)
        self._timeout = 2.0

    def get_latest_price(self, symbol: str) -> float | None:
        """Return latest mid/ask/bid price. Returns None on any failure."""
        if _os.environ.get("ALPACA_SUPPLEMENT_ENABLED", "true").lower() in ("false", "0", "no"):
            return None
        if not self._bucket.consume():
            return None
        url = (
            f"{self._BASE_URL}/v2/stocks/{symbol.upper()}/quotes/latest"
            f"?feed={self._data_feed}"
        )
        req = _urllib_request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._api_secret,
            },
        )
        try:
            with _urllib_request.urlopen(req, timeout=self._timeout) as resp:
                data = _json.loads(resp.read())
            quote = data.get("quote") or {}
            ask = quote.get("ap")
            bid = quote.get("bp")
            if ask and bid:
                return (float(ask) + float(bid)) / 2.0
            if ask:
                return float(ask)
            if bid:
                return float(bid)
        except Exception:
            pass
        return None
# ── End Alpaca supplement ────────────────────────────────────────────────────
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_alpaca_data_supplement.py -v
```
Expected: all tests pass.

---

### Task 7: Wire AlpacaMarketDataClient into enrich_pending_with_market_data

**Files:**
- Modify: `live_scoring.py`

The integration adds Alpaca latest-price fetching **inside the existing parallel prefetch pool** for today's tickers, then uses results in the row loop.

- [ ] **Step 1: Add alpaca_api_key / alpaca_api_secret / no_alpaca_supplement args to enrich_pending_with_market_data**

Find the function signature (around line 604):

```python
def enrich_pending_with_market_data(
    pending_df: pd.DataFrame,
    api_key: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
```

Change to:

```python
def enrich_pending_with_market_data(
    pending_df: pd.DataFrame,
    api_key: str,
    cache_dir: Path,
    logger: logging.Logger,
    *,
    alpaca_api_key: str = "",
    alpaca_api_secret: str = "",
    alpaca_supplement_enabled: bool = True,
) -> pd.DataFrame:
```

- [ ] **Step 2: Build the AlpacaMarketDataClient inside the function**

Add after the `client = RESTClient(api_key=api_key, retries=3)` line:

```python
    _alpaca_client: AlpacaMarketDataClient | None = None
    if alpaca_supplement_enabled and alpaca_api_key and alpaca_api_secret:
        from live_trading.strategy_settings import ALPACA_CONFIG
        _alpaca_client = AlpacaMarketDataClient(
            api_key=alpaca_api_key,
            api_secret=alpaca_api_secret,
            data_feed=ALPACA_CONFIG.data_feed,
            rate_limit_per_minute=ALPACA_CONFIG.data_rate_limit_per_minute,
        )
```

- [ ] **Step 3: Add Alpaca futures to the prefetch pool**

In the prefetch section, find the line:
```python
    n_prefetch = len(min_tasks) + len(lk_tasks) + len(fb_tasks)
```

Add before it:
```python
    # Collect unique tickers needing Alpaca latest-price (today's signals only)
    alpaca_tickers: set[str] = set()
    if _alpaca_client is not None:
        for _, row in out.iterrows():
            ticker_a = str(row["ticker"])
            txn_ts_a = pd.to_datetime(row["transaction_date"], errors="coerce")
            if bool(pd.isna(txn_ts_a)):
                continue
            buy_dt_a = compute_buy_datetime(txn_ts_a)  # type: ignore[arg-type]
            if buy_dt_a.date() == today_et_d:
                alpaca_tickers.add(ticker_a)
```

Then find the existing block (around line 670 in `live_scoring.py`):
```python
    n_prefetch = len(min_tasks) + len(lk_tasks) + len(fb_tasks)
    min_futs: Dict[Tuple[str, str], object] = {}
    if n_prefetch > 0:
        logger.info("Prefetching %d uncached Polygon tasks in parallel ...", n_prefetch)
        with ThreadPoolExecutor(max_workers=min(16, n_prefetch)) as pool:
```

Replace those five lines with:
```python
    n_prefetch = len(min_tasks) + len(lk_tasks) + len(fb_tasks)
    n_total_tasks = n_prefetch + len(alpaca_tickers)
    alpaca_futs: dict[str, object] = {}
    min_futs: Dict[Tuple[str, str], object] = {}
    if n_total_tasks > 0:
        logger.info("Prefetching %d uncached Polygon tasks + %d Alpaca quotes in parallel ...", n_prefetch, len(alpaca_tickers))
        with ThreadPoolExecutor(max_workers=min(16, max(1, n_total_tasks))) as pool:
```

Then inside the pool context, after the existing `pool.submit` calls for `fb_tasks`, add:
```python
            if _alpaca_client is not None:
                for aticker in alpaca_tickers:
                    alpaca_futs[aticker] = pool.submit(_alpaca_client.get_latest_price, aticker)
```

After the pool closes, collect Alpaca results:
```python
    alpaca_prices: dict[str, float] = {}
    for aticker, afut in alpaca_futs.items():  # type: ignore[assignment]
        try:
            price = afut.result()  # type: ignore[union-attr]
            if price is not None:
                alpaca_prices[aticker] = float(price)
        except Exception as exc:
            logger.debug("Alpaca price fetch %s: %s", aticker, exc)
    if alpaca_prices:
        logger.info("Alpaca real-time prices fetched for %d tickers: %s", len(alpaca_prices), sorted(alpaca_prices))
```

- [ ] **Step 4: Use Alpaca prices in the row loop**

In the row loop, after `buy_price` is computed via Polygon (around line 715-718), add:

```python
        # Override with real-time Alpaca price for today's signals
        if buy_date == today_et_d and ticker in alpaca_prices:
            buy_price = alpaca_prices[ticker]
```

This replaces the Polygon-derived price (which may be 15 min stale) with the real-time Alpaca quote.

- [ ] **Step 5: Wire new args through to the main() call**

Find where `enrich_pending_with_market_data` is called in `live_scoring.py` (search for `enrich_pending_with_market_data(`). Add the new keyword args:

```python
enrich_pending_with_market_data(
    pending_df,
    api_key,
    cache_dir,
    logger,
    alpaca_api_key=args.alpaca_api_key,
    alpaca_api_secret=args.alpaca_api_secret,
    alpaca_supplement_enabled=not args.no_alpaca_supplement,
)
```

- [ ] **Step 6: Add CLI args to live_scoring.py argument parser**

Find `p.add_argument("--polygon-api-key", ...)` and add below it:

```python
    p.add_argument("--alpaca-api-key", default="", help="Alpaca API key for real-time price supplement.")
    p.add_argument("--alpaca-api-secret", default="", help="Alpaca API secret for real-time price supplement.")
    p.add_argument("--no-alpaca-supplement", action="store_true", help="Disable Alpaca real-time price supplement.")
```

Also populate from env vars in the same block where `POLYGON_API_KEY` is resolved:

```python
    if not args.alpaca_api_key:
        args.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
    if not args.alpaca_api_secret:
        args.alpaca_api_secret = os.getenv("ALPACA_API_SECRET", "")
```

- [ ] **Step 7: Run the data supplement tests**

```bash
python -m pytest tests/test_alpaca_data_supplement.py -v
```
Expected: all pass (no regressions from wiring changes).

- [ ] **Step 8: Pyright check on live_scoring.py**

```bash
"C:/Users/XavierFriesen/AppData/Local/Programs/Python/Python313/python.exe" -m pyright live_scoring.py
```
Expected: 0 errors (fix any type errors before continuing).

- [ ] **Step 9: Smoke test live_scoring with no Alpaca key (should behave identically to before)**

```bash
python live_scoring.py --help
```
Expected: help text includes `--alpaca-api-key`, `--alpaca-api-secret`, `--no-alpaca-supplement`. No crash.

- [ ] **Step 10: Commit**

```bash
git add live_scoring.py tests/test_alpaca_data_supplement.py
git commit -m "feat: add AlpacaMarketDataClient and wire into live_scoring buy_price enrichment"
```

---

## Final Verification

- [ ] **Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all pass.

- [ ] **Pyright full check**

```bash
"C:/Users/XavierFriesen/AppData/Local/Programs/Python/Python313/python.exe" -m pyright live_trading/strategy_settings.py live_trading/alpaca_broker.py live_trading/alpaca_trader.py live_scoring.py
```
Expected: 0 errors across all four files.

- [ ] **Dry-run alpaca trader end-to-end**

```bash
python live_trading/run_alpaca_trader.py --dry-run --once
```
Expected: exits 0, logs show "state_initialized" and "Cycle complete."

- [ ] **Final commit**

```bash
git add -u
git commit -m "feat: Alpaca live trader + real-time price supplement complete"
```
