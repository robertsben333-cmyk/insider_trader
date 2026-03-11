from __future__ import annotations

from datetime import UTC, datetime
import json as _json
import logging
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
    from alpaca.data.enums import DataFeed
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


def _enum_str(raw: object) -> str:
    """Return the .value of an enum or str(raw), handling None gracefully."""
    if raw is None:
        return ""
    if hasattr(raw, "value"):
        return str(raw.value)  # type: ignore[union-attr]
    return str(raw)


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
        self._data_feed: DataFeed | None = DataFeed(data_feed) if data_feed else None
        self._connect_timeout_seconds = connect_timeout_seconds
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
            side_str = _enum_str(side_raw).upper()
            status_raw = getattr(order, "status", None)
            status_str = _enum_str(status_raw)
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
        status_str = _enum_str(getattr(order, "status", None))
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
        """Fetch fill activities via direct REST call."""
        url = f"{self._base_url}/v2/account/activities?activity_type=FILL"
        req = _urllib_request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._api_secret,
            },
        )
        try:
            with _urllib_request.urlopen(req, timeout=self._connect_timeout_seconds) as resp:
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
