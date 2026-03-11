from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from typing import Protocol


def _ensure_asyncio_event_loop() -> None:
    """Create a thread-local event loop when libraries expect one at import time."""
    policy = asyncio.get_event_loop_policy()
    try:
        policy.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(policy.new_event_loop())


@dataclass
class QuoteSnapshot:
    symbol: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    close: float | None = None
    captured_at: str | None = None

    def reference_price(self, side: str, fallback: float | None = None) -> float | None:
        if side.upper() == "BUY":
            candidates = [self.ask, self.last, self.close, self.bid, fallback]
        else:
            candidates = [self.bid, self.last, self.close, self.ask, fallback]
        for value in candidates:
            if value is None:
                continue
            if math.isfinite(value) and value > 0:
                return float(value)
        return None


@dataclass
class AccountSnapshot:
    account_id: str
    net_liquidation: float
    available_funds: float
    total_cash_value: float


@dataclass
class BrokerOrderView:
    broker_order_id: int
    order_ref: str
    symbol: str
    side: str
    quantity: int
    limit_price: float
    filled_quantity: int
    remaining_quantity: int
    status: str
    placed_at: str
    order_type: str = "LIMIT"


@dataclass
class BrokerPositionView:
    symbol: str
    quantity: int
    avg_cost: float
    market_price: float | None
    market_value: float | None


@dataclass
class BrokerFillView:
    execution_id: str
    broker_order_id: int | None
    order_ref: str
    symbol: str
    side: str
    quantity: int
    price: float
    filled_at: str
    commission: float = 0.0


@dataclass
class BrokerOrderRequest:
    order_ref: str
    symbol: str
    side: str
    quantity: int
    limit_price: float
    order_type: str = "LIMIT"
    tif: str = "DAY"
    outside_rth: bool = False


class BrokerAdapter(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def get_account_snapshot(self) -> AccountSnapshot: ...

    def get_positions(self) -> list[BrokerPositionView]: ...

    def get_open_orders(self) -> list[BrokerOrderView]: ...

    def get_recent_fills(self) -> list[BrokerFillView]: ...

    def list_orders(self, include_closed: bool = False) -> list[BrokerOrderView]: ...

    def list_fills(self) -> list[BrokerFillView]: ...

    def get_quote(self, symbol: str) -> QuoteSnapshot: ...

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderView: ...

    def cancel_order(self, broker_order_id: int) -> None: ...


class DryRunBrokerAdapter:
    def __init__(self) -> None:
        self._connected = False
        self._next_order_id = 1
        self._orders: dict[int, BrokerOrderView] = {}
        self._quotes: dict[str, QuoteSnapshot] = {}

    def set_quote(self, quote: QuoteSnapshot) -> None:
        self._quotes[quote.symbol.upper()] = quote

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(account_id="DRYRUN", net_liquidation=0.0, available_funds=0.0, total_cash_value=0.0)

    def get_positions(self) -> list[BrokerPositionView]:
        return []

    def get_open_orders(self) -> list[BrokerOrderView]:
        return self.list_orders(include_closed=False)

    def get_recent_fills(self) -> list[BrokerFillView]:
        return self.list_fills()

    def list_orders(self, include_closed: bool = False) -> list[BrokerOrderView]:
        if include_closed:
            return list(self._orders.values())
        return [row for row in self._orders.values() if row.status not in {"Cancelled", "Filled"}]

    def list_fills(self) -> list[BrokerFillView]:
        return []

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        return self._quotes.get(symbol.upper(), QuoteSnapshot(symbol=symbol.upper()))

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderView:
        order = BrokerOrderView(
            broker_order_id=self._next_order_id,
            order_ref=request.order_ref,
            symbol=request.symbol.upper(),
            side=request.side.upper(),
            quantity=int(request.quantity),
            limit_price=float(request.limit_price),
            order_type=str(request.order_type).upper(),
            filled_quantity=0,
            remaining_quantity=int(request.quantity),
            status="Submitted",
            placed_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        self._orders[self._next_order_id] = order
        self._next_order_id += 1
        return order

    def cancel_order(self, broker_order_id: int) -> None:
        order = self._orders.get(int(broker_order_id))
        if order is not None:
            order.status = "Cancelled"


class IbkrBrokerAdapter:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        client_id: int,
        account_id: str,
        connect_timeout_seconds: float,
        readonly: bool,
        routing_exchange: str,
        currency: str,
        quote_wait_seconds: float,
    ) -> None:
        try:
            _ensure_asyncio_event_loop()
            from ib_insync import IB
        except ImportError as exc:
            raise RuntimeError(
                "ib-insync is not installed. Install dependencies from requirements.txt before using the IBKR trader."
            ) from exc
        self._IB = IB
        self._host = host
        self._port = int(port)
        self._client_id = int(client_id)
        self._account_id = str(account_id).strip()
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        self._readonly = bool(readonly)
        self._routing_exchange = routing_exchange
        self._currency = currency
        self._quote_wait_seconds = float(quote_wait_seconds)
        self._ib = self._IB()
        self._contracts: dict[str, object] = {}

    def connect(self) -> None:
        if self._ib.isConnected():
            return
        self._ib.connect(
            host=self._host,
            port=self._port,
            clientId=self._client_id,
            timeout=self._connect_timeout_seconds,
            readonly=self._readonly,
            account=self._account_id or "",
        )

    def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    def is_connected(self) -> bool:
        return bool(self._ib.isConnected())

    def _stock_contract(self, symbol: str):
        from ib_insync import Stock

        key = str(symbol).upper()
        if key not in self._contracts:
            contract = Stock(key, self._routing_exchange, self._currency)
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                raise RuntimeError(f"Unable to qualify IBKR contract for {key}.")
            self._contracts[key] = qualified[0]
        return self._contracts[key]

    def get_account_snapshot(self) -> AccountSnapshot:
        rows = self._ib.accountSummary(self._account_id or None)
        tag_map: dict[str, float] = {}
        account_id = self._account_id
        for row in rows:
            if account_id and str(row.account) != account_id:
                continue
            account_id = str(row.account)
            try:
                tag_map[str(row.tag)] = float(row.value)
            except Exception:
                continue
        return AccountSnapshot(
            account_id=account_id or "",
            net_liquidation=float(tag_map.get("NetLiquidation", 0.0)),
            available_funds=float(tag_map.get("AvailableFunds", 0.0)),
            total_cash_value=float(tag_map.get("TotalCashValue", 0.0)),
        )

    def get_positions(self) -> list[BrokerPositionView]:
        out: list[BrokerPositionView] = []
        for row in self._ib.positions(self._account_id or None):
            contract = getattr(row, "contract", None)
            symbol = getattr(contract, "symbol", "")
            if not symbol:
                continue
            out.append(
                BrokerPositionView(
                    symbol=str(symbol),
                    quantity=int(getattr(row, "position", 0)),
                    avg_cost=float(getattr(row, "avgCost", 0.0) or 0.0),
                    market_price=None,
                    market_value=None,
                )
            )
        return out

    def _trade_to_order_view(self, trade) -> BrokerOrderView:
        order = trade.order
        status = trade.orderStatus
        contract = trade.contract
        total_qty = int(round(float(getattr(order, "totalQuantity", 0) or 0)))
        filled_qty = int(round(float(getattr(status, "filled", 0) or 0)))
        order_type = str(getattr(order, "orderType", "") or "LIMIT").upper()
        return BrokerOrderView(
            broker_order_id=int(getattr(order, "orderId", 0) or 0),
            order_ref=str(getattr(order, "orderRef", "") or ""),
            symbol=str(getattr(contract, "symbol", "") or ""),
            side=str(getattr(order, "action", "") or ""),
            quantity=total_qty,
            limit_price=float(getattr(order, "lmtPrice", 0.0) or 0.0),
            filled_quantity=filled_qty,
            remaining_quantity=max(0, total_qty - filled_qty),
            status=str(getattr(status, "status", "") or ""),
            placed_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            order_type=order_type,
        )

    def list_orders(self, include_closed: bool = False) -> list[BrokerOrderView]:
        trades = self._ib.trades() if include_closed else self._ib.openTrades()
        return [self._trade_to_order_view(trade) for trade in trades]

    def get_open_orders(self) -> list[BrokerOrderView]:
        return self.list_orders(include_closed=False)

    def get_recent_fills(self) -> list[BrokerFillView]:
        return self.list_fills()

    def list_fills(self) -> list[BrokerFillView]:
        out: list[BrokerFillView] = []
        for fill in self._ib.fills():
            execution = getattr(fill, "execution", None)
            contract = getattr(fill, "contract", None)
            if execution is None or contract is None:
                continue
            commission_report = getattr(fill, "commissionReport", None)
            commission = getattr(commission_report, "commission", 0.0) or 0.0
            out.append(
                BrokerFillView(
                    execution_id=str(getattr(execution, "execId", "") or ""),
                    broker_order_id=int(getattr(execution, "orderId", 0) or 0),
                    order_ref=str(getattr(execution, "orderRef", "") or ""),
                    symbol=str(getattr(contract, "symbol", "") or ""),
                    side=str(getattr(execution, "side", "") or ""),
                    quantity=int(round(float(getattr(execution, "shares", 0) or 0))),
                    price=float(getattr(execution, "price", 0.0) or 0.0),
                    filled_at=str(getattr(execution, "time", "") or ""),
                    commission=float(commission),
                )
            )
        return out

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        contract = self._stock_contract(symbol)
        ticker = self._ib.reqTickers(contract)[0]
        self._ib.sleep(self._quote_wait_seconds)
        return QuoteSnapshot(
            symbol=str(symbol).upper(),
            bid=float(ticker.bid) if getattr(ticker, "bid", None) else None,
            ask=float(ticker.ask) if getattr(ticker, "ask", None) else None,
            last=float(ticker.last) if getattr(ticker, "last", None) else None,
            close=float(ticker.close) if getattr(ticker, "close", None) else None,
            captured_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderView:
        from ib_insync import LimitOrder, MarketOrder

        contract = self._stock_contract(request.symbol)
        order_type = str(request.order_type or "LIMIT").upper()
        if order_type == "MARKET":
            order = MarketOrder(
                action=request.side.upper(),
                totalQuantity=int(request.quantity),
                tif=request.tif,
                outsideRth=bool(request.outside_rth),
            )
        else:
            order = LimitOrder(
                action=request.side.upper(),
                totalQuantity=int(request.quantity),
                lmtPrice=float(request.limit_price),
                tif=request.tif,
                outsideRth=bool(request.outside_rth),
            )
        order.orderRef = request.order_ref
        if self._account_id:
            order.account = self._account_id
        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.5)
        return self._trade_to_order_view(trade)

    def cancel_order(self, broker_order_id: int) -> None:
        target = int(broker_order_id)
        for trade in self._ib.trades():
            order = getattr(trade, "order", None)
            if int(getattr(order, "orderId", 0) or 0) == target:
                self._ib.cancelOrder(order)
                self._ib.sleep(0.2)
                return
