from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from live_trading.broker import (
    AccountSnapshot,
    BrokerAdapter,
    BrokerFillView,
    BrokerOrderView,
    BrokerPositionView,
    DryRunBrokerAdapter,
    IbkrBrokerAdapter,
)
from live_trading.strategy_settings import (
    ACTIVE_STRATEGY,
    DASHBOARD_CONFIG,
    EXECUTION_POLICY,
    IBKR_CONFIG,
    LIVE_PATHS,
    TRADING_BUDGET,
)
from live_trading.trader_state import utc_now_iso


DEFAULT_DASHBOARD_CLIENT_ID = IBKR_CONFIG.client_id + 1


def _parse_timestamp(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return datetime.now(UTC)
    return ts.to_pydatetime()


@dataclass
class BaselinePosition:
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float


@dataclass
class BaselineRecord:
    mode: str
    established_at: str
    baseline_equity: float
    baseline_cash: float
    baseline_market_value: float
    account_id: str
    strategy_id: str
    positions: list[BaselinePosition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineRecord":
        return cls(
            mode=str(data.get("mode", "flat_start")),
            established_at=str(data.get("established_at", "")),
            baseline_equity=float(data.get("baseline_equity", 0.0)),
            baseline_cash=float(data.get("baseline_cash", 0.0)),
            baseline_market_value=float(data.get("baseline_market_value", 0.0)),
            account_id=str(data.get("account_id", "")),
            strategy_id=str(data.get("strategy_id", "")),
            positions=[BaselinePosition(**row) for row in data.get("positions", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionRecord:
    execution_id: str
    broker_order_id: int | None
    order_ref: str
    symbol: str
    side: str
    quantity: int
    price: float
    filled_at: str
    commission: float
    gross_notional: float
    net_cash_flow: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionRecord":
        broker_order_id = data.get("broker_order_id")
        return cls(
            execution_id=str(data.get("execution_id", "")),
            broker_order_id=int(broker_order_id) if broker_order_id not in {None, ""} else None,
            order_ref=str(data.get("order_ref", "")),
            symbol=str(data.get("symbol", "")),
            side=str(data.get("side", "")),
            quantity=int(data.get("quantity", 0)),
            price=float(data.get("price", 0.0)),
            filled_at=str(data.get("filled_at", "")),
            commission=float(data.get("commission", 0.0)),
            gross_notional=float(data.get("gross_notional", 0.0)),
            net_cash_flow=float(data.get("net_cash_flow", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioRow:
    symbol: str
    quantity: int
    avg_cost: float
    broker_avg_cost: float
    mark_price: float
    market_value: float
    unrealized_pnl: float
    last_updated_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortfolioRow":
        return cls(
            symbol=str(data.get("symbol", "")),
            quantity=int(data.get("quantity", 0)),
            avg_cost=float(data.get("avg_cost", 0.0)),
            broker_avg_cost=float(data.get("broker_avg_cost", 0.0)),
            mark_price=float(data.get("mark_price", 0.0)),
            market_value=float(data.get("market_value", 0.0)),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            last_updated_at=str(data.get("last_updated_at", "")),
        )


@dataclass
class TradeRow:
    execution_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    gross_notional: float
    commission: float
    net_cash_flow: float
    filled_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeRow":
        return cls(
            execution_id=str(data.get("execution_id", "")),
            symbol=str(data.get("symbol", "")),
            side=str(data.get("side", "")),
            quantity=int(data.get("quantity", 0)),
            price=float(data.get("price", 0.0)),
            gross_notional=float(data.get("gross_notional", 0.0)),
            commission=float(data.get("commission", 0.0)),
            net_cash_flow=float(data.get("net_cash_flow", 0.0)),
            filled_at=str(data.get("filled_at", "")),
        )


@dataclass
class OrderRow:
    broker_order_id: int | None
    order_ref: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    remaining_quantity: int
    limit_price: float
    status: str
    placed_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderRow":
        broker_order_id = data.get("broker_order_id")
        return cls(
            broker_order_id=int(broker_order_id) if broker_order_id not in {None, ""} else None,
            order_ref=str(data.get("order_ref", "")),
            symbol=str(data.get("symbol", "")),
            side=str(data.get("side", "")),
            quantity=int(data.get("quantity", 0)),
            filled_quantity=int(data.get("filled_quantity", 0)),
            remaining_quantity=int(data.get("remaining_quantity", 0)),
            limit_price=float(data.get("limit_price", 0.0)),
            status=str(data.get("status", "")),
            placed_at=str(data.get("placed_at", "")),
        )


@dataclass
class HeadlineMetrics:
    total_equity: float = 0.0
    total_return_value: float = 0.0
    total_return_pct: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    cash_balance: float = 0.0
    market_value: float = 0.0
    open_positions: int = 0
    baseline_equity: float = 0.0
    broker_available_funds: float = 0.0
    broker_net_liquidation: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadlineMetrics":
        return cls(
            total_equity=float(data.get("total_equity", 0.0)),
            total_return_value=float(data.get("total_return_value", 0.0)),
            total_return_pct=float(data.get("total_return_pct", 0.0)),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            cash_balance=float(data.get("cash_balance", 0.0)),
            market_value=float(data.get("market_value", 0.0)),
            open_positions=int(data.get("open_positions", 0)),
            baseline_equity=float(data.get("baseline_equity", 0.0)),
            broker_available_funds=float(data.get("broker_available_funds", 0.0)),
            broker_net_liquidation=float(data.get("broker_net_liquidation", 0.0)),
        )


@dataclass
class StrategyDashboardSnapshot:
    strategy_id: str
    account_id: str
    generated_at: str
    connection_status: str
    stale_data: bool
    warning_message: str
    baseline_mode: str
    baseline_started_at: str
    headline_metrics: HeadlineMetrics = field(default_factory=HeadlineMetrics)
    portfolio_rows: list[PortfolioRow] = field(default_factory=list)
    trade_rows: list[TradeRow] = field(default_factory=list)
    order_rows: list[OrderRow] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyDashboardSnapshot":
        return cls(
            strategy_id=str(data.get("strategy_id", ACTIVE_STRATEGY.strategy_id)),
            account_id=str(data.get("account_id", "")),
            generated_at=str(data.get("generated_at", "")),
            connection_status=str(data.get("connection_status", "unknown")),
            stale_data=bool(data.get("stale_data", True)),
            warning_message=str(data.get("warning_message", "")),
            baseline_mode=str(data.get("baseline_mode", "")),
            baseline_started_at=str(data.get("baseline_started_at", "")),
            headline_metrics=HeadlineMetrics.from_dict(data.get("headline_metrics", {})),
            portfolio_rows=[PortfolioRow.from_dict(row) for row in data.get("portfolio_rows", [])],
            trade_rows=[TradeRow.from_dict(row) for row in data.get("trade_rows", [])],
            order_rows=[OrderRow.from_dict(row) for row in data.get("order_rows", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarkedBrokerPosition:
    symbol: str
    quantity: int
    broker_avg_cost: float
    mark_price: float
    market_value: float


@dataclass
class LedgerPosition:
    symbol: str
    quantity: int
    total_cost: float

    @property
    def avg_cost(self) -> float:
        if self.quantity <= 0:
            return 0.0
        return self.total_cost / float(self.quantity)


@dataclass
class LedgerSnapshot:
    cash_balance: float
    realized_pnl: float
    positions: dict[str, LedgerPosition]


class DashboardStore:
    def __init__(
        self,
        *,
        baseline_path: Path,
        executions_path: Path,
        equity_history_path: Path,
        latest_snapshot_path: Path,
    ) -> None:
        self.baseline_path = baseline_path
        self.executions_path = executions_path
        self.equity_history_path = equity_history_path
        self.latest_snapshot_path = latest_snapshot_path

    def load_baseline(self) -> BaselineRecord | None:
        if not self.baseline_path.exists():
            return None
        raw = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        return BaselineRecord.from_dict(raw)

    def save_baseline(self, baseline: BaselineRecord) -> None:
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_path.write_text(json.dumps(baseline.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def load_execution_ids(self) -> set[str]:
        return {row.execution_id for row in self.load_executions()}

    def load_executions(self) -> list[ExecutionRecord]:
        if not self.executions_path.exists():
            return []
        rows: list[ExecutionRecord] = []
        with self.executions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(ExecutionRecord.from_dict(json.loads(line)))
        rows.sort(key=lambda row: (_parse_timestamp(row.filled_at), row.execution_id))
        return rows

    def append_executions(self, records: list[ExecutionRecord]) -> None:
        if not records:
            return
        self.executions_path.parent.mkdir(parents=True, exist_ok=True)
        with self.executions_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def append_equity_snapshot(self, snapshot: StrategyDashboardSnapshot) -> None:
        self.equity_history_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = snapshot.headline_metrics
        row = {
            "generated_at": snapshot.generated_at,
            "strategy_id": snapshot.strategy_id,
            "account_id": snapshot.account_id,
            "baseline_mode": snapshot.baseline_mode,
            "baseline_started_at": snapshot.baseline_started_at,
            "total_equity": metrics.total_equity,
            "total_return_value": metrics.total_return_value,
            "total_return_pct": metrics.total_return_pct,
            "realized_pnl": metrics.realized_pnl,
            "unrealized_pnl": metrics.unrealized_pnl,
            "cash_balance": metrics.cash_balance,
            "market_value": metrics.market_value,
            "open_positions": metrics.open_positions,
            "broker_available_funds": metrics.broker_available_funds,
            "broker_net_liquidation": metrics.broker_net_liquidation,
        }
        write_header = not self.equity_history_path.exists()
        with self.equity_history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def save_latest_snapshot(self, snapshot: StrategyDashboardSnapshot) -> None:
        self.latest_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_snapshot_path.write_text(
            json.dumps(snapshot.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_latest_snapshot(self) -> StrategyDashboardSnapshot | None:
        if not self.latest_snapshot_path.exists():
            return None
        raw = json.loads(self.latest_snapshot_path.read_text(encoding="utf-8"))
        return StrategyDashboardSnapshot.from_dict(raw)


class StrategyDashboardService:
    def __init__(
        self,
        *,
        broker: BrokerAdapter,
        store: DashboardStore,
        logger: logging.Logger | None = None,
        recent_trades_limit: int = DASHBOARD_CONFIG.recent_trades_limit,
        strategy_budget: float = TRADING_BUDGET.initial_strategy_budget,
    ) -> None:
        self.broker = broker
        self.store = store
        self.logger = logger or logging.getLogger("strategy_dashboard")
        self.recent_trades_limit = int(recent_trades_limit)
        self.strategy_budget = float(strategy_budget)

    def sync_broker_state(self) -> StrategyDashboardSnapshot:
        try:
            return self._sync_broker_state()
        except Exception as exc:
            self.logger.exception("Dashboard sync failed.")
            snapshot = self.load_latest_snapshot()
            snapshot.generated_at = utc_now_iso()
            snapshot.connection_status = "error"
            snapshot.stale_data = True
            snapshot.warning_message = f"IBKR sync failed: {exc}"
            return snapshot

    def load_latest_snapshot(self) -> StrategyDashboardSnapshot:
        snapshot = self.store.load_latest_snapshot()
        if snapshot is not None:
            return snapshot
        return StrategyDashboardSnapshot(
            strategy_id=ACTIVE_STRATEGY.strategy_id,
            account_id=IBKR_CONFIG.account_id,
            generated_at=utc_now_iso(),
            connection_status="uninitialized",
            stale_data=True,
            warning_message="No dashboard snapshot has been captured yet.",
            baseline_mode="",
            baseline_started_at="",
        )

    def _sync_broker_state(self) -> StrategyDashboardSnapshot:
        if not self.broker.is_connected():
            self.broker.connect()

        account = self.broker.get_account_snapshot()
        marked_positions = self._mark_positions(self.broker.get_positions())
        baseline = self._ensure_baseline(account, marked_positions)
        self._append_new_executions(self.broker.get_recent_fills())
        all_executions = self.store.load_executions()
        ledger = self._build_ledger(baseline, all_executions)
        open_orders = self.broker.get_open_orders()

        portfolio_rows = self._build_portfolio_rows(marked_positions, ledger)
        headline_metrics = self._build_headline_metrics(account, baseline, ledger, portfolio_rows)
        trade_rows = self._build_trade_rows(all_executions)
        order_rows = self._build_order_rows(open_orders)
        snapshot = StrategyDashboardSnapshot(
            strategy_id=ACTIVE_STRATEGY.strategy_id,
            account_id=account.account_id,
            generated_at=utc_now_iso(),
            connection_status="connected",
            stale_data=False,
            warning_message="",
            baseline_mode=baseline.mode,
            baseline_started_at=baseline.established_at,
            headline_metrics=headline_metrics,
            portfolio_rows=portfolio_rows,
            trade_rows=trade_rows,
            order_rows=order_rows,
        )
        self.store.append_equity_snapshot(snapshot)
        self.store.save_latest_snapshot(snapshot)
        return snapshot

    def _mark_positions(self, positions: list[BrokerPositionView]) -> list[MarkedBrokerPosition]:
        out: list[MarkedBrokerPosition] = []
        for position in positions:
            if int(position.quantity) == 0:
                continue
            symbol = str(position.symbol).upper()
            fallback = position.market_price if position.market_price is not None else position.avg_cost
            quote = self.broker.get_quote(symbol)
            mark_price = quote.reference_price("SELL", fallback)
            if mark_price is None:
                mark_price = float(fallback or 0.0)
            out.append(
                MarkedBrokerPosition(
                    symbol=symbol,
                    quantity=int(position.quantity),
                    broker_avg_cost=float(position.avg_cost),
                    mark_price=float(mark_price),
                    market_value=float(mark_price) * float(position.quantity),
                )
            )
        out.sort(key=lambda row: row.symbol)
        return out

    def _ensure_baseline(self, account: AccountSnapshot, positions: list[MarkedBrokerPosition]) -> BaselineRecord:
        existing = self.store.load_baseline()
        if existing is not None:
            return existing

        has_open_positions = any(position.quantity != 0 for position in positions)
        if not has_open_positions:
            baseline = BaselineRecord(
                mode="flat_start",
                established_at=utc_now_iso(),
                baseline_equity=self.strategy_budget,
                baseline_cash=self.strategy_budget,
                baseline_market_value=0.0,
                account_id=account.account_id,
                strategy_id=ACTIVE_STRATEGY.strategy_id,
                positions=[],
            )
        else:
            baseline_positions = [
                BaselinePosition(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_cost=position.mark_price,
                    market_value=position.market_value,
                )
                for position in positions
            ]
            baseline_market_value = sum(position.market_value for position in positions)
            baseline_cash = float(account.total_cash_value)
            baseline = BaselineRecord(
                mode="first_sync",
                established_at=utc_now_iso(),
                baseline_equity=baseline_cash + baseline_market_value,
                baseline_cash=baseline_cash,
                baseline_market_value=baseline_market_value,
                account_id=account.account_id,
                strategy_id=ACTIVE_STRATEGY.strategy_id,
                positions=baseline_positions,
            )
        self.store.save_baseline(baseline)
        return baseline

    def _append_new_executions(self, fills: list[BrokerFillView]) -> None:
        known_ids = self.store.load_execution_ids()
        new_rows: list[ExecutionRecord] = []
        for fill in sorted(fills, key=lambda row: (_parse_timestamp(row.filled_at), row.execution_id)):
            if not fill.execution_id or fill.execution_id in known_ids:
                continue
            new_rows.append(self._normalize_fill(fill))
            known_ids.add(fill.execution_id)
        self.store.append_executions(new_rows)

    def _normalize_fill(self, fill: BrokerFillView) -> ExecutionRecord:
        side = str(fill.side).upper()
        gross_notional = float(fill.quantity) * float(fill.price)
        if side == "BUY":
            net_cash_flow = -(gross_notional + float(fill.commission))
        else:
            net_cash_flow = gross_notional - float(fill.commission)
        return ExecutionRecord(
            execution_id=str(fill.execution_id),
            broker_order_id=fill.broker_order_id,
            order_ref=str(fill.order_ref),
            symbol=str(fill.symbol).upper(),
            side=side,
            quantity=int(fill.quantity),
            price=float(fill.price),
            filled_at=str(fill.filled_at),
            commission=float(fill.commission),
            gross_notional=gross_notional,
            net_cash_flow=net_cash_flow,
        )

    def _build_ledger(self, baseline: BaselineRecord, executions: list[ExecutionRecord]) -> LedgerSnapshot:
        positions = {
            row.symbol: LedgerPosition(
                symbol=row.symbol,
                quantity=int(row.quantity),
                total_cost=float(row.avg_cost) * float(row.quantity),
            )
            for row in baseline.positions
            if int(row.quantity) != 0
        }
        cash_balance = float(baseline.baseline_cash)
        realized_pnl = 0.0

        for execution in executions:
            symbol = execution.symbol.upper()
            side = execution.side.upper()
            quantity = int(execution.quantity)
            if quantity <= 0:
                continue
            position = positions.get(symbol)
            if position is None:
                position = LedgerPosition(symbol=symbol, quantity=0, total_cost=0.0)
                positions[symbol] = position
            gross = float(quantity) * float(execution.price)
            commission = float(execution.commission)
            if side == "BUY":
                position.quantity += quantity
                position.total_cost += gross + commission
                cash_balance -= gross + commission
                continue

            avg_cost = position.avg_cost
            sold_cost = avg_cost * float(quantity)
            proceeds = gross - commission
            realized_pnl += proceeds - sold_cost
            cash_balance += proceeds
            position.quantity -= quantity
            position.total_cost = max(0.0, position.total_cost - sold_cost)
            if position.quantity <= 0:
                positions.pop(symbol, None)

        return LedgerSnapshot(
            cash_balance=cash_balance,
            realized_pnl=realized_pnl,
            positions=positions,
        )

    def _build_portfolio_rows(
        self,
        positions: list[MarkedBrokerPosition],
        ledger: LedgerSnapshot,
    ) -> list[PortfolioRow]:
        last_updated_at = utc_now_iso()
        rows: list[PortfolioRow] = []
        for position in positions:
            ledger_position = ledger.positions.get(position.symbol)
            avg_cost = ledger_position.avg_cost if ledger_position is not None and ledger_position.quantity > 0 else float(
                position.broker_avg_cost
            )
            unrealized_pnl = float(position.market_value) - (avg_cost * float(position.quantity))
            rows.append(
                PortfolioRow(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_cost=avg_cost,
                    broker_avg_cost=float(position.broker_avg_cost),
                    mark_price=float(position.mark_price),
                    market_value=float(position.market_value),
                    unrealized_pnl=unrealized_pnl,
                    last_updated_at=last_updated_at,
                )
            )
        rows.sort(key=lambda row: row.symbol)
        return rows

    def _build_headline_metrics(
        self,
        account: AccountSnapshot,
        baseline: BaselineRecord,
        ledger: LedgerSnapshot,
        portfolio_rows: list[PortfolioRow],
    ) -> HeadlineMetrics:
        market_value = float(sum(row.market_value for row in portfolio_rows))
        unrealized_pnl = float(sum(row.unrealized_pnl for row in portfolio_rows))
        cash_balance = float(ledger.cash_balance)
        total_equity = cash_balance + market_value
        total_return_value = total_equity - float(baseline.baseline_equity)
        baseline_equity = float(baseline.baseline_equity)
        total_return_pct = 0.0 if abs(baseline_equity) < 1e-9 else (total_return_value / baseline_equity) * 100.0
        return HeadlineMetrics(
            total_equity=total_equity,
            total_return_value=total_return_value,
            total_return_pct=total_return_pct,
            realized_pnl=float(ledger.realized_pnl),
            unrealized_pnl=unrealized_pnl,
            cash_balance=cash_balance,
            market_value=market_value,
            open_positions=len(portfolio_rows),
            baseline_equity=baseline_equity,
            broker_available_funds=float(account.available_funds),
            broker_net_liquidation=float(account.net_liquidation),
        )

    def _build_trade_rows(self, executions: list[ExecutionRecord]) -> list[TradeRow]:
        recent = sorted(executions, key=lambda row: (_parse_timestamp(row.filled_at), row.execution_id), reverse=True)
        return [
            TradeRow(
                execution_id=row.execution_id,
                symbol=row.symbol,
                side=row.side,
                quantity=row.quantity,
                price=row.price,
                gross_notional=row.gross_notional,
                commission=row.commission,
                net_cash_flow=row.net_cash_flow,
                filled_at=row.filled_at,
            )
            for row in recent[: self.recent_trades_limit]
        ]

    def _build_order_rows(self, orders: list[BrokerOrderView]) -> list[OrderRow]:
        return [
            OrderRow(
                broker_order_id=row.broker_order_id,
                order_ref=row.order_ref,
                symbol=row.symbol,
                side=row.side,
                quantity=row.quantity,
                filled_quantity=row.filled_quantity,
                remaining_quantity=row.remaining_quantity,
                limit_price=row.limit_price,
                status=row.status,
                placed_at=row.placed_at,
            )
            for row in sorted(orders, key=lambda order: (_parse_timestamp(order.placed_at), order.symbol), reverse=True)
        ]


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("dashboard_sync")


def build_dashboard_store() -> DashboardStore:
    return DashboardStore(
        baseline_path=Path(LIVE_PATHS.dashboard_baseline_file),
        executions_path=Path(LIVE_PATHS.dashboard_executions_file),
        equity_history_path=Path(LIVE_PATHS.dashboard_equity_history_file),
        latest_snapshot_path=Path(LIVE_PATHS.dashboard_latest_snapshot_file),
    )


def build_dashboard_service(
    *,
    broker: BrokerAdapter | None = None,
    logger: logging.Logger | None = None,
) -> StrategyDashboardService:
    return StrategyDashboardService(
        broker=broker
        or IbkrBrokerAdapter(
            host=IBKR_CONFIG.host,
            port=IBKR_CONFIG.port,
            client_id=DEFAULT_DASHBOARD_CLIENT_ID,
            account_id=IBKR_CONFIG.account_id,
            connect_timeout_seconds=IBKR_CONFIG.connect_timeout_seconds,
            readonly=True,
            routing_exchange=EXECUTION_POLICY.routing_exchange,
            currency=EXECUTION_POLICY.currency,
            quote_wait_seconds=EXECUTION_POLICY.quote_wait_seconds,
        ),
        store=build_dashboard_store(),
        logger=logger,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only IBKR dashboard sync service.")
    parser.add_argument("--host", default=IBKR_CONFIG.host)
    parser.add_argument("--port", type=int, default=IBKR_CONFIG.port)
    parser.add_argument("--client-id", type=int, default=DEFAULT_DASHBOARD_CLIENT_ID)
    parser.add_argument("--account-id", default=IBKR_CONFIG.account_id)
    parser.add_argument("--cycle-seconds", type=int, default=DASHBOARD_CONFIG.sync_interval_seconds)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Use the in-memory broker adapter instead of IBKR.")
    return parser


def make_broker(args) -> BrokerAdapter:
    if args.dry_run:
        broker = DryRunBrokerAdapter()
        broker.connect()
        return broker
    return IbkrBrokerAdapter(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        account_id=args.account_id,
        connect_timeout_seconds=IBKR_CONFIG.connect_timeout_seconds,
        readonly=True,
        routing_exchange=EXECUTION_POLICY.routing_exchange,
        currency=EXECUTION_POLICY.currency,
        quote_wait_seconds=EXECUTION_POLICY.quote_wait_seconds,
    )


def main() -> None:
    load_dotenv()
    logger = setup_logger()
    args = build_arg_parser().parse_args()
    broker = make_broker(args)
    service = StrategyDashboardService(
        broker=broker,
        store=build_dashboard_store(),
        logger=logger,
    )
    try:
        while True:
            snapshot = service.sync_broker_state()
            logger.info(
                "Dashboard sync complete: status=%s stale=%s equity=%.2f",
                snapshot.connection_status,
                snapshot.stale_data,
                snapshot.headline_metrics.total_equity,
            )
            if args.once:
                break
            time.sleep(max(1, int(args.cycle_seconds)))
    finally:
        if not args.dry_run and broker.is_connected():
            broker.disconnect()


if __name__ == "__main__":
    main()
