from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import unittest

from live_trading.broker import (
    AccountSnapshot,
    BrokerFillView,
    BrokerOrderRequest,
    BrokerOrderView,
    BrokerPositionView,
    QuoteSnapshot,
)
from live_trading.dashboard_service import DashboardStore, StrategyDashboardService
from live_trading.strategy_dashboard import render_dashboard


class FakeDashboardBroker:
    def __init__(
        self,
        *,
        account: AccountSnapshot,
        positions: list[BrokerPositionView] | None = None,
        orders: list[BrokerOrderView] | None = None,
        fills: list[BrokerFillView] | None = None,
        quotes: dict[str, QuoteSnapshot] | None = None,
        fail: bool = False,
    ) -> None:
        self.account = account
        self.positions = positions or []
        self.orders = orders or []
        self.fills = fills or []
        self.quotes = quotes or {}
        self.fail = fail
        self.connected = False

    def connect(self) -> None:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def get_account_snapshot(self) -> AccountSnapshot:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        return self.account

    def get_positions(self) -> list[BrokerPositionView]:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        return list(self.positions)

    def get_open_orders(self) -> list[BrokerOrderView]:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        return list(self.orders)

    def get_recent_fills(self) -> list[BrokerFillView]:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        return list(self.fills)

    def list_orders(self, include_closed: bool = False) -> list[BrokerOrderView]:
        return self.get_open_orders()

    def list_fills(self) -> list[BrokerFillView]:
        return self.get_recent_fills()

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        if self.fail:
            raise RuntimeError("gateway unavailable")
        return self.quotes.get(symbol.upper(), QuoteSnapshot(symbol=symbol.upper()))

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderView:
        raise NotImplementedError

    def cancel_order(self, broker_order_id: int) -> None:
        raise NotImplementedError


class FakeColumn:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, str, str | None]] = []

    def metric(self, label: str, value: str, delta: str | None = None) -> None:
        self.metrics.append((label, value, delta))


class FakeStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.columns_seen: list[FakeColumn] = []

    def set_page_config(self, **kwargs) -> None:
        self.calls.append(("set_page_config", kwargs))

    def title(self, value: str) -> None:
        self.calls.append(("title", value))

    def caption(self, value: str) -> None:
        self.calls.append(("caption", value))

    def warning(self, value: str) -> None:
        self.calls.append(("warning", value))

    def success(self, value: str) -> None:
        self.calls.append(("success", value))

    def subheader(self, value: str) -> None:
        self.calls.append(("subheader", value))

    def dataframe(self, value, **kwargs) -> None:
        self.calls.append(("dataframe", {"shape": getattr(value, "shape", None), **kwargs}))

    def info(self, value: str) -> None:
        self.calls.append(("info", value))

    def columns(self, count: int) -> list[FakeColumn]:
        cols = [FakeColumn() for _ in range(count)]
        self.columns_seen.extend(cols)
        return cols


class StrategyDashboardTests(unittest.TestCase):
    def _make_store(self, tmpdir: str) -> DashboardStore:
        root = Path(tmpdir)
        return DashboardStore(
            baseline_path=root / "baseline.json",
            executions_path=root / "executions.jsonl",
            equity_history_path=root / "equity_history.csv",
            latest_snapshot_path=root / "latest_snapshot.json",
        )

    def _make_service(self, tmpdir: str, broker: FakeDashboardBroker) -> StrategyDashboardService:
        return StrategyDashboardService(
            broker=broker,
            store=self._make_store(tmpdir),
            logger=logging.getLogger("test_strategy_dashboard"),
            recent_trades_limit=10,
            strategy_budget=1000.0,
        )

    def test_baseline_initializes_flat_start_with_strategy_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = FakeDashboardBroker(
                account=AccountSnapshot(
                    account_id="DU123",
                    net_liquidation=25000.0,
                    available_funds=25000.0,
                    total_cash_value=25000.0,
                ),
            )
            service = self._make_service(tmpdir, broker)
            snapshot = service.sync_broker_state()

        self.assertEqual(snapshot.baseline_mode, "flat_start")
        self.assertAlmostEqual(snapshot.headline_metrics.baseline_equity, 1000.0)
        self.assertAlmostEqual(snapshot.headline_metrics.total_equity, 1000.0)
        self.assertAlmostEqual(snapshot.headline_metrics.cash_balance, 1000.0)
        self.assertEqual(snapshot.headline_metrics.open_positions, 0)

    def test_non_flat_baseline_starts_at_first_sync_and_tracks_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = FakeDashboardBroker(
                account=AccountSnapshot(
                    account_id="DU123",
                    net_liquidation=1000.0,
                    available_funds=880.0,
                    total_cash_value=880.0,
                ),
                positions=[BrokerPositionView(symbol="AAPL", quantity=10, avg_cost=9.5, market_price=None, market_value=None)],
                quotes={"AAPL": QuoteSnapshot(symbol="AAPL", bid=12.0, last=12.0)},
            )
            service = self._make_service(tmpdir, broker)
            first_snapshot = service.sync_broker_state()
            broker.quotes["AAPL"] = QuoteSnapshot(symbol="AAPL", bid=13.0, last=13.0)
            second_snapshot = service.sync_broker_state()

        self.assertEqual(first_snapshot.baseline_mode, "first_sync")
        self.assertAlmostEqual(first_snapshot.headline_metrics.total_return_value, 0.0)
        self.assertAlmostEqual(second_snapshot.headline_metrics.unrealized_pnl, 10.0)
        self.assertAlmostEqual(second_snapshot.headline_metrics.total_return_value, 10.0)

    def test_execution_deduplicates_partial_fills_across_syncs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fills = [
                BrokerFillView(
                    execution_id="exec1",
                    broker_order_id=1,
                    order_ref="ord1",
                    symbol="AAPL",
                    side="BUY",
                    quantity=5,
                    price=10.0,
                    filled_at="2026-03-07T10:00:00Z",
                    commission=1.0,
                ),
                BrokerFillView(
                    execution_id="exec2",
                    broker_order_id=1,
                    order_ref="ord1",
                    symbol="AAPL",
                    side="BUY",
                    quantity=5,
                    price=10.5,
                    filled_at="2026-03-07T10:01:00Z",
                    commission=1.0,
                ),
            ]
            broker = FakeDashboardBroker(
                account=AccountSnapshot(
                    account_id="DU123",
                    net_liquidation=1000.0,
                    available_funds=1000.0,
                    total_cash_value=1000.0,
                ),
                fills=fills,
                quotes={"AAPL": QuoteSnapshot(symbol="AAPL", bid=11.0, last=11.0)},
            )
            service = self._make_service(tmpdir, broker)
            service.sync_broker_state()
            service.sync_broker_state()
            stored = service.store.load_executions()

        self.assertEqual(len(stored), 2)
        self.assertEqual([row.execution_id for row in stored], ["exec1", "exec2"])

    def test_return_math_after_buy_and_partial_sell(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = FakeDashboardBroker(
                account=AccountSnapshot(
                    account_id="DU123",
                    net_liquidation=1000.0,
                    available_funds=1000.0,
                    total_cash_value=1000.0,
                ),
                quotes={"AAPL": QuoteSnapshot(symbol="AAPL", bid=12.0, last=12.0)},
            )
            service = self._make_service(tmpdir, broker)
            service.sync_broker_state()

            broker.fills = [
                BrokerFillView(
                    execution_id="buy1",
                    broker_order_id=1,
                    order_ref="ord1",
                    symbol="AAPL",
                    side="BUY",
                    quantity=10,
                    price=10.0,
                    filled_at="2026-03-07T10:00:00Z",
                    commission=1.0,
                )
            ]
            broker.positions = [BrokerPositionView(symbol="AAPL", quantity=10, avg_cost=10.0, market_price=None, market_value=None)]
            first_snapshot = service.sync_broker_state()

            broker.fills.append(
                BrokerFillView(
                    execution_id="sell1",
                    broker_order_id=2,
                    order_ref="ord2",
                    symbol="AAPL",
                    side="SELL",
                    quantity=5,
                    price=13.0,
                    filled_at="2026-03-07T11:00:00Z",
                    commission=1.0,
                )
            )
            broker.positions = [BrokerPositionView(symbol="AAPL", quantity=5, avg_cost=10.0, market_price=None, market_value=None)]
            second_snapshot = service.sync_broker_state()

        self.assertAlmostEqual(first_snapshot.headline_metrics.total_equity, 1019.0)
        self.assertAlmostEqual(first_snapshot.headline_metrics.unrealized_pnl, 19.0)
        self.assertAlmostEqual(second_snapshot.headline_metrics.cash_balance, 963.0)
        self.assertAlmostEqual(second_snapshot.headline_metrics.realized_pnl, 13.5)
        self.assertAlmostEqual(second_snapshot.headline_metrics.unrealized_pnl, 9.5)
        self.assertAlmostEqual(second_snapshot.headline_metrics.total_equity, 1023.0)

    def test_portfolio_rows_include_live_marks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = FakeDashboardBroker(
                account=AccountSnapshot(
                    account_id="DU123",
                    net_liquidation=1000.0,
                    available_funds=1000.0,
                    total_cash_value=1000.0,
                ),
                quotes={"MSFT": QuoteSnapshot(symbol="MSFT", bid=12.0, last=12.0)},
            )
            service = self._make_service(tmpdir, broker)
            service.sync_broker_state()
            broker.fills = [
                BrokerFillView(
                    execution_id="buy1",
                    broker_order_id=1,
                    order_ref="ord1",
                    symbol="MSFT",
                    side="BUY",
                    quantity=10,
                    price=10.0,
                    filled_at="2026-03-07T10:00:00Z",
                    commission=1.0,
                )
            ]
            broker.positions = [BrokerPositionView(symbol="MSFT", quantity=10, avg_cost=10.0, market_price=None, market_value=None)]
            snapshot = service.sync_broker_state()

        self.assertEqual(len(snapshot.portfolio_rows), 1)
        row = snapshot.portfolio_rows[0]
        self.assertEqual(row.symbol, "MSFT")
        self.assertEqual(row.quantity, 10)
        self.assertAlmostEqual(row.mark_price, 12.0)
        self.assertAlmostEqual(row.market_value, 120.0)
        self.assertAlmostEqual(row.unrealized_pnl, 19.0)

    def test_sync_failure_returns_cached_snapshot_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            healthy = FakeDashboardBroker(
                account=AccountSnapshot(
                    account_id="DU123",
                    net_liquidation=1000.0,
                    available_funds=1000.0,
                    total_cash_value=1000.0,
                ),
            )
            service = self._make_service(tmpdir, healthy)
            initial_snapshot = service.sync_broker_state()

            failing_service = StrategyDashboardService(
                broker=FakeDashboardBroker(account=healthy.account, fail=True),
                store=service.store,
                logger=logging.getLogger("test_strategy_dashboard"),
                recent_trades_limit=10,
                strategy_budget=1000.0,
            )
            fallback = failing_service.sync_broker_state()

        self.assertEqual(initial_snapshot.headline_metrics.total_equity, fallback.headline_metrics.total_equity)
        self.assertTrue(fallback.stale_data)
        self.assertEqual(fallback.connection_status, "error")
        self.assertIn("IBKR sync failed", fallback.warning_message)

    def test_render_dashboard_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = FakeDashboardBroker(
                account=AccountSnapshot(
                    account_id="DU123",
                    net_liquidation=1000.0,
                    available_funds=1000.0,
                    total_cash_value=1000.0,
                ),
            )
            service = self._make_service(tmpdir, broker)
            snapshot = service.sync_broker_state()
            fake_st = FakeStreamlit()

            render_dashboard(snapshot, fake_st)

        call_names = [name for name, _ in fake_st.calls]
        self.assertIn("title", call_names)
        self.assertIn("success", call_names)
        self.assertGreaterEqual(len(fake_st.columns_seen), 6)


if __name__ == "__main__":
    unittest.main()
