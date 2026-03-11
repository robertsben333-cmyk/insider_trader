from __future__ import annotations

from datetime import UTC, datetime
from unittest import TestCase
from unittest.mock import MagicMock, patch
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

    def test_regular_hours_orders_use_market_order(self) -> None:
        from alpaca.trading.requests import MarketOrderRequest as AlpacaMarketOrder
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
        view = adapter.place_order(broker_req)
        self.assertIsInstance(captured[0], AlpacaMarketOrder)
        self.assertFalse(bool(captured[0].extended_hours))
        self.assertEqual(view.order_type, "MARKET")

    def test_outside_rth_orders_are_rejected(self) -> None:
        adapter = self._connected_adapter()
        broker_req = BrokerOrderRequest(
            order_ref="r1", symbol="AAPL", side="BUY",
            quantity=5, limit_price=150.0, outside_rth=True,
        )
        with self.assertRaisesRegex(ValueError, "outside_rth orders are disabled"):
            adapter.place_order(broker_req)
