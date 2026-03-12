from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from live_trading.broker import DryRunBrokerAdapter, QuoteSnapshot
from live_trading.ibkr_paper_trader import IbkrPaperTrader
from live_trading.market_calendar import (
    ET,
    exit_at_tplus_open,
    is_weekend_shutdown_window,
    sleeve_id_for_trade_day,
)
from live_trading.signal_intake import load_signal_candidates
from live_trading.strategy_settings import EXECUTION_POLICY, TRADING_BUDGET
from live_trading.trader_state import FillEvent, PositionLot, SignalCandidate, SleeveState, StateStore, TraderStateSnapshot


class IbkrPaperTraderTests(unittest.TestCase):
    def test_exit_at_tplus_open(self) -> None:
        entry = datetime(2026, 3, 2, 10, 15, tzinfo=ET)
        exit_dt = exit_at_tplus_open(entry, 2)
        self.assertEqual(exit_dt, datetime(2026, 3, 4, 9, 30, tzinfo=ET))

    def test_two_sleeve_rotation_assignment(self) -> None:
        anchor = TRADING_BUDGET.anchor_date()
        sleeve_a = sleeve_id_for_trade_day(anchor, anchor, TRADING_BUDGET.sleeve_count)
        sleeve_b = sleeve_id_for_trade_day(anchor.replace(day=anchor.day + 1), anchor, TRADING_BUDGET.sleeve_count)
        self.assertEqual(sleeve_a, "sleeve_0")
        self.assertEqual(sleeve_b, "sleeve_1")

    def test_signal_intake_builds_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "latest_alert_candidates.csv"
            pd.DataFrame(
                [
                    {
                        "scored_at": "2026-03-02 14:31:00",
                        "event_key": "AAA|2026-03-02",
                        "ticker": "AAA",
                        "score_1d": 1.2,
                        "pred_mean4": 1.2,
                        "estimated_decile_score": 0.94,
                        "advised_allocation_fraction": 0.4,
                        "buy_price": 12.5,
                        "is_tradable": 1,
                    }
                ]
            ).to_csv(path, index=False)
            candidates = load_signal_candidates(
                path,
                budget_config=TRADING_BUDGET,
                execution_policy=EXECUTION_POLICY,
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].ticker, "AAA")
        self.assertEqual(candidates[0].sleeve_id, "sleeve_0")
        self.assertEqual(candidates[0].entry_bucket, "intraday")

    def test_signal_intake_queues_premarket_candidates_for_open_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "latest_alert_candidates.csv"
            pd.DataFrame(
                [
                    {
                        "scored_at": "2026-03-02 13:00:00",
                        "event_key": "AAA|2026-03-02",
                        "ticker": "AAA",
                        "score_1d": 1.2,
                        "pred_mean4": 1.2,
                        "estimated_decile_score": 0.94,
                        "advised_allocation_fraction": 0.4,
                        "buy_price": 12.5,
                        "is_tradable": 1,
                    }
                ]
            ).to_csv(path, index=False)
            candidates = load_signal_candidates(
                path,
                budget_config=TRADING_BUDGET,
                execution_policy=EXECUTION_POLICY,
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].entry_bucket, "open")
        self.assertEqual(
            candidates[0].intended_entry_at,
            datetime(2026, 3, 2, 9, 30, tzinfo=ET).isoformat(),
        )

    def test_state_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            journal_path = Path(tmpdir) / "journal.jsonl"
            store = StateStore(state_path, journal_path)
            snapshot = TraderStateSnapshot(
                sleeves=[SleeveState(sleeve_id="sleeve_0", starting_cash=5000.0, cash_balance=4900.0)],
            )
            store.save(snapshot)
            reloaded = store.load()
        self.assertEqual(len(reloaded.sleeves), 1)
        self.assertEqual(reloaded.sleeves[0].cash_balance, 4900.0)

    def test_score_priority_uses_cash_on_higher_rank_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", ask=500.0, last=500.0))
            broker.set_quote(QuoteSnapshot(symbol="BBB", ask=500.0, last=500.0))

            store = StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl")
            trader = IbkrPaperTrader(
                broker=broker,
                store=store,
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            now = datetime(2026, 3, 2, 10, 0, tzinfo=ET)
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=1000.0,
                        cash_balance=1000.0,
                        last_equity=1000.0,
                    )
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|1",
                        event_key="AAA",
                        ticker="AAA",
                        scored_at="2026-03-02 14:31:00",
                        intended_entry_at=datetime(2026, 3, 2, 9, 45, tzinfo=ET).isoformat(),
                        expires_at=datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.60,
                        score_column="score_1d",
                        step_up_from_prev_close_pct=1.0,
                    ),
                    SignalCandidate(
                        candidate_id="BBB|1",
                        event_key="BBB",
                        ticker="BBB",
                        scored_at="2026-03-02 14:32:00",
                        intended_entry_at=datetime(2026, 3, 2, 9, 45, tzinfo=ET).isoformat(),
                        expires_at=datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.0,
                        estimated_decile_score=0.91,
                        advised_allocation_fraction=0.60,
                        score_column="score_1d",
                        step_up_from_prev_close_pct=1.0,
                    ),
                ],
            )
            trader._manage_entry_orders(snapshot, now)
        self.assertIsNotNone(snapshot.candidates[0].active_order_id)
        self.assertIsNone(snapshot.candidates[1].active_order_id)
        self.assertLess(snapshot.sleeves[0].cash_balance, 1000.0)

    def test_open_batch_fully_deploys_cash_across_identified_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", ask=100.0, last=100.0))
            broker.set_quote(QuoteSnapshot(symbol="BBB", ask=100.0, last=100.0))

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            intended = datetime(2026, 3, 2, 9, 30, tzinfo=ET).isoformat()
            expiry = datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat()
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=1000.0,
                        cash_balance=1000.0,
                        last_equity=1000.0,
                    )
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|1",
                        event_key="AAA",
                        ticker="AAA",
                        scored_at="2026-03-02 14:00:00",
                        intended_entry_at=intended,
                        expires_at=expiry,
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-02",
                        step_up_from_prev_close_pct=1.0,
                    ),
                    SignalCandidate(
                        candidate_id="BBB|1",
                        event_key="BBB",
                        ticker="BBB",
                        scored_at="2026-03-02 14:01:00",
                        intended_entry_at=intended,
                        expires_at=expiry,
                        sleeve_id="sleeve_0",
                        signal_score=1.0,
                        estimated_decile_score=0.91,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-02",
                        step_up_from_prev_close_pct=1.0,
                    ),
                ],
            )
            trader._manage_entry_orders(snapshot, datetime(2026, 3, 2, 9, 30, tzinfo=ET))

        self.assertEqual(len(snapshot.pending_orders), 2)
        self.assertTrue(all(order.order_type == "MARKET" for order in snapshot.pending_orders))
        self.assertLess(snapshot.sleeves[0].cash_balance, snapshot.pending_orders[0].limit_price)
        self.assertLess(snapshot.sleeves[0].cash_balance, snapshot.pending_orders[1].limit_price)

    def test_open_batch_is_prepared_at_0929_without_submitting_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", ask=100.0, last=100.0))
            broker.set_quote(QuoteSnapshot(symbol="BBB", ask=100.0, last=100.0))

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            intended = datetime(2026, 3, 2, 9, 30, tzinfo=ET).isoformat()
            expiry = datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat()
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=1000.0,
                        cash_balance=1000.0,
                        last_equity=1000.0,
                    )
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|1",
                        event_key="AAA",
                        ticker="AAA",
                        scored_at="2026-03-02 13:00:00",
                        intended_entry_at=intended,
                        expires_at=expiry,
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-02",
                        step_up_from_prev_close_pct=1.0,
                    ),
                    SignalCandidate(
                        candidate_id="BBB|1",
                        event_key="BBB",
                        ticker="BBB",
                        scored_at="2026-03-02 13:01:00",
                        intended_entry_at=intended,
                        expires_at=expiry,
                        sleeve_id="sleeve_0",
                        signal_score=1.0,
                        estimated_decile_score=0.91,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-02",
                        step_up_from_prev_close_pct=1.0,
                    ),
                ],
            )

            trader._manage_entry_orders(snapshot, datetime(2026, 3, 2, 9, 29, tzinfo=ET))

        self.assertEqual(len(snapshot.pending_orders), 0)
        self.assertIn("open_batch_plans", snapshot.metadata)
        self.assertIn(f"sleeve_0|{intended}", snapshot.metadata["open_batch_plans"])

    def test_open_batch_prioritizes_best_candidates_while_exit_fills_are_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", ask=100.0, last=100.0))
            broker.set_quote(QuoteSnapshot(symbol="BBB", ask=100.0, last=100.0))

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            intended_dt = datetime(2026, 3, 4, 9, 30, tzinfo=ET)
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=1000.0,
                        cash_balance=150.0,
                        last_equity=1000.0,
                    )
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|1",
                        event_key="AAA",
                        ticker="AAA",
                        scored_at="2026-03-04 13:00:00",
                        intended_entry_at=intended_dt.isoformat(),
                        expires_at=datetime(2026, 3, 4, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-04",
                        step_up_from_prev_close_pct=1.0,
                    ),
                    SignalCandidate(
                        candidate_id="BBB|1",
                        event_key="BBB",
                        ticker="BBB",
                        scored_at="2026-03-04 13:01:00",
                        intended_entry_at=intended_dt.isoformat(),
                        expires_at=datetime(2026, 3, 4, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=0.8,
                        estimated_decile_score=0.91,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-04",
                        step_up_from_prev_close_pct=1.0,
                    )
                ],
                lots=[
                    PositionLot(
                        lot_id="lot_old",
                        candidate_id="OLD|1",
                        ticker="OLD",
                        sleeve_id="sleeve_0",
                        entry_order_id="ord_old",
                        opened_at=datetime(2026, 3, 2, 10, 0, tzinfo=ET).isoformat(),
                        due_exit_at=intended_dt.isoformat(),
                        entry_quantity=10,
                        quantity=10,
                        entry_value=1000.0,
                        entry_estimated_decile_score=0.88,
                        entry_trade_day="2026-03-02",
                    )
                ],
            )

            trader._ensure_planned_exits(snapshot)
            trader._manage_entry_orders(snapshot, intended_dt)

        self.assertEqual(len(snapshot.pending_orders), 1)
        self.assertEqual(snapshot.pending_orders[0].ticker, "AAA")
        self.assertEqual(snapshot.candidates[1].last_reason, "waiting_for_exit_fills")

    def test_candidate_expires_after_cutoff_when_unfunded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trader = IbkrPaperTrader(
                broker=DryRunBrokerAdapter(),
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            snapshot = TraderStateSnapshot(
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|1",
                        event_key="AAA",
                        ticker="AAA",
                        scored_at="2026-03-02 14:31:00",
                        intended_entry_at=datetime(2026, 3, 2, 9, 45, tzinfo=ET).isoformat(),
                        expires_at=datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.60,
                        score_column="score_1d",
                        step_up_from_prev_close_pct=1.0,
                    )
                ]
            )
            trader._expire_stale_candidates(snapshot, datetime(2026, 3, 2, 15, 45, tzinfo=ET))
        self.assertEqual(snapshot.candidates[0].status, "expired")

    def test_weekend_shutdown_starts_after_friday_close(self) -> None:
        self.assertFalse(is_weekend_shutdown_window(datetime(2026, 3, 6, 15, 59, tzinfo=ET)))
        self.assertTrue(is_weekend_shutdown_window(datetime(2026, 3, 6, 16, 0, tzinfo=ET)))
        self.assertTrue(is_weekend_shutdown_window(datetime(2026, 3, 7, 12, 0, tzinfo=ET)))
        self.assertFalse(is_weekend_shutdown_window(datetime(2026, 3, 9, 0, 0, tzinfo=ET)))

    def test_intraday_replacement_selects_lowest_decile_prev_day_lot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", bid=100.0, last=100.0))

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(sleeve_id="sleeve_0", starting_cash=5000.0, cash_balance=0.0, last_equity=5000.0),
                    SleeveState(sleeve_id="sleeve_1", starting_cash=5000.0, cash_balance=0.0, last_equity=5000.0),
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="NEW|1",
                        event_key="NEW",
                        ticker="NEW",
                        scored_at="2026-03-03 15:00:00",
                        intended_entry_at=datetime(2026, 3, 3, 10, 0, tzinfo=ET).isoformat(),
                        expires_at=datetime(2026, 3, 3, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.5,
                        estimated_decile_score=0.96,
                        advised_allocation_fraction=0.60,
                        score_column="score_1d",
                        entry_bucket="intraday",
                        entry_trade_day="2026-03-03",
                        buy_price_hint=50.0,
                        step_up_from_prev_close_pct=1.0,
                    )
                ],
                lots=[
                    PositionLot(
                        lot_id="lot_low",
                        candidate_id="OLD_LOW|1",
                        ticker="AAA",
                        sleeve_id="sleeve_1",
                        entry_order_id="ord_old_low",
                        opened_at=datetime(2026, 3, 2, 10, 0, tzinfo=ET).isoformat(),
                        due_exit_at=datetime(2026, 3, 4, 9, 30, tzinfo=ET).isoformat(),
                        entry_quantity=10,
                        quantity=10,
                        entry_value=1000.0,
                        entry_estimated_decile_score=0.88,
                        entry_trade_day="2026-03-02",
                    ),
                    PositionLot(
                        lot_id="lot_high",
                        candidate_id="OLD_HIGH|1",
                        ticker="BBB",
                        sleeve_id="sleeve_0",
                        entry_order_id="ord_old_high",
                        opened_at=datetime(2026, 3, 2, 10, 5, tzinfo=ET).isoformat(),
                        due_exit_at=datetime(2026, 3, 4, 9, 30, tzinfo=ET).isoformat(),
                        entry_quantity=10,
                        quantity=10,
                        entry_value=1000.0,
                        entry_estimated_decile_score=0.93,
                        entry_trade_day="2026-03-02",
                    ),
                ],
            )

            trader._ensure_planned_exits(snapshot)
            trader._manage_intraday_replacements(snapshot, datetime(2026, 3, 3, 10, 1, tzinfo=ET))

        self.assertEqual(snapshot.candidates[0].replacement_for_lot_id, "lot_low")
        self.assertEqual(snapshot.candidates[0].sleeve_id, "sleeve_1")
        self.assertIsNotNone(snapshot.lots[0].active_exit_order_id)
        self.assertEqual(len(snapshot.pending_orders), 1)
        self.assertEqual(snapshot.pending_orders[0].ticker, "AAA")

    def test_intraday_candidate_above_step_up_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", ask=100.0, last=100.0))

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=1000.0,
                        cash_balance=1000.0,
                        last_equity=1000.0,
                    )
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|1",
                        event_key="AAA",
                        ticker="AAA",
                        scored_at="2026-03-02 14:31:00",
                        intended_entry_at=datetime(2026, 3, 2, 9, 45, tzinfo=ET).isoformat(),
                        expires_at=datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.60,
                        score_column="score_1d",
                        step_up_from_prev_close_pct=2.5,
                    )
                ],
            )

            trader._manage_entry_orders(snapshot, datetime(2026, 3, 2, 10, 0, tzinfo=ET))

        self.assertEqual(snapshot.candidates[0].status, "rejected")
        self.assertEqual(snapshot.candidates[0].last_reason, "step_up_exceeds_live_max_pct")
        self.assertEqual(len(snapshot.pending_orders), 0)

    def test_open_batch_caps_to_top_ten_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            candidates: list[SignalCandidate] = []
            intended = datetime(2026, 3, 2, 9, 30, tzinfo=ET).isoformat()
            expiry = datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat()
            for idx in range(11):
                ticker = f"T{idx:02d}"
                broker.set_quote(QuoteSnapshot(symbol=ticker, ask=100.0, last=100.0))
                candidates.append(
                    SignalCandidate(
                        candidate_id=f"{ticker}|1",
                        event_key=ticker,
                        ticker=ticker,
                        scored_at=f"2026-03-02 13:{idx:02d}:00",
                        intended_entry_at=intended,
                        expires_at=expiry,
                        sleeve_id="sleeve_0",
                        signal_score=float(20 - idx),
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.10,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-02",
                        step_up_from_prev_close_pct=1.0,
                    )
                )

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=10000.0,
                        cash_balance=10000.0,
                        last_equity=10000.0,
                    )
                ],
                candidates=candidates,
            )

            trader._manage_entry_orders(snapshot, datetime(2026, 3, 2, 9, 30, tzinfo=ET))

        self.assertEqual(len(snapshot.pending_orders), 10)
        self.assertTrue(all(order.order_type == "MARKET" for order in snapshot.pending_orders))
        self.assertEqual(snapshot.candidates[-1].status, "rejected")
        self.assertEqual(snapshot.candidates[-1].last_reason, "batch_rank_exceeds_live_max")

    def test_ingest_signals_replaces_same_event_with_newer_refresh_even_if_score_is_lower(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "latest_alert_candidates.csv"
            pd.DataFrame(
                [
                    {
                        "scored_at": "2026-03-02 13:20:00",
                        "event_key": "AAA|2026-03-02",
                        "ticker": "AAA",
                        "score_1d": 0.9,
                        "pred_mean4": 0.9,
                        "estimated_decile_score": 0.95,
                        "advised_allocation_fraction": 0.4,
                        "buy_price": 12.5,
                        "step_up_from_prev_close_pct": 1.0,
                        "is_tradable": 1,
                    }
                ]
            ).to_csv(snapshot_path, index=False)

            trader = IbkrPaperTrader(
                broker=DryRunBrokerAdapter(),
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=snapshot_path,
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            snapshot = TraderStateSnapshot(
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|2026-03-02|2026-03-02 13:00:00",
                        event_key="AAA|2026-03-02",
                        ticker="AAA",
                        scored_at="2026-03-02 13:00:00",
                        intended_entry_at=datetime(2026, 3, 2, 9, 30, tzinfo=ET).isoformat(),
                        expires_at=datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.96,
                        advised_allocation_fraction=0.4,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-02",
                        step_up_from_prev_close_pct=1.0,
                    )
                ]
            )

            trader._ingest_signals(snapshot)

        self.assertEqual(len(snapshot.candidates), 2)
        self.assertEqual(snapshot.candidates[0].status, "expired")
        self.assertEqual(snapshot.candidates[0].last_reason, "superseded_by_newer_score_refresh")
        self.assertEqual(snapshot.candidates[1].signal_score, 0.9)

    def test_preopen_refresh_expires_today_open_candidate_missing_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "latest_alert_candidates.csv"
            pd.DataFrame(columns=["event_key", "ticker", "scored_at"]).to_csv(snapshot_path, index=False)

            trader = IbkrPaperTrader(
                broker=DryRunBrokerAdapter(),
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=snapshot_path,
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            snapshot = TraderStateSnapshot(
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|2026-03-02|2026-03-02 13:00:00",
                        event_key="AAA|2026-03-02",
                        ticker="AAA",
                        scored_at="2026-03-02 13:00:00",
                        intended_entry_at=datetime(2026, 3, 2, 9, 30, tzinfo=ET).isoformat(),
                        expires_at=datetime(2026, 3, 2, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.96,
                        advised_allocation_fraction=0.4,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-02",
                        step_up_from_prev_close_pct=1.0,
                    )
                ]
            )

            trader._reconcile_today_open_candidates_with_snapshot(
                snapshot,
                datetime(2026, 3, 2, 9, 20, tzinfo=ET),
            )

        self.assertEqual(snapshot.candidates[0].status, "expired")
        self.assertEqual(snapshot.candidates[0].last_reason, "removed_by_preopen_refresh")

    def test_same_day_buy_nets_against_due_exit_and_only_sells_remainder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", bid=99.5, ask=99.5, last=99.5))

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            intended_dt = datetime(2026, 3, 4, 9, 30, tzinfo=ET)
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=1000.0,
                        cash_balance=1000.0,
                        last_equity=1000.0,
                    )
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|new",
                        event_key="AAA|2026-03-04",
                        ticker="AAA",
                        scored_at="2026-03-04 13:20:00",
                        intended_entry_at=intended_dt.isoformat(),
                        expires_at=datetime(2026, 3, 4, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-04",
                        buy_price_hint=99.5,
                        step_up_from_prev_close_pct=1.0,
                    )
                ],
                lots=[
                    PositionLot(
                        lot_id="lot_old",
                        candidate_id="AAA|old",
                        ticker="AAA",
                        sleeve_id="sleeve_0",
                        entry_order_id="ord_old",
                        opened_at=datetime(2026, 3, 2, 10, 0, tzinfo=ET).isoformat(),
                        due_exit_at=intended_dt.isoformat(),
                        entry_quantity=10,
                        quantity=10,
                        entry_value=1000.0,
                        entry_estimated_decile_score=0.88,
                        entry_trade_day="2026-03-02",
                    )
                ],
            )

            trader._ensure_planned_exits(snapshot)
            trader._net_same_day_buys_and_sells(snapshot, intended_dt)
            trader._manage_exit_orders(snapshot, intended_dt)
            trader._manage_entry_orders(snapshot, intended_dt)

        candidate = snapshot.candidates[0]
        old_lot = next(row for row in snapshot.lots if row.lot_id == "lot_old")
        new_lot = next(row for row in snapshot.lots if row.candidate_id == candidate.candidate_id)
        sell_orders = [row for row in snapshot.pending_orders if row.side == "SELL"]
        buy_orders = [row for row in snapshot.pending_orders if row.side == "BUY"]

        self.assertEqual(candidate.status, "filled")
        self.assertEqual(old_lot.quantity, 2)
        self.assertEqual(new_lot.quantity, 8)
        self.assertEqual(len(sell_orders), 1)
        self.assertEqual(sell_orders[0].quantity, 2)
        self.assertEqual(len(buy_orders), 0)

    def test_same_day_buy_partial_net_still_submits_buy_for_remainder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", bid=99.5, ask=99.5, last=99.5))

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            intended_dt = datetime(2026, 3, 4, 9, 30, tzinfo=ET)
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=1000.0,
                        cash_balance=1000.0,
                        last_equity=1000.0,
                    )
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|new",
                        event_key="AAA|2026-03-04",
                        ticker="AAA",
                        scored_at="2026-03-04 13:20:00",
                        intended_entry_at=intended_dt.isoformat(),
                        expires_at=datetime(2026, 3, 4, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-04",
                        buy_price_hint=99.5,
                        step_up_from_prev_close_pct=1.0,
                    )
                ],
                lots=[
                    PositionLot(
                        lot_id="lot_old",
                        candidate_id="AAA|old",
                        ticker="AAA",
                        sleeve_id="sleeve_0",
                        entry_order_id="ord_old",
                        opened_at=datetime(2026, 3, 2, 10, 0, tzinfo=ET).isoformat(),
                        due_exit_at=intended_dt.isoformat(),
                        entry_quantity=5,
                        quantity=5,
                        entry_value=500.0,
                        entry_estimated_decile_score=0.88,
                        entry_trade_day="2026-03-02",
                    )
                ],
            )

            trader._ensure_planned_exits(snapshot)
            trader._net_same_day_buys_and_sells(snapshot, intended_dt)
            trader._manage_exit_orders(snapshot, intended_dt)
            trader._manage_entry_orders(snapshot, intended_dt)

        candidate = snapshot.candidates[0]
        old_lot = next(row for row in snapshot.lots if row.lot_id == "lot_old")
        new_lot = next(row for row in snapshot.lots if row.candidate_id == candidate.candidate_id)
        sell_orders = [row for row in snapshot.pending_orders if row.side == "SELL"]
        buy_orders = [row for row in snapshot.pending_orders if row.side == "BUY"]

        self.assertEqual(old_lot.status, "closed")
        self.assertEqual(old_lot.quantity, 0)
        self.assertEqual(new_lot.quantity, 5)
        self.assertEqual(len(sell_orders), 0)
        self.assertEqual(len(buy_orders), 1)
        self.assertEqual(buy_orders[0].quantity, 3)
        self.assertEqual(buy_orders[0].order_type, "MARKET")

    def test_entry_fill_appends_into_existing_netted_candidate_lot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = DryRunBrokerAdapter()
            broker.connect()
            broker.set_quote(QuoteSnapshot(symbol="AAA", bid=99.5, ask=99.5, last=99.5))

            trader = IbkrPaperTrader(
                broker=broker,
                store=StateStore(Path(tmpdir) / "state.json", Path(tmpdir) / "journal.jsonl"),
                alert_snapshot_path=Path(tmpdir) / "missing.csv",
                signal_archive_path=Path(tmpdir) / "archive.csv",
                logger=logging.getLogger("test_ibkr_paper_trader"),
            )
            intended_dt = datetime(2026, 3, 4, 9, 30, tzinfo=ET)
            snapshot = TraderStateSnapshot(
                sleeves=[
                    SleeveState(
                        sleeve_id="sleeve_0",
                        starting_cash=1000.0,
                        cash_balance=1000.0,
                        last_equity=1000.0,
                    )
                ],
                candidates=[
                    SignalCandidate(
                        candidate_id="AAA|new",
                        event_key="AAA|2026-03-04",
                        ticker="AAA",
                        scored_at="2026-03-04 13:20:00",
                        intended_entry_at=intended_dt.isoformat(),
                        expires_at=datetime(2026, 3, 4, 15, 30, tzinfo=ET).isoformat(),
                        sleeve_id="sleeve_0",
                        signal_score=1.2,
                        estimated_decile_score=0.95,
                        advised_allocation_fraction=0.25,
                        score_column="score_1d",
                        entry_bucket="open",
                        entry_trade_day="2026-03-04",
                        buy_price_hint=99.5,
                        step_up_from_prev_close_pct=1.0,
                    )
                ],
                lots=[
                    PositionLot(
                        lot_id="lot_old",
                        candidate_id="AAA|old",
                        ticker="AAA",
                        sleeve_id="sleeve_0",
                        entry_order_id="ord_old",
                        opened_at=datetime(2026, 3, 2, 10, 0, tzinfo=ET).isoformat(),
                        due_exit_at=intended_dt.isoformat(),
                        entry_quantity=5,
                        quantity=5,
                        entry_value=500.0,
                        entry_estimated_decile_score=0.88,
                        entry_trade_day="2026-03-02",
                    )
                ],
            )

            trader._ensure_planned_exits(snapshot)
            trader._net_same_day_buys_and_sells(snapshot, intended_dt)
            trader._manage_entry_orders(snapshot, intended_dt)

            buy_order = next(row for row in snapshot.pending_orders if row.side == "BUY")
            fill = FillEvent(
                fill_id="fill_1",
                execution_id="exec_1",
                local_order_id=buy_order.local_order_id,
                broker_order_id=buy_order.broker_order_id,
                ticker="AAA",
                side="BUY",
                quantity=3,
                price=100.0,
                filled_at=intended_dt.isoformat(),
            )
            trader._apply_entry_fill(snapshot, buy_order, fill)

        candidate = snapshot.candidates[0]
        candidate_lot = next(row for row in snapshot.lots if row.lot_id == candidate.linked_lot_id)

        self.assertEqual(len([row for row in snapshot.lots if row.candidate_id == candidate.candidate_id]), 1)
        self.assertEqual(candidate_lot.entry_quantity, 8)
        self.assertEqual(candidate_lot.quantity, 8)
        self.assertEqual(candidate_lot.entry_value, 800.0)


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


class AlpacaTraderWiringTests(unittest.TestCase):
    def test_main_dry_run_completes_one_cycle(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "live_trading/run_alpaca_trader.py",
                "--dry-run",
                "--once",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.fail(
                f"run_alpaca_trader.py --dry-run --once exited with code "
                f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )


if __name__ == "__main__":
    unittest.main()
