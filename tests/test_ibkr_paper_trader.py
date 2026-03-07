from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from live_trading.broker import DryRunBrokerAdapter, QuoteSnapshot
from live_trading.ibkr_paper_trader import IbkrPaperTrader
from live_trading.market_calendar import ET, exit_at_tplus_open, sleeve_id_for_trade_day
from live_trading.signal_intake import load_signal_candidates
from live_trading.strategy_settings import EXECUTION_POLICY, TRADING_BUDGET
from live_trading.trader_state import SignalCandidate, SleeveState, StateStore, TraderStateSnapshot


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
                        "scored_at": "2026-03-02 13:31:00",
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
                    ),
                ],
            )
            trader._manage_entry_orders(snapshot, now)
        self.assertIsNotNone(snapshot.candidates[0].active_order_id)
        self.assertIsNone(snapshot.candidates[1].active_order_id)
        self.assertLess(snapshot.sleeves[0].cash_balance, 1000.0)

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
                    )
                ]
            )
            trader._expire_stale_candidates(snapshot, datetime(2026, 3, 2, 15, 45, tzinfo=ET))
        self.assertEqual(snapshot.candidates[0].status, "expired")


if __name__ == "__main__":
    unittest.main()
