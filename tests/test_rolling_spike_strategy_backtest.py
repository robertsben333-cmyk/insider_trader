from __future__ import annotations

from datetime import date
import sys
import unittest
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from research.scripts import backtest_rolling_spike_strategy as strategy


class RollingSpikeStrategyExitTests(unittest.TestCase):
    def test_open_snapshot_exits_same_day_close_when_threshold_hits(self) -> None:
        row = pd.Series(
            {
                "snapshot_date": "2026-03-16",
                "label_end_date": "2026-03-18",
                "snapshot_kind": "open",
                "snapshot_price": 100.0,
                "snapshot_timestamp": "2026-03-16 09:30:00",
            }
        )
        trading_dates = [date(2026, 3, 16), date(2026, 3, 17), date(2026, 3, 18)]
        bar_by_date = {
            date(2026, 3, 16): {"h": 109.0, "c": 105.0},
            date(2026, 3, 17): {"h": 102.0, "c": 101.0},
            date(2026, 3, 18): {"h": 101.0, "c": 100.0},
        }

        out = strategy.resolve_trade_exit(row, trading_dates, bar_by_date)

        self.assertIsNotNone(out)
        self.assertEqual(out["exit_date"], date(2026, 3, 16))
        self.assertEqual(out["exit_reason"], "target_hit_close")
        self.assertAlmostEqual(out["ret_pct"], 5.0)

    def test_no_hit_exits_on_last_allowed_close(self) -> None:
        row = pd.Series(
            {
                "snapshot_date": "2026-03-16",
                "label_end_date": "2026-03-18",
                "snapshot_kind": "close",
                "snapshot_price": 100.0,
                "snapshot_timestamp": "2026-03-16 16:00:00",
            }
        )
        trading_dates = [date(2026, 3, 16), date(2026, 3, 17), date(2026, 3, 18)]
        bar_by_date = {
            date(2026, 3, 16): {"h": 101.0, "c": 100.0},
            date(2026, 3, 17): {"h": 105.0, "c": 104.0},
            date(2026, 3, 18): {"h": 107.0, "c": 103.0},
        }

        out = strategy.resolve_trade_exit(row, trading_dates, bar_by_date)

        self.assertIsNotNone(out)
        self.assertEqual(out["exit_date"], date(2026, 3, 18))
        self.assertEqual(out["exit_reason"], "time_exit_close")
        self.assertAlmostEqual(out["ret_pct"], 3.0)


class RollingSpikeStrategySimulationTests(unittest.TestCase):
    def test_simulation_skips_overlapping_same_event_until_exit(self) -> None:
        picks = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "event_key": "AAA|2026-03-16",
                    "entry_time": "2026-03-16 09:30:00",
                    "exit_time": "2026-03-17 16:00:00",
                    "ret_pct": 5.0,
                    "prob_ensemble_calibrated": 0.30,
                    "snapshot_price": 100.0,
                    "exit_price": 105.0,
                    "exit_reason": "target_hit_close",
                    "snapshot_day_idx": 0,
                },
                {
                    "ticker": "AAA",
                    "event_key": "AAA|2026-03-16",
                    "entry_time": "2026-03-16 16:00:00",
                    "exit_time": "2026-03-18 16:00:00",
                    "ret_pct": 2.0,
                    "prob_ensemble_calibrated": 0.25,
                    "snapshot_price": 105.0,
                    "exit_price": 107.1,
                    "exit_reason": "time_exit_close",
                    "snapshot_day_idx": 1,
                },
                {
                    "ticker": "AAA",
                    "event_key": "AAA|2026-03-16",
                    "entry_time": "2026-03-18 16:00:00",
                    "exit_time": "2026-03-19 16:00:00",
                    "ret_pct": 3.0,
                    "prob_ensemble_calibrated": 0.20,
                    "snapshot_price": 107.1,
                    "exit_price": 110.313,
                    "exit_reason": "time_exit_close",
                    "snapshot_day_idx": 2,
                },
            ]
        )

        curve_df, trades_df = strategy.simulate_budget_curve(
            picks,
            score_col="prob_ensemble_calibrated",
            start_budget_usd=10_000.0,
            max_ticker_weight=1.0,
        )

        self.assertFalse(curve_df.empty)
        self.assertEqual(len(trades_df), 2)
        self.assertEqual(trades_df["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(), ["2026-03-16 09:30:00", "2026-03-18 16:00:00"])


if __name__ == "__main__":
    unittest.main()
