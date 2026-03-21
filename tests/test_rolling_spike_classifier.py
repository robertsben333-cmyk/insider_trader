from __future__ import annotations

from datetime import date
import sys
import unittest
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from research.scripts import evaluate_rolling_spike_classifier as rolling


class RollingSpikeSnapshotTests(unittest.TestCase):
    def test_first_snapshot_session_respects_intraday_vs_outside_hours(self) -> None:
        trading_dates = [
            date(2026, 3, 16),
            date(2026, 3, 17),
            date(2026, 3, 18),
        ]

        intraday = pd.Timestamp("2026-03-16 11:00:00")
        premarket = pd.Timestamp("2026-03-16 08:10:00")
        after_hours = pd.Timestamp("2026-03-16 17:10:00")
        weekend = pd.Timestamp("2026-03-14 10:00:00")

        self.assertEqual(rolling.first_snapshot_session(intraday, trading_dates), (date(2026, 3, 16), "close"))
        self.assertEqual(rolling.first_snapshot_session(premarket, trading_dates), (date(2026, 3, 16), "open"))
        self.assertEqual(rolling.first_snapshot_session(after_hours, trading_dates), (date(2026, 3, 17), "open"))
        self.assertEqual(rolling.first_snapshot_session(weekend, trading_dates), (date(2026, 3, 16), "open"))

    def test_open_snapshot_stops_after_same_day_hit(self) -> None:
        row = pd.Series(
            {
                "event_key": "AAA|2026-03-16",
                "ticker": "AAA",
                "transaction_date": "2026-03-16 08:05:00",
                "trade_date": "2026-03-13",
                "company_name": "AAA Inc",
                "owner_name": "Owner",
                "title": "CEO",
                "last_price_clean": 100.0,
            }
        )
        trading_dates = [
            date(2026, 3, 16),
            date(2026, 3, 17),
            date(2026, 3, 18),
            date(2026, 3, 19),
        ]
        bar_by_date = {
            date(2026, 3, 16): {"o": 100.0, "c": 104.0, "h": 109.5},
            date(2026, 3, 17): {"o": 103.0, "c": 102.0, "h": 103.5},
            date(2026, 3, 18): {"o": 102.0, "c": 101.0, "h": 102.5},
            date(2026, 3, 19): {"o": 101.0, "c": 100.0, "h": 101.5},
        }

        rows = rolling.build_snapshot_rows_for_event(row, trading_dates, bar_by_date)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["snapshot_kind"], "open")
        self.assertEqual(rows[0]["snapshot_day_idx"], 0)
        self.assertEqual(rows[0]["lookahead_days_used"], 3)
        self.assertEqual(rows[0][rolling.TARGET_COL], 1)

    def test_late_rows_use_truncated_lookahead(self) -> None:
        row = pd.Series(
            {
                "event_key": "BBB|2026-03-16",
                "ticker": "BBB",
                "transaction_date": "2026-03-16 11:15:00",
                "trade_date": "2026-03-13",
                "company_name": "BBB Inc",
                "owner_name": "Owner",
                "title": "CEO",
                "last_price_clean": 100.0,
            }
        )
        trading_dates = [
            date(2026, 3, 16),
            date(2026, 3, 17),
            date(2026, 3, 18),
        ]
        bar_by_date = {
            date(2026, 3, 16): {"o": 100.0, "c": 101.0, "h": 101.5},
            date(2026, 3, 17): {"o": 101.0, "c": 102.0, "h": 102.2},
            date(2026, 3, 18): {"o": 102.0, "c": 103.0, "h": 103.1},
        }

        rows = rolling.build_snapshot_rows_for_event(row, trading_dates, bar_by_date)

        self.assertEqual(len(rows), 2)
        self.assertEqual([r["lookahead_days_used"] for r in rows], [2, 1])
        self.assertEqual([r["snapshot_kind"] for r in rows], ["close", "close"])


class RollingSpikeSplitTests(unittest.TestCase):
    def test_chronological_event_split_has_no_overlap_or_time_leakage(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=15, freq="5B")
        rows = []
        for idx, d in enumerate(dates):
            snap = d.date()
            label_end = (d + pd.offsets.BDay(1)).date()
            rows.append(
                {
                    "event_key": f"T{idx:02d}|{snap}",
                    "snapshot_date": snap,
                    "label_end_date": label_end,
                    "event_last_label_date": label_end,
                    rolling.TARGET_COL: idx % 2,
                }
            )
        frame = pd.DataFrame(rows)

        split = rolling.chronological_event_split(frame)

        train_keys = set(split.train["event_key"])
        val_keys = set(split.val["event_key"])
        test_keys = set(split.test["event_key"])
        self.assertFalse(train_keys & val_keys)
        self.assertFalse(train_keys & test_keys)
        self.assertFalse(val_keys & test_keys)

        val_start = split.val["snapshot_date"].min()
        test_start = split.test["snapshot_date"].min()
        self.assertTrue((split.train["label_end_date"] < val_start).all())
        self.assertTrue((split.val["label_end_date"] < test_start).all())


if __name__ == "__main__":
    unittest.main()
