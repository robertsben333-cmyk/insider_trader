from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


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
        client = self._client()
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            price = client.get_latest_price("AAPL")
        self.assertIsNone(price)

    def test_falls_back_to_latest_trade_when_quote_is_empty(self) -> None:
        client = self._client()
        quote_body = {"quote": {}}
        trade_body = {"trade": {"p": 151.25}}
        with patch(
            "urllib.request.urlopen",
            side_effect=[self._mock_response(quote_body), self._mock_response(trade_body)],
        ):
            price = client.get_latest_price("AAPL")
        self.assertAlmostEqual(price, 151.25)  # type: ignore[arg-type]

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

    def test_latest_available_close_returns_last_bar_up_to_as_of_date(self) -> None:
        client = self._client()
        body = {"bars": [{"c": 99.0}, {"c": 101.5}]}
        with patch("urllib.request.urlopen", return_value=self._mock_response(body)):
            price = client.get_latest_available_close("AAPL", datetime(2026, 3, 16).date())
        self.assertAlmostEqual(price, 101.5)  # type: ignore[arg-type]


class LiveScoringRefreshTests(unittest.TestCase):
    def test_preopen_refresh_window_uses_one_minute_polling(self) -> None:
        from live_scoring import compute_sleep_interval_minutes
        from live_trading.market_calendar import ET

        now_et = datetime(2026, 3, 12, 9, 20, tzinfo=ET)
        self.assertEqual(compute_sleep_interval_minutes(now_et), 1)

    def test_select_preopen_refresh_events_targets_today_open_candidates(self) -> None:
        from live_scoring import select_preopen_refresh_events
        from live_trading.market_calendar import ET

        candidates = pd.DataFrame(
            [
                {
                    "event_key": "AFTER|2026-03-11",
                    "representative_transaction_date": "2026-03-11 18:30:00",
                },
                {
                    "event_key": "INTRA|2026-03-12",
                    "representative_transaction_date": "2026-03-12 10:05:00",
                },
                {
                    "event_key": "NEXT|2026-03-12",
                    "representative_transaction_date": "2026-03-12 18:30:00",
                },
            ]
        )

        refreshed = select_preopen_refresh_events(
            candidates,
            datetime(2026, 3, 12, 9, 20, tzinfo=ET),
        )

        self.assertEqual(refreshed["event_key"].tolist(), ["AFTER|2026-03-11"])

    def test_filter_active_alert_candidates_removes_stale_rows_from_latest_snapshot(self) -> None:
        from live_scoring import filter_active_alert_candidates
        from live_trading.market_calendar import ET

        export_df = pd.DataFrame(
            [
                {
                    "scored_at": "2026-02-25 17:01:09",
                    "event_key": "VEEE|2026-02-23",
                    "ticker": "VEEE",
                },
                {
                    "scored_at": "2026-03-25 13:20:00",
                    "event_key": "AAA|2026-03-25",
                    "ticker": "AAA",
                    "representative_transaction_date": "2026-03-25 13:20:00",
                },
                {
                    "scored_at": "2026-03-25 14:02:05",
                    "event_key": "OLD|2026-03-10",
                    "ticker": "OLD",
                    "representative_transaction_date": "2026-03-10 13:20:00",
                },
            ]
        )

        latest = filter_active_alert_candidates(
            export_df,
            now_et=datetime(2026, 3, 25, 14, 30, tzinfo=ET),
        )

        self.assertEqual(latest["ticker"].tolist(), ["AAA"])

    def test_update_alert_candidate_exports_preserves_filing_timestamps(self) -> None:
        from live_scoring import read_csv_or_empty, update_alert_candidate_exports

        export_df = pd.DataFrame(
            [
                {
                    "scored_at": "2026-03-25 14:02:05",
                    "representative_transaction_date": "2026-03-25 13:20:00",
                    "transaction_date": "2026-03-25 13:20:00",
                    "event_key": "CUR|2026-03-25",
                    "ticker": "CUR",
                },
                {
                    "scored_at": "2026-03-25 14:02:05",
                    "representative_transaction_date": "2026-03-10 13:20:00",
                    "transaction_date": "2026-03-10 13:20:00",
                    "event_key": "OLD|2026-03-10",
                    "ticker": "OLD",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            latest_path = Path(tmpdir) / "latest.csv"
            history_path = Path(tmpdir) / "history.csv"

            update_alert_candidate_exports(
                latest_path,
                history_path,
                export_df,
                logging.getLogger("test"),
            )

            latest = read_csv_or_empty(latest_path)
            history = read_csv_or_empty(history_path)

        self.assertIn("representative_transaction_date", latest.columns)
        self.assertIn("transaction_date", latest.columns)
        self.assertEqual(latest.loc[0, "representative_transaction_date"], "2026-03-25 13:20:00")
        self.assertEqual(latest.loc[0, "transaction_date"], "2026-03-25 13:20:00")
        self.assertEqual(
            history.loc[history["ticker"] == "OLD", "transaction_date"].iloc[0],
            "2026-03-10 13:20:00",
        )

    def test_refresh_empty_minute_cache_only_after_open_for_today(self) -> None:
        from live_scoring import _should_refresh_empty_minute_cache
        from live_trading.market_calendar import ET

        target_date = datetime(2026, 3, 12, 0, 0, tzinfo=ET).date()
        self.assertFalse(
            _should_refresh_empty_minute_cache(
                target_date,
                now_et=datetime(2026, 3, 12, 9, 20, tzinfo=ET),
            )
        )
        self.assertTrue(
            _should_refresh_empty_minute_cache(
                target_date,
                now_et=datetime(2026, 3, 12, 9, 35, tzinfo=ET),
            )
        )

    def test_refresh_recent_empty_daily_cache_for_previous_close(self) -> None:
        from live_scoring import _should_refresh_empty_daily_cache

        self.assertTrue(
            _should_refresh_empty_daily_cache(
                datetime(2026, 3, 11).date(),
                today_et=datetime(2026, 3, 12).date(),
            )
        )
        self.assertFalse(
            _should_refresh_empty_daily_cache(
                datetime(2026, 3, 1).date(),
                today_et=datetime(2026, 3, 12).date(),
            )
        )
