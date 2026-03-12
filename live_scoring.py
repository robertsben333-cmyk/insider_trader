"""
Live insider-trade scoring loop.

What this script does every cycle:
1. Rescrapes recent OpenInsider filings (current + previous month).
2. Merges into raw dataset and detects newly arrived filings.
3. Builds event-level rows (one row per ticker/trade_date), using the same
   cluster representative rule as training data (2nd filing if clustered).
4. Enriches pending events with buy price + lookback cache data.
5. Reuses train_models.load_and_merge() + engineer_features() for feature parity.
6. Scores all 4 models for horizons 1d/3d and computes mean score.
7. Upserts predictions into a separate dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import smtplib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
# ── Alpaca real-time data supplement ────────────────────────────────────────
import json as _json
import os as _os
import threading as _threading
import time as _time
import urllib.request as _urllib_request


class _TokenBucket:
    """Non-blocking token bucket for rate limiting (thread-safe)."""

    def __init__(self, rate_per_minute: int) -> None:
        self._tokens = float(rate_per_minute)
        self._max = float(rate_per_minute)
        self._refill_per_sec = float(rate_per_minute) / 60.0
        self._last = _time.monotonic()
        self._lock = _threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            now = _time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._max, self._tokens + elapsed * self._refill_per_sec)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class AlpacaMarketDataClient:
    """
    Fetches the latest quote for a symbol from Alpaca's market data API.

    Uses stdlib urllib only (no SDK dependency in live_scoring.py).
    Non-blocking: returns None on timeout, HTTP error, or rate-limit exhaustion.
    Disabled entirely when env var ALPACA_SUPPLEMENT_ENABLED=false.
    """

    _BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        data_feed: str = "iex",
        rate_limit_per_minute: int = 200,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._data_feed = data_feed
        self._bucket = _TokenBucket(rate_limit_per_minute)
        self._timeout = 2.0

    def get_latest_price(self, symbol: str) -> float | None:
        """Return latest mid/ask/bid price. Returns None on any failure."""
        if _os.environ.get("ALPACA_SUPPLEMENT_ENABLED", "true").lower() in ("false", "0", "no"):
            return None
        if not self._bucket.consume():
            return None
        url = (
            f"{self._BASE_URL}/v2/stocks/{symbol.upper()}/quotes/latest"
            f"?feed={self._data_feed}"
        )
        req = _urllib_request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._api_secret,
            },
        )
        try:
            with _urllib_request.urlopen(req, timeout=self._timeout) as resp:
                data = _json.loads(resp.read())
            quote = data.get("quote") or {}
            ask = quote.get("ap")
            bid = quote.get("bp")
            if ask and bid:
                return (float(ask) + float(bid)) / 2.0
            if ask:
                return float(ask)
            if bid:
                return float(bid)
        except Exception:
            pass
        return None
# ── End Alpaca supplement ────────────────────────────────────────────────────

from polygon import RESTClient

import train_models
from live_trading.market_calendar import is_weekend_shutdown_window, seconds_until_weekend_shutdown_end
from live_trading.strategy_settings import ACTIVE_STRATEGY, LIVE_PATHS, RUNTIME_DEFAULTS
from openinsider_scraper import OpenInsiderScraper

RAW_COLUMNS = [
    "transaction_date",
    "trade_date",
    "ticker",
    "company_name",
    "owner_name",
    "title",
    "transaction_type",
    "last_price",
    "qty",
    "shares_held",
    "owned_pct",
    "value",
]

MODEL_NAMES = ["HGBR", "XGBoost", "ElasticNet", "SplineElasticNet"]
HORIZONS = [1, 3]
DEFAULT_ALERT_RECIPIENT = RUNTIME_DEFAULTS.alert_recipient
DEFAULT_DAY1_DECILE_SCORE_THRESHOLD = ACTIVE_STRATEGY.day1_decile_score_threshold
DEFAULT_DAY1_RAW_THRESHOLD = ACTIVE_STRATEGY.day1_raw_threshold_fallback
DEFAULT_ADVICE_BASE_ALLOC_FRACTION = ACTIVE_STRATEGY.advice_base_alloc_fraction
DEFAULT_ADVICE_BONUS_FRACTION = ACTIVE_STRATEGY.advice_bonus_fraction
EXIT_POLICY_REVIEW_DATE = ACTIVE_STRATEGY.exit_policy_review_date
ALERT_EXPORT_COLUMNS = [
    "scored_at",
    "event_key",
    "ticker",
    "company_name",
    "owner_name",
    "title",
    "trade_date",
    "buy_price",
    "prev_regular_close",
    "step_up_from_prev_close_pct",
    "score_1d",
    "score_3d",
    "score_5d",
    "score_10d",
    "pred_mean4",
    "estimated_decile_score",
    "decile_strength",
    "advised_allocation_fraction",
    "advised_allocation_pct",
    "market_type",
    "is_tradable",
    "raw_alert_threshold",
    "decile_score_threshold",
    "threshold_source",
    "alert_score_column",
    "target_return_mode",
    "benchmark_ticker",
]
OFFICER_PATTERN = re.compile(
    r"\b(COB|Chairman|CEO|Co-CEO|Pres|President|COO|CFO|GC|VP|SVP|EVP)\b",
    re.IGNORECASE,
)
DIRECTOR_PATTERN = re.compile(r"\bdirector\b|\bdir\b", re.IGNORECASE)
TEN_PCT_PATTERN = re.compile(r"10%|10 percent|10pct|ten percent", re.IGNORECASE)

MAX_FILING_GAP_DAYS = 5
MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0
ET = ZoneInfo("America/New_York")

# Polling intervals: high-frequency window covers 2 h before/after market open.
NEAR_OPEN_WINDOW_HOURS = RUNTIME_DEFAULTS.near_open_window_hours
MARKET_HOURS_INTERVAL_MINUTES = RUNTIME_DEFAULTS.market_hours_interval_minutes
NEAR_OPEN_INTERVAL_MINUTES = RUNTIME_DEFAULTS.near_open_interval_minutes
FAR_INTERVAL_MINUTES = RUNTIME_DEFAULTS.far_interval_minutes
PREOPEN_REFRESH_WINDOW_MINUTES = 15


def is_market_open(now_et: datetime) -> bool:
    """True if now_et is within regular trading hours (9:30–16:00 ET, Mon–Fri)."""
    if now_et.weekday() >= 5:
        return False
    open_t  = now_et.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    close_t = now_et.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    return open_t <= now_et < close_t


def seconds_until_next_open(now_et: datetime) -> float:
    """Seconds from now until 9:30 AM ET on the next trading weekday."""
    candidate = now_et.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    if candidate <= now_et:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return max(0.0, (candidate - now_et).total_seconds())


def is_preopen_refresh_window(now_et: datetime) -> bool:
    """True during the final minutes before the regular-session open on weekdays."""
    if now_et.weekday() >= 5:
        return False
    open_t = now_et.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    refresh_start = open_t - timedelta(minutes=PREOPEN_REFRESH_WINDOW_MINUTES)
    return refresh_start <= now_et < open_t


def compute_sleep_interval_minutes(now_et: datetime) -> int:
    """Return polling interval in minutes based on proximity to market open.

    - MARKET_HOURS_INTERVAL_MINUTES (1) during regular trading hours.
    - MARKET_HOURS_INTERVAL_MINUTES (1) in the final PREOPEN_REFRESH_WINDOW_MINUTES
      before 9:30 ET so queued open-entry names can be rescored with fresh gap data.
    - NEAR_OPEN_INTERVAL_MINUTES (30) within NEAR_OPEN_WINDOW_HOURS hours
      before *or* after 9:30 ET on a weekday (i.e. 7:30–11:30 ET).
    - FAR_INTERVAL_MINUTES (120) at all other times (overnight, weekends,
      midday).  This avoids blocking order creation at market open while still
      capturing pre-market and early-session filings frequently.
    """
    if now_et.weekday() >= 5:
        return FAR_INTERVAL_MINUTES
    if is_market_open(now_et):
        return MARKET_HOURS_INTERVAL_MINUTES
    if is_preopen_refresh_window(now_et):
        return MARKET_HOURS_INTERVAL_MINUTES
    open_t = now_et.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    near_start = open_t - timedelta(hours=NEAR_OPEN_WINDOW_HOURS)
    near_end   = open_t + timedelta(hours=NEAR_OPEN_WINDOW_HOURS)
    if near_start <= now_et < near_end:
        return NEAR_OPEN_INTERVAL_MINUTES
    return FAR_INTERVAL_MINUTES

def _to_float(val: object) -> float:
    """Coerce *val* to a Python float; returns nan on failure.

    Using this helper avoids repeated type-ignore comments at call sites where
    pd.to_numeric() returns a broad union that float() can't statically accept.
    """
    try:
        return float(pd.to_numeric(val, errors="coerce"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


EXPECTED_AGG_COLUMNS = [
    "transaction_date",
    "trade_date",
    "ticker",
    "company_name",
    "owner_name",
    "title",
    "last_price",
    "qty",
    "value",
    "filing_gap_days",
    "buy_datetime",
    "buy_price",
    "close_1d",
    "close_3d",
    "close_5d",
    "close_10d",
    "return_1d_pct",
    "return_3d_pct",
    "return_5d_pct",
    "return_10d_pct",
    "trade_date_d",
    "n_insiders",
    "cluster_buy",
    "n_insiders_label",
    "n_insiders_in_cluster",
]


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("live_scoring")


def read_csv_or_empty(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = list(columns) if columns else []
    return pd.DataFrame(columns=pd.Index(cols))


def make_event_key_series(ticker: pd.Series, trade_date_like: pd.Series) -> pd.Series:
    trade_date = pd.to_datetime(trade_date_like, errors="coerce").dt.strftime("%Y-%m-%d")  # type: ignore[union-attr]
    return ticker.fillna("").astype(str) + "|" + trade_date.fillna("")  # type: ignore[return-value]


def row_signature(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return df[list(columns)].fillna("").astype(str).agg("||".join, axis=1)  # type: ignore[return-value]


def coerce_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RAW_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[RAW_COLUMNS]  # type: ignore[return-value]


def month_offsets(now: datetime, months_back: int) -> List[Tuple[int, int]]:
    out = []
    for off in range(months_back + 1):
        y, m = now.year, now.month - off
        while m <= 0:
            y -= 1
            m += 12
        out.append((y, m))
    return out


def _scrape_state_path(raw_file: str) -> Path:
    return Path(raw_file).with_suffix(".scrape_state.json")


def load_scrape_state(raw_file: str) -> datetime | None:
    """Return the timestamp of the last completed scrape, or None if unknown."""
    path = _scrape_state_path(raw_file)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return datetime.fromisoformat(data["last_scraped_at"])
    except Exception:
        return None


def save_scrape_state(raw_file: str, ts: datetime) -> None:
    path = _scrape_state_path(raw_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_scraped_at": ts.isoformat()}), encoding="utf-8")


def compute_months_back(last_scraped_at: datetime | None, now: datetime, default: int, max_months: int = 12) -> int:
    """Return how many previous months to scrape given the last scrape time.

    - Same calendar month as last scrape → 0 (current month only).
    - N calendar months since last scrape → N (covers the full gap).
    - No prior state → ``default`` (first-run behaviour).
    """
    if last_scraped_at is None:
        return default
    month_diff = (now.year * 12 + now.month) - (last_scraped_at.year * 12 + last_scraped_at.month)
    return min(month_diff, max_months)


def scrape_recent_filings(scraper: OpenInsiderScraper, months_back: int, logger: logging.Logger) -> pd.DataFrame:
    now = datetime.now()
    offsets = month_offsets(now, months_back)

    def _fetch_month(ym: Tuple[int, int]) -> list[tuple]:
        year, month = ym
        logger.info("Scraping OpenInsider month %04d-%02d ...", year, month)
        return list(scraper._get_data_for_month(year, month))

    with ThreadPoolExecutor(max_workers=len(offsets)) as pool:
        results = list(pool.map(_fetch_month, offsets))

    frames = [pd.DataFrame(list(rows), columns=pd.Index(RAW_COLUMNS)) for rows in results if rows]
    if not frames:
        return pd.DataFrame(columns=pd.Index(RAW_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out = coerce_raw_columns(out).drop_duplicates(subset=RAW_COLUMNS).reset_index(drop=True)
    return out


def merge_scraped_into_raw(
    raw_path: Path,
    scraped_df: pd.DataFrame,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    existing = coerce_raw_columns(read_csv_or_empty(raw_path, RAW_COLUMNS))
    scraped = coerce_raw_columns(scraped_df)
    if scraped.empty:
        logger.info("Scraper returned 0 rows.")
        return existing, pd.DataFrame(columns=pd.Index(RAW_COLUMNS))

    existing_sig = set(row_signature(existing, RAW_COLUMNS).tolist())
    scraped_sig = row_signature(scraped, RAW_COLUMNS)
    new_rows = scraped.loc[~scraped_sig.isin(existing_sig)].copy()

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=RAW_COLUMNS, keep="first")

    combined["_trade_date_sort"] = pd.to_datetime(combined["trade_date"], errors="coerce")
    combined["_txn_date_sort"] = pd.to_datetime(combined["transaction_date"], errors="coerce")
    combined = (
        combined.sort_values(["_trade_date_sort", "_txn_date_sort"], ascending=True)
        .drop(columns=["_trade_date_sort", "_txn_date_sort"])
        .reset_index(drop=True)
    )

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(raw_path, index=False)
    logger.info(
        "Raw merge complete. Existing=%d, scraped=%d, new=%d, total=%d",
        len(existing),
        len(scraped),
        len(new_rows),
        len(combined),
    )
    return combined, new_rows


def build_candidate_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_raw_columns(raw_df).copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date", "trade_date", "ticker"]).copy()

    title = df["title"].astype(str)
    mask_supported = (
        title.str.contains(OFFICER_PATTERN, na=False)
        | title.str.contains(DIRECTOR_PATTERN, na=False)
        | title.str.contains(TEN_PCT_PATTERN, na=False)
    )
    df = df[mask_supported].copy()
    df["filing_gap_days"] = (df["transaction_date"] - df["trade_date"]).dt.days  # type: ignore[union-attr]
    df = df[(df["filing_gap_days"] >= 0) & (df["filing_gap_days"] <= MAX_FILING_GAP_DAYS)].copy()
    df["trade_date_d"] = df["trade_date"].dt.date  # type: ignore[union-attr]

    df["n_insiders_in_cluster"] = (
        df.groupby(["ticker", "trade_date_d"])["ticker"]  # type: ignore[union-attr]
        .transform("count")
        .astype(int)
    )
    df = df.sort_values(["ticker", "trade_date_d", "transaction_date"]).reset_index(drop=True)  # type: ignore[call-overload]

    def pick_representative(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) == 1:
            return group.iloc[[0]]
        return group.iloc[[1]]

    rep = pd.DataFrame(
        df.groupby(["ticker", "trade_date_d"], group_keys=False, sort=False)
        .apply(pick_representative)
        .reset_index(drop=True)
    )

    rep["n_insiders"] = rep["n_insiders_in_cluster"].astype(int)
    rep["cluster_buy"] = rep["n_insiders_in_cluster"].astype(int) >= 2
    rep["n_insiders_label"] = rep["n_insiders_in_cluster"].astype(int).clip(upper=5).map(
        {1: "1 (solo)", 2: "2", 3: "3", 4: "4", 5: "5+"}  # type: ignore[arg-type]
    )
    rep["event_key"] = make_event_key_series(rep["ticker"], rep["trade_date_d"].astype(str))  # type: ignore[arg-type]
    rep["representative_transaction_date"] = rep["transaction_date"].dt.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore[union-attr]
    rep["trade_date"] = rep["trade_date"].dt.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore[union-attr]
    return rep.reset_index(drop=True)


def new_event_keys_from_rows(new_raw_rows: pd.DataFrame) -> set[str]:
    if new_raw_rows.empty:
        return set()
    keys = make_event_key_series(new_raw_rows["ticker"], new_raw_rows["trade_date"])  # type: ignore[arg-type]
    return {k for k in keys.tolist() if k and not k.endswith("|")}


def load_latest_scored_state(predictions_path: Path) -> pd.DataFrame:
    _state_cols = pd.Index(["event_key", "representative_transaction_date"])
    if not predictions_path.exists():
        return pd.DataFrame(columns=_state_cols)
    df = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
    if df.empty or "event_key" not in df.columns:
        return pd.DataFrame(columns=_state_cols)
    for col in ("scored_at", "representative_transaction_date"):
        if col not in df.columns:
            df[col] = ""
    df["_score_sort"] = pd.to_datetime(df["scored_at"], errors="coerce")
    df = df.sort_values(["event_key", "_score_sort"]).drop_duplicates(subset=["event_key"], keep="last")
    return df[["event_key", "representative_transaction_date"]].reset_index(drop=True)  # type: ignore[return-value]


def select_pending_events(
    candidates: pd.DataFrame,
    impacted_event_keys: set[str],
    latest_state: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty or not impacted_event_keys:
        return pd.DataFrame(columns=candidates.columns)
    pending = candidates[candidates["event_key"].isin(list(impacted_event_keys))].copy()
    if pending.empty:
        return pending  # type: ignore[return-value]

    if latest_state.empty:
        return pending  # type: ignore[return-value]

    latest = latest_state.rename(
        columns={"representative_transaction_date": "last_scored_representative_transaction_date"}
    )
    pending = pending.merge(latest, on="event_key", how="left")
    needs_score = (
        pending["last_scored_representative_transaction_date"].isna()
        | (
            pending["last_scored_representative_transaction_date"]
            != pending["representative_transaction_date"]
        )
    )
    pending = pending.loc[needs_score].copy()
    return pending.drop(columns=["last_scored_representative_transaction_date"], errors="ignore")


def select_preopen_refresh_events(
    candidates: pd.DataFrame,
    now_et: datetime,
) -> pd.DataFrame:
    if candidates.empty or not is_preopen_refresh_window(now_et):
        return pd.DataFrame(columns=candidates.columns)

    target_open = now_et.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    pending = candidates.copy()
    pending["_representative_ts"] = pd.to_datetime(pending["representative_transaction_date"], errors="coerce")
    pending = pending.loc[pending["_representative_ts"].notna()].copy()
    pending["_intended_entry_at"] = pending["_representative_ts"].apply(compute_buy_datetime)
    pending = pending.loc[pending["_intended_entry_at"] == target_open].copy()
    pending = pending.drop(columns=["_representative_ts", "_intended_entry_at"], errors="ignore")
    return pending.reset_index(drop=True)


def _json_load(path: Path) -> List[dict] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _json_save(path: Path, payload: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _aggs_to_dicts(aggs) -> List[dict]:
    return [
        {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
        for a in aggs
        if getattr(a, "timestamp", None) is not None and getattr(a, "close", None) is not None
    ]


def minute_cache_path(cache_dir: Path, ticker: str, d: date) -> Path:
    ds = d.strftime("%Y-%m-%d")
    return cache_dir / f"{ticker}_min_{ds}_{ds}.json"


def lookback_cache_path(cache_dir: Path, ticker: str, from_d: date, to_d: date) -> Path:
    return cache_dir / f"{ticker}_lkbk_{from_d:%Y-%m-%d}_{to_d:%Y-%m-%d}.json"


def fetch_minute_bars(client: RESTClient, cache_dir: Path, ticker: str, target_date: date) -> List[dict]:
    path = minute_cache_path(cache_dir, ticker, target_date)
    cached = _json_load(path)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=target_date,
            to=target_date,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
        bars = _aggs_to_dicts(aggs)
    except Exception:
        bars = []
    _json_save(path, bars)
    return bars


def ensure_lookback_cache(
    client: RESTClient,
    cache_dir: Path,
    ticker: str,
    from_d: date,
    to_d: date,
) -> None:
    path = lookback_cache_path(cache_dir, ticker, from_d, to_d)
    if path.exists():
        return
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=from_d,
            to=to_d,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
        bars = _aggs_to_dicts(aggs)
    except Exception:
        bars = []
    _json_save(path, bars)


def find_price_at_or_after(bars: Iterable[dict], target_ts_ms: int) -> float | None:
    for bar in bars:
        if bar.get("t", -1) >= target_ts_ms:
            c = bar.get("c")
            return float(c) if c is not None else None
    return None


def find_last_close(bars: Iterable[dict]) -> float | None:
    last_close = None
    for bar in bars:
        close = bar.get("c")
        if close is not None:
            last_close = float(close)
    return last_close


def fetch_latest_available_close(
    client: RESTClient,
    cache_dir: Path,
    ticker: str,
    as_of_date: date,
) -> float | None:
    today_et = datetime.now(ET).date()
    to_d = min(as_of_date, today_et)
    from_d = to_d - timedelta(days=30)
    path = lookback_cache_path(cache_dir, ticker, from_d, to_d)

    bars = _json_load(path)
    # Empty caches for the current/future boundary can be stale; refresh once.
    if bars is not None and len(bars) == 0 and to_d >= today_et:
        bars = None

    if bars is None:
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=from_d,
                to=to_d,
                adjusted=True,
                sort="asc",
                limit=50000,
            )
            bars = _aggs_to_dicts(aggs)
        except Exception:
            bars = []
        _json_save(path, bars)

    return find_last_close(bars)


def compute_buy_datetime(filing_dt: pd.Timestamp) -> datetime:
    """
    Two-track entry timing based on research findings:

    - Intraday filing (9:30–16:00 ET, weekday): buy immediately — return the
      filing timestamp itself so the live loop executes within the current
      minute bar.  The market is live and every minute of delay costs ~0.29 pp.

    - Pre-market / after-hours / weekend filing: buy at 9:30 AM on the next
      trading day (market open).  The 9:30 vs 9:45 gap is worth ~0.19 pp but
      sub-minute precision adds no edge once the opening bar has been chosen.
    """
    filing = filing_dt.to_pydatetime()
    if filing.tzinfo is None:
        filing_et = filing.replace(tzinfo=ET)
    else:
        filing_et = filing.astimezone(ET)

    def next_weekday(dt: datetime) -> datetime:
        out = dt
        while out.weekday() >= 5:
            out += timedelta(days=1)
        return out

    open_time  = filing_et.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    close_time = filing_et.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)

    # Weekend → next Monday open
    if filing_et.weekday() >= 5:
        return next_weekday(filing_et).replace(
            hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0
        )
    # Pre-market → today's open
    if filing_et < open_time:
        return open_time
    # After-hours → next trading day's open
    if filing_et >= close_time:
        return next_weekday(filing_et + timedelta(days=1)).replace(
            hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0
        )
    # Intraday → immediate (filing time; loop fires within ≤1 minute)
    return filing_et


def enrich_pending_with_market_data(
    pending_df: pd.DataFrame,
    api_key: str,
    cache_dir: Path,
    logger: logging.Logger,
    *,
    alpaca_api_key: str = "",
    alpaca_api_secret: str = "",
    alpaca_supplement_enabled: bool = True,
) -> pd.DataFrame:
    if pending_df.empty:
        return pending_df

    cache_dir.mkdir(parents=True, exist_ok=True)
    client = RESTClient(api_key=api_key, retries=3)
    _alpaca_client: AlpacaMarketDataClient | None = None
    if alpaca_supplement_enabled and alpaca_api_key and alpaca_api_secret:
        from live_trading.strategy_settings import ALPACA_CONFIG
        _alpaca_client = AlpacaMarketDataClient(
            api_key=alpaca_api_key,
            api_secret=alpaca_api_secret,
            data_feed=ALPACA_CONFIG.data_feed,
            rate_limit_per_minute=ALPACA_CONFIG.data_rate_limit_per_minute,
        )
    out = pending_df.copy()

    buy_dt_col = []
    buy_px_col = []
    prev_close_col = []
    step_up_col = []

    minute_cache: Dict[Tuple[str, str], List[dict]] = {}
    lookback_seen: set[Tuple[str, str, str]] = set()
    prev_close_cache: Dict[Tuple[str, str], float | None] = {}
    fallback_same_day_last = 0
    fallback_latest_daily = 0
    unresolved_buy_price = 0

    # ── Parallel prefetch: fire all uncached API calls before the row loop ──
    _tls = threading.local()

    def _tls_client() -> RESTClient:
        if not hasattr(_tls, "client"):
            _tls.client = RESTClient(api_key=api_key, retries=3)
        return _tls.client

    def _pfetch_minutes(ticker: str, bd: date) -> Tuple[str, str, List[dict]]:
        return ticker, bd.strftime("%Y-%m-%d"), fetch_minute_bars(_tls_client(), cache_dir, ticker, bd)

    def _pfetch_lookback(ticker: str, from_d: date, to_d: date) -> None:
        ensure_lookback_cache(_tls_client(), cache_dir, ticker, from_d, to_d)

    today_et_d = datetime.now(ET).date()
    min_tasks: Dict[Tuple[str, str], date] = {}
    lk_tasks: Dict[Tuple[str, str, str], Tuple[date, date]] = {}
    fb_tasks: Dict[Tuple[str, str, str], Tuple[date, date]] = {}

    for _, row in out.iterrows():
        ticker_p = str(row["ticker"])
        txn_ts_p = pd.to_datetime(row["transaction_date"], errors="coerce")
        trade_ts_p = pd.to_datetime(row["trade_date"], errors="coerce")
        if bool(pd.isna(txn_ts_p)) or bool(pd.isna(trade_ts_p)):
            continue
        buy_dt_p = compute_buy_datetime(txn_ts_p)  # type: ignore[arg-type]
        buy_date_p = buy_dt_p.date()

        mkey = (ticker_p, buy_date_p.strftime("%Y-%m-%d"))
        if mkey not in min_tasks and not minute_cache_path(cache_dir, ticker_p, buy_date_p).exists():
            min_tasks[mkey] = buy_date_p

        td_p = trade_ts_p.date()  # type: ignore[union-attr]
        from_d_p = td_p - timedelta(days=train_models.FETCH_WINDOW)
        lkey = (ticker_p, from_d_p.strftime("%Y-%m-%d"), td_p.strftime("%Y-%m-%d"))
        if lkey not in lk_tasks and not lookback_cache_path(cache_dir, ticker_p, from_d_p, td_p).exists():
            lk_tasks[lkey] = (from_d_p, td_p)

        fb_to = min(buy_date_p, today_et_d)
        fb_from = fb_to - timedelta(days=30)
        fbkey = (ticker_p, fb_from.strftime("%Y-%m-%d"), fb_to.strftime("%Y-%m-%d"))
        if fbkey not in fb_tasks and not lookback_cache_path(cache_dir, ticker_p, fb_from, fb_to).exists():
            fb_tasks[fbkey] = (fb_from, fb_to)

    # Collect unique tickers needing Alpaca latest-price (today's signals only)
    alpaca_tickers: set[str] = set()
    if _alpaca_client is not None:
        for _, row in out.iterrows():
            ticker_a = str(row["ticker"])
            txn_ts_a = pd.to_datetime(row["transaction_date"], errors="coerce")
            if bool(pd.isna(txn_ts_a)):
                continue
            buy_dt_a = compute_buy_datetime(txn_ts_a)  # type: ignore[arg-type]
            if buy_dt_a.date() == today_et_d:
                alpaca_tickers.add(ticker_a)

    n_prefetch = len(min_tasks) + len(lk_tasks) + len(fb_tasks)
    n_total_tasks = n_prefetch + len(alpaca_tickers)
    alpaca_futs: dict[str, object] = {}
    min_futs: Dict[Tuple[str, str], object] = {}
    if n_total_tasks > 0:
        logger.info(
            "Prefetching %d uncached Polygon tasks + %d Alpaca quotes in parallel ...",
            n_prefetch, len(alpaca_tickers),
        )
        with ThreadPoolExecutor(max_workers=min(16, max(1, n_total_tasks))) as pool:
            for mkey, bd in min_tasks.items():
                min_futs[mkey] = pool.submit(_pfetch_minutes, mkey[0], bd)
            for lkey, (from_d_l, to_d_l) in lk_tasks.items():
                pool.submit(_pfetch_lookback, lkey[0], from_d_l, to_d_l)
            for fbkey, (fb_from, fb_to) in fb_tasks.items():
                pool.submit(_pfetch_lookback, fbkey[0], fb_from, fb_to)
            if _alpaca_client is not None:
                for aticker in alpaca_tickers:
                    alpaca_futs[aticker] = pool.submit(_alpaca_client.get_latest_price, aticker)
        for mkey, fut in min_futs.items():  # type: ignore[assignment]
            try:
                _, _ds, bars = fut.result()  # type: ignore[union-attr]
                minute_cache[mkey] = bars
            except Exception as exc:
                logger.warning("Prefetch minute bars %s: %s", mkey, exc)
    alpaca_prices: dict[str, float] = {}
    for aticker, afut in alpaca_futs.items():  # type: ignore[assignment]
        try:
            price = afut.result()  # type: ignore[union-attr]
            if price is not None:
                alpaca_prices[aticker] = float(price)
        except Exception as exc:
            logger.debug("Alpaca price fetch %s: %s", aticker, exc)
    if alpaca_prices:
        logger.info(
            "Alpaca real-time prices fetched for %d tickers: %s",
            len(alpaca_prices), sorted(alpaca_prices),
        )
    # ── End parallel prefetch ──

    total_rows = len(out)
    for pos, (_, row) in enumerate(out.iterrows(), start=1):
        ticker = str(row["ticker"])
        txn_ts = pd.to_datetime(row["transaction_date"], errors="coerce")
        trade_ts = pd.to_datetime(row["trade_date"], errors="coerce")
        if bool(pd.isna(txn_ts)) or bool(pd.isna(trade_ts)):
            buy_dt_col.append("")
            buy_px_col.append(np.nan)
            prev_close_col.append(np.nan)
            step_up_col.append(np.nan)
            continue

        buy_dt = compute_buy_datetime(txn_ts)  # type: ignore[arg-type]
        buy_date = buy_dt.date()
        minute_key = (ticker, buy_date.strftime("%Y-%m-%d"))
        if minute_key not in minute_cache:
            minute_cache[minute_key] = fetch_minute_bars(client, cache_dir, ticker, buy_date)

        buy_price = find_price_at_or_after(minute_cache[minute_key], int(buy_dt.timestamp() * 1000))
        if buy_price is None:
            buy_price = find_last_close(minute_cache[minute_key])
            if buy_price is not None:
                fallback_same_day_last += 1
        if buy_price is None:
            buy_price = fetch_latest_available_close(client, cache_dir, ticker, buy_date)
            if buy_price is not None:
                fallback_latest_daily += 1
        if buy_price is None:
            unresolved_buy_price += 1

        # Override with real-time Alpaca price for today's signals (avoids 15-min Polygon delay)
        if buy_date == today_et_d and ticker in alpaca_prices:
            buy_price = alpaca_prices[ticker]

        buy_dt_col.append(buy_dt.strftime("%Y-%m-%d %H:%M:%S"))
        buy_px_col.append(float(buy_price) if buy_price is not None else np.nan)
        prev_close_key = (ticker, (buy_date - timedelta(days=1)).strftime("%Y-%m-%d"))
        if prev_close_key not in prev_close_cache:
            prev_close_cache[prev_close_key] = fetch_latest_available_close(
                client,
                cache_dir,
                ticker,
                buy_date - timedelta(days=1),
            )
        prev_close = prev_close_cache[prev_close_key]
        prev_close_col.append(float(prev_close) if prev_close is not None else np.nan)
        if (
            buy_price is not None
            and prev_close is not None
            and np.isfinite(float(buy_price))
            and np.isfinite(float(prev_close))
            and float(buy_price) > 0
            and float(prev_close) > 0
        ):
            step_up_col.append(((float(buy_price) / float(prev_close)) - 1.0) * 100.0)
        else:
            step_up_col.append(np.nan)

        td = trade_ts.date()  # type: ignore[union-attr]
        from_d = td - timedelta(days=train_models.FETCH_WINDOW)
        lk = (ticker, from_d.strftime("%Y-%m-%d"), td.strftime("%Y-%m-%d"))
        if lk not in lookback_seen:
            ensure_lookback_cache(client, cache_dir, ticker, from_d, td)
            lookback_seen.add(lk)

        if pos % 25 == 0 or pos == total_rows:
            logger.info("Market data enrichment progress: %d / %d", pos, total_rows)

    logger.info(
        "Buy-price fallback usage: same_day_last=%d, latest_daily=%d, unresolved=%d",
        fallback_same_day_last,
        fallback_latest_daily,
        unresolved_buy_price,
    )

    out["buy_datetime"] = buy_dt_col
    out["buy_price"] = buy_px_col
    out["prev_regular_close"] = prev_close_col
    out["step_up_from_prev_close_pct"] = step_up_col
    out["trade_date_d"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def build_pending_aggregate_rows(pending_enriched: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    base = pd.DataFrame(index=np.arange(len(pending_enriched)), columns=pd.Index(list(columns)))
    for col in pending_enriched.columns:
        if col in base.columns:
            base[col] = pending_enriched[col].values
    if "n_insiders" in base.columns and "n_insiders_in_cluster" in pending_enriched.columns:
        base["n_insiders"] = pending_enriched["n_insiders_in_cluster"].values
    return base


def prepare_temp_aggregate_file(
    aggregated_path: Path,
    pending_enriched: pd.DataFrame,
    temp_path: Path,
) -> None:
    hist = read_csv_or_empty(aggregated_path)
    if hist.empty:
        hist = pd.DataFrame(columns=pd.Index(EXPECTED_AGG_COLUMNS))

    cols = list(dict.fromkeys(list(hist.columns) + EXPECTED_AGG_COLUMNS))
    hist = hist.reindex(columns=cols)

    # Limit history to last 400 days so load_and_merge iterates far fewer price-cache files
    if not hist.empty and "trade_date" in hist.columns:
        cutoff_dt = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        hist = hist[pd.to_datetime(hist["trade_date"], errors="coerce") >= cutoff_dt].copy()

    pending_rows = build_pending_aggregate_rows(pending_enriched, cols).reindex(columns=cols)

    combined = pd.concat([hist, pending_rows], ignore_index=True)
    combined["_event_key"] = make_event_key_series(combined["ticker"], combined["trade_date"])  # type: ignore[arg-type]
    combined["_txn_sort"] = pd.to_datetime(combined["transaction_date"], errors="coerce")
    combined = (
        combined.sort_values(["_event_key", "_txn_sort"])  # type: ignore[call-overload]
        .drop_duplicates(subset=["_event_key"], keep="last")
        .drop(columns=["_event_key", "_txn_sort"])
        .reset_index(drop=True)
    )

    temp_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(temp_path, index=False)


def compute_features_for_pending(
    temp_agg_path: Path,
    pending_event_keys: Sequence[str],
    raw_file: Path,
    cache_dir: Path,
    sector_cache_file: Path,
) -> pd.DataFrame:
    pending_set = set(pending_event_keys)
    if not pending_set:
        return pd.DataFrame()

    original_agg_path = train_models.AGGREGATED_CSV
    original_raw_path = train_models.ORIGINAL_CSV
    original_cache_dir = train_models.CACHE_DIR
    original_sector_cache = train_models.SECTOR_CACHE
    try:
        train_models.AGGREGATED_CSV = str(temp_agg_path)
        train_models.ORIGINAL_CSV = str(raw_file)
        train_models.CACHE_DIR = str(cache_dir)
        train_models.SECTOR_CACHE = str(sector_cache_file)
        merged = train_models.load_and_merge()
        feat_df, _, _ = train_models.engineer_features(merged)
    finally:
        train_models.AGGREGATED_CSV = original_agg_path
        train_models.ORIGINAL_CSV = original_raw_path
        train_models.CACHE_DIR = original_cache_dir
        train_models.SECTOR_CACHE = original_sector_cache

    feat_df["event_key"] = make_event_key_series(feat_df["ticker"], feat_df["trade_date"])
    feat_df["representative_transaction_date"] = pd.to_datetime(
        feat_df["transaction_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    out = feat_df[feat_df["event_key"].isin(pending_set)].copy()
    return out.reset_index(drop=True)


def to_xgb(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    if "officer_type_enc" in Xc.columns:
        Xc["officer_type_enc"] = Xc["officer_type_enc"].astype("category")
    if "market_type_enc" in Xc.columns:
        Xc["market_type_enc"] = Xc["market_type_enc"].astype("category")
    if "sector_enc" in Xc.columns:
        Xc["sector_enc"] = Xc["sector_enc"].astype("category")
    return Xc


def to_linear_numeric(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for col in Xn.columns:
        if isinstance(Xn[col].dtype, pd.CategoricalDtype):
            codes = Xn[col].cat.codes.astype(float)
            codes[codes < 0] = np.nan
            Xn[col] = codes
    return Xn.astype(float)


def align_features_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Keep live scoring backward-compatible while feature sets evolve.
    If a persisted model carries feature_names_in_, predict with that exact order.
    """
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        return X
    cols = [str(c) for c in cols]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Model expects missing feature columns: {missing}")
    return X[cols].copy()  # type: ignore[return-value]


def predict_model(model_name: str, model, X: pd.DataFrame) -> np.ndarray:
    X_use = align_features_to_model(model, X)
    if model_name == "HGBR":
        return model.predict(X_use)
    if model_name == "XGBoost":
        return model.predict(to_xgb(X_use))
    if model_name == "ElasticNet":
        return model.predict(to_linear_numeric(X_use))
    if model_name == "SplineElasticNet":
        return model.predict(X_use.copy())
    raise ValueError(f"Unknown model name: {model_name}")


def expected_daily_from_horizon_pct(
    pred_pct: np.ndarray,
    horizon_days: int,
    multi_day_targets_are_daily: bool,
) -> np.ndarray:
    """
    Convert horizon return prediction (%) into expected daily return (%).
    If targets are already daily for multi-day horizons, return unchanged.
    """
    vals = np.asarray(pred_pct, dtype=float)
    if horizon_days <= 1:
        return vals
    if multi_day_targets_are_daily:
        return vals

    gross = 1.0 + (vals / 100.0)
    out = np.full_like(vals, np.nan, dtype=float)
    valid = gross > 0.0
    out[valid] = (np.power(gross[valid], 1.0 / float(horizon_days)) - 1.0) * 100.0
    return out


def load_models_and_policy(model_dir: Path) -> Tuple[Dict[int, Dict[str, object]], dict]:
    models: Dict[int, Dict[str, object]] = {}
    for h in HORIZONS:
        models[h] = {}
        for model_name in MODEL_NAMES:
            path = model_dir / f"model_{h}d_{model_name}.pkl"
            models[h][model_name] = joblib.load(path)

    policy_path = model_dir / "ensemble_policy.json"
    with policy_path.open("r", encoding="utf-8") as f:
        policy = json.load(f)
    return models, policy


def score_features(
    feat_df: pd.DataFrame,
    models_by_horizon: Dict[int, Dict[str, object]],
    policy: dict,
) -> pd.DataFrame:
    if feat_df.empty:
        return pd.DataFrame()

    if "is_tradable" in feat_df.columns:
        feat_df = feat_df[feat_df["is_tradable"].astype(int) == 1].copy()  # type: ignore[assignment]
        if feat_df.empty:
            return pd.DataFrame()

    missing = [f for f in train_models.FEATURES if f not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing features required for scoring: {missing}")

    X: pd.DataFrame = feat_df[train_models.FEATURES].copy()  # type: ignore[assignment]
    if "market_type" not in feat_df.columns:
        feat_df["market_type"] = "UNKNOWN"
    if "is_tradable" not in feat_df.columns:
        feat_df["is_tradable"] = 1

    meta = feat_df[
        [
            "event_key",
            "representative_transaction_date",
            "transaction_date",
            "trade_date",
            "ticker",
            "company_name",
            "owner_name",
            "title",
            "market_type",
            "is_tradable",
            "buy_datetime",
            "buy_price",
            "filing_gap_days",
            "n_insiders_in_cluster",
            "days_since_last_buy_any",
            "days_since_last_buy_same_ticker",
            "same_ticker_insider_count_365d",
            "filing_hour_et",
        ]
    ].copy()
    meta["market_type"] = meta["market_type"].fillna("UNKNOWN").astype(str)  # type: ignore[union-attr]
    meta["is_tradable"] = pd.to_numeric(meta["is_tradable"], errors="coerce").fillna(0).astype(int)  # type: ignore[union-attr]
    meta["trade_date"] = pd.to_datetime(meta["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")  # type: ignore[union-attr]
    multi_day_targets_are_daily = bool(policy.get("multi_day_targets_are_daily", False))

    scored_frames = []
    for h in HORIZONS:
        preds = {}
        for model_name, model in models_by_horizon[h].items():
            preds[model_name] = predict_model(model_name, model, X)

        pred_matrix = np.column_stack([preds[m] for m in MODEL_NAMES])
        mean4 = pred_matrix.mean(axis=1)

        w = policy.get("weights_by_horizon", {}).get(str(h), {}).get("weights", {})
        if not w:
            w = {m: 1.0 / len(MODEL_NAMES) for m in MODEL_NAMES}
        weighted: np.ndarray = sum(preds[m] * float(w.get(m, 0.0)) for m in MODEL_NAMES)  # type: ignore[assignment]
        expected_daily = expected_daily_from_horizon_pct(
            pred_pct=weighted,
            horizon_days=h,
            multi_day_targets_are_daily=multi_day_targets_are_daily,
        )

        out = meta.copy()
        out["horizon_days"] = h
        out["pred_HGBR"] = preds["HGBR"]
        out["pred_XGBoost"] = preds["XGBoost"]
        out["pred_ElasticNet"] = preds["ElasticNet"]
        out["pred_SplineElasticNet"] = preds["SplineElasticNet"]
        out["pred_mean4"] = mean4
        out["pred_weighted_policy"] = weighted
        out["expected_daily_return_pct"] = expected_daily
        scored_frames.append(out)

    scored = pd.concat(scored_frames, ignore_index=True)
    scored["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return scored


def upsert_predictions(predictions_path: Path, scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return read_csv_or_empty(predictions_path)

    old = read_csv_or_empty(predictions_path)
    if not old.empty and "event_key" in old.columns:
        keys = list(scored_df["event_key"].astype(str).tolist())
        old = old[~old["event_key"].astype(str).isin(keys)].copy()
    combined = pd.concat([old, scored_df], ignore_index=True)
    combined = combined.sort_values(["event_key", "horizon_days", "scored_at"], na_position="last")  # type: ignore[call-overload]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(predictions_path, index=False)
    return combined


def print_report(scored_df: pd.DataFrame) -> None:
    if scored_df.empty:
        print("\nNo pending events scored in this cycle.\n")
        return

    scored_at = scored_df["scored_at"].iloc[0]
    n_events = scored_df["event_key"].nunique()
    print("\n" + "=" * 120)
    print(f"LIVE SCORING REPORT | scored_at={scored_at} | events={n_events} | rows={len(scored_df)}")
    print("=" * 120)

    for _, grp in scored_df.sort_values(["event_key", "horizon_days"]).groupby("event_key", sort=True):
        head = grp.iloc[0]
        insiders = head["n_insiders_in_cluster"]
        buy_price = _to_float(head["buy_price"])
        buy_price_txt = "nan" if np.isnan(buy_price) else f"{buy_price:.4f}"
        print(
            f"{head['ticker']} | trade_date={head['trade_date']} | filed={head['representative_transaction_date']} "
            f"| owner={head['owner_name']} | market={head.get('market_type', 'UNKNOWN')} "
            f"| n_insiders={insiders} | buy_price={buy_price_txt}"
        )
        for _, row in grp.sort_values("horizon_days").iterrows():
            print(
                f"  {int(row['horizon_days']):>2}d  "
                f"HGBR={row['pred_HGBR']:+7.3f}%  "
                f"XGB={row['pred_XGBoost']:+7.3f}%  "
                f"EN={row['pred_ElasticNet']:+7.3f}%  "
                f"SPL={row['pred_SplineElasticNet']:+7.3f}%  "
                f"mean4={row['pred_mean4']:+7.3f}%  "
                f"weighted={row['pred_weighted_policy']:+7.3f}%  "
                f"daily={row['expected_daily_return_pct']:+7.3f}%"
            )
        print("-" * 120)
    print()


def load_day1_pred_mean_threshold(
    decile_score_threshold: float,
    cutoff_csv: Path,
    benchmark_csv: Path,
    logger: logging.Logger,
) -> Tuple[float, str]:
    score = float(np.clip(decile_score_threshold, 0.0, 1.0))

    if cutoff_csv.exists():
        try:
            sweep = pd.read_csv(cutoff_csv)
            sweep["decile_score_threshold"] = pd.to_numeric(
                sweep.get("decile_score_threshold"), errors="coerce"
            )
            sweep["raw_pred_mean4_cutoff"] = pd.to_numeric(
                sweep.get("raw_pred_mean4_cutoff"), errors="coerce"
            )
            sweep = sweep.dropna(subset=["decile_score_threshold", "raw_pred_mean4_cutoff"]).copy()
            if not sweep.empty:
                sweep = sweep.sort_values("decile_score_threshold")
                xs = sweep["decile_score_threshold"].to_numpy(dtype=float)
                ys = sweep["raw_pred_mean4_cutoff"].to_numpy(dtype=float)
                raw_cut = float(np.interp(score, xs, ys))
                src = f"{cutoff_csv} (decile_score={score:.2f} -> raw_cut={raw_cut:.3f})"
                return raw_cut, src
        except Exception as exc:
            logger.warning("Failed loading decile cutoff map from %s: %s", cutoff_csv, exc)

    if benchmark_csv.exists():
        try:
            dec = pd.read_csv(benchmark_csv)
            dec["horizon_days"] = pd.to_numeric(dec.get("horizon_days"), errors="coerce")
            dec["decile"] = pd.to_numeric(dec.get("decile"), errors="coerce")
            dec["mean_pred"] = pd.to_numeric(dec.get("mean_pred"), errors="coerce")
            day1 = dec[dec["horizon_days"] == 1].copy()
            d9 = float(day1.loc[day1["decile"] == 9, "mean_pred"].dropna().iloc[0])
            source = f"{benchmark_csv} (legacy D9 mean_pred={d9:.3f})"
            return d9, source
        except Exception as exc:
            logger.warning("Failed loading day-1 benchmark from %s: %s", benchmark_csv, exc)

    logger.warning(
        "Using fallback day-1 raw threshold %.6f (decile_score=%.2f).",
        DEFAULT_DAY1_RAW_THRESHOLD,
        score,
    )
    return DEFAULT_DAY1_RAW_THRESHOLD, "fallback constant"


def load_day1_decile_curve(
    cutoff_csv: Path,
    benchmark_csv: Path,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if cutoff_csv.exists():
        try:
            sweep = pd.read_csv(cutoff_csv)
            sweep["decile_score_threshold"] = pd.to_numeric(
                sweep.get("decile_score_threshold"), errors="coerce"
            )
            sweep["raw_pred_mean4_cutoff"] = pd.to_numeric(
                sweep.get("raw_pred_mean4_cutoff"), errors="coerce"
            )
            sweep = sweep.dropna(subset=["decile_score_threshold", "raw_pred_mean4_cutoff"]).copy()
            if not sweep.empty:
                sweep = sweep.sort_values("decile_score_threshold")
                xs = sweep["decile_score_threshold"].to_numpy(dtype=float)
                ys = sweep["raw_pred_mean4_cutoff"].to_numpy(dtype=float)
                if len(xs) >= 2:
                    return xs, ys, str(cutoff_csv)
        except Exception as exc:
            logger.warning("Failed loading decile curve from %s: %s", cutoff_csv, exc)

    if benchmark_csv.exists():
        try:
            dec = pd.read_csv(benchmark_csv)
            dec["horizon_days"] = pd.to_numeric(dec.get("horizon_days"), errors="coerce")
            dec["decile"] = pd.to_numeric(dec.get("decile"), errors="coerce")
            dec["mean_pred"] = pd.to_numeric(dec.get("mean_pred"), errors="coerce")
            day1 = dec[dec["horizon_days"] == 1].dropna(subset=["decile", "mean_pred"]).copy()  # type: ignore[call-overload]
            day1 = day1[(day1["decile"] >= 1) & (day1["decile"] <= 10)].sort_values("decile")  # type: ignore[call-overload]
            if len(day1) >= 2:
                xs = (day1["decile"].to_numpy(dtype=float) / 10.0).astype(float)
                ys = day1["mean_pred"].to_numpy(dtype=float)
                return xs, ys, str(benchmark_csv)
        except Exception as exc:
            logger.warning("Failed loading benchmark decile curve from %s: %s", benchmark_csv, exc)

    xs = np.asarray([0.0, 1.0], dtype=float)
    ys = np.asarray([DEFAULT_DAY1_RAW_THRESHOLD - 1.0, DEFAULT_DAY1_RAW_THRESHOLD + 1.0], dtype=float)
    return xs, ys, "fallback synthetic curve"


def estimate_decile_score_from_raw_pred(
    raw_pred_values: np.ndarray,
    decile_scores: np.ndarray,
    raw_cutoffs: np.ndarray,
) -> np.ndarray:
    curve = pd.DataFrame({"decile_score": decile_scores, "raw_cutoff": raw_cutoffs}).dropna().copy()
    if curve.empty:
        return np.full_like(raw_pred_values, np.nan, dtype=float)

    curve = curve.sort_values("raw_cutoff")
    curve["raw_cutoff"] = np.maximum.accumulate(curve["raw_cutoff"].to_numpy(dtype=float))
    curve = curve.drop_duplicates(subset=["raw_cutoff"], keep="last")
    if len(curve) < 2:
        return np.full_like(raw_pred_values, float(curve["decile_score"].iloc[0]), dtype=float)

    x_raw = curve["raw_cutoff"].to_numpy(dtype=float)
    y_dec = curve["decile_score"].to_numpy(dtype=float)
    return np.interp(raw_pred_values, x_raw, y_dec, left=float(y_dec.min()), right=float(y_dec.max()))


def apply_linear_allocation_advice(
    picks_df: pd.DataFrame,
    pred_col: str,
    decile_score_threshold: float,
    decile_scores: np.ndarray,
    raw_cutoffs: np.ndarray,
    base_alloc_fraction: float,
    bonus_fraction: float,
) -> pd.DataFrame:
    if picks_df.empty:
        return picks_df.copy()

    out = picks_df.copy()
    pred_series = pd.to_numeric(out[pred_col], errors="coerce") if pred_col in out.columns else pd.Series(np.nan, index=out.index)
    p = np.asarray(pred_series, dtype=float)
    est_dec = estimate_decile_score_from_raw_pred(p, decile_scores, raw_cutoffs)

    base_alloc = float(max(0.0, base_alloc_fraction))
    bonus = float(max(0.0, bonus_fraction))
    max_alloc = base_alloc + bonus
    score_floor = float(np.clip(decile_score_threshold, 0.0, 1.0))
    denom = max(1e-9, 1.0 - score_floor)
    strength = np.clip((est_dec - score_floor) / denom, 0.0, 1.0)
    advised_alloc = np.clip(base_alloc + bonus * strength, base_alloc, max_alloc)

    out["estimated_decile_score"] = est_dec
    out["decile_strength"] = strength
    out["advised_allocation_fraction"] = advised_alloc
    out["advised_allocation_pct"] = advised_alloc * 100.0
    return out


def select_day1_pred_mean_candidates(
    scored_df: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, str]:
    if scored_df.empty:
        return pd.DataFrame(), "pred_mean4"

    pred_col = "pred_mean4" if "pred_mean4" in scored_df.columns else "pred_mean"
    if pred_col not in scored_df.columns:
        return pd.DataFrame(), pred_col

    day1 = scored_df.copy()
    day1["horizon_days"] = pd.to_numeric(day1["horizon_days"], errors="coerce")
    day1[pred_col] = pd.to_numeric(day1[pred_col], errors="coerce")
    day1 = day1[day1["horizon_days"] == 1].copy()
    if day1.empty:
        return pd.DataFrame(), pred_col

    picks = pd.DataFrame(day1[day1[pred_col] > float(threshold)].copy())
    if picks.empty:
        return picks, pred_col
    picks = picks.sort_values(pred_col, ascending=False).drop_duplicates(subset=["event_key"], keep="first")
    return picks.reset_index(drop=True), pred_col


def print_day1_investment_findings(
    picks_df: pd.DataFrame,
    threshold: float,
    decile_score_threshold: float,
    pred_col: str,
    threshold_source: str,
) -> None:
    print("\n" + "=" * 120)
    print(
        "DAY-1 INVESTMENT FINDINGS | "
        f"rule={pred_col} > {threshold:.3f}% excess return (decile_score>={decile_score_threshold:.2f}) "
        f"| benchmark_source={threshold_source}"
    )
    print("=" * 120)
    if picks_df.empty:
        print("No day-1 candidates above the configured decile-score threshold.\n")
        return

    for _, row in picks_df.iterrows():
        alloc_frac = _to_float(row.get("advised_allocation_fraction"))
        alloc_txt = "n/a" if np.isnan(alloc_frac) else f"{alloc_frac * 100.0:.1f}%"
        dec_est = _to_float(row.get("estimated_decile_score"))
        dec_txt = "n/a" if np.isnan(dec_est) else f"{dec_est:.3f}"
        found_at = str(row.get("scored_at", ""))
        print(
            f"{row['ticker']} | {row['company_name']} | owner={row['owner_name']} | "
            f"trade_date={row['trade_date']} | {pred_col}={_to_float(row[pred_col]):+.3f}% | "
            f"est_decile={dec_txt} | advised_alloc={alloc_txt} of budget | found_at={found_at}"
        )
    print()


def build_alert_detail_rows(
    scored_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    pred_col: str,
) -> pd.DataFrame:
    if scored_df.empty or picks_df.empty:
        return pd.DataFrame()

    score_col = pred_col if pred_col in scored_df.columns else "pred_mean4"
    if score_col not in scored_df.columns:
        return pd.DataFrame()

    tmp = scored_df.copy()
    tmp["horizon_days"] = pd.to_numeric(tmp["horizon_days"], errors="coerce")
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")

    piv = (
        tmp[tmp["horizon_days"].isin(HORIZONS)]
        .pivot_table(index="event_key", columns="horizon_days", values=score_col, aggfunc="first")
        .rename(columns={1: "score_1d", 3: "score_3d", 5: "score_5d", 10: "score_10d"})
        .reset_index()
    )
    for c in ["score_1d", "score_3d", "score_5d", "score_10d"]:
        if c not in piv.columns:
            piv[c] = np.nan

    cols = ["event_key", "ticker", "company_name", "owner_name", "trade_date", "buy_price"]
    optional_cols = [
        "title",
        "scored_at",
        "pred_mean4",
        "prev_regular_close",
        "step_up_from_prev_close_pct",
        "estimated_decile_score",
        "decile_strength",
        "advised_allocation_fraction",
        "advised_allocation_pct",
        "market_type",
        "is_tradable",
    ]
    cols = cols + [c for c in optional_cols if c in picks_df.columns]
    base = picks_df[cols].drop_duplicates(subset=["event_key"], keep="first").copy()  # type: ignore[call-overload]
    out = base.merge(piv, on="event_key", how="left")
    return out.sort_values("score_1d", ascending=False).reset_index(drop=True)


def build_alert_export_rows(
    scored_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    pred_col: str,
    threshold: float,
    decile_score_threshold: float,
    threshold_source: str,
) -> pd.DataFrame:
    details_df = build_alert_detail_rows(scored_df, picks_df, pred_col)
    if details_df.empty:
        return pd.DataFrame(columns=pd.Index(ALERT_EXPORT_COLUMNS))

    out = details_df.copy()
    out["raw_alert_threshold"] = float(threshold)
    out["decile_score_threshold"] = float(decile_score_threshold)
    out["threshold_source"] = str(threshold_source)
    out["alert_score_column"] = pred_col
    out["target_return_mode"] = train_models.TARGET_RETURN_MODE
    out["benchmark_ticker"] = train_models.BENCHMARK_TICKER
    for col in ALERT_EXPORT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[ALERT_EXPORT_COLUMNS].copy()
    return out.sort_values(["scored_at", "score_1d", "ticker"], ascending=[False, False, True]).reset_index(drop=True)


def update_alert_candidate_exports(
    latest_path: Path,
    history_path: Path,
    export_df: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    latest = export_df.copy() if not export_df.empty else pd.DataFrame(columns=pd.Index(ALERT_EXPORT_COLUMNS))
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(latest_path, index=False)
    logger.info("Alert snapshot updated -> %s (rows=%d)", latest_path, len(latest))

    old_history = read_csv_or_empty(history_path, ALERT_EXPORT_COLUMNS)
    if export_df.empty:
        history = old_history
    else:
        history = pd.concat([old_history, export_df], ignore_index=True)
        history = history.drop_duplicates(subset=["event_key", "scored_at"], keep="last")
        history = history.sort_values(["scored_at", "score_1d", "ticker"], ascending=[False, False, True])  # type: ignore[call-overload]

    history = history.reindex(columns=ALERT_EXPORT_COLUMNS)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    logger.info("Alert history updated -> %s (rows=%d)", history_path, len(history))


def build_exit_policy_html(review_date: str) -> str:
    return f"""
    <p><b>Exit timing policy (reviewed {review_date})</b></p>
    <p>Active live rule: <code>{ACTIVE_STRATEGY.sell_rule_label}</code></p>
    <table border="1" cellpadding="6" cellspacing="0">
      <tr>
        <th>Filing bucket</th><th>Recommended exit</th><th>Reason</th><th>Fast-turnover option</th>
      </tr>
      <tr>
        <td>Pre-market filing</td>
        <td><b>second_day_close</b></td>
        <td>Highest average edge in backtest slice; slower capital turnover.</td>
        <td>same_day_close when cash-constrained</td>
      </tr>
      <tr>
        <td>Intraday filing</td>
        <td><b>next_day_close</b></td>
        <td>Most robust balance of return and stability in the slice.</td>
        <td>Keep next_day_close (alternatives were weaker)</td>
      </tr>
      <tr>
        <td>Post-market/weekend filing</td>
        <td><b>next_day_12_00</b></td>
        <td>Slight return edge vs next_day_close with faster reuse of capital.</td>
        <td>next_day_open for max turnover</td>
      </tr>
    </table>
    """


def send_email(
    scored_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    threshold: float,
    decile_score_threshold: float,
    pred_col: str,
    threshold_source: str,
    advice_base_alloc_fraction: float,
    advice_bonus_fraction: float,
    advice_curve_source: str,
    recipient: str,
    logger: logging.Logger,
) -> bool:
    if picks_df.empty:
        return False

    details_df = build_alert_detail_rows(scored_df, picks_df, pred_col)
    if details_df.empty:
        logger.warning("No alert detail rows available; skipping email send.")
        return False

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    if not smtp_user or not smtp_pass:
        logger.warning("SMTP_USER/SMTP_PASS not set; skipping email send.")
        return False

    now_txt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tickers = ", ".join(details_df["ticker"].astype(str).tolist())
    exit_policy_html = build_exit_policy_html(EXIT_POLICY_REVIEW_DATE)
    subject = (
        f"Insider Alert {datetime.now().strftime('%Y-%m-%d')}: "
        f"{len(details_df)} day-1 candidate(s) above decile threshold"
    )

    rows_html = []
    for _, row in details_df.iterrows():
        buy_px  = _to_float(row.get("buy_price"))
        s1      = _to_float(row.get("score_1d"))
        s3      = _to_float(row.get("score_3d"))
        s5      = _to_float(row.get("score_5d"))
        s10     = _to_float(row.get("score_10d"))
        dec_est = _to_float(row.get("estimated_decile_score"))
        adv_alloc_pct = _to_float(row.get("advised_allocation_pct"))
        found_at      = str(row.get("scored_at", ""))
        buy_px_txt    = "nan" if np.isnan(buy_px)       else f"{buy_px:.4f}"
        s1_txt        = "nan" if np.isnan(s1)           else f"{s1:+.3f}%"
        s3_txt        = "nan" if np.isnan(s3)           else f"{s3:+.3f}%"
        s5_txt        = "nan" if np.isnan(s5)           else f"{s5:+.3f}%"
        s10_txt       = "nan" if np.isnan(s10)          else f"{s10:+.3f}%"
        dec_txt       = "nan" if np.isnan(dec_est)      else f"{dec_est:.3f}"
        adv_alloc_txt = "nan" if np.isnan(adv_alloc_pct) else f"{adv_alloc_pct:.1f}%"
        rows_html.append(
            "<tr>"
            f"<td>{row['ticker']}</td>"
            f"<td>{row['company_name']}</td>"
            f"<td>{row['owner_name']}</td>"
            f"<td>{row['trade_date']}</td>"
            f"<td>{buy_px_txt}</td>"
            f"<td>{s1_txt}</td>"
            f"<td>{s3_txt}</td>"
            f"<td>{s5_txt}</td>"
            f"<td>{s10_txt}</td>"
            f"<td>{dec_txt}</td>"
            f"<td>{adv_alloc_txt}</td>"
            f"<td>{found_at}</td>"
            "</tr>"
        )

    html_body = f"""
    <html>
      <body>
        <p><b>Day-1 investment findings</b></p>
        <p>Run time: {now_txt}</p>
        <p>Rule: {pred_col} &gt; {threshold:.3f}% (decile_score &gt;= {decile_score_threshold:.2f})</p>
        <p>Benchmark source: {threshold_source}</p>
        <p>
          Allocation advice: base {advice_base_alloc_fraction * 100:.1f}% + linear bonus up to
          {advice_bonus_fraction * 100:.1f}% by estimated decile strength.
        </p>
        <p>Allocation decile-curve source: {advice_curve_source}</p>
        <p>Tickers: {tickers}</p>
        {exit_policy_html}
        <table border="1" cellpadding="6" cellspacing="0">
          <tr>
            <th>Ticker</th><th>Company</th><th>Owner</th><th>Trade Date</th><th>Buy Price</th>
            <th>Score 1d</th><th>Score 3d</th><th>Score 5d</th><th>Score 10d</th>
            <th>Est Decile</th><th>Adv Alloc % of Budget</th><th>Found At</th>
          </tr>
          {''.join(rows_html)}
        </table>
      </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipient, msg.as_string())
        server.quit()
        logger.info("Alert email sent to %s for tickers: %s", recipient, tickers)
        return True
    except Exception as exc:
        logger.warning("Failed to send alert email to %s: %s", recipient, exc)
        return False


def run_cycle(
    scraper: OpenInsiderScraper,
    models_by_horizon: Dict[int, Dict[str, object]],
    policy: dict,
    args,
    logger: logging.Logger,
    *,
    now_et: datetime,
    threshold: float,
    threshold_source: str,
    advice_deciles: np.ndarray,
    advice_raw_cutoffs: np.ndarray,
    advice_curve_source: str,
    months_back: int,
) -> int:
    scraped = scrape_recent_filings(scraper, months_back, logger)
    combined_raw, new_raw_rows = merge_scraped_into_raw(Path(args.raw_file), scraped, logger)

    impacted_event_keys = new_event_keys_from_rows(new_raw_rows)
    candidates = build_candidate_events(combined_raw)
    latest_state = load_latest_scored_state(Path(args.predictions_file))
    pending_new = select_pending_events(candidates, impacted_event_keys, latest_state)
    pending_refresh = select_preopen_refresh_events(candidates, now_et)
    if not impacted_event_keys and pending_refresh.empty:
        logger.info("No new filings detected and no queued open-entry candidates need refresh this cycle.")
        return 0

    pending_parts = [frame for frame in (pending_new, pending_refresh) if not frame.empty]
    if pending_parts:
        pending = pd.concat(pending_parts, ignore_index=True)
        pending = pending.drop_duplicates(subset=["event_key"], keep="last").reset_index(drop=True)
    else:
        pending = pd.DataFrame(columns=candidates.columns)
    logger.info(
        "Impacted events=%d, pre-open refresh=%d, pending for scoring=%d",
        len(impacted_event_keys),
        len(pending_refresh),
        len(pending),
    )
    if pending.empty:
        return 0

    api_key = args.polygon_api_key
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not found (.env or environment).")

    pending = enrich_pending_with_market_data(
        pending,
        api_key,
        Path(args.cache_dir),
        logger,
        alpaca_api_key=args.alpaca_api_key,
        alpaca_api_secret=args.alpaca_api_secret,
        alpaca_supplement_enabled=not args.no_alpaca_supplement,
    )

    temp_agg_path = Path(args.temp_aggregate_file)
    prepare_temp_aggregate_file(Path(args.aggregated_file), pending, temp_agg_path)

    feat_pending = compute_features_for_pending(
        temp_agg_path=temp_agg_path,
        pending_event_keys=pending["event_key"].tolist(),
        raw_file=Path(args.raw_file),
        cache_dir=Path(args.cache_dir),
        sector_cache_file=Path(args.sector_cache_file),
    )
    logger.info("Feature rows ready for scoring: %d", len(feat_pending))
    if feat_pending.empty:
        return 0

    scored = score_features(feat_pending, models_by_horizon, policy)
    pending_market_meta = pending[
        [
            "event_key",
            "prev_regular_close",
            "step_up_from_prev_close_pct",
        ]
    ].drop_duplicates(subset=["event_key"], keep="last")
    scored = scored.merge(pending_market_meta, on="event_key", how="left")
    print_report(scored)

    picks_df, pred_col = select_day1_pred_mean_candidates(scored, threshold)
    if not picks_df.empty:
        picks_df = apply_linear_allocation_advice(
            picks_df=picks_df,
            pred_col=pred_col,
            decile_score_threshold=args.day1_decile_score_threshold,
            decile_scores=advice_deciles,
            raw_cutoffs=advice_raw_cutoffs,
            base_alloc_fraction=args.advice_base_alloc_fraction,
            bonus_fraction=args.advice_bonus_fraction,
        )
    print_day1_investment_findings(
        picks_df,
        threshold,
        args.day1_decile_score_threshold,
        pred_col,
        threshold_source,
    )
    alert_export_df = build_alert_export_rows(
        scored_df=scored,
        picks_df=picks_df,
        pred_col=pred_col,
        threshold=threshold,
        decile_score_threshold=args.day1_decile_score_threshold,
        threshold_source=threshold_source,
    )
    update_alert_candidate_exports(
        latest_path=Path(args.alert_snapshot_file),
        history_path=Path(args.alert_history_file),
        export_df=alert_export_df,
        logger=logger,
    )
    if not picks_df.empty and not args.no_email:
        send_email(
            scored_df=scored,
            picks_df=picks_df,
            threshold=threshold,
            decile_score_threshold=args.day1_decile_score_threshold,
            pred_col=pred_col,
            threshold_source=threshold_source,
            advice_base_alloc_fraction=args.advice_base_alloc_fraction,
            advice_bonus_fraction=args.advice_bonus_fraction,
            advice_curve_source=advice_curve_source,
            recipient=args.alert_recipient,
            logger=logger,
        )

    combined_pred = upsert_predictions(Path(args.predictions_file), scored)
    logger.info(
        "Predictions upserted -> %s (total rows=%d)",
        args.predictions_file,
        len(combined_pred),
    )
    return int(scored["event_key"].nunique())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live scoring loop for insider-trade models.")
    p.add_argument("--config", default=LIVE_PATHS.scraper_config, help="Path to scraper config.")
    p.add_argument("--raw-file", default=LIVE_PATHS.raw_file)
    p.add_argument("--aggregated-file", default=LIVE_PATHS.aggregated_file)
    p.add_argument("--predictions-file", default=LIVE_PATHS.predictions_file)
    p.add_argument("--cache-dir", default=LIVE_PATHS.cache_dir)
    p.add_argument("--sector-cache-file", default=LIVE_PATHS.sector_cache_file)
    p.add_argument("--model-dir", default=ACTIVE_STRATEGY.model_dir)
    p.add_argument("--temp-aggregate-file", default=LIVE_PATHS.temp_aggregate_file)
    p.add_argument("--alert-snapshot-file", default=LIVE_PATHS.alert_snapshot_file)
    p.add_argument("--alert-history-file", default=LIVE_PATHS.alert_history_file)
    p.add_argument("--months-back", type=int, default=RUNTIME_DEFAULTS.months_back, help="Current month + N previous months.")
    p.add_argument("--once", action="store_true", help="Run exactly one cycle.")
    p.add_argument("--polygon-api-key", default="", help="Optional override for POLYGON_API_KEY.")
    p.add_argument("--alpaca-api-key", default="", help="Alpaca API key for real-time price supplement.")
    p.add_argument("--alpaca-api-secret", default="", help="Alpaca API secret for real-time price supplement.")
    p.add_argument("--no-alpaca-supplement", action="store_true", help="Disable Alpaca real-time price supplement.")
    p.add_argument(
        "--day1-decile-score-threshold",
        type=float,
        default=DEFAULT_DAY1_DECILE_SCORE_THRESHOLD,
        help="Investable threshold on decile score scale [0,1].",
    )
    p.add_argument(
        "--day1-decile-cutoff-file",
        default=ACTIVE_STRATEGY.day1_decile_cutoff_file,
        help="CSV mapping decile_score_threshold -> raw_pred_mean4_cutoff.",
    )
    p.add_argument("--day1-benchmark-file", default=ACTIVE_STRATEGY.day1_benchmark_file)
    p.add_argument(
        "--advice-base-alloc-fraction",
        type=float,
        default=DEFAULT_ADVICE_BASE_ALLOC_FRACTION,
        help="Base advised allocation fraction for live alerts (e.g. 0.25 = 25%%).",
    )
    p.add_argument(
        "--advice-bonus-fraction",
        type=float,
        default=DEFAULT_ADVICE_BONUS_FRACTION,
        help="Linear bonus allocation fraction added at max decile strength.",
    )
    p.add_argument("--alert-recipient", default=DEFAULT_ALERT_RECIPIENT)
    p.add_argument("--no-email", action="store_true", help="Skip sending SMTP email alerts.")
    return p


def main() -> None:
    load_dotenv()
    logger = setup_logger()
    args = build_arg_parser().parse_args()

    if not args.polygon_api_key:
        args.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
    if not args.alpaca_api_key:
        args.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
    if not args.alpaca_api_secret:
        args.alpaca_api_secret = os.getenv("ALPACA_API_SECRET", "")

    model_dir = Path(args.model_dir)
    models_by_horizon, policy = load_models_and_policy(model_dir)
    logger.info("Loaded models from %s", model_dir)

    threshold, threshold_source = load_day1_pred_mean_threshold(
        decile_score_threshold=args.day1_decile_score_threshold,
        cutoff_csv=Path(args.day1_decile_cutoff_file),
        benchmark_csv=Path(args.day1_benchmark_file),
        logger=logger,
    )
    advice_deciles, advice_raw_cutoffs, advice_curve_source = load_day1_decile_curve(
        cutoff_csv=Path(args.day1_decile_cutoff_file),
        benchmark_csv=Path(args.day1_benchmark_file),
        logger=logger,
    )
    logger.info("Day-1 threshold: %.6f (%s)", threshold, threshold_source)

    scraper = OpenInsiderScraper(args.config)
    # Force fresh rescrape every cycle.
    scraper.config.cache_enabled = False

    last_scraped_at: datetime | None = load_scrape_state(args.raw_file)
    if last_scraped_at is not None:
        logger.info("Last scrape state: %s", last_scraped_at.isoformat())
    else:
        logger.info("No prior scrape state found; first run will fetch %d month(s) back.", args.months_back)

    while True:
        now_et = datetime.now(ET)
        if is_weekend_shutdown_window(now_et):
            sleep_seconds = seconds_until_weekend_shutdown_end(now_et)
            logger.info("Weekend shutdown active. Sleeping %.1f hours until Monday 00:00 ET.", sleep_seconds / 3600.0)
            if args.once:
                break
            time.sleep(sleep_seconds)
            continue
        interval_minutes = compute_sleep_interval_minutes(now_et)
        cycle_start_time = datetime.now()
        months_back = compute_months_back(last_scraped_at, cycle_start_time, args.months_back)
        logger.info(
            "Cycle start (interval=%d min, window=%s). Scraping %d month(s) back (last_scraped_at=%s).",
            interval_minutes,
            "near-open" if interval_minutes == NEAR_OPEN_INTERVAL_MINUTES else "far",
            months_back,
            last_scraped_at,
        )

        start = time.time()
        try:
            scored_events = run_cycle(
                scraper, models_by_horizon, policy, args, logger,
                now_et=now_et,
                threshold=threshold,
                threshold_source=threshold_source,
                advice_deciles=advice_deciles,
                advice_raw_cutoffs=advice_raw_cutoffs,
                advice_curve_source=advice_curve_source,
                months_back=months_back,
            )
            last_scraped_at = cycle_start_time
            save_scrape_state(args.raw_file, last_scraped_at)
            logger.info("Cycle complete. Scored events=%d", scored_events)
        except Exception as exc:
            logger.exception("Cycle failed: %s", exc)

        if args.once:
            break

        elapsed = time.time() - start
        sleep_seconds = max(0.0, interval_minutes * 60 - elapsed)
        logger.info("Sleeping %.1f min until next cycle...", sleep_seconds / 60.0)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
