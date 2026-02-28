"""
Insider Purchase Backtesting Script
====================================
Filters OpenInsider purchase data to officer-level trades filed within 5 days,
then backtests a strategy: buy 15 minutes after SEC filing, sell at close on
T+1, T+3, T+5, T+10 trading days. Uses Polygon API for price data.

Optimised with ThreadPoolExecutor for concurrent API fetching.
"""

import os
import re
import json
import time
import logging
import threading
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import urllib3
from polygon import RESTClient
from dotenv import load_dotenv
from tqdm import tqdm

# Suppress urllib3 connection pool warnings (harmless with thread-per-client)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Connection pool is full")

# ──────────────────────────── Configuration ────────────────────────────

OFFICER_PATTERN = re.compile(
    r'\b(COB|Chairman|CEO|Co-CEO|Pres|President|COO|CFO|GC|VP|SVP|EVP)\b',
    re.IGNORECASE,
)
MAX_FILING_GAP_DAYS = 5
MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0
BUY_DELAY_MINUTES = 15
ET = ZoneInfo("America/New_York")
SELL_HORIZONS = [1, 3, 5, 10]
MAX_WORKERS = 10              # Concurrent API threads
INPUT_CSV = "data/insider_purchases.csv"
OUTPUT_CSV = "data/backtest_results.csv"
CACHE_DIR = "data/price_cache"

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")
# Suppress noisy urllib3 pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# ──────────────────────────── Helpers ──────────────────────────────────


def clean_numeric(val) -> float:
    if not isinstance(val, str):
        return 0.0
    val = val.replace("$", "").replace(",", "").replace("+", "")
    if "%" in val or val.lower() in ("n/a", "new", ""):
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def compute_buy_datetime(filing_dt: pd.Timestamp) -> datetime:
    """
    Buy = filing + 15 min, snapped to market hours (9:30–16:00 ET).
    If outside market hours or weekend, push to 9:45 AM next weekday.
    """
    filing_et = filing_dt.to_pydatetime().replace(tzinfo=ET)
    buy_dt = filing_et + timedelta(minutes=BUY_DELAY_MINUTES)

    open_time = buy_dt.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    close_time = buy_dt.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    earliest_buy = open_time + timedelta(minutes=BUY_DELAY_MINUTES)  # 9:45 AM

    def _next_weekday(dt):
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        return dt

    if buy_dt.weekday() >= 5:
        return _next_weekday(buy_dt).replace(hour=9, minute=45, second=0, microsecond=0)
    if buy_dt < earliest_buy:
        return earliest_buy
    if buy_dt >= close_time:
        return _next_weekday(buy_dt + timedelta(days=1)).replace(hour=9, minute=45, second=0, microsecond=0)
    return buy_dt


# ──────────────────────────── Thread-safe Price Cache ─────────────────


class PriceCache:
    def __init__(self, cache_dir: str):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, ticker: str, kind: str, f: str, t: str) -> Path:
        return self.dir / f"{ticker}_{kind}_{f}_{t}.json"

    def get(self, ticker, kind, f, t) -> Optional[list]:
        p = self._path(ticker, kind, f, t)
        if p.exists():
            try:
                with open(p) as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def put(self, ticker, kind, f, t, data: list):
        p = self._path(ticker, kind, f, t)
        with self._lock:
            with open(p, "w") as fh:
                json.dump(data, fh)


# ──────────────────────────── Polygon Fetchers ────────────────────────


def _aggs_to_dicts(aggs) -> List[dict]:
    return [
        {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
        for a in aggs
        if a.timestamp and a.close
    ]


def fetch_minute_bars(client: RESTClient, cache: PriceCache, ticker: str, target_date: date) -> List[dict]:
    ds = target_date.strftime("%Y-%m-%d")
    cached = cache.get(ticker, "min", ds, ds)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker, multiplier=1, timespan="minute",
            from_=target_date, to=target_date,
            adjusted=True, sort="asc", limit=50000,
        )
        bars = _aggs_to_dicts(aggs)
    except Exception:
        bars = []
    cache.put(ticker, "min", ds, ds, bars)
    return bars


def fetch_daily_bars(client: RESTClient, cache: PriceCache, ticker: str, from_d: date, to_d: date) -> List[dict]:
    fs, ts = from_d.strftime("%Y-%m-%d"), to_d.strftime("%Y-%m-%d")
    cached = cache.get(ticker, "day", fs, ts)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker, multiplier=1, timespan="day",
            from_=from_d, to=to_d,
            adjusted=True, sort="asc", limit=50000,
        )
        bars = _aggs_to_dicts(aggs)
    except Exception:
        bars = []
    cache.put(ticker, "day", fs, ts, bars)
    return bars


# ──────────────────────────── Price Lookup ─────────────────────────────


def find_price_at_or_after(bars: List[dict], target_ts_ms: int) -> Optional[float]:
    for bar in bars:
        if bar["t"] >= target_ts_ms:
            return bar["c"]
    return None


def find_nth_trading_day_close(daily_bars: List[dict], buy_date: date, n: int) -> Optional[float]:
    count = 0
    for bar in daily_bars:
        bar_date = datetime.fromtimestamp(bar["t"] / 1000, tz=ET).date()
        if bar_date > buy_date:
            count += 1
            if count == n:
                return bar["c"]
    return None


# ──────────────────────────── Filtering ────────────────────────────────


def load_and_filter(csv_path: str) -> pd.DataFrame:
    logger.info(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    total = len(df)

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    mask_officer = df["title"].str.contains(OFFICER_PATTERN, na=False)
    df = df[mask_officer].copy()
    logger.info(f"Officer filter: {len(df):,} / {total:,} kept")

    df["filing_gap"] = (df["transaction_date"] - df["trade_date"]).dt.days
    df = df[(df["filing_gap"] >= 0) & (df["filing_gap"] <= MAX_FILING_GAP_DAYS)].copy()
    logger.info(f"Filing-gap filter (≤{MAX_FILING_GAP_DAYS}d): {len(df):,} kept")

    return df.reset_index(drop=True)


# ──────────── Thread-local Polygon clients ────────────────────────────

_thread_local = threading.local()


def _get_client(api_key: str) -> RESTClient:
    """One RESTClient per thread — avoids connection pool contention."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = RESTClient(api_key=api_key, retries=3)
    return _thread_local.client


# ──────────── Per-ticker worker (runs in thread pool) ─────────────────


def process_ticker(
    ticker: str,
    ticker_rows: List[dict],
    api_key: str,
    cache: PriceCache,
) -> Tuple[List[dict], int]:
    """
    Fetch prices and compute returns for all trades of one ticker.
    Returns (results_list, skipped_count).
    """
    client = _get_client(api_key)
    results = []
    skipped = 0

    # Collect buy dates and date range
    buy_dates = set()
    min_buy = None
    max_buy = None
    for r in ticker_rows:
        bd = r["buy_date"]
        buy_dates.add(bd)
        if min_buy is None or bd < min_buy:
            min_buy = bd
        if max_buy is None or bd > max_buy:
            max_buy = bd

    # One daily-bar call per ticker
    daily_bars = fetch_daily_bars(client, cache, ticker, min_buy, max_buy + timedelta(days=20))

    # Minute bars – one call per unique buy date
    minute_bars_by_date: Dict[date, List[dict]] = {}
    for bd in buy_dates:
        minute_bars_by_date[bd] = fetch_minute_bars(client, cache, ticker, bd)

    # Process each trade
    for r in ticker_rows:
        buy_dt = r["buy_datetime"]
        bd = r["buy_date"]
        minute_bars = minute_bars_by_date.get(bd, [])

        buy_ts_ms = int(buy_dt.timestamp() * 1000)
        buy_price = find_price_at_or_after(minute_bars, buy_ts_ms)
        if buy_price is None:
            skipped += 1
            continue

        closes = {}
        rets = {}
        for h in SELL_HORIZONS:
            cp = find_nth_trading_day_close(daily_bars, bd, h)
            closes[h] = cp
            rets[h] = round((cp - buy_price) / buy_price * 100, 4) if cp else None

        results.append({
            "transaction_date": r["transaction_date"],
            "trade_date": r["trade_date"],
            "ticker": ticker,
            "company_name": r["company_name"],
            "owner_name": r["owner_name"],
            "title": r["title"],
            "last_price": r["last_price"],
            "qty": r["qty"],
            "value": r["value"],
            "filing_gap_days": r["filing_gap"],
            "buy_datetime": buy_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "buy_price": buy_price,
            "close_1d": closes[1],
            "close_3d": closes[3],
            "close_5d": closes[5],
            "close_10d": closes[10],
            "return_1d_pct": rets[1],
            "return_3d_pct": rets[3],
            "return_5d_pct": rets[5],
            "return_10d_pct": rets[10],
        })

    return results, skipped


# ──────────────────────────── Main Pipeline ───────────────────────────


def run_backtest():
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise SystemExit("Set POLYGON_API_KEY in .env")

    cache = PriceCache(CACHE_DIR)

    # ── Filter ──
    df = load_and_filter(INPUT_CSV)

    # ── Buy datetimes ──
    logger.info("Computing buy datetimes …")
    df["buy_datetime"] = df["transaction_date"].apply(compute_buy_datetime)
    df["buy_date"] = df["buy_datetime"].apply(lambda dt: dt.date())

    # ── Group by ticker ──
    tickers = df["ticker"].unique()
    logger.info(f"Fetching prices for {len(tickers):,} tickers with {MAX_WORKERS} threads …")

    # Pre-group rows into dicts for each ticker (avoids pandas overhead in threads)
    ticker_groups: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        if t not in ticker_groups:
            ticker_groups[t] = []
        ticker_groups[t].append({
            "transaction_date": str(row["transaction_date"]),
            "trade_date": str(row["trade_date"]),
            "company_name": row["company_name"],
            "owner_name": row["owner_name"],
            "title": row["title"],
            "last_price": row["last_price"],
            "qty": row["qty"],
            "value": row["value"],
            "filing_gap": row["filing_gap"],
            "buy_datetime": row["buy_datetime"],
            "buy_date": row["buy_date"],
        })

    # ── Concurrent price fetching ──
    all_results: List[dict] = []
    total_skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_ticker, ticker, rows, api_key, cache): ticker
            for ticker, rows in ticker_groups.items()
        }

        with tqdm(total=len(futures), desc="Tickers", unit="tkr") as pbar:
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results, skipped = future.result()
                    all_results.extend(results)
                    total_skipped += skipped
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
                pbar.update(1)

    # ── Save ──
    out = pd.DataFrame(all_results)
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved {len(out):,} trades → {OUTPUT_CSV}  (skipped {total_skipped:,})")

    # ── Summary ──
    if len(out) > 0:
        print_summary(out)


# ──────────────────────────── Summary Stats ───────────────────────────


def print_summary(df: pd.DataFrame):
    w = 70
    print("\n" + "=" * w)
    print("  INSIDER PURCHASE BACKTEST — SUMMARY")
    print("=" * w)
    print(f"  Total trades with price data : {len(df):,}")

    for h in SELL_HORIZONS:
        col = f"return_{h}d_pct"
        v = df[col].dropna()
        if v.empty:
            continue
        print(f"\n  ── {h}-Day Hold ──")
        print(f"    Trades   : {len(v):>7,}")
        print(f"    Mean     : {v.mean():>+8.2f}%")
        print(f"    Median   : {v.median():>+8.2f}%")
        print(f"    Win rate : {(v > 0).mean() * 100:>7.1f}%")
        print(f"    Std dev  : {v.std():>8.2f}%")
        print(f"    Best     : {v.max():>+8.2f}%")
        print(f"    Worst    : {v.min():>+8.2f}%")

    print(f"\n  ── Returns by Officer Title ──")
    print(f"  {'Title':>10}  {'n':>6}  {'1d':>8}  {'3d':>8}  {'5d':>8}  {'10d':>8}")
    print(f"  {'─' * 10}  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for kw in ["CEO", "CFO", "COO", "COB", "Pres", "VP", "GC"]:
        mask = df["title"].str.contains(kw, case=False, na=False)
        s = df[mask]
        if len(s) < 5:
            continue
        vals = []
        for h in SELL_HORIZONS:
            c = f"return_{h}d_pct"
            m = s[c].dropna().mean()
            vals.append(f"{m:>+7.2f}%" if pd.notna(m) else "     N/A")
        print(f"  {kw:>10}  {len(s):>6,}  {'  '.join(vals)}")

    print("=" * w + "\n")


if __name__ == "__main__":
    run_backtest()
