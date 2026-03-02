"""
Build an aggregated backtest dataset from the unfiltered insider purchases file.

Pipeline:
  1) Load raw insider purchases (no officer-title filter), filing gap in [0, 5].
  2) Aggregate to one representative row per (ticker, trade_date), using the
     same cluster rule as training data (2nd filing if cluster, otherwise 1st).
  3) Compute buy price and horizon returns:
     - buy price from minute cache when present
     - fallback buy price = Form-4 reported last_price
     - sell prices from daily bars (cache first, Polygon API fallback)
  4) Save aggregated event-level file ready for train_models.load_and_merge().
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

ET = ZoneInfo("America/New_York")
SELL_HORIZONS = [1, 3, 5, 10]
MAX_FILING_GAP_DAYS = 5
BUY_DELAY_MINUTES = 15
MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("build_unfiltered_backtest")


def clean_numeric(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return 0.0
    s = val.replace("$", "").replace(",", "").replace("+", "").strip()
    if "%" in s or s.lower() in ("n/a", "new", ""):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def compute_buy_datetime(filing_dt: pd.Timestamp) -> datetime:
    filing_et = filing_dt.to_pydatetime().replace(tzinfo=ET)
    buy_dt = filing_et + timedelta(minutes=BUY_DELAY_MINUTES)

    open_time = buy_dt.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    close_time = buy_dt.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    earliest_buy = open_time + timedelta(minutes=BUY_DELAY_MINUTES)

    def _next_weekday(dt: datetime) -> datetime:
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


def find_price_at_or_after(bars: List[dict], target_ts_ms: int) -> Optional[float]:
    for bar in bars:
        if bar.get("t", 0) >= target_ts_ms:
            return bar.get("c")
    return None


def find_nth_trading_day_close(daily_bars: List[dict], buy_date: date, n: int) -> Optional[float]:
    count = 0
    for bar in daily_bars:
        ts = bar.get("t")
        if not ts:
            continue
        bar_date = datetime.fromtimestamp(ts / 1000, tz=ET).date()
        if bar_date > buy_date:
            count += 1
            if count == n:
                return bar.get("c")
    return None


def find_close_on_or_before(daily_bars: List[dict], target_date: date) -> Optional[float]:
    result = None
    for bar in daily_bars:
        ts = bar.get("t")
        if not ts:
            continue
        bar_date = datetime.fromtimestamp(ts / 1000, tz=ET).date()
        if bar_date <= target_date:
            result = bar.get("c")
        else:
            break
    return result


def compute_horizon_return_pct(buy_price: float, sell_price: Optional[float], horizon_days: int) -> Optional[float]:
    if sell_price is None or buy_price is None or buy_price <= 0:
        return None
    gross = float(sell_price) / float(buy_price)
    if gross < 0:
        return None
    if horizon_days <= 1:
        ret = gross - 1.0
    else:
        ret = gross ** (1.0 / float(horizon_days)) - 1.0
    return round(ret * 100.0, 4)


@dataclass
class TickerStats:
    events: int = 0
    used_minute_cache: int = 0
    used_daily_close_fallback: int = 0
    used_last_price_fallback: int = 0
    skipped_no_buy_price: int = 0
    daily_missing: int = 0


class PriceCache:
    DAY_RE = re.compile(r"^(?P<ticker>.+)_day_(?P<from>\d{4}-\d{2}-\d{2})_(?P<to>\d{4}-\d{2}-\d{2})\.json$")

    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._day_index: Dict[str, List[Tuple[date, date, Path]]] = {}
        self._build_day_index()

    def _build_day_index(self) -> None:
        for p in self.dir.glob("*_day_*.json"):
            m = self.DAY_RE.match(p.name)
            if not m:
                continue
            try:
                d0 = datetime.strptime(m.group("from"), "%Y-%m-%d").date()
                d1 = datetime.strptime(m.group("to"), "%Y-%m-%d").date()
            except ValueError:
                continue
            self._day_index.setdefault(m.group("ticker"), []).append((d0, d1, p))

    def _path(self, ticker: str, kind: str, d0: str, d1: str) -> Path:
        return self.dir / f"{ticker}_{kind}_{d0}_{d1}.json"

    def get_minute(self, ticker: str, d: date) -> Optional[List[dict]]:
        ds = d.strftime("%Y-%m-%d")
        p = self._path(ticker, "min", ds, ds)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_day_exact(self, ticker: str, d0: date, d1: date) -> Optional[List[dict]]:
        p = self._path(ticker, "day", d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d"))
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_day_covering(self, ticker: str, d0: date, d1: date) -> Optional[List[dict]]:
        candidates = []
        for s, e, p in self._day_index.get(ticker, []):
            if s <= d0 and e >= d1:
                candidates.append((e.toordinal() - s.toordinal(), p))
        if not candidates:
            return None
        _, best = sorted(candidates, key=lambda x: x[0])[0]
        try:
            return json.loads(best.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put_day(self, ticker: str, d0: date, d1: date, data: List[dict]) -> None:
        p = self._path(ticker, "day", d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d"))
        with self._lock:
            p.write_text(json.dumps(data), encoding="utf-8")
            self._day_index.setdefault(ticker, []).append((d0, d1, p))


def _aggs_to_dicts(aggs) -> List[dict]:
    return [
        {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
        for a in aggs
        if getattr(a, "timestamp", None) and getattr(a, "close", None)
    ]


_thread_local = threading.local()


def _get_client(api_key: str) -> RESTClient:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = RESTClient(api_key=api_key, retries=3)
    return _thread_local.client


def fetch_daily_bars(client: RESTClient, cache: PriceCache, ticker: str, d0: date, d1: date) -> List[dict]:
    hit = cache.get_day_exact(ticker, d0, d1)
    if hit is not None:
        return hit
    hit = cache.get_day_covering(ticker, d0, d1)
    if hit is not None:
        return hit
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=d0,
            to=d1,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
        bars = _aggs_to_dicts(aggs)
    except Exception:
        bars = []
    cache.put_day(ticker, d0, d1, bars)
    return bars


def get_daily_bars(
    client: RESTClient,
    cache: PriceCache,
    ticker: str,
    d0: date,
    d1: date,
    cache_only_day: bool,
) -> List[dict]:
    hit = cache.get_day_exact(ticker, d0, d1)
    if hit is not None:
        return hit
    hit = cache.get_day_covering(ticker, d0, d1)
    if hit is not None:
        return hit
    if cache_only_day:
        return []
    return fetch_daily_bars(client, cache, ticker, d0, d1)


def load_and_aggregate(raw_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading raw insider purchases (unfiltered titles): %s", raw_csv)
    df = pd.read_csv(raw_csv)

    required = [
        "transaction_date",
        "trade_date",
        "ticker",
        "company_name",
        "owner_name",
        "title",
        "last_price",
        "qty",
        "value",
    ]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing required column in raw CSV: {c}")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date", "trade_date", "ticker"]).copy()

    total = len(df)
    df["filing_gap_days"] = (df["transaction_date"] - df["trade_date"]).dt.days
    df = df[(df["filing_gap_days"] >= 0) & (df["filing_gap_days"] <= MAX_FILING_GAP_DAYS)].copy()
    logger.info("Filing-gap filter [0,%d]: %d / %d rows", MAX_FILING_GAP_DAYS, len(df), total)

    df["trade_date_d"] = df["trade_date"].dt.date
    counts = (
        df.groupby(["ticker", "trade_date_d"], as_index=False)
        .size()
        .rename(columns={"size": "n_insiders_in_cluster"})
    )
    df = df.merge(counts, on=["ticker", "trade_date_d"], how="left")
    df = df.sort_values(["ticker", "trade_date_d", "transaction_date"]).reset_index(drop=True)

    def pick_representative(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) == 1:
            return group.iloc[[0]]
        return group.iloc[[1]]

    rep = (
        df.groupby(["ticker", "trade_date_d"], group_keys=False, sort=False)
        .apply(pick_representative)
        .reset_index(drop=True)
    )

    rep["n_insiders"] = rep["n_insiders_in_cluster"].astype(int)
    rep["cluster_buy"] = rep["n_insiders_in_cluster"].astype(int) >= 2
    rep["n_insiders_label"] = rep["n_insiders_in_cluster"].astype(int).clip(upper=5).map(
        {1: "1 (solo)", 2: "2", 3: "3", 4: "4", 5: "5+"}
    )

    logger.info("Aggregated to %d unique ticker/trade-date events", len(rep))
    return rep


def process_ticker(
    ticker: str,
    rows: List[dict],
    api_key: str,
    cache: PriceCache,
    allow_last_price_fallback: bool,
    cache_only_day: bool,
) -> Tuple[List[dict], TickerStats]:
    stats = TickerStats(events=len(rows))
    client = _get_client(api_key)

    min_buy = min(r["buy_date"] for r in rows)
    max_buy = max(r["buy_date"] for r in rows)
    d0 = min_buy
    d1 = max_buy + timedelta(days=25)
    daily_bars = get_daily_bars(client, cache, ticker, d0, d1, cache_only_day)

    out = []
    for r in rows:
        buy_date = r["buy_date"]
        buy_dt = r["buy_datetime"]

        minute_bars = cache.get_minute(ticker, buy_date) or []
        buy_ts_ms = int(buy_dt.timestamp() * 1000)
        buy_price = find_price_at_or_after(minute_bars, buy_ts_ms)

        buy_source = "minute_cache"
        if buy_price is None or buy_price <= 0:
            dp = find_close_on_or_before(daily_bars, buy_date)
            if dp is not None and dp > 0:
                buy_price = dp
                buy_source = "daily_close_fallback"
                stats.used_daily_close_fallback += 1
            else:
                if not allow_last_price_fallback:
                    stats.skipped_no_buy_price += 1
                    continue
                lp = clean_numeric(r["last_price"])
                if lp > 0:
                    buy_price = lp
                    buy_source = "last_price_fallback"
                    stats.used_last_price_fallback += 1
                else:
                    stats.skipped_no_buy_price += 1
                    continue
        else:
            stats.used_minute_cache += 1

        closes = {}
        rets = {}
        missing_any_daily = False
        for h in SELL_HORIZONS:
            cp = find_nth_trading_day_close(daily_bars, buy_date, h)
            closes[h] = cp
            rets[h] = compute_horizon_return_pct(buy_price, cp, h)
            if cp is None:
                missing_any_daily = True
        if missing_any_daily:
            stats.daily_missing += 1

        out.append(
            {
                "transaction_date": r["transaction_date"],
                "trade_date": r["trade_date"],
                "ticker": ticker,
                "company_name": r["company_name"],
                "owner_name": r["owner_name"],
                "title": r["title"],
                "last_price": r["last_price"],
                "qty": r["qty"],
                "value": r["value"],
                "filing_gap_days": r["filing_gap_days"],
                "buy_datetime": buy_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "buy_price": buy_price,
                "buy_price_source": buy_source,
                "close_1d": closes[1],
                "close_3d": closes[3],
                "close_5d": closes[5],
                "close_10d": closes[10],
                "return_1d_pct": rets[1],
                "return_3d_pct": rets[3],
                "return_5d_pct": rets[5],
                "return_10d_pct": rets[10],
                "returns_are_per_day_adjusted": True,
                "trade_date_d": str(r["trade_date_d"]),
                "n_insiders": int(r["n_insiders"]),
                "cluster_buy": bool(r["cluster_buy"]),
                "n_insiders_label": r["n_insiders_label"],
                "n_insiders_in_cluster": int(r["n_insiders_in_cluster"]),
            }
        )

    return out, stats


def build_dataset(
    input_csv: Path,
    output_csv: Path,
    cache_dir: Path,
    max_workers: int,
    allow_last_price_fallback: bool,
    cache_only_day: bool,
    logger: logging.Logger,
) -> None:
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("POLYGON_API_KEY is required in environment/.env")

    rep = load_and_aggregate(input_csv, logger)
    rep["buy_datetime"] = rep["transaction_date"].apply(compute_buy_datetime)
    rep["buy_date"] = rep["buy_datetime"].apply(lambda x: x.date())

    ticker_rows: Dict[str, List[dict]] = {}
    for _, row in rep.iterrows():
        t = row["ticker"]
        ticker_rows.setdefault(t, []).append(
            {
                "transaction_date": str(row["transaction_date"]),
                "trade_date": str(row["trade_date"]),
                "trade_date_d": row["trade_date_d"],
                "company_name": row["company_name"],
                "owner_name": row["owner_name"],
                "title": row["title"],
                "last_price": row["last_price"],
                "qty": row["qty"],
                "value": row["value"],
                "filing_gap_days": int(row["filing_gap_days"]),
                "buy_datetime": row["buy_datetime"],
                "buy_date": row["buy_date"],
                "n_insiders": int(row["n_insiders"]),
                "cluster_buy": bool(row["cluster_buy"]),
                "n_insiders_label": row["n_insiders_label"],
                "n_insiders_in_cluster": int(row["n_insiders_in_cluster"]),
            }
        )

    logger.info("Pricing %d aggregated events across %d tickers", len(rep), len(ticker_rows))
    cache = PriceCache(cache_dir)
    all_rows: List[dict] = []
    stats = TickerStats()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_ticker,
                ticker,
                rows,
                api_key,
                cache,
                allow_last_price_fallback,
                cache_only_day,
            ): ticker
            for ticker, rows in ticker_rows.items()
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 250 == 0:
                logger.info("Tickers processed: %d / %d", done, len(futures))
            rows, st = fut.result()
            all_rows.extend(rows)
            stats.events += st.events
            stats.used_minute_cache += st.used_minute_cache
            stats.used_daily_close_fallback += st.used_daily_close_fallback
            stats.used_last_price_fallback += st.used_last_price_fallback
            stats.skipped_no_buy_price += st.skipped_no_buy_price
            stats.daily_missing += st.daily_missing

    out = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    logger.info("Saved %d rows to %s", len(out), output_csv)
    logger.info(
        "Buy price source: minute_cache=%d, daily_close_fallback=%d, last_price_fallback=%d, skipped_no_buy=%d",
        stats.used_minute_cache,
        stats.used_daily_close_fallback,
        stats.used_last_price_fallback,
        stats.skipped_no_buy_price,
    )
    logger.info("Rows with at least one missing horizon close: %d", stats.daily_missing)


def main() -> None:
    p = argparse.ArgumentParser(description="Build unfiltered aggregated backtest dataset.")
    p.add_argument("--input-csv", default="backtest/data/insider_purchases.csv")
    p.add_argument("--output-csv", default="backtest/data/backtest_results_aggregated_unfiltered.csv")
    p.add_argument("--cache-dir", default="backtest/data/price_cache")
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument(
        "--allow-last-price-fallback",
        action="store_true",
        help="Use raw Form-4 last_price when minute and daily-close fallback are unavailable.",
    )
    p.add_argument(
        "--cache-only-day",
        action="store_true",
        help="Do not call Polygon for daily bars; use cache only.",
    )
    args = p.parse_args()

    logger = setup_logger()
    build_dataset(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        cache_dir=Path(args.cache_dir),
        max_workers=args.max_workers,
        allow_last_price_fallback=bool(args.allow_last_price_fallback),
        cache_only_day=bool(args.cache_only_day),
        logger=logger,
    )


if __name__ == "__main__":
    main()
