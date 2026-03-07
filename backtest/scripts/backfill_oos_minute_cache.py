from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

BASE = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUT_DIR = BASE / "backtest" / "out"

if str(BASE) not in os.sys.path:
    os.sys.path.insert(0, str(BASE))

from backtest.scripts.evaluate_investable_risk_rules import (  # noqa: E402
    chrono_split_60_20_20,
    load_scored_day1,
)


class PriceCache:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, ticker: str, d_str: str) -> Path:
        return self.dir / f"{ticker}_min_{d_str}_{d_str}.json"

    def exists(self, ticker: str, d_str: str) -> bool:
        return self.path(ticker, d_str).exists()

    def put(self, ticker: str, d_str: str, data: list[dict]) -> None:
        p = self.path(ticker, d_str)
        with self._lock:
            p.write_text(json.dumps(data), encoding="utf-8")


_thread_local = threading.local()


def get_client(api_key: str) -> RESTClient:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = RESTClient(api_key=api_key, retries=3)
    return _thread_local.client


def fetch_one(api_key: str, cache: PriceCache, ticker: str, d_str: str) -> tuple[str, str, int, str]:
    if cache.exists(ticker, d_str):
        return ticker, d_str, 0, "cached"

    target_date = date.fromisoformat(d_str)
    client = get_client(api_key)
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
        bars = [
            {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
            for a in aggs
            if getattr(a, "timestamp", None) is not None and getattr(a, "close", None) is not None
        ]
        cache.put(ticker, d_str, bars)
        return ticker, d_str, len(bars), "fetched"
    except Exception as exc:
        return ticker, d_str, 0, f"error:{type(exc).__name__}"


def build_missing_pairs(cache: PriceCache, day_field: str) -> pd.DataFrame:
    if day_field not in {"buy_datetime", "exit_datetime", "both"}:
        raise ValueError(f"Unsupported day_field: {day_field}")

    cols = ["ticker", "buy_datetime", "exit_datetime"]
    scored = load_scored_day1()[cols].copy()
    scored["buy_datetime"] = pd.to_datetime(scored["buy_datetime"], errors="coerce")
    scored["exit_datetime"] = pd.to_datetime(scored["exit_datetime"], errors="coerce")
    if day_field == "buy_datetime":
        scored = scored.dropna(subset=["buy_datetime"]).copy()
    elif day_field == "exit_datetime":
        scored = scored.dropna(subset=["exit_datetime"]).copy()
    else:
        scored = scored.dropna(subset=["buy_datetime", "exit_datetime"], how="all").copy()
    n_train, split_80 = chrono_split_60_20_20(len(scored))
    subset = scored.iloc[n_train:].copy()

    pair_frames: list[pd.DataFrame] = []
    if day_field in {"buy_datetime", "both"}:
        buy_pairs = subset.dropna(subset=["buy_datetime"])[["ticker", "buy_datetime"]].copy()
        buy_pairs["trade_day"] = buy_pairs["buy_datetime"].dt.strftime("%Y-%m-%d")
        pair_frames.append(buy_pairs[["ticker", "trade_day"]])
    if day_field in {"exit_datetime", "both"}:
        exit_pairs = subset.dropna(subset=["exit_datetime"])[["ticker", "exit_datetime"]].copy()
        exit_pairs["trade_day"] = exit_pairs["exit_datetime"].dt.strftime("%Y-%m-%d")
        pair_frames.append(exit_pairs[["ticker", "trade_day"]])

    if not pair_frames:
        return pd.DataFrame(columns=["ticker", "trade_day", "cache_exists"])

    pairs = (
        pd.concat(pair_frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["ticker", "trade_day"])
        .reset_index(drop=True)
    )
    pairs["cache_exists"] = [
        cache.exists(str(r.ticker), str(r.trade_day))
        for r in pairs.itertuples(index=False)
    ]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing minute cache files for the original all-company OOS split.")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of missing pairs to fetch.")
    parser.add_argument(
        "--day-field",
        choices=["buy_datetime", "exit_datetime", "both"],
        default="buy_datetime",
        help="Which OOS trading day to backfill minute bars for.",
    )
    parser.add_argument("--manifest-out", default=str(OUT_DIR / "oos_minute_cache_backfill_manifest.csv"))
    args = parser.parse_args()

    load_dotenv(BASE / ".env")
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    cache = PriceCache(Path(args.cache_dir))
    pairs = build_missing_pairs(cache, args.day_field)
    missing = pairs[~pairs["cache_exists"]].copy().reset_index(drop=True)
    if args.limit and args.limit > 0:
        missing = missing.iloc[: args.limit].copy()

    print(f"total_pairs={len(pairs)}")
    print(f"missing_pairs={len(missing)}")
    print(f"day_field={args.day_field}")
    print(f"workers={args.workers}")

    results: list[dict] = []
    fetched = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, api_key, cache, str(r.ticker), str(r.trade_day)): (str(r.ticker), str(r.trade_day))
            for r in missing.itertuples(index=False)
        }
        total = len(futures)
        for idx, fut in enumerate(as_completed(futures), start=1):
            ticker, d_str = futures[fut]
            t, d, bar_count, status = fut.result()
            if status == "fetched":
                fetched += 1
            elif status.startswith("error:"):
                errors += 1
            results.append(
                {
                    "ticker": t,
                    "trade_day": d,
                    "bar_count": bar_count,
                    "status": status,
                }
            )
            if idx % 100 == 0 or idx == total:
                print(f"done={idx}/{total} fetched={fetched} errors={errors}")

    manifest = pd.DataFrame(results).sort_values(["status", "ticker", "trade_day"]).reset_index(drop=True)
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    print(f"fetched={fetched}")
    print(f"errors={errors}")
    print(f"manifest_out={manifest_path}")


if __name__ == "__main__":
    main()
