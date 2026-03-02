"""
Entry Timing Analysis
=====================
Compare returns when entering 1, 5, 10, 15, 30, or 60 minutes after the
insider filing is published.

For each of the ~17,500 officer/director trades in backtest_results.csv we:
  1. Compute the buy datetime for each delay (clamped to market hours).
  2. Fetch minute bars for the buy date (mostly already cached).
  3. Look up the entry price = open of the first bar at-or-after the entry ts.
  4. Sell = T+1 trading-day close (taken from the CSV's close_1d, which is
     relative to the 15-minute baseline buy date).
     - Trades where a delay's buy date differs from the 15-min baseline buy
       date are flagged; we fall back to the CSV close_1d anyway and note
       the potential horizon mismatch (affects ~5 % of rows).
  5. Return = (close_1d / entry_price - 1) * 100.

Key questions answered:
  - Does the market price in the filing immediately, or is there a window
    where earlier entry gives a meaningfully lower price?
  - Which delay gives the best average return and win-rate?

Usage:
    python backtest/scripts/analyze_entry_timing.py
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

# ─────────────────────── Config ─────────────────────────────────────────

BASE        = Path(__file__).resolve().parents[2]
RESULTS_CSV = BASE / "backtest" / "data" / "backtest_results.csv"
CACHE_DIR   = BASE / "backtest" / "data" / "price_cache"
OUT_DIR     = BASE / "backtest" / "out"

ET = ZoneInfo("America/New_York")

DELAYS_MIN = [1, 5, 10, 15, 30, 60]      # minutes after filing
MAX_WORKERS = 10

# ─────────────────────── Price Cache ────────────────────────────────────


class PriceCache:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
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


def fetch_minute_bars(
    client: RESTClient, cache: PriceCache, ticker: str, d: date
) -> list:
    ds = d.strftime("%Y-%m-%d")
    cached = cache.get(ticker, "min", ds, ds)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker, multiplier=1, timespan="minute",
            from_=d, to=d, adjusted=True, sort="asc", limit=50000,
        )
        bars = [
            {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
            for a in aggs if a.timestamp and a.close
        ]
    except Exception as e:
        print(f"  [WARN] fetch_minute {ticker} {ds}: {e}")
        bars = []
    cache.put(ticker, "min", ds, ds, bars)
    return bars


# ─────────────────────── Entry Time Logic ───────────────────────────────


def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def compute_entry(filing_dt: datetime, delay_min: int) -> Tuple[Optional[datetime], Optional[date]]:
    """
    Compute (entry_datetime, buy_date) for filing_dt + delay_min minutes.
    Rules (mirrors research/scripts/analyze_entry_timing_sensitivity.py):
      - Tentative = filing_dt + delay_min
      - Earliest valid entry on a day = 9:30 + delay_min (so a 1-min delay
        never executes before 9:31, a 15-min delay never before 9:45, etc.)
      - If tentative < earliest valid entry on that day -> snap to earliest
      - If tentative >= 16:00 or weekend -> push to next trading day at 9:30+delay
    """
    if filing_dt.tzinfo is None:
        filing_dt = filing_dt.replace(tzinfo=ET)
    else:
        filing_dt = filing_dt.astimezone(ET)

    tentative = filing_dt + timedelta(minutes=delay_min)

    # Resolve to a trading day
    d = tentative.date()
    if d.weekday() >= 5:
        d = _next_weekday(d)
        entry = datetime.combine(d, time(9, 30), tzinfo=ET) + timedelta(minutes=delay_min)
        return entry, d

    earliest = datetime.combine(d, time(9, 30), tzinfo=ET) + timedelta(minutes=delay_min)
    mclose   = datetime.combine(d, time(16, 0),  tzinfo=ET)

    if tentative < earliest:
        return earliest, d
    if tentative >= mclose:
        d2    = _next_weekday(d + timedelta(days=1))
        entry = datetime.combine(d2, time(9, 30), tzinfo=ET) + timedelta(minutes=delay_min)
        return entry, d2
    return tentative, d


def ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def get_price_at_or_after(bars: list, target_ms: int) -> Optional[float]:
    """Open of the first minute bar whose timestamp >= target_ms."""
    for bar in bars:
        if bar["t"] >= target_ms:
            return bar["o"]
    return None


# ─────────────────────── Per-ticker Worker ──────────────────────────────


def process_ticker(
    ticker: str,
    rows: List[dict],   # list of row dicts from backtest_results.csv
    api_key: str,
    cache: PriceCache,
) -> List[dict]:
    """
    For each trade row, compute entry prices for all delay scenarios using
    cached (or freshly fetched) minute bars.
    Returns list of result dicts, one per (row, delay).
    """
    client = RESTClient(api_key=api_key, retries=3)
    out = []

    # Collect unique buy dates needed across all rows and all delays
    needed_dates: Dict[int, Dict[date, Optional[datetime]]] = {}
    # Map (row_id -> delay -> (entry_dt, buy_date))
    entry_map: Dict[int, Dict[int, Tuple[Optional[datetime], Optional[date]]]] = {}

    for row in rows:
        rid = row["row_id"]
        filing_dt = row["filing_dt"]
        entry_map[rid] = {}
        for delay in DELAYS_MIN:
            edt, bd = compute_entry(filing_dt, delay)
            entry_map[rid][delay] = (edt, bd)

    # Unique buy dates needed for minute bar fetches
    unique_dates: set = set()
    for rid, delay_map in entry_map.items():
        for delay, (edt, bd) in delay_map.items():
            if bd is not None:
                unique_dates.add(bd)

    # Fetch minute bars once per date
    bars_by_date: Dict[date, list] = {}
    for bd in sorted(unique_dates):
        bars_by_date[bd] = fetch_minute_bars(client, cache, ticker, bd)

    # Build results
    for row in rows:
        rid          = row["row_id"]
        close_1d     = row["close_1d"]
        baseline_bd  = row["baseline_buy_date"]  # buy_date for 15-min scenario from CSV

        if close_1d is None or math.isnan(float(close_1d)):
            continue  # can't compute return without sell price

        for delay in DELAYS_MIN:
            edt, bd = entry_map[rid].get(delay, (None, None))
            if edt is None or bd is None:
                continue

            bars = bars_by_date.get(bd, [])
            entry_price = get_price_at_or_after(bars, ts_ms(edt))
            if entry_price is None or entry_price <= 0:
                continue

            ret_pct = (float(close_1d) / entry_price - 1.0) * 100.0

            out.append({
                "row_id":           rid,
                "ticker":           ticker,
                "delay_min":        delay,
                "entry_dt":         edt.isoformat(),
                "buy_date":         bd.isoformat(),
                "different_bd":     bd != baseline_bd,   # horizon mismatch flag
                "entry_price":      entry_price,
                "close_1d":         float(close_1d),
                "ret_pct":          round(ret_pct, 4),
            })

    return out


# ─────────────────────── Stats ──────────────────────────────────────────


def summary_stats(df_d: pd.DataFrame, label: str) -> dict:
    r = df_d["ret_pct"]
    return {
        "delay":      label,
        "n_trades":   len(r),
        "avg_ret":    float(r.mean()),
        "median_ret": float(r.median()),
        "win_rate":   float((r > 0).mean() * 100),
        "std_ret":    float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "sum_ret":    float(r.sum()),
        "pct_diff_bd": float(df_d["different_bd"].mean() * 100),
    }


def paired_vs_15m(all_df: pd.DataFrame, delay: int) -> dict:
    """Paired difference vs 15-min baseline, matching on row_id."""
    base  = all_df[all_df["delay_min"] == 15][["row_id", "ret_pct"]].rename(columns={"ret_pct": "ret_15m"})
    other = all_df[all_df["delay_min"] == delay][["row_id", "ret_pct"]].rename(columns={"ret_pct": f"ret_{delay}m"})
    merged = base.merge(other, on="row_id")
    diff = merged[f"ret_{delay}m"] - merged["ret_15m"]
    n = len(diff.dropna())
    if n == 0:
        return {}
    mean_d = float(diff.mean())
    se     = float(diff.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
    if math.isnan(se) or se == 0:
        z, p, ci_lo, ci_hi = float("nan"), float("nan"), float("nan"), float("nan")
    else:
        z    = mean_d / se
        p    = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        ci_lo = mean_d - 1.96 * se
        ci_hi = mean_d + 1.96 * se
    return {
        "delay_min":    delay,
        "shared_n":     n,
        "mean_delta":   round(mean_d, 4),
        "ci95_low":     round(ci_lo, 4),
        "ci95_high":    round(ci_hi, 4),
        "z":            round(z, 3),
        "p_value":      round(p, 6),
    }


# ─────────────────────── Main ────────────────────────────────────────────


def main():
    load_dotenv(BASE / ".env")
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    print("Loading backtest results...")
    df = pd.read_csv(RESULTS_CSV, parse_dates=["transaction_date", "buy_datetime"])
    df = df.dropna(subset=["close_1d"]).reset_index(drop=True)
    df["row_id"] = df.index
    print(f"  {len(df):,} rows loaded")

    # Baseline buy date (15-min scenario already computed in CSV)
    df["baseline_buy_date"] = pd.to_datetime(df["buy_datetime"]).dt.date

    # Group by ticker
    groups: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        ticker = row["ticker"]
        groups.setdefault(ticker, []).append({
            "row_id":            int(row["row_id"]),
            "filing_dt":         row["transaction_date"].to_pydatetime(),
            "close_1d":          row["close_1d"],
            "baseline_buy_date": row["baseline_buy_date"],
        })

    cache = PriceCache(CACHE_DIR)

    print(f"Processing {len(groups):,} tickers with {MAX_WORKERS} threads...")
    all_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_ticker, ticker, rows, api_key, cache): ticker
            for ticker, rows in groups.items()
        }
        done = 0
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                print(f"  [WARN] {ticker}: {e}")
            done += 1
            if done % 100 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} tickers done")

    if not all_rows:
        raise RuntimeError("No results produced.")

    results = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "entry_timing_detail.csv"
    results.to_csv(detail_path, index=False)
    print(f"\nDetail saved -> {detail_path}")

    # ── Summary per delay ─────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("ENTRY TIMING SUMMARY  (all trades, sell = T+1 close from baseline buy date)")
    print(f"{'='*75}")
    print(f"\n{'Delay':>8}  {'N':>6}  {'AvgRet':>7}  {'MedRet':>7}  {'WinRate':>7}  {'SumRet':>9}  {'DiffBD%':>7}")
    print("-" * 65)

    summary_rows = []
    for delay in DELAYS_MIN:
        sub = results[results["delay_min"] == delay]
        if len(sub) == 0:
            continue
        s = summary_stats(sub, f"{delay}m")
        summary_rows.append(s)
        star = " *" if delay == 15 else "  "
        print(
            f"{delay:>6}m{star}  {s['n_trades']:>6}  {s['avg_ret']:>+7.3f}  "
            f"{s['median_ret']:>+7.3f}  {s['win_rate']:>6.1f}%  "
            f"{s['sum_ret']:>+9.1f}  {s['pct_diff_bd']:>6.1f}%"
        )
    print("  (* = current default)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "entry_timing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved -> {summary_path}")

    # ── Paired comparison vs 15m ──────────────────────────────────────────
    print(f"\n{'='*75}")
    print("PAIRED DELTA VS 15-MIN BASELINE  (same trades, different entry times)")
    print(f"{'='*75}")
    print(f"\n{'Delay':>8}  {'N':>6}  {'MeanDelta':>9}  {'95% CI':>22}  {'z':>6}  {'p':>8}")
    print("-" * 70)

    paired_rows = []
    for delay in DELAYS_MIN:
        if delay == 15:
            continue
        p = paired_vs_15m(results, delay)
        if not p:
            continue
        paired_rows.append(p)
        ci = f"[{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}]"
        sig = " **" if p["p_value"] < 0.01 else (" *" if p["p_value"] < 0.05 else "")
        print(
            f"{delay:>6}m   {p['shared_n']:>6}  {p['mean_delta']:>+9.4f}  {ci:>22}  "
            f"{p['z']:>6.2f}  {p['p_value']:>8.4f}{sig}"
        )

    paired_df = pd.DataFrame(paired_rows)
    paired_path = OUT_DIR / "entry_timing_paired_vs_15m.csv"
    paired_df.to_csv(paired_path, index=False)
    print(f"\nPaired stats saved -> {paired_path}")

    # ── Distribution of entry-price spread across delays ─────────────────
    print(f"\n{'='*75}")
    print("PRICE SPREAD ACROSS DELAYS  (how much does price move in the first hour?)")
    print(f"{'='*75}")

    # Compute entry price for 1m and 60m on the same rows, pivot
    ep_1m  = results[results["delay_min"] == 1][["row_id", "entry_price"]].rename(columns={"entry_price": "ep_1m"})
    ep_60m = results[results["delay_min"] == 60][["row_id", "entry_price"]].rename(columns={"entry_price": "ep_60m"})
    ep_15m = results[results["delay_min"] == 15][["row_id", "entry_price"]].rename(columns={"entry_price": "ep_15m"})

    spread = ep_1m.merge(ep_60m, on="row_id").merge(ep_15m, on="row_id")
    spread["drift_1m_to_60m"]  = (spread["ep_60m"]  / spread["ep_1m"]  - 1.0) * 100.0
    spread["drift_1m_to_15m"]  = (spread["ep_15m"]  / spread["ep_1m"]  - 1.0) * 100.0

    for col, label in [("drift_1m_to_15m", "1m->15m entry drift"), ("drift_1m_to_60m", "1m->60m entry drift")]:
        d = spread[col].dropna()
        print(f"\n  {label} (n={len(d):,}):")
        for p in [5, 25, 50, 75, 95]:
            print(f"    p{p:02d}: {np.percentile(d, p):+.3f}%")
        print(f"    mean: {d.mean():+.3f}%")

    print()


if __name__ == "__main__":
    main()
