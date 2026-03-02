"""
Stop-Loss Analysis on Testset Trade Log
========================================
Phase 1: Fixed stop-loss sweep
  - For each trade in testset_trade_log.csv, fetch minute bars for the
    holding period and find the intraday low relative to entry price.
  - Simulate stop levels: -3%, -5%, -7%, -10%, -15%, -20%.
  - Key nuance: even winning trades may have dipped below the stop before
    recovering. We use actual intraday lows (full bar l) to check this.
  - Reports total return, win-rate, max drawdown per stop level.

Phase 2: Variable stop-loss (training data) — to be added separately.

Usage:
    python backtest/scripts/analyze_stoploss.py
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

# ─────────────────────── Config ────────────────────────────────────────

BASE = Path(__file__).resolve().parents[2]
TRADE_LOG = BASE / "backtest" / "out" / "testset_trade_log.csv"
CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUT_DIR   = BASE / "backtest" / "out"

ET = ZoneInfo("America/New_York")

STOP_LEVELS = [-3.0, -5.0, -7.0, -10.0, -15.0, -20.0]   # % from entry

# ─────────────────────── Price Cache (mirrors run_backtest.py) ──────────


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


def fetch_minute_bars(client: RESTClient, cache: PriceCache, ticker: str, d: date) -> list:
    ds = d.strftime("%Y-%m-%d")
    cached = cache.get(ticker, "min", ds, ds)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker, multiplier=1, timespan="minute",
            from_=d, to=d, adjusted=True, sort="asc", limit=50000,
        )
        bars = [{"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
                for a in aggs if a.timestamp and a.close]
    except Exception as e:
        print(f"  [WARN] fetch failed {ticker} {ds}: {e}")
        bars = []
    cache.put(ticker, "min", ds, ds, bars)
    return bars


# ─────────────────────── Helpers ────────────────────────────────────────


def ts_ms(dt: datetime) -> int:
    """UTC-aware datetime → milliseconds since epoch."""
    return int(dt.timestamp() * 1000)


def trading_days_between(start: date, end: date) -> List[date]:
    """All weekdays from start to end inclusive."""
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def get_entry_price(bars: list, entry_ts_ms: int) -> Optional[float]:
    """
    Find the open price of the first bar AT or AFTER entry_ts_ms.
    This approximates the fill price when we enter the position.
    """
    for bar in bars:
        if bar["t"] >= entry_ts_ms:
            return bar["o"]
    return None


def get_intraday_path(
    all_bars: List[dict],
    entry_ts_ms: int,
    exit_ts_ms: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    From all minute bars during the holding period (after entry, up to exit),
    return (min_low, max_high) as percentage moves from the entry bar open.

    We start from the ENTRY BAR (the bar whose open we used as entry price)
    to capture the full price path, including any same-candle spike.
    """
    # Collect bars from entry bar onwards until exit
    relevant = [b for b in all_bars if b["t"] >= entry_ts_ms and b["t"] <= exit_ts_ms]
    if not relevant:
        return None, None
    entry_price = relevant[0]["o"]
    if not entry_price or entry_price <= 0:
        return None, None
    min_low  = min(b["l"] for b in relevant)
    max_high = max(b["h"] for b in relevant)
    low_pct  = (min_low  / entry_price - 1.0) * 100.0
    high_pct = (max_high / entry_price - 1.0) * 100.0
    return low_pct, high_pct


# ─────────────────────── Core Simulation ────────────────────────────────


def simulate_stop(
    original_ret: float,
    low_pct: float,
    stop_level: float,     # negative, e.g. -5.0
) -> float:
    """
    Return the simulated trade return given a stop level.
    - If the intraday low dipped below stop_level → exit at stop_level.
    - Otherwise keep original_ret (position held to scheduled close).
    Note: stop_level is defined from entry (e.g. -5.0 means -5%).
    """
    if low_pct <= stop_level:
        return stop_level           # stopped out; assume fill exactly at stop
    return original_ret


def portfolio_stats(returns: List[float], start_budget: float = 10_000.0) -> dict:
    """
    Compute aggregate stats from a list of per-trade return percentages.
    Assumes trades are sequential with compound growth on a single-unit budget
    (simplified: average position size ≈ proportional share, but for summary
    stats we just use arithmetic metrics over the return series).
    """
    arr = np.array(returns, dtype=float)
    win_rate    = float(np.mean(arr > 0)) * 100.0
    avg_ret     = float(np.mean(arr))
    median_ret  = float(np.median(arr))
    total_pnl   = float(np.sum(arr))     # sum of %-returns (not compounded)
    worst_trade = float(np.min(arr))
    best_trade  = float(np.max(arr))
    n_stopped   = 0   # filled externally
    return dict(
        n_trades=len(arr),
        win_rate=win_rate,
        avg_ret=avg_ret,
        median_ret=median_ret,
        sum_ret=total_pnl,
        worst=worst_trade,
        best=best_trade,
    )


# ─────────────────────── Main ────────────────────────────────────────────


def main():
    load_dotenv(BASE / ".env")
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    trades = pd.read_csv(TRADE_LOG, parse_dates=["entry_time", "exit_time"])
    print(f"Loaded {len(trades)} trades from {TRADE_LOG.name}\n")

    cache  = PriceCache(CACHE_DIR)
    client = RESTClient(api_key)

    # ── Fetch intraday data and compute path stats for every trade ──────
    results = []   # one dict per trade

    for idx, row in trades.iterrows():
        ticker     = row["ticker"]
        entry_dt   = row["entry_time"].to_pydatetime().replace(tzinfo=ET)
        exit_dt    = row["exit_time"].to_pydatetime().replace(tzinfo=ET)
        orig_ret   = float(row["ret_pct"])

        # Gather ALL minute bars from entry date through exit date
        hold_dates = trading_days_between(entry_dt.date(), exit_dt.date())
        all_bars: List[dict] = []
        for d in hold_dates:
            all_bars.extend(fetch_minute_bars(client, cache, ticker, d))
        all_bars.sort(key=lambda b: b["t"])

        entry_ts = ts_ms(entry_dt)
        # Exit at scheduled close of exit date (use very end of day)
        exit_ts  = ts_ms(exit_dt)

        low_pct, high_pct = get_intraday_path(all_bars, entry_ts, exit_ts)

        rec = {
            "idx":       idx,
            "ticker":    ticker,
            "entry_time": row["entry_time"],
            "exit_time":  row["exit_time"],
            "orig_ret":  orig_ret,
            "low_pct":   low_pct,
            "high_pct":  high_pct,
            "no_data":   low_pct is None,
        }
        # Simulate each fixed stop level
        for sl in STOP_LEVELS:
            key = f"ret_sl{abs(int(sl))}"
            if low_pct is None:
                rec[key] = orig_ret          # can't evaluate → keep original
                rec[f"stopped_sl{abs(int(sl))}"] = False
            else:
                stopped = low_pct <= sl
                rec[key] = sl if stopped else orig_ret
                rec[f"stopped_sl{abs(int(sl))}"] = stopped

        results.append(rec)
        status = f"low={low_pct:+.1f}%  high={high_pct:+.1f}%" if low_pct is not None else "NO DATA"
        print(f"  [{idx+1:3d}/{len(trades)}] {ticker:<8} orig={orig_ret:+6.2f}%  {status}")

    df = pd.DataFrame(results)

    # ── Save detailed per-trade table ────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "stoploss_trade_detail.csv"
    df.to_csv(detail_path, index=False)
    print(f"\nDetailed results saved -> {detail_path}")

    # ── Summary table across stop levels ────────────────────────────────
    no_data_count = int(df["no_data"].sum())
    print(f"\n{'='*70}")
    print(f"FIXED STOP-LOSS SUMMARY  ({len(df)} trades, {no_data_count} missing intraday data)")
    print(f"{'='*70}")

    rows = []
    # Baseline (no stop)
    base_stats = portfolio_stats(df["orig_ret"].tolist())
    base_stats["stop_level"] = "none"
    base_stats["n_stopped"]  = 0
    rows.append(base_stats)

    for sl in STOP_LEVELS:
        key  = f"ret_sl{abs(int(sl))}"
        skey = f"stopped_sl{abs(int(sl))}"
        rets = df[key].tolist()
        stats = portfolio_stats(rets)
        stats["stop_level"] = f"{sl:.0f}%"
        stats["n_stopped"]  = int(df[skey].sum())
        rows.append(stats)

    summary = pd.DataFrame(rows)[[
        "stop_level", "n_trades", "n_stopped", "win_rate",
        "avg_ret", "median_ret", "sum_ret", "worst", "best"
    ]]

    summary_path = OUT_DIR / "stoploss_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Pretty-print
    print(f"\n{'Stop':>8}  {'N':>5}  {'Stopped':>7}  {'WinRate':>7}  {'AvgRet':>7}  {'MedRet':>7}  {'SumRet':>8}  {'Worst':>7}  {'Best':>7}")
    print("-" * 75)
    for _, r in summary.iterrows():
        print(
            f"{r['stop_level']:>8}  {int(r['n_trades']):>5}  {int(r['n_stopped']):>7}  "
            f"{r['win_rate']:>6.1f}%  {r['avg_ret']:>+7.2f}  {r['median_ret']:>+7.2f}  "
            f"{r['sum_ret']:>+8.1f}  {r['worst']:>+7.2f}  {r['best']:>+7.2f}"
        )

    print(f"\nSummary saved -> {summary_path}")

    # ── Stop interaction analysis: winners stopped vs losers stopped ─────
    print(f"\n{'='*70}")
    print("STOP INTERACTION: How often are WINNERS cut vs LOSERS saved?")
    print(f"{'='*70}")
    print(f"\n{'Stop':>8}  {'WinnersHit':>10}  {'LosersHit':>9}  {'NetBenefit':>10}")
    print("-" * 50)

    orig_winners = df["orig_ret"] > 0
    orig_losers  = df["orig_ret"] <= 0

    for sl in STOP_LEVELS:
        skey = f"stopped_sl{abs(int(sl))}"
        stopped = df[skey]
        winners_hit = int((stopped & orig_winners).sum())
        losers_hit  = int((stopped & orig_losers).sum())
        # Net PnL change for stopped trades
        key = f"ret_sl{abs(int(sl))}"
        pnl_change = float((df[key] - df["orig_ret"]).sum())
        print(f"{sl:>+8.0f}%  {winners_hit:>10}  {losers_hit:>9}  {pnl_change:>+10.1f}pp")

    # ── Distribution of intraday lows ────────────────────────────────────
    has_data = df[~df["no_data"]]
    print(f"\n{'='*70}")
    print(f"INTRADAY LOW DISTRIBUTION (n={len(has_data)} trades with minute data)")
    print(f"{'='*70}")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    lows = has_data["low_pct"].dropna()
    print("\nIntraday low pct from entry (percentiles):")
    for p in percentiles:
        print(f"  p{p:02d}: {np.percentile(lows, p):+.2f}%")
    print(f"  mean: {lows.mean():+.2f}%")

    print(f"\nIntraday high pct from entry (percentiles):")
    highs = has_data["high_pct"].dropna()
    for p in percentiles:
        print(f"  p{p:02d}: {np.percentile(highs, p):+.2f}%")
    print(f"  mean: {highs.mean():+.2f}%")


if __name__ == "__main__":
    main()
