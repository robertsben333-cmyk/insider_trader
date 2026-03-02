"""
Entry timing sensitivity analysis for insider purchase backtests.

Compares these entry rules on the same officer-filtered, filing-gap-filtered trades:
  - delay_15m      (current default)
  - delay_30m
  - delay_60m
  - same_day_close
  - next_day_open

For each scenario, computes return_1d/3d/5d/10d using the same sell logic as the
main backtest (sell at close on Nth trading day after buy date). For 3d/5d/10d,
returns are stored as per-day compounded equivalents, then reports:
  1) per-scenario summary stats
  2) paired differences vs delay_15m (mean delta + 95% CI + approx p-value)
"""

import math
import os
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient
from tqdm import tqdm

from backtest_insider_purchases import (
    CACHE_DIR,
    ET,
    INPUT_CSV,
    MAX_WORKERS,
    SELL_HORIZONS,
    PriceCache,
    fetch_daily_bars,
    fetch_minute_bars,
    find_nth_trading_day_close,
    find_price_at_or_after,
    load_and_filter,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str  # "delay", "close", "open"
    delay_min: int = 0


SCENARIOS: List[Scenario] = [
    Scenario("delay_15m", "delay", 15),
    Scenario("delay_30m", "delay", 30),
    Scenario("delay_60m", "delay", 60),
    Scenario("same_day_close", "close", 0),
    Scenario("next_day_open", "open", 0),
]

BASELINE = "delay_15m"
SUMMARY_CSV = "data/entry_timing_sensitivity_summary.csv"
PAIRED_CSV = "data/entry_timing_sensitivity_paired_vs_15m.csv"
RETURNS_CSV = "data/entry_timing_sensitivity_returns.csv"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else float("nan")


def _safe_median(s: pd.Series) -> float:
    return float(s.median()) if len(s) else float("nan")


def _safe_std(s: pd.Series) -> float:
    return float(s.std(ddof=1)) if len(s) > 1 else float("nan")

def compute_horizon_return_pct(buy_price: float, sell_price: Optional[float], horizon_days: int) -> Optional[float]:
    """
    Return percentage used for timing comparison.
    - 1d: standard one-day return
    - 3d/5d/10d: per-day compounded equivalent return
    """
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


def _trading_dates_from_daily(daily_bars: List[dict]) -> List[date]:
    dts = []
    for bar in daily_bars:
        d = datetime.fromtimestamp(bar["t"] / 1000, tz=ET).date()
        dts.append(d)
    return sorted(set(dts))


def _daily_map(daily_bars: List[dict]) -> Dict[date, dict]:
    out = {}
    for bar in daily_bars:
        d = datetime.fromtimestamp(bar["t"] / 1000, tz=ET).date()
        out[d] = bar
    return out


def _first_trading_on_or_after(trading_dates: List[date], d: date) -> Optional[date]:
    i = bisect_left(trading_dates, d)
    return trading_dates[i] if i < len(trading_dates) else None


def _first_trading_after(trading_dates: List[date], d: date) -> Optional[date]:
    i = bisect_right(trading_dates, d)
    return trading_dates[i] if i < len(trading_dates) else None


def _build_delay_entry(
    filing_dt: datetime,
    delay_min: int,
    trading_dates: List[date],
) -> Tuple[Optional[datetime], Optional[date]]:
    tentative = filing_dt + timedelta(minutes=delay_min)

    trade_day = _first_trading_on_or_after(trading_dates, tentative.date())
    if trade_day is None:
        return None, None

    open_delay_dt = datetime.combine(
        trade_day,
        time(hour=9, minute=30),
        tzinfo=ET,
    ) + timedelta(minutes=delay_min)
    close_dt = datetime.combine(trade_day, time(hour=16, minute=0), tzinfo=ET)

    if trade_day > tentative.date():
        return open_delay_dt, trade_day

    if tentative.tzinfo is None:
        tentative = tentative.replace(tzinfo=ET)
    else:
        tentative = tentative.astimezone(ET)

    if tentative < open_delay_dt:
        return open_delay_dt, trade_day
    if tentative >= close_dt:
        next_trade = _first_trading_after(trading_dates, trade_day)
        if next_trade is None:
            return None, None
        ndt = datetime.combine(next_trade, time(hour=9, minute=30), tzinfo=ET) + timedelta(minutes=delay_min)
        return ndt, next_trade
    return tentative, trade_day


def _build_non_delay_entry(
    filing_dt: datetime,
    kind: str,
    trading_dates: List[date],
) -> Tuple[Optional[date], str]:
    if kind == "close":
        d = _first_trading_on_or_after(trading_dates, filing_dt.date())
        return d, "daily_close"
    if kind == "open":
        d = _first_trading_after(trading_dates, filing_dt.date())
        return d, "daily_open"
    return None, ""


def _compute_paired_stats(diff: pd.Series) -> Dict[str, float]:
    n = int(diff.notna().sum())
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "z": float("nan"), "p_approx": float("nan")}

    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = std_diff / math.sqrt(n) if n > 1 else float("nan")

    if n > 1 and se > 0:
        z = mean_diff / se
        p = 2.0 * (1.0 - _normal_cdf(abs(z)))
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se
    else:
        z = float("nan")
        p = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "z": z,
        "p_approx": p,
    }


def process_ticker_timing(
    ticker: str,
    rows: List[dict],
    api_key: str,
    cache: PriceCache,
) -> List[dict]:
    client = RESTClient(api_key=api_key, retries=3)

    filing_dates = [r["transaction_date"].date() for r in rows]
    min_f = min(filing_dates)
    max_f = max(filing_dates)

    # Extend range to support next-day-open entries and 10-day exits.
    daily_bars = fetch_daily_bars(client, cache, ticker, min_f - timedelta(days=3), max_f + timedelta(days=30))
    if not daily_bars:
        return []

    trading_dates = _trading_dates_from_daily(daily_bars)
    daily_by_date = _daily_map(daily_bars)

    # Collect minute-bar dates needed by delay scenarios.
    needed_minute_dates = set()
    entry_specs_by_row: Dict[int, Dict[str, dict]] = {}

    for r in rows:
        rid = r["row_id"]
        filing_dt = r["transaction_date"]
        specs = {}
        for sc in SCENARIOS:
            if sc.kind == "delay":
                buy_dt, buy_d = _build_delay_entry(filing_dt, sc.delay_min, trading_dates)
                if buy_dt is not None and buy_d is not None:
                    specs[sc.name] = {"method": "minute_after_ts", "buy_dt": buy_dt, "buy_date": buy_d}
                    needed_minute_dates.add(buy_d)
            else:
                buy_d, method = _build_non_delay_entry(filing_dt, sc.kind, trading_dates)
                if buy_d is not None:
                    specs[sc.name] = {"method": method, "buy_date": buy_d}
        entry_specs_by_row[rid] = specs

    minute_bars_by_date: Dict[date, List[dict]] = {}
    for d in needed_minute_dates:
        minute_bars_by_date[d] = fetch_minute_bars(client, cache, ticker, d)

    out_rows: List[dict] = []

    for r in rows:
        rid = r["row_id"]
        specs = entry_specs_by_row.get(rid, {})
        for sc in SCENARIOS:
            spec = specs.get(sc.name)
            if spec is None:
                continue

            buy_date = spec["buy_date"]
            buy_price = None

            if spec["method"] == "minute_after_ts":
                minute_bars = minute_bars_by_date.get(buy_date, [])
                ts = int(spec["buy_dt"].timestamp() * 1000)
                buy_price = find_price_at_or_after(minute_bars, ts)
            elif spec["method"] == "daily_close":
                b = daily_by_date.get(buy_date)
                buy_price = b["c"] if b else None
            elif spec["method"] == "daily_open":
                b = daily_by_date.get(buy_date)
                buy_price = b["o"] if b else None

            if buy_price is None:
                continue

            rec = {
                "row_id": rid,
                "ticker": ticker,
                "scenario": sc.name,
                "buy_date": buy_date.isoformat(),
                "buy_price": float(buy_price),
            }

            for h in SELL_HORIZONS:
                cp = find_nth_trading_day_close(daily_bars, buy_date, h)
                rec[f"return_{h}d_pct"] = compute_horizon_return_pct(buy_price, cp, h)

            out_rows.append(rec)

    return out_rows


def main():
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise SystemExit("Set POLYGON_API_KEY in .env")

    cache = PriceCache(CACHE_DIR)
    df = load_and_filter(INPUT_CSV)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    groups: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        groups.setdefault(t, []).append(
            {
                "row_id": int(row["row_id"]),
                "transaction_date": row["transaction_date"].to_pydatetime().replace(tzinfo=ET),
            }
        )

    all_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_ticker_timing, ticker, rows, api_key, cache): ticker
            for ticker, rows in groups.items()
        }
        with tqdm(total=len(futures), desc="TimingSensitivity", unit="tkr") as pbar:
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    all_rows.extend(fut.result())
                except Exception as e:
                    print(f"[WARN] {ticker}: {e}")
                pbar.update(1)

    if not all_rows:
        raise SystemExit("No scenario rows produced.")

    ret_df = pd.DataFrame(all_rows)
    ret_df.to_csv(RETURNS_CSV, index=False)

    summary_rows = []
    for sc in SCENARIOS:
        s = ret_df[ret_df["scenario"] == sc.name]
        rec = {"scenario": sc.name, "trades": int(len(s))}
        for h in SELL_HORIZONS:
            col = f"return_{h}d_pct"
            v = s[col].dropna()
            rec[f"mean_{h}d"] = _safe_mean(v)
            rec[f"median_{h}d"] = _safe_median(v)
            rec[f"winrate_{h}d"] = float((v > 0).mean() * 100) if len(v) else float("nan")
            rec[f"std_{h}d"] = _safe_std(v)
        summary_rows.append(rec)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # Paired deltas vs baseline on shared row_id.
    pair_rows = []
    base = ret_df[ret_df["scenario"] == BASELINE].copy()
    for sc in SCENARIOS:
        if sc.name == BASELINE:
            continue
        cur = ret_df[ret_df["scenario"] == sc.name].copy()
        merged = base.merge(cur, on="row_id", suffixes=("_base", "_alt"))
        for h in SELL_HORIZONS:
            bcol = f"return_{h}d_pct_base"
            acol = f"return_{h}d_pct_alt"
            d = (merged[acol] - merged[bcol]).dropna()
            stats = _compute_paired_stats(d)
            pair_rows.append(
                {
                    "scenario_vs_15m": sc.name,
                    "horizon_days": h,
                    "shared_trades": int(stats["n"]),
                    "mean_delta_pct": stats["mean_diff"],
                    "ci95_low": stats["ci_low"],
                    "ci95_high": stats["ci_high"],
                    "z_stat_approx": stats["z"],
                    "p_value_approx": stats["p_approx"],
                }
            )
    paired_df = pd.DataFrame(pair_rows)
    paired_df.to_csv(PAIRED_CSV, index=False)

    print("\nSaved:")
    print(f"  {RETURNS_CSV}")
    print(f"  {SUMMARY_CSV}")
    print(f"  {PAIRED_CSV}")
    print("\nScenario summary:")
    print(summary_df.to_string(index=False))
    print("\nPaired deltas vs 15m:")
    print(paired_df.to_string(index=False))


if __name__ == "__main__":
    main()
