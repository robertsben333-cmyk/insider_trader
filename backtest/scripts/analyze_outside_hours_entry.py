"""
Compare the current regular-hours entry policy against an extended-hours
attempt policy for filings that arrive outside 9:30-16:00 ET.

Baseline:
  Use the existing buy_datetime / buy_price from backtest_results.csv.

Extended-hours attempt:
  Try to buy on the first minute bar at or after the filing timestamp, even if
  that bar is pre-market or after-hours. If the filing day has no eligible bar,
  fall forward to the earliest minute bar on the next trading day.

Sell logic matches the main backtest: sell at the close on the Nth trading day
after the buy date.
"""

from __future__ import annotations

import math
import os
import sys
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from backtest.scripts.run_backtest import (
    CACHE_DIR,
    ET,
    MAX_WORKERS,
    PriceCache,
    SELL_HORIZONS,
    fetch_daily_bars,
    fetch_minute_bars,
    find_nth_trading_day_close,
)


INPUT_CSV = BASE / "backtest" / "data" / "backtest_results.csv"
OUT_DIR = BASE / "backtest" / "out"
SUMMARY_CSV = OUT_DIR / "outside_hours_entry_summary.csv"
DETAIL_CSV = OUT_DIR / "outside_hours_entry_detail.csv"
FILL_CSV = OUT_DIR / "outside_hours_entry_fill_breakdown.csv"
FIXED_SUMMARY_CSV = OUT_DIR / "outside_hours_entry_fixed_exit_summary.csv"


def filing_bucket(filing_dt: datetime) -> str:
    if filing_dt.weekday() >= 5:
        return "weekend"
    hm = (filing_dt.hour, filing_dt.minute)
    if hm < (9, 30):
        return "pre_market"
    if hm >= (16, 0):
        return "after_hours"
    return "intraday"


def compute_return_pct(buy_price: float, sell_price: Optional[float], horizon_days: int) -> Optional[float]:
    if sell_price is None or buy_price is None or buy_price <= 0:
        return None
    gross = float(sell_price) / float(buy_price)
    if gross <= 0:
        return None
    if horizon_days <= 1:
        ret = gross - 1.0
    else:
        ret = gross ** (1.0 / float(horizon_days)) - 1.0
    return ret * 100.0


def trading_dates_from_daily(daily_bars: List[dict]) -> List[date]:
    out: List[date] = []
    for bar in daily_bars:
        bar_date = datetime.fromtimestamp(bar["t"] / 1000, tz=ET).date()
        out.append(bar_date)
    return sorted(set(out))


def next_trading_dates(trading_dates: List[date], start_date: date, n: int = 3) -> List[date]:
    idx = bisect_left(trading_dates, start_date)
    return trading_dates[idx : idx + n]


def first_bar_on_or_after(
    client: RESTClient,
    cache: PriceCache,
    ticker: str,
    filing_dt: datetime,
    trading_dates: List[date],
) -> Optional[dict]:
    candidate_dates = next_trading_dates(trading_dates, filing_dt.date(), n=3)
    if not candidate_dates:
        return None

    filing_ts_ms = int(filing_dt.timestamp() * 1000)
    for i, d in enumerate(candidate_dates):
        bars = fetch_minute_bars(client, cache, ticker, d)
        threshold = filing_ts_ms if i == 0 else 0
        for bar in bars:
            if bar.get("t", 0) >= threshold and bar.get("c"):
                fill_dt = datetime.fromtimestamp(bar["t"] / 1000, tz=ET)
                regular = time(9, 30) <= fill_dt.timetz().replace(tzinfo=None) < time(16, 0)
                return {
                    "fill_dt": fill_dt,
                    "fill_price": float(bar["c"]),
                    "fill_bucket": "regular_hours" if regular else "extended_hours",
                    "fill_date": fill_dt.date(),
                }
    return None


def process_ticker(
    ticker: str,
    rows: List[dict],
    api_key: str,
    cache: PriceCache,
) -> List[dict]:
    client = RESTClient(api_key=api_key, retries=3)
    filing_dates = [row["transaction_date"].date() for row in rows]
    daily_bars = fetch_daily_bars(
        client,
        cache,
        ticker,
        min(filing_dates) - timedelta(days=3),
        max(filing_dates) + timedelta(days=30),
    )
    if not daily_bars:
        return []

    trading_dates = trading_dates_from_daily(daily_bars)
    out: List[dict] = []

    for row in rows:
        filing_dt = row["transaction_date"]
        fill = first_bar_on_or_after(client, cache, ticker, filing_dt, trading_dates)
        if fill is None:
            continue

        rec = {
            "row_id": row["row_id"],
            "ticker": ticker,
            "filing_bucket": row["filing_bucket"],
            "transaction_date": filing_dt.isoformat(),
            "baseline_buy_datetime": row["buy_datetime"].isoformat(),
            "baseline_buy_price": float(row["buy_price"]),
            "alt_buy_datetime": fill["fill_dt"].isoformat(),
            "alt_buy_price": float(fill["fill_price"]),
            "alt_fill_bucket": fill["fill_bucket"],
            "alt_buy_date": fill["fill_date"].isoformat(),
        }
        baseline_buy_date = row["buy_datetime"].date()

        for horizon in SELL_HORIZONS:
            base_ret = row[f"return_{horizon}d_pct"]
            alt_close = find_nth_trading_day_close(daily_bars, fill["fill_date"], horizon)
            alt_ret = compute_return_pct(fill["fill_price"], alt_close, horizon)
            fixed_close = find_nth_trading_day_close(daily_bars, baseline_buy_date, horizon)
            alt_fixed_ret = compute_return_pct(fill["fill_price"], fixed_close, horizon)
            rec[f"baseline_return_{horizon}d_pct"] = float(base_ret) if pd.notna(base_ret) else math.nan
            rec[f"alt_return_{horizon}d_pct"] = float(alt_ret) if alt_ret is not None else math.nan
            rec[f"delta_{horizon}d_pct"] = (
                float(alt_ret) - float(base_ret)
                if alt_ret is not None and pd.notna(base_ret)
                else math.nan
            )
            rec[f"alt_fixed_exit_return_{horizon}d_pct"] = (
                float(alt_fixed_ret) if alt_fixed_ret is not None else math.nan
            )
            rec[f"fixed_exit_delta_{horizon}d_pct"] = (
                float(alt_fixed_ret) - float(base_ret)
                if alt_fixed_ret is not None and pd.notna(base_ret)
                else math.nan
            )

        out.append(rec)
    return out


def summarize(detail_df: pd.DataFrame, mode: str = "shifted") -> pd.DataFrame:
    if mode == "shifted":
        alt_prefix = "alt_return"
        delta_prefix = "delta"
    elif mode == "fixed":
        alt_prefix = "alt_fixed_exit_return"
        delta_prefix = "fixed_exit_delta"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    rows = []
    for filing_bucket in sorted(detail_df["filing_bucket"].dropna().unique()):
        sub = detail_df[detail_df["filing_bucket"] == filing_bucket].copy()
        rec = {"filing_bucket": filing_bucket, "trades": int(len(sub))}
        ext_share = (sub["alt_fill_bucket"] == "extended_hours").mean() * 100.0 if len(sub) else math.nan
        rec["alt_extended_fill_rate_pct"] = ext_share
        for horizon in SELL_HORIZONS:
            bcol = f"baseline_return_{horizon}d_pct"
            acol = f"{alt_prefix}_{horizon}d_pct"
            dcol = f"{delta_prefix}_{horizon}d_pct"
            b = sub[bcol].dropna()
            a = sub[acol].dropna()
            d = sub[dcol].dropna()
            rec[f"baseline_mean_{horizon}d"] = float(b.mean()) if len(b) else math.nan
            rec[f"alt_mean_{horizon}d"] = float(a.mean()) if len(a) else math.nan
            rec[f"mean_delta_{horizon}d"] = float(d.mean()) if len(d) else math.nan
            rec[f"median_delta_{horizon}d"] = float(d.median()) if len(d) else math.nan
            rec[f"win_rate_delta_{horizon}d_pct"] = float((d > 0).mean() * 100.0) if len(d) else math.nan
        rows.append(rec)

    overall = {"filing_bucket": "overall", "trades": int(len(detail_df))}
    ext_share = (detail_df["alt_fill_bucket"] == "extended_hours").mean() * 100.0 if len(detail_df) else math.nan
    overall["alt_extended_fill_rate_pct"] = ext_share
    for horizon in SELL_HORIZONS:
        bcol = f"baseline_return_{horizon}d_pct"
        acol = f"{alt_prefix}_{horizon}d_pct"
        dcol = f"{delta_prefix}_{horizon}d_pct"
        b = detail_df[bcol].dropna()
        a = detail_df[acol].dropna()
        d = detail_df[dcol].dropna()
        overall[f"baseline_mean_{horizon}d"] = float(b.mean()) if len(b) else math.nan
        overall[f"alt_mean_{horizon}d"] = float(a.mean()) if len(a) else math.nan
        overall[f"mean_delta_{horizon}d"] = float(d.mean()) if len(d) else math.nan
        overall[f"median_delta_{horizon}d"] = float(d.median()) if len(d) else math.nan
        overall[f"win_rate_delta_{horizon}d_pct"] = float((d > 0).mean() * 100.0) if len(d) else math.nan
    rows.append(overall)
    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise SystemExit("Set POLYGON_API_KEY in .env")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = PriceCache(CACHE_DIR)

    df = pd.read_csv(INPUT_CSV)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["buy_datetime"] = pd.to_datetime(df["buy_datetime"], errors="coerce")
    df = df.dropna(subset=["transaction_date", "buy_datetime", "buy_price"]).copy()
    df["transaction_date"] = df["transaction_date"].apply(lambda ts: ts.to_pydatetime().replace(tzinfo=ET))
    df["buy_datetime"] = df["buy_datetime"].apply(lambda ts: ts.to_pydatetime().replace(tzinfo=ET))
    df["filing_bucket"] = df["transaction_date"].apply(filing_bucket)
    df = df[df["filing_bucket"].isin(["pre_market", "after_hours", "weekend"])].copy()
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    groups: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        groups.setdefault(row["ticker"], []).append(
            {
                "row_id": int(row["row_id"]),
                "transaction_date": row["transaction_date"],
                "buy_datetime": row["buy_datetime"],
                "buy_price": float(row["buy_price"]),
                "filing_bucket": row["filing_bucket"],
                **{f"return_{h}d_pct": row.get(f"return_{h}d_pct") for h in SELL_HORIZONS},
            }
        )

    all_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_ticker, ticker, rows, api_key, cache): ticker
            for ticker, rows in groups.items()
        }
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                print(f"[WARN] {ticker}: {exc}")

    if not all_rows:
        raise SystemExit("No outside-hours comparison rows produced.")

    detail_df = pd.DataFrame(all_rows).sort_values(["filing_bucket", "transaction_date", "ticker"])
    summary_df = summarize(detail_df, mode="shifted")
    fixed_summary_df = summarize(detail_df, mode="fixed")
    fill_df = (
        detail_df.groupby(["filing_bucket", "alt_fill_bucket"], dropna=False)
        .size()
        .reset_index(name="trades")
        .sort_values(["filing_bucket", "alt_fill_bucket"])
    )

    detail_df.to_csv(DETAIL_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    fixed_summary_df.to_csv(FIXED_SUMMARY_CSV, index=False)
    fill_df.to_csv(FILL_CSV, index=False)

    print("\nSaved:")
    print(f"  {DETAIL_CSV}")
    print(f"  {SUMMARY_CSV}")
    print(f"  {FIXED_SUMMARY_CSV}")
    print(f"  {FILL_CSV}")
    print("\nSummary (shifted exit):")
    print(summary_df.to_string(index=False))
    print("\nSummary (fixed baseline exit):")
    print(fixed_summary_df.to_string(index=False))
    print("\nFill breakdown:")
    print(fill_df.to_string(index=False))


if __name__ == "__main__":
    main()
