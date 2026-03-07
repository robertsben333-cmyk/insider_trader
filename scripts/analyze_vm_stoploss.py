from __future__ import annotations

import argparse
import csv
from datetime import datetime, time, timedelta
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo
import sys

from dotenv import load_dotenv
from polygon import RESTClient
import os

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_vm_recommendations import (
    ET,
    UTC,
    fetch_day_bars,
    fetch_minute_bars,
    intended_entry_from_score,
    load_rows,
    next_weekday,
    parse_scored_at_utc,
    resolve_entry,
    safe_float,
)
from live_trading.strategy_settings import LIVE_PATHS


def next_business_day_close(entry_dt_et: datetime) -> datetime:
    d = next_weekday(entry_dt_et.date() + timedelta(days=1))
    return datetime.combine(d, time(16, 0), tzinfo=ET)


def simulate_stop_return(
    minute_bars: list[dict],
    entry_dt_et: datetime,
    entry_price: float,
    exit_dt_et: datetime,
    scheduled_exit_close: float,
    stop_loss_pct: float | None,
) -> tuple[float, str]:
    if stop_loss_pct is None:
        return ((scheduled_exit_close / entry_price) - 1.0) * 100.0, "scheduled_exit"

    stop_price = entry_price * (1.0 - stop_loss_pct / 100.0)
    entry_ts_ms = int(entry_dt_et.timestamp() * 1000)
    exit_ts_ms = int(exit_dt_et.timestamp() * 1000)

    bars = [b for b in minute_bars if int(b["t"]) > entry_ts_ms and int(b["t"]) <= exit_ts_ms]
    for bar in bars:
        bar_open = safe_float(str(bar.get("o", "")))
        bar_low = safe_float(str(bar.get("l", "")))
        if bar_open is None or bar_low is None:
            continue
        if bar_open <= stop_price:
            return ((bar_open / entry_price) - 1.0) * 100.0, "gap_stop_open"
        if bar_low <= stop_price:
            return ((stop_price / entry_price) - 1.0) * 100.0, "stop_intraday"

    return ((scheduled_exit_close / entry_price) - 1.0) * 100.0, "scheduled_exit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stop-loss sensitivity for VM historical recommendations.")
    parser.add_argument("--input", default=LIVE_PATHS.vm_predictions_file)
    parser.add_argument("--cache-dir", default=LIVE_PATHS.cache_dir)
    parser.add_argument("--summary-out", default=LIVE_PATHS.vm_stoploss_summary_file)
    parser.add_argument("--detail-out", default=LIVE_PATHS.vm_stoploss_detail_file)
    parser.add_argument("--stop-levels", nargs="+", type=float, default=[2, 3, 5, 7, 10, 15])
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    rows = load_rows(Path(args.input))
    cache_dir = Path(args.cache_dir)
    client = RESTClient(api_key=api_key, retries=3)

    detail_rows: list[dict] = []
    day_cache: dict[tuple[str, str], list[dict]] = {}
    minute_cache: dict[tuple[str, str], list[dict]] = {}

    stop_levels: list[float | None] = [None] + list(args.stop_levels)

    for row in rows:
        ticker = str(row["ticker"])
        scored_utc = parse_scored_at_utc(str(row["scored_at"]))
        entry_target = intended_entry_from_score(scored_utc)
        entry_dt, entry_price, entry_source = resolve_entry(client, cache_dir, ticker, entry_target)
        if entry_dt is None or entry_price is None or entry_price <= 0:
            continue

        exit_dt = next_business_day_close(entry_dt)

        hold_dates = []
        cur = entry_dt.date()
        while cur <= exit_dt.date():
            if cur.weekday() < 5:
                hold_dates.append(cur)
            cur += timedelta(days=1)

        all_minute_bars: list[dict] = []
        for d in hold_dates:
            mkey = (ticker, f"{d:%Y-%m-%d}")
            if mkey not in minute_cache:
                minute_cache[mkey] = fetch_minute_bars(client, cache_dir, ticker, d)
            all_minute_bars.extend(minute_cache[mkey])
        all_minute_bars.sort(key=lambda b: int(b["t"]))

        from_d = entry_dt.date()
        to_d = exit_dt.date()
        dkey = (ticker, f"{from_d:%Y-%m-%d}_{to_d:%Y-%m-%d}")
        if dkey not in day_cache:
            day_cache[dkey] = fetch_day_bars(client, cache_dir, ticker, from_d, to_d)
        exit_close = None
        for bar in day_cache[dkey]:
            ts = datetime.fromtimestamp(int(bar["t"]) / 1000, tz=UTC).astimezone(ET)
            if ts.date() == exit_dt.date():
                exit_close = safe_float(str(bar.get("c", "")))
                break
        if exit_close is None or exit_close <= 0:
            continue

        base = {
            "scored_at": row["scored_at"],
            "ticker": ticker,
            "event_key": row["event_key"],
            "score_1d": row.get("score_1d", ""),
            "score_3d": row.get("score_3d", ""),
            "entry_dt_et": entry_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "entry_price": entry_price,
            "entry_price_source": entry_source,
            "scheduled_exit_dt_et": exit_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "scheduled_exit_close": exit_close,
        }

        for stop in stop_levels:
            ret_pct, exit_reason = simulate_stop_return(
                minute_bars=all_minute_bars,
                entry_dt_et=entry_dt,
                entry_price=entry_price,
                exit_dt_et=exit_dt,
                scheduled_exit_close=exit_close,
                stop_loss_pct=stop,
            )
            row_out = dict(base)
            row_out["stop_loss_pct"] = "" if stop is None else stop
            row_out["ret_pct"] = ret_pct
            row_out["exit_reason"] = exit_reason
            detail_rows.append(row_out)

    detail_rows.sort(
        key=lambda r: (
            str(r["stop_loss_pct"]),
            str(r["scored_at"]),
            str(r["ticker"]),
        ),
        reverse=True,
    )

    summary_rows: list[dict] = []
    for stop in stop_levels:
        subset = [r for r in detail_rows if ((r["stop_loss_pct"] == "") if stop is None else (r["stop_loss_pct"] == stop))]
        vals = [float(r["ret_pct"]) for r in subset]
        if not vals:
            continue
        stopped = sum(r["exit_reason"] != "scheduled_exit" for r in subset)
        summary_rows.append(
            {
                "stop_loss_pct": "none" if stop is None else stop,
                "n_trades": len(vals),
                "mean_return_pct": sum(vals) / len(vals),
                "median_return_pct": median(vals),
                "win_rate_pct": 100.0 * sum(v > 0 for v in vals) / len(vals),
                "stopped_pct": 100.0 * stopped / len(vals),
                "min_return_pct": min(vals),
                "max_return_pct": max(vals),
            }
        )

    detail_out = Path(args.detail_out)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    with detail_out.open("w", newline="", encoding="utf-8") as f:
        if detail_rows:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stop_loss_pct",
                "n_trades",
                "mean_return_pct",
                "median_return_pct",
                "win_rate_pct",
                "stopped_pct",
                "min_return_pct",
                "max_return_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"detail_rows={len(detail_rows)}")
    print(f"detail_out={detail_out}")
    print(f"summary_out={summary_out}")
    for row in summary_rows:
        print(
            "stop={stop_loss_pct} n={n_trades} mean={mean_return_pct} median={median_return_pct} "
            "win_rate={win_rate_pct} stopped={stopped_pct}".format(**row)
        )


if __name__ == "__main__":
    main()
