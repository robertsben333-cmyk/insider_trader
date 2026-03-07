from __future__ import annotations

import argparse
import csv
from datetime import datetime, time, timedelta
from pathlib import Path
from statistics import median
import os
import sys

from dotenv import load_dotenv
from polygon import RESTClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_vm_recommendations import (
    ET,
    MARKET_CLOSE,
    MARKET_OPEN,
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
from live_trading.strategy_settings import LIVE_PATHS, RUNTIME_DEFAULTS


def next_business_day_close(entry_dt_et: datetime) -> datetime:
    d = next_weekday(entry_dt_et.date() + timedelta(days=1))
    return datetime.combine(d, time(16, 0), tzinfo=ET)


def trading_dates_between(start_date, end_date) -> list:
    out = []
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def bar_dt_et(bar: dict) -> datetime | None:
    ts = bar.get("t")
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts) / 1000, tz=UTC).astimezone(ET)


def find_entry_bar_index(minute_bars: list[dict], entry_dt_et: datetime) -> int | None:
    entry_ts_ms = int(entry_dt_et.timestamp() * 1000)
    for idx, bar in enumerate(minute_bars):
        ts = bar.get("t")
        close = safe_float(str(bar.get("c", "")))
        if ts is None or close is None:
            continue
        if int(ts) >= entry_ts_ms:
            return idx
    return None


def find_same_day_close_index(minute_bars: list[dict], entry_idx: int) -> int | None:
    entry_dt = bar_dt_et(minute_bars[entry_idx])
    if entry_dt is None:
        return None
    entry_date = entry_dt.date()
    last_idx = None
    for idx in range(entry_idx, len(minute_bars)):
        current_dt = bar_dt_et(minute_bars[idx])
        if current_dt is None:
            continue
        if current_dt.date() != entry_date:
            break
        last_idx = idx
    return last_idx


def find_next_day_open_index(minute_bars: list[dict], entry_idx: int) -> int | None:
    entry_dt = bar_dt_et(minute_bars[entry_idx])
    if entry_dt is None:
        return None
    entry_date = entry_dt.date()
    for idx in range(entry_idx + 1, len(minute_bars)):
        current_dt = bar_dt_et(minute_bars[idx])
        if current_dt is None:
            continue
        if current_dt.date() > entry_date:
            return idx
    return None


def is_regular_session_bar(bar: dict) -> bool:
    dt = bar_dt_et(bar)
    if dt is None:
        return False
    if dt.weekday() >= 5:
        return False
    return MARKET_OPEN <= dt.time() < MARKET_CLOSE


def collect_minute_bars(
    client: RESTClient,
    cache_dir: Path,
    ticker: str,
    from_date,
    to_date,
    minute_cache: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    out: list[dict] = []
    for day in trading_dates_between(from_date, to_date):
        cache_key = (ticker, f"{day:%Y-%m-%d}")
        if cache_key not in minute_cache:
            minute_cache[cache_key] = fetch_minute_bars(client, cache_dir, ticker, day)
        out.extend(minute_cache[cache_key])
    out = [bar for bar in out if is_regular_session_bar(bar)]
    out.sort(key=lambda b: int(b["t"]))
    return out


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pct_positive(values: list[float]) -> float | None:
    if not values:
        return None
    return 100.0 * sum(v > 0 for v in values) / len(values)


def fmt(value: float | None) -> float | str:
    if value is None:
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze early loser-identification signals for VM historical recommendations."
    )
    parser.add_argument("--input", default=LIVE_PATHS.vm_predictions_file)
    parser.add_argument("--cache-dir", default=LIVE_PATHS.cache_dir)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[15, 30, 60, 120],
        help="Trading-minute checkpoints after entry.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(RUNTIME_DEFAULTS.early_exit_thresholds),
        help="Trigger threshold on checkpoint return percentage.",
    )
    parser.add_argument(
        "--summary-out",
        default=LIVE_PATHS.vm_early_exit_summary_file,
    )
    parser.add_argument(
        "--detail-out",
        default=LIVE_PATHS.vm_early_exit_detail_file,
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    rows = load_rows(Path(args.input))
    cache_dir = Path(args.cache_dir)
    client = RESTClient(api_key=api_key, retries=3)
    minute_cache: dict[tuple[str, str], list[dict]] = {}
    day_cache: dict[tuple[str, str], list[dict]] = {}

    detail_rows: list[dict] = []

    for row in rows:
        ticker = str(row["ticker"])
        scored_utc = parse_scored_at_utc(str(row["scored_at"]))
        entry_target = intended_entry_from_score(scored_utc)
        entry_dt, entry_price, entry_source = resolve_entry(client, cache_dir, ticker, entry_target)
        if entry_dt is None or entry_price is None or entry_price <= 0:
            continue

        exit_dt = next_business_day_close(entry_dt)
        minute_bars = collect_minute_bars(
            client=client,
            cache_dir=cache_dir,
            ticker=ticker,
            from_date=entry_dt.date(),
            to_date=exit_dt.date(),
            minute_cache=minute_cache,
        )
        if not minute_bars:
            continue

        entry_idx = find_entry_bar_index(minute_bars, entry_dt)
        if entry_idx is None:
            continue

        day_key = (ticker, f"{entry_dt.date():%Y-%m-%d}_{exit_dt.date():%Y-%m-%d}")
        if day_key not in day_cache:
            day_cache[day_key] = fetch_day_bars(client, cache_dir, ticker, entry_dt.date(), exit_dt.date())

        final_close = None
        for bar in day_cache[day_key]:
            bar_dt = bar_dt_et(bar)
            if bar_dt is None:
                continue
            if bar_dt.date() == exit_dt.date():
                final_close = safe_float(str(bar.get("c", "")))
                break
        if final_close is None or final_close <= 0:
            continue

        entry_bar_dt = bar_dt_et(minute_bars[entry_idx])
        if entry_bar_dt is None:
            continue

        same_day_close_idx = find_same_day_close_index(minute_bars, entry_idx)
        next_day_open_idx = find_next_day_open_index(minute_bars, entry_idx)
        final_ret = ((final_close / entry_price) - 1.0) * 100.0

        out = {
            "scored_at": row["scored_at"],
            "scored_at_et": scored_utc.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S%z"),
            "ticker": ticker,
            "event_key": row["event_key"],
            "trade_date": row.get("trade_date", ""),
            "score_1d": row.get("score_1d", ""),
            "score_3d": row.get("score_3d", ""),
            "entry_dt_et": entry_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "entry_bar_dt_et": entry_bar_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "entry_price": entry_price,
            "entry_price_source": entry_source,
            "scheduled_exit_dt_et": exit_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "scheduled_exit_close": final_close,
            "final_ret_1d_pct": final_ret,
        }

        if same_day_close_idx is not None:
            same_day_close = safe_float(str(minute_bars[same_day_close_idx].get("c", "")))
            if same_day_close is not None:
                out["same_day_close_dt_et"] = bar_dt_et(minute_bars[same_day_close_idx]).strftime("%Y-%m-%d %H:%M:%S%z")
                out["ret_same_day_close_pct"] = ((same_day_close / entry_price) - 1.0) * 100.0
            else:
                out["same_day_close_dt_et"] = ""
                out["ret_same_day_close_pct"] = ""
        else:
            out["same_day_close_dt_et"] = ""
            out["ret_same_day_close_pct"] = ""

        if next_day_open_idx is not None:
            next_day_open = safe_float(str(minute_bars[next_day_open_idx].get("c", "")))
            if next_day_open is not None:
                out["next_day_open_dt_et"] = bar_dt_et(minute_bars[next_day_open_idx]).strftime("%Y-%m-%d %H:%M:%S%z")
                out["ret_next_day_open_pct"] = ((next_day_open / entry_price) - 1.0) * 100.0
            else:
                out["next_day_open_dt_et"] = ""
                out["ret_next_day_open_pct"] = ""
        else:
            out["next_day_open_dt_et"] = ""
            out["ret_next_day_open_pct"] = ""

        for checkpoint in args.checkpoints:
            idx = entry_idx + checkpoint
            dt_col = f"checkpoint_{checkpoint}m_dt_et"
            ret_col = f"ret_{checkpoint}m_pct"
            if idx < len(minute_bars):
                checkpoint_close = safe_float(str(minute_bars[idx].get("c", "")))
                checkpoint_dt = bar_dt_et(minute_bars[idx])
                out[dt_col] = checkpoint_dt.strftime("%Y-%m-%d %H:%M:%S%z") if checkpoint_dt is not None else ""
                out[ret_col] = ((checkpoint_close / entry_price) - 1.0) * 100.0 if checkpoint_close is not None else ""
            else:
                out[dt_col] = ""
                out[ret_col] = ""

        detail_rows.append(out)

    detail_rows.sort(key=lambda r: (r["scored_at"], str(r["ticker"])), reverse=True)

    signal_specs: list[tuple[str, str, int | str]] = []
    for checkpoint in args.checkpoints:
        signal_specs.append((f"ret_{checkpoint}m_pct", "checkpoint_return", checkpoint))
    signal_specs.extend(
        [
            ("ret_same_day_close_pct", "session_return", "same_day_close"),
            ("ret_next_day_open_pct", "session_return", "next_day_open"),
        ]
    )

    summary_rows: list[dict] = []
    for signal_col, signal_family, signal_label in signal_specs:
        eligible = [r for r in detail_rows if str(r.get(signal_col, "")) != "" and str(r.get("final_ret_1d_pct", "")) != ""]
        if not eligible:
            continue

        baseline_returns = [float(r["final_ret_1d_pct"]) for r in eligible]
        total_losers = sum(float(r["final_ret_1d_pct"]) < 0 for r in eligible)

        for threshold in args.thresholds:
            flagged = [r for r in eligible if float(r[signal_col]) <= threshold]
            unflagged = [r for r in eligible if float(r[signal_col]) > threshold]
            strategy_returns = [
                float(r[signal_col]) if float(r[signal_col]) <= threshold else float(r["final_ret_1d_pct"])
                for r in eligible
            ]
            flagged_final_returns = [float(r["final_ret_1d_pct"]) for r in flagged]
            unflagged_final_returns = [float(r["final_ret_1d_pct"]) for r in unflagged]
            flagged_losers = sum(v < 0 for v in flagged_final_returns)

            summary_rows.append(
                {
                    "signal_family": signal_family,
                    "signal": signal_col,
                    "signal_label": signal_label,
                    "threshold_pct": threshold,
                    "n_trades": len(eligible),
                    "n_flagged": len(flagged),
                    "flag_rate_pct": 100.0 * len(flagged) / len(eligible),
                    "loser_precision_pct": fmt(100.0 * flagged_losers / len(flagged) if flagged else None),
                    "loser_recall_pct": fmt(100.0 * flagged_losers / total_losers if total_losers else None),
                    "baseline_mean_ret_pct": mean(baseline_returns),
                    "strategy_mean_ret_pct": mean(strategy_returns),
                    "delta_mean_ret_pct": fmt(mean(strategy_returns) - mean(baseline_returns) if baseline_returns else None),
                    "baseline_median_ret_pct": median(baseline_returns),
                    "strategy_median_ret_pct": median(strategy_returns),
                    "baseline_win_rate_pct": pct_positive(baseline_returns),
                    "strategy_win_rate_pct": pct_positive(strategy_returns),
                    "mean_final_ret_flagged_pct": fmt(mean(flagged_final_returns)),
                    "mean_signal_ret_flagged_pct": fmt(mean([float(r[signal_col]) for r in flagged])),
                    "mean_final_ret_unflagged_pct": fmt(mean(unflagged_final_returns)),
                }
            )

    summary_rows.sort(
        key=lambda r: (
            float(r["delta_mean_ret_pct"]) if r["delta_mean_ret_pct"] != "" else float("-inf"),
            int(r["n_flagged"]),
        ),
        reverse=True,
    )

    detail_out = Path(args.detail_out)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    with detail_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()) if detail_rows else [])
        if detail_rows:
            writer.writeheader()
            writer.writerows(detail_rows)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "signal_family",
                "signal",
                "signal_label",
                "threshold_pct",
                "n_trades",
                "n_flagged",
                "flag_rate_pct",
                "loser_precision_pct",
                "loser_recall_pct",
                "baseline_mean_ret_pct",
                "strategy_mean_ret_pct",
                "delta_mean_ret_pct",
                "baseline_median_ret_pct",
                "strategy_median_ret_pct",
                "baseline_win_rate_pct",
                "strategy_win_rate_pct",
                "mean_final_ret_flagged_pct",
                "mean_signal_ret_flagged_pct",
                "mean_final_ret_unflagged_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"detail_rows={len(detail_rows)}")
    print(f"summary_rows={len(summary_rows)}")
    print(f"detail_out={detail_out}")
    print(f"summary_out={summary_out}")
    for row in summary_rows[:12]:
        print(
            "signal={signal} threshold={threshold_pct} n={n_flagged}/{n_trades} "
            "precision={loser_precision_pct} recall={loser_recall_pct} "
            "strategy_mean={strategy_mean_ret_pct} delta={delta_mean_ret_pct}".format(**row)
        )


if __name__ == "__main__":
    main()
