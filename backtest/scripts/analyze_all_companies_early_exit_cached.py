from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUT_DIR = BASE / "backtest" / "out"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

if str(BASE) not in os.sys.path:
    os.sys.path.insert(0, str(BASE))

from backtest.scripts.evaluate_investable_risk_rules import (  # noqa: E402
    chrono_split_60_20_20,
    load_scored_day1,
)


def cache_path(cache_dir: Path, ticker: str, d_str: str) -> Path:
    return cache_dir / f"{ticker}_min_{d_str}_{d_str}.json"


def load_cached_minute_bars(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def bar_dt_et(bar: dict) -> datetime:
    return datetime.fromtimestamp(int(bar["t"]) / 1000, tz=UTC).astimezone(ET)


def is_regular_session_bar(bar: dict) -> bool:
    dt = bar_dt_et(bar)
    return dt.weekday() < 5 and MARKET_OPEN <= dt.time() < MARKET_CLOSE


def find_entry_bar_index(minute_bars: list[dict], entry_dt: datetime) -> int | None:
    entry_ts = int(entry_dt.timestamp() * 1000)
    for idx, bar in enumerate(minute_bars):
        if int(bar["t"]) >= entry_ts:
            return idx
    return None


def find_same_day_close_index(minute_bars: list[dict], entry_idx: int) -> int | None:
    entry_date = bar_dt_et(minute_bars[entry_idx]).date()
    last_idx = None
    for idx in range(entry_idx, len(minute_bars)):
        if bar_dt_et(minute_bars[idx]).date() != entry_date:
            break
        last_idx = idx
    return last_idx


def compute_signal_row(row: pd.Series, checkpoints: list[int], cache_dir: Path) -> dict | None:
    ticker = str(row["ticker"])
    buy_dt = pd.Timestamp(row["buy_datetime"]).to_pydatetime().replace(tzinfo=ET)
    day_str = pd.Timestamp(row["buy_datetime"]).strftime("%Y-%m-%d")
    path = cache_path(cache_dir, ticker, day_str)
    if not path.exists():
        return None

    bars = [b for b in load_cached_minute_bars(path) if is_regular_session_bar(b)]
    bars.sort(key=lambda b: int(b["t"]))
    if not bars:
        return None

    entry_idx = find_entry_bar_index(bars, buy_dt)
    if entry_idx is None:
        return None

    entry_price = float(bars[entry_idx]["o"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    out = {
        "split": row["split"],
        "ticker": ticker,
        "buy_datetime": pd.Timestamp(row["buy_datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
        "ret_1d_pct": float(row["ret_pct"]),
        "entry_price": entry_price,
        "entry_bar_time": bar_dt_et(bars[entry_idx]).strftime("%Y-%m-%d %H:%M:%S"),
    }

    for checkpoint in checkpoints:
        idx = entry_idx + checkpoint
        ret_col = f"ret_{checkpoint}m_pct"
        dt_col = f"checkpoint_{checkpoint}m_dt"
        if idx < len(bars):
            px = float(bars[idx]["c"])
            out[ret_col] = (px / entry_price - 1.0) * 100.0
            out[dt_col] = bar_dt_et(bars[idx]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            out[ret_col] = np.nan
            out[dt_col] = ""

    same_day_close_idx = find_same_day_close_index(bars, entry_idx)
    if same_day_close_idx is not None:
        px = float(bars[same_day_close_idx]["c"])
        out["ret_same_day_close_pct"] = (px / entry_price - 1.0) * 100.0
        out["same_day_close_dt"] = bar_dt_et(bars[same_day_close_idx]).strftime("%Y-%m-%d %H:%M:%S")
    else:
        out["ret_same_day_close_pct"] = np.nan
        out["same_day_close_dt"] = ""

    return out


def summarize_rule(detail: pd.DataFrame, signal_col: str, threshold: float) -> dict | None:
    eligible = detail[detail[signal_col].notna()].copy()
    if eligible.empty:
        return None

    baseline = eligible["ret_1d_pct"].to_numpy(dtype=float)
    signal = eligible[signal_col].to_numpy(dtype=float)
    flagged = signal <= float(threshold)
    strategy = np.where(flagged, signal, baseline)
    losers = baseline < 0
    flagged_count = int(flagged.sum())
    flagged_losers = int(np.logical_and(flagged, losers).sum())
    total_losers = int(losers.sum())

    return {
        "signal": signal_col,
        "threshold_pct": float(threshold),
        "n_trades": int(len(eligible)),
        "n_flagged": flagged_count,
        "flag_rate_pct": float(flagged_count / len(eligible) * 100.0),
        "loser_precision_pct": float(flagged_losers / flagged_count * 100.0) if flagged_count else np.nan,
        "loser_recall_pct": float(flagged_losers / total_losers * 100.0) if total_losers else np.nan,
        "baseline_mean_ret_pct": float(np.mean(baseline)),
        "strategy_mean_ret_pct": float(np.mean(strategy)),
        "delta_mean_ret_pct": float(np.mean(strategy) - np.mean(baseline)),
        "baseline_median_ret_pct": float(np.median(baseline)),
        "strategy_median_ret_pct": float(np.median(strategy)),
    }


def apply_rule(detail: pd.DataFrame, signal_col: str, threshold: float) -> pd.DataFrame:
    out = detail.copy()
    if signal_col == "none":
        out["selected_signal"] = "none"
        out["selected_threshold_pct"] = ""
        out["flagged"] = False
        out["strategy_ret_pct"] = out["ret_1d_pct"]
        out["improvement_pct"] = 0.0
        return out

    signal = pd.to_numeric(out[signal_col], errors="coerce")
    out["selected_signal"] = signal_col
    out["selected_threshold_pct"] = float(threshold)
    out["flagged"] = (signal <= float(threshold)).fillna(False)
    out["strategy_ret_pct"] = np.where(out["flagged"], signal, out["ret_1d_pct"])
    out["improvement_pct"] = out["strategy_ret_pct"] - out["ret_1d_pct"]
    return out


def fmt(value) -> str | float:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache-only early-exit analysis on all companies using the original chrono val/test split."
    )
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[15, 30, 60, 120])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-0.5, -1.0, -2.0, -3.0, -5.0])
    parser.add_argument("--detail-out", default=str(OUT_DIR / "all_companies_early_exit_cached_detail.csv"))
    parser.add_argument("--rule-out", default=str(OUT_DIR / "all_companies_early_exit_cached_rule_grid.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "all_companies_early_exit_cached_summary.csv"))
    args = parser.parse_args()

    scored = load_scored_day1()
    n_train, split_80 = chrono_split_60_20_20(len(scored))
    subset = scored.iloc[n_train:].copy()
    subset["split"] = np.where(np.arange(len(subset)) < (split_80 - n_train), "val", "test")
    subset["ret_pct"] = pd.to_numeric(subset["ret_pct"], errors="coerce")
    subset = subset.dropna(subset=["buy_datetime", "ret_pct"]).copy()

    cache_dir = Path(args.cache_dir)
    detail_rows = []
    missing_cache = 0
    for _, row in subset.iterrows():
        out = compute_signal_row(row, args.checkpoints, cache_dir)
        if out is None:
            missing_cache += 1
            continue
        detail_rows.append(out)

    detail = pd.DataFrame(detail_rows)
    signal_cols = [f"ret_{checkpoint}m_pct" for checkpoint in args.checkpoints] + ["ret_same_day_close_pct"]

    val_detail = detail[detail["split"] == "val"].copy()
    test_detail = detail[detail["split"] == "test"].copy()

    grid_rows = []
    for signal_col in signal_cols:
        for threshold in args.thresholds:
            val_metrics = summarize_rule(val_detail, signal_col, threshold)
            if val_metrics is None:
                continue
            test_metrics = summarize_rule(test_detail, signal_col, threshold)
            grid_rows.append(
                {
                    "signal": signal_col,
                    "threshold_pct": threshold,
                    "val_n_trades": val_metrics["n_trades"],
                    "val_n_flagged": val_metrics["n_flagged"],
                    "val_flag_rate_pct": val_metrics["flag_rate_pct"],
                    "val_loser_precision_pct": fmt(val_metrics["loser_precision_pct"]),
                    "val_loser_recall_pct": fmt(val_metrics["loser_recall_pct"]),
                    "val_baseline_mean_ret_pct": val_metrics["baseline_mean_ret_pct"],
                    "val_strategy_mean_ret_pct": val_metrics["strategy_mean_ret_pct"],
                    "val_delta_mean_ret_pct": val_metrics["delta_mean_ret_pct"],
                    "test_n_trades": test_metrics["n_trades"] if test_metrics else 0,
                    "test_n_flagged": test_metrics["n_flagged"] if test_metrics else 0,
                    "test_flag_rate_pct": fmt(test_metrics["flag_rate_pct"] if test_metrics else np.nan),
                    "test_loser_precision_pct": fmt(test_metrics["loser_precision_pct"] if test_metrics else np.nan),
                    "test_loser_recall_pct": fmt(test_metrics["loser_recall_pct"] if test_metrics else np.nan),
                    "test_baseline_mean_ret_pct": fmt(test_metrics["baseline_mean_ret_pct"] if test_metrics else np.nan),
                    "test_strategy_mean_ret_pct": fmt(test_metrics["strategy_mean_ret_pct"] if test_metrics else np.nan),
                    "test_delta_mean_ret_pct": fmt(test_metrics["delta_mean_ret_pct"] if test_metrics else np.nan),
                }
            )

    grid = pd.DataFrame(grid_rows).sort_values(
        ["val_strategy_mean_ret_pct", "val_loser_precision_pct", "val_n_flagged"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = grid.iloc[0]
    if float(best["val_delta_mean_ret_pct"]) <= 0:
        chosen_signal = "none"
        chosen_threshold = np.nan
        best_row = {
            "test_baseline_mean_ret_pct": float(test_detail["ret_1d_pct"].mean()),
            "test_strategy_mean_ret_pct": float(test_detail["ret_1d_pct"].mean()),
            "test_delta_mean_ret_pct": 0.0,
            "test_n_flagged": 0,
            "test_flag_rate_pct": 0.0,
            "test_loser_precision_pct": np.nan,
            "test_loser_recall_pct": np.nan,
            "val_baseline_mean_ret_pct": float(val_detail["ret_1d_pct"].mean()),
            "val_strategy_mean_ret_pct": float(val_detail["ret_1d_pct"].mean()),
            "val_delta_mean_ret_pct": 0.0,
        }
    else:
        chosen_signal = str(best["signal"])
        chosen_threshold = float(best["threshold_pct"])
        best_row = best

    detail_with_rule = apply_rule(detail, chosen_signal, chosen_threshold)

    summary_rows = [
        {"metric": "all_oos_rows_before_cache_filter", "value": int(len(subset))},
        {"metric": "rows_with_cached_entry_day_minutes", "value": int(len(detail))},
        {"metric": "cached_row_pct", "value": float(len(detail) / len(subset) * 100.0) if len(subset) else 0.0},
        {"metric": "val_rows_cached", "value": int(len(val_detail))},
        {"metric": "test_rows_cached", "value": int(len(test_detail))},
        {"metric": "selected_signal", "value": chosen_signal},
        {"metric": "selected_threshold_pct", "value": "" if not np.isfinite(chosen_threshold) else chosen_threshold},
        {"metric": "val_baseline_mean_ret_pct", "value": float(best_row["val_baseline_mean_ret_pct"])},
        {"metric": "val_strategy_mean_ret_pct", "value": float(best_row["val_strategy_mean_ret_pct"])},
        {"metric": "val_delta_mean_ret_pct", "value": float(best_row["val_delta_mean_ret_pct"])},
        {"metric": "test_baseline_mean_ret_pct", "value": float(best_row["test_baseline_mean_ret_pct"])},
        {"metric": "test_strategy_mean_ret_pct", "value": float(best_row["test_strategy_mean_ret_pct"])},
        {"metric": "test_delta_mean_ret_pct", "value": float(best_row["test_delta_mean_ret_pct"])},
        {"metric": "test_n_flagged", "value": int(best_row["test_n_flagged"])},
        {"metric": "test_flag_rate_pct", "value": float(best_row["test_flag_rate_pct"])},
        {"metric": "test_loser_precision_pct", "value": fmt(best_row["test_loser_precision_pct"])},
        {"metric": "test_loser_recall_pct", "value": fmt(best_row["test_loser_recall_pct"])},
        {"metric": "missing_cache_rows", "value": int(missing_cache)},
    ]

    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail_with_rule.to_csv(args.detail_out, index=False)
    grid.to_csv(args.rule_out, index=False)
    pd.DataFrame(summary_rows).to_csv(args.summary_out, index=False)

    print(f"all_oos_rows={len(subset)}")
    print(f"cached_rows={len(detail)}")
    print(f"cached_row_pct={len(detail) / len(subset) * 100.0 if len(subset) else 0.0:.2f}")
    if chosen_signal == "none":
        print("selected_rule=none")
    else:
        print(f"selected_rule={chosen_signal} <= {chosen_threshold}")
    print(f"val_delta_mean_ret_pct={float(best_row['val_delta_mean_ret_pct']):.4f}")
    print(f"test_delta_mean_ret_pct={float(best_row['test_delta_mean_ret_pct']):.4f}")
    print(f"detail_out={args.detail_out}")
    print(f"rule_out={args.rule_out}")
    print(f"summary_out={args.summary_out}")


if __name__ == "__main__":
    main()
