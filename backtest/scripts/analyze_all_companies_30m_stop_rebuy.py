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
STOP_CHECKPOINT_MIN = 30
STOP_THRESHOLD_PCT = -1.0

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


def safe_close_1d(row: pd.Series, entry_price: float) -> float | None:
    close_1d = pd.to_numeric(pd.Series([row.get("close_1d")]), errors="coerce").iloc[0]
    if np.isfinite(close_1d) and close_1d > 0:
        return float(close_1d)
    ret_pct = pd.to_numeric(pd.Series([row.get("ret_pct")]), errors="coerce").iloc[0]
    if np.isfinite(ret_pct):
        return float(entry_price * (1.0 + float(ret_pct) / 100.0))
    return None


def total_return_pct(entry_price: float, interim_exit_price: float, final_exit_price: float, reentry_price: float | None) -> float:
    if (
        not np.isfinite(entry_price)
        or entry_price <= 0
        or not np.isfinite(interim_exit_price)
        or interim_exit_price <= 0
        or not np.isfinite(final_exit_price)
        or final_exit_price <= 0
    ):
        return np.nan
    if reentry_price is None:
        return (interim_exit_price / entry_price - 1.0) * 100.0
    if not np.isfinite(reentry_price) or reentry_price <= 0:
        return np.nan
    gross = (interim_exit_price / entry_price) * (final_exit_price / reentry_price)
    return (gross - 1.0) * 100.0


def first_recovery_close(
    bars: list[dict],
    entry_idx: int,
    start_idx: int,
    last_idx: int,
    entry_price: float,
    recovery_threshold_pct: float,
) -> tuple[float | None, str]:
    for idx in range(start_idx, last_idx + 1):
        px = float(bars[idx]["c"])
        ret_pct = (px / entry_price - 1.0) * 100.0
        if ret_pct >= recovery_threshold_pct:
            return px, bar_dt_et(bars[idx]).strftime("%Y-%m-%d %H:%M:%S")
    return None, ""


def candidate_reentry_prices(
    bars: list[dict],
    entry_idx: int,
    stop_idx: int,
    close_idx: int,
    entry_price: float,
) -> dict[str, tuple[float | None, str]]:
    out: dict[str, tuple[float | None, str]] = {}

    for checkpoint in [60, 120]:
        idx = entry_idx + checkpoint
        if idx <= close_idx and idx > stop_idx:
            out[f"fixed_{checkpoint}m"] = (
                float(bars[idx]["c"]),
                bar_dt_et(bars[idx]).strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            out[f"fixed_{checkpoint}m"] = (None, "")

    out["fixed_close"] = (
        float(bars[close_idx]["c"]) if close_idx > stop_idx else None,
        bar_dt_et(bars[close_idx]).strftime("%Y-%m-%d %H:%M:%S") if close_idx > stop_idx else "",
    )

    for threshold in [-1.0, -0.5, 0.0]:
        price, ts = first_recovery_close(
            bars=bars,
            entry_idx=entry_idx,
            start_idx=stop_idx + 1,
            last_idx=close_idx,
            entry_price=entry_price,
            recovery_threshold_pct=threshold,
        )
        name = f"recover_{threshold:+.1f}pct"
        out[f"{name}_else_out"] = (price, ts)
        out[f"{name}_else_close"] = (price if price is not None else float(bars[close_idx]["c"]), ts if price is not None else bar_dt_et(bars[close_idx]).strftime("%Y-%m-%d %H:%M:%S"))

    return out


def compute_row(row: pd.Series, cache_dir: Path) -> dict | None:
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

    stop_idx = entry_idx + STOP_CHECKPOINT_MIN
    if stop_idx >= len(bars):
        return None

    close_idx = find_same_day_close_index(bars, entry_idx)
    if close_idx is None or close_idx <= stop_idx:
        return None

    entry_price = float(bars[entry_idx]["o"])
    stop_price = float(bars[stop_idx]["c"])
    close_1d = safe_close_1d(row, entry_price)
    if not np.isfinite(entry_price) or entry_price <= 0 or close_1d is None:
        return None

    stop_ret_pct = (stop_price / entry_price - 1.0) * 100.0
    baseline_ret_pct = float(row["ret_pct"])
    flagged = bool(stop_ret_pct <= STOP_THRESHOLD_PCT)

    out = {
        "split": row["split"],
        "ticker": ticker,
        "buy_datetime": pd.Timestamp(row["buy_datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_ret_pct": baseline_ret_pct,
        "entry_price": entry_price,
        "close_1d_price": float(close_1d),
        "entry_bar_time": bar_dt_et(bars[entry_idx]).strftime("%Y-%m-%d %H:%M:%S"),
        "stop_30m_price": stop_price,
        "stop_30m_dt": bar_dt_et(bars[stop_idx]).strftime("%Y-%m-%d %H:%M:%S"),
        "stop_30m_ret_pct": stop_ret_pct,
        "flagged_30m_lt_neg1": flagged,
        "same_day_close_price": float(bars[close_idx]["c"]),
        "same_day_close_dt": bar_dt_et(bars[close_idx]).strftime("%Y-%m-%d %H:%M:%S"),
    }

    candidates = {"stay_out": (None, "")}
    candidates.update(candidate_reentry_prices(bars, entry_idx, stop_idx, close_idx, entry_price))

    for name, (reentry_price, reentry_dt) in candidates.items():
        out[f"{name}_reentry_price"] = "" if reentry_price is None else float(reentry_price)
        out[f"{name}_reentry_dt"] = reentry_dt
        if flagged:
            strat_ret = total_return_pct(
                entry_price=entry_price,
                interim_exit_price=stop_price,
                final_exit_price=float(close_1d),
                reentry_price=reentry_price,
            )
            out[f"{name}_strategy_ret_pct"] = strat_ret
            out[f"{name}_reentered"] = bool(reentry_price is not None)
        else:
            out[f"{name}_strategy_ret_pct"] = baseline_ret_pct
            out[f"{name}_reentered"] = False

    return out


def summarize_candidate(detail: pd.DataFrame, candidate: str) -> dict:
    eligible = detail[detail[f"{candidate}_strategy_ret_pct"].notna()].copy()
    baseline = eligible["baseline_ret_pct"].to_numpy(dtype=float)
    strategy = eligible[f"{candidate}_strategy_ret_pct"].to_numpy(dtype=float)
    flagged = eligible["flagged_30m_lt_neg1"].to_numpy(dtype=bool)
    reentered = eligible[f"{candidate}_reentered"].to_numpy(dtype=bool)
    return {
        "n_trades": int(len(eligible)),
        "n_flagged": int(flagged.sum()),
        "flag_rate_pct": float(flagged.mean() * 100.0) if len(flagged) else 0.0,
        "n_reentered": int(np.logical_and(flagged, reentered).sum()),
        "reentry_rate_among_flagged_pct": float(np.logical_and(flagged, reentered).sum() / flagged.sum() * 100.0) if flagged.sum() else np.nan,
        "baseline_mean_ret_pct": float(np.mean(baseline)),
        "strategy_mean_ret_pct": float(np.mean(strategy)),
        "delta_mean_ret_pct": float(np.mean(strategy) - np.mean(baseline)),
        "baseline_win_rate_pct": float((baseline > 0).mean() * 100.0),
        "strategy_win_rate_pct": float((strategy > 0).mean() * 100.0),
        "delta_win_rate_pct": float((strategy > 0).mean() * 100.0 - (baseline > 0).mean() * 100.0),
    }


def apply_selected(detail: pd.DataFrame, candidate: str) -> pd.DataFrame:
    out = detail.copy()
    if candidate == "none":
        out["selected_candidate"] = "none"
        out["selected_strategy_ret_pct"] = out["baseline_ret_pct"]
        out["selected_improvement_pct"] = 0.0
        out["selected_reentry_dt"] = ""
        out["selected_reentry_price"] = ""
        out["selected_reentered"] = False
        return out

    out["selected_candidate"] = candidate
    out["selected_strategy_ret_pct"] = out[f"{candidate}_strategy_ret_pct"]
    out["selected_improvement_pct"] = out["selected_strategy_ret_pct"] - out["baseline_ret_pct"]
    out["selected_reentry_dt"] = out[f"{candidate}_reentry_dt"]
    out["selected_reentry_price"] = out[f"{candidate}_reentry_price"]
    out["selected_reentered"] = out[f"{candidate}_reentered"]
    return out


def fmt(value) -> str | float:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 30-minute stop and later re-buy variants on all companies using the original chrono val/test split."
    )
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--detail-out", default=str(OUT_DIR / "all_companies_30m_stop_rebuy_detail.csv"))
    parser.add_argument("--grid-out", default=str(OUT_DIR / "all_companies_30m_stop_rebuy_grid.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "all_companies_30m_stop_rebuy_summary.csv"))
    args = parser.parse_args()

    scored = load_scored_day1()
    n_train, split_80 = chrono_split_60_20_20(len(scored))
    subset = scored.iloc[n_train:].copy()
    subset["split"] = np.where(np.arange(len(subset)) < (split_80 - n_train), "val", "test")
    subset = subset.dropna(subset=["buy_datetime", "ret_pct"]).copy()

    cache_dir = Path(args.cache_dir)
    detail_rows = []
    missing_rows = 0
    for _, row in subset.iterrows():
        out = compute_row(row, cache_dir)
        if out is None:
            missing_rows += 1
            continue
        detail_rows.append(out)

    detail = pd.DataFrame(detail_rows)
    candidate_names = [
        "stay_out",
        "fixed_60m",
        "fixed_120m",
        "fixed_close",
        "recover_-1.0pct_else_out",
        "recover_-1.0pct_else_close",
        "recover_-0.5pct_else_out",
        "recover_-0.5pct_else_close",
        "recover_+0.0pct_else_out",
        "recover_+0.0pct_else_close",
    ]

    val_detail = detail[detail["split"] == "val"].copy()
    test_detail = detail[detail["split"] == "test"].copy()

    grid_rows = []
    for candidate in candidate_names:
        val_metrics = summarize_candidate(val_detail, candidate)
        test_metrics = summarize_candidate(test_detail, candidate)
        grid_rows.append(
            {
                "candidate": candidate,
                "val_n_trades": val_metrics["n_trades"],
                "val_n_flagged": val_metrics["n_flagged"],
                "val_flag_rate_pct": val_metrics["flag_rate_pct"],
                "val_n_reentered": val_metrics["n_reentered"],
                "val_reentry_rate_among_flagged_pct": fmt(val_metrics["reentry_rate_among_flagged_pct"]),
                "val_baseline_mean_ret_pct": val_metrics["baseline_mean_ret_pct"],
                "val_strategy_mean_ret_pct": val_metrics["strategy_mean_ret_pct"],
                "val_delta_mean_ret_pct": val_metrics["delta_mean_ret_pct"],
                "val_baseline_win_rate_pct": val_metrics["baseline_win_rate_pct"],
                "val_strategy_win_rate_pct": val_metrics["strategy_win_rate_pct"],
                "val_delta_win_rate_pct": val_metrics["delta_win_rate_pct"],
                "test_n_trades": test_metrics["n_trades"],
                "test_n_flagged": test_metrics["n_flagged"],
                "test_flag_rate_pct": test_metrics["flag_rate_pct"],
                "test_n_reentered": test_metrics["n_reentered"],
                "test_reentry_rate_among_flagged_pct": fmt(test_metrics["reentry_rate_among_flagged_pct"]),
                "test_baseline_mean_ret_pct": test_metrics["baseline_mean_ret_pct"],
                "test_strategy_mean_ret_pct": test_metrics["strategy_mean_ret_pct"],
                "test_delta_mean_ret_pct": test_metrics["delta_mean_ret_pct"],
                "test_baseline_win_rate_pct": test_metrics["baseline_win_rate_pct"],
                "test_strategy_win_rate_pct": test_metrics["strategy_win_rate_pct"],
                "test_delta_win_rate_pct": test_metrics["delta_win_rate_pct"],
            }
        )

    grid = pd.DataFrame(grid_rows).sort_values(
        ["val_strategy_mean_ret_pct", "val_delta_mean_ret_pct", "test_strategy_mean_ret_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = grid.iloc[0]
    if float(best["val_delta_mean_ret_pct"]) <= 0:
        chosen_candidate = "none"
        best_row = {
            "val_baseline_mean_ret_pct": float(val_detail["baseline_ret_pct"].mean()),
            "val_strategy_mean_ret_pct": float(val_detail["baseline_ret_pct"].mean()),
            "val_delta_mean_ret_pct": 0.0,
            "test_baseline_mean_ret_pct": float(test_detail["baseline_ret_pct"].mean()),
            "test_strategy_mean_ret_pct": float(test_detail["baseline_ret_pct"].mean()),
            "test_delta_mean_ret_pct": 0.0,
            "test_n_flagged": int(test_detail["flagged_30m_lt_neg1"].sum()),
            "test_flag_rate_pct": float(test_detail["flagged_30m_lt_neg1"].mean() * 100.0),
            "test_n_reentered": 0,
            "test_reentry_rate_among_flagged_pct": np.nan,
        }
    else:
        chosen_candidate = str(best["candidate"])
        best_row = best

    detail_selected = apply_selected(detail, chosen_candidate)

    summary_rows = [
        {"metric": "all_oos_rows_before_filter", "value": int(len(subset))},
        {"metric": "rows_with_usable_30m_and_same_day_minutes", "value": int(len(detail))},
        {"metric": "usable_row_pct", "value": float(len(detail) / len(subset) * 100.0) if len(subset) else 0.0},
        {"metric": "val_rows", "value": int(len(val_detail))},
        {"metric": "test_rows", "value": int(len(test_detail))},
        {"metric": "stop_checkpoint_min", "value": STOP_CHECKPOINT_MIN},
        {"metric": "stop_threshold_pct", "value": STOP_THRESHOLD_PCT},
        {"metric": "selected_candidate", "value": chosen_candidate},
        {"metric": "val_baseline_mean_ret_pct", "value": float(best_row["val_baseline_mean_ret_pct"])},
        {"metric": "val_strategy_mean_ret_pct", "value": float(best_row["val_strategy_mean_ret_pct"])},
        {"metric": "val_delta_mean_ret_pct", "value": float(best_row["val_delta_mean_ret_pct"])},
        {"metric": "test_baseline_mean_ret_pct", "value": float(best_row["test_baseline_mean_ret_pct"])},
        {"metric": "test_strategy_mean_ret_pct", "value": float(best_row["test_strategy_mean_ret_pct"])},
        {"metric": "test_delta_mean_ret_pct", "value": float(best_row["test_delta_mean_ret_pct"])},
        {"metric": "test_n_flagged", "value": int(best_row["test_n_flagged"])},
        {"metric": "test_flag_rate_pct", "value": float(best_row["test_flag_rate_pct"])},
        {"metric": "test_n_reentered", "value": int(best_row["test_n_reentered"])},
        {"metric": "test_reentry_rate_among_flagged_pct", "value": fmt(best_row["test_reentry_rate_among_flagged_pct"])},
        {"metric": "missing_or_unusable_rows", "value": int(missing_rows)},
    ]

    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail_selected.to_csv(args.detail_out, index=False)
    grid.to_csv(args.grid_out, index=False)
    pd.DataFrame(summary_rows).to_csv(args.summary_out, index=False)

    print(f"all_oos_rows={len(subset)}")
    print(f"usable_rows={len(detail)}")
    print(f"usable_row_pct={len(detail) / len(subset) * 100.0 if len(subset) else 0.0:.2f}")
    print(f"selected_candidate={chosen_candidate}")
    print(f"val_delta_mean_ret_pct={float(best_row['val_delta_mean_ret_pct']):.4f}")
    print(f"test_delta_mean_ret_pct={float(best_row['test_delta_mean_ret_pct']):.4f}")
    print(f"detail_out={args.detail_out}")
    print(f"grid_out={args.grid_out}")
    print(f"summary_out={args.summary_out}")


if __name__ == "__main__":
    main()
