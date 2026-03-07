from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
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


@dataclass(frozen=True)
class Candidate:
    arm_gain_pct: float
    trail_drawdown_pct: float
    reentry_mode: str

    @property
    def name(self) -> str:
        return f"trail_after_{self.arm_gain_pct:g}_dd_{self.trail_drawdown_pct:g}_{self.reentry_mode}"


def percentile_rank(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_reference.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    idx = np.searchsorted(sorted_reference, values, side="right")
    return idx / float(sorted_reference.size)


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


def find_last_bar_index_at_or_before(minute_bars: list[dict], ts: datetime) -> int | None:
    cutoff_ts = int(ts.timestamp() * 1000)
    last_idx = None
    for idx, bar in enumerate(minute_bars):
        bar_ts = int(bar["t"])
        if bar_ts <= cutoff_ts:
            last_idx = idx
            continue
        break
    return last_idx


def trading_days_between(start: date, end: date) -> list[date]:
    days: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def load_trade_bars(cache_dir: Path, ticker: str, entry_dt: datetime, exit_dt: datetime, path_horizon: str) -> list[dict]:
    if path_horizon == "same_day":
        days = [entry_dt.date()]
    elif path_horizon == "through_exit":
        days = trading_days_between(entry_dt.date(), exit_dt.date())
    else:
        raise ValueError(f"Unknown path_horizon: {path_horizon}")

    bars: list[dict] = []
    for d in days:
        path = cache_path(cache_dir, ticker, d.strftime("%Y-%m-%d"))
        if not path.exists():
            return []
        day_bars = [b for b in load_cached_minute_bars(path) if is_regular_session_bar(b)]
        bars.extend(day_bars)
    bars.sort(key=lambda b: int(b["t"]))
    return bars


def implied_exit_price_from_ret(entry_price: float, ret_pct: float) -> float | None:
    if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(ret_pct):
        return float(entry_price * (1.0 + float(ret_pct) / 100.0))
    return None


def total_return_pct(entry_price: float, first_exit_price: float, final_exit_price: float, reentry_price: float | None) -> float:
    if (
        not np.isfinite(entry_price)
        or entry_price <= 0
        or not np.isfinite(first_exit_price)
        or first_exit_price <= 0
        or not np.isfinite(final_exit_price)
        or final_exit_price <= 0
    ):
        return np.nan
    if reentry_price is None:
        return (first_exit_price / entry_price - 1.0) * 100.0
    if not np.isfinite(reentry_price) or reentry_price <= 0:
        return np.nan
    gross = (first_exit_price / entry_price) * (final_exit_price / reentry_price)
    return (gross - 1.0) * 100.0


def first_reentry_price(
    closes: np.ndarray,
    times: list[str],
    start_idx: int,
    end_idx: int,
    threshold_price: float,
    fallback_close: bool,
) -> tuple[float | None, str]:
    for idx in range(start_idx, end_idx + 1):
        px = float(closes[idx])
        if px >= threshold_price:
            return px, times[idx]
    if fallback_close:
        return float(closes[end_idx]), times[end_idx]
    return None, ""


def simulate_candidate(
    closes: np.ndarray,
    times: list[str],
    entry_idx: int,
    close_idx: int,
    entry_price: float,
    close_1d_price: float,
    baseline_ret_pct: float,
    candidate: Candidate,
) -> dict:
    armed = False
    peak_price = entry_price
    peak_idx = entry_idx
    stop_idx = None

    for idx in range(entry_idx + 1, close_idx + 1):
        px = float(closes[idx])
        if px > peak_price:
            peak_price = px
            peak_idx = idx
        gain_pct = (peak_price / entry_price - 1.0) * 100.0
        if gain_pct >= candidate.arm_gain_pct:
            armed = True
        if not armed:
            continue
        drawdown_pct = (px / peak_price - 1.0) * 100.0
        if drawdown_pct <= -candidate.trail_drawdown_pct:
            stop_idx = idx
            break

    if stop_idx is None:
        return {
            "strategy_ret_pct": baseline_ret_pct,
            "stopped": False,
            "reentered": False,
            "stop_dt": "",
            "stop_price": np.nan,
            "peak_ret_pct_before_stop": np.nan,
            "reentry_dt": "",
            "reentry_price": np.nan,
        }

    stop_price = float(closes[stop_idx])
    peak_ret_pct = (peak_price / entry_price - 1.0) * 100.0
    fallback_close = candidate.reentry_mode.endswith("_else_close")

    if candidate.reentry_mode == "stay_out":
        reentry_price, reentry_dt = None, ""
    elif candidate.reentry_mode.startswith("reclaim_entry"):
        reentry_price, reentry_dt = first_reentry_price(
            closes=closes,
            times=times,
            start_idx=stop_idx + 1,
            end_idx=close_idx,
            threshold_price=entry_price,
            fallback_close=fallback_close,
        )
    elif candidate.reentry_mode.startswith("reclaim_peak"):
        reentry_price, reentry_dt = first_reentry_price(
            closes=closes,
            times=times,
            start_idx=stop_idx + 1,
            end_idx=close_idx,
            threshold_price=peak_price,
            fallback_close=fallback_close,
        )
    else:
        raise ValueError(f"Unknown reentry mode: {candidate.reentry_mode}")

    strategy_ret_pct = total_return_pct(
        entry_price=entry_price,
        first_exit_price=stop_price,
        final_exit_price=close_1d_price,
        reentry_price=reentry_price,
    )
    return {
        "strategy_ret_pct": strategy_ret_pct,
        "stopped": True,
        "reentered": bool(reentry_price is not None),
        "stop_dt": times[stop_idx],
        "stop_price": stop_price,
        "peak_ret_pct_before_stop": peak_ret_pct,
        "reentry_dt": reentry_dt,
        "reentry_price": reentry_price if reentry_price is not None else np.nan,
    }


def compute_row(row: pd.Series, cache_dir: Path, candidates: list[Candidate]) -> dict | None:
    ticker = str(row["ticker"])
    buy_dt = pd.Timestamp(row["buy_datetime"]).to_pydatetime().replace(tzinfo=ET)
    exit_dt = pd.Timestamp(row["exit_datetime"]).to_pydatetime().replace(tzinfo=ET)
    bars = load_trade_bars(cache_dir=cache_dir, ticker=ticker, entry_dt=buy_dt, exit_dt=exit_dt, path_horizon=str(row["path_horizon"]))
    if not bars:
        return None

    entry_idx = find_entry_bar_index(bars, buy_dt)
    if entry_idx is None:
        return None

    same_day_close_idx = find_same_day_close_index(bars, entry_idx)
    if same_day_close_idx is None or same_day_close_idx <= entry_idx:
        return None
    if str(row["path_horizon"]) == "same_day":
        path_end_idx = same_day_close_idx
    else:
        path_end_idx = find_last_bar_index_at_or_before(bars, exit_dt)
        if path_end_idx is None or path_end_idx <= entry_idx:
            return None

    entry_price = float(bars[entry_idx]["o"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    baseline_ret_pct = float(row["ret_pct"])
    close_1d_price = implied_exit_price_from_ret(entry_price, baseline_ret_pct)
    if close_1d_price is None or not np.isfinite(close_1d_price) or close_1d_price <= 0:
        return None
    closes = np.array([float(b["c"]) for b in bars], dtype=float)
    times = [bar_dt_et(b).strftime("%Y-%m-%d %H:%M:%S") for b in bars]

    out = {
        "split": row["split"],
        "ticker": ticker,
        "buy_datetime": pd.Timestamp(row["buy_datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
        "pred_mean4": float(row["pred_mean4"]) if np.isfinite(row["pred_mean4"]) else np.nan,
        "decile_score": float(row["decile_score"]) if np.isfinite(row["decile_score"]) else np.nan,
        "baseline_ret_pct": baseline_ret_pct,
        "entry_price": entry_price,
        "close_1d_price": float(close_1d_price),
        "entry_bar_time": times[entry_idx],
        "same_day_close_time": times[same_day_close_idx],
        "same_day_close_ret_pct": (closes[same_day_close_idx] / entry_price - 1.0) * 100.0,
        "path_horizon": row["path_horizon"],
        "path_end_time": times[path_end_idx],
        "path_end_ret_pct": (closes[path_end_idx] / entry_price - 1.0) * 100.0,
    }

    for candidate in candidates:
        result = simulate_candidate(
            closes=closes,
            times=times,
            entry_idx=entry_idx,
            close_idx=path_end_idx,
            entry_price=entry_price,
            close_1d_price=float(close_1d_price),
            baseline_ret_pct=baseline_ret_pct,
            candidate=candidate,
        )
        prefix = candidate.name
        out[f"{prefix}_strategy_ret_pct"] = result["strategy_ret_pct"]
        out[f"{prefix}_stopped"] = result["stopped"]
        out[f"{prefix}_reentered"] = result["reentered"]
        out[f"{prefix}_stop_dt"] = result["stop_dt"]
        out[f"{prefix}_stop_price"] = result["stop_price"]
        out[f"{prefix}_peak_ret_pct_before_stop"] = result["peak_ret_pct_before_stop"]
        out[f"{prefix}_reentry_dt"] = result["reentry_dt"]
        out[f"{prefix}_reentry_price"] = result["reentry_price"]

    return out


def summarize_candidate(detail: pd.DataFrame, candidate: Candidate) -> dict:
    prefix = candidate.name
    eligible = detail[detail[f"{prefix}_strategy_ret_pct"].notna()].copy()
    baseline = eligible["baseline_ret_pct"].to_numpy(dtype=float)
    strategy = eligible[f"{prefix}_strategy_ret_pct"].to_numpy(dtype=float)
    stopped = eligible[f"{prefix}_stopped"].to_numpy(dtype=bool)
    reentered = eligible[f"{prefix}_reentered"].to_numpy(dtype=bool)
    return {
        "n_trades": int(len(eligible)),
        "n_stopped": int(stopped.sum()),
        "stop_rate_pct": float(stopped.mean() * 100.0) if len(stopped) else 0.0,
        "n_reentered": int(np.logical_and(stopped, reentered).sum()),
        "reentry_rate_among_stopped_pct": float(np.logical_and(stopped, reentered).sum() / stopped.sum() * 100.0) if stopped.sum() else np.nan,
        "baseline_mean_ret_pct": float(np.mean(baseline)),
        "strategy_mean_ret_pct": float(np.mean(strategy)),
        "delta_mean_ret_pct": float(np.mean(strategy) - np.mean(baseline)),
        "baseline_win_rate_pct": float((baseline > 0).mean() * 100.0),
        "strategy_win_rate_pct": float((strategy > 0).mean() * 100.0),
        "delta_win_rate_pct": float((strategy > 0).mean() * 100.0 - (baseline > 0).mean() * 100.0),
    }


def apply_selected(detail: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    out = detail.copy()
    if candidate_name == "none":
        out["selected_candidate"] = "none"
        out["selected_strategy_ret_pct"] = out["baseline_ret_pct"]
        out["selected_improvement_pct"] = 0.0
        out["selected_stopped"] = False
        out["selected_reentered"] = False
        out["selected_stop_dt"] = ""
        out["selected_reentry_dt"] = ""
        return out

    out["selected_candidate"] = candidate_name
    out["selected_strategy_ret_pct"] = out[f"{candidate_name}_strategy_ret_pct"]
    out["selected_improvement_pct"] = out["selected_strategy_ret_pct"] - out["baseline_ret_pct"]
    out["selected_stopped"] = out[f"{candidate_name}_stopped"]
    out["selected_reentered"] = out[f"{candidate_name}_reentered"]
    out["selected_stop_dt"] = out[f"{candidate_name}_stop_dt"]
    out["selected_reentry_dt"] = out[f"{candidate_name}_reentry_dt"]
    return out


def fmt(value) -> str | float:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def build_candidates() -> list[Candidate]:
    arm_gains = [0.5, 1.0, 2.0]
    trail_drawdowns = [0.5, 1.0, 1.5, 2.0]
    reentry_modes = [
        "stay_out",
        "reclaim_entry_else_out",
        "reclaim_entry_else_close",
        "reclaim_peak_else_out",
        "reclaim_peak_else_close",
    ]
    return [
        Candidate(arm_gain_pct=arm, trail_drawdown_pct=dd, reentry_mode=mode)
        for arm in arm_gains
        for dd in trail_drawdowns
        for mode in reentry_modes
    ]


def prepare_analysis_subset(
    scored: pd.DataFrame,
    n_train: int,
    split_80: int,
    analysis_scope: str,
    test_subset_tune_fraction: float,
) -> pd.DataFrame:
    if analysis_scope == "oos":
        subset = scored.iloc[n_train:].copy()
        subset["source_split"] = np.where(np.arange(len(subset)) < (split_80 - n_train), "val", "test")
        subset["split"] = subset["source_split"]
        return subset

    if analysis_scope != "test_only":
        raise ValueError(f"Unknown analysis scope: {analysis_scope}")

    subset = scored.iloc[split_80:].copy()
    subset["source_split"] = "test"
    n_subset = len(subset)
    if n_subset < 2:
        raise ValueError("Need at least 2 rows in the original test subset for test_only analysis.")
    n_tune = int(round(n_subset * test_subset_tune_fraction))
    n_tune = max(1, min(n_tune, n_subset - 1))
    subset["split"] = "test"
    subset.iloc[:n_tune, subset.columns.get_loc("split")] = "val"
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic path-based exit and rebuy analysis on all companies using the original chrono val/test split."
    )
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument(
        "--path-horizon",
        choices=["same_day", "through_exit"],
        default="same_day",
        help="Whether the dynamic rule can only use same-day bars or all regular-session bars through exit_datetime.",
    )
    parser.add_argument(
        "--analysis-scope",
        choices=["oos", "test_only"],
        default="oos",
        help="`oos` uses the original validation/test split; `test_only` uses only the original test rows and splits them chronologically into tune/holdout slices.",
    )
    parser.add_argument(
        "--test-subset-tune-fraction",
        type=float,
        default=0.5,
        help="When --analysis-scope=test_only, fraction of the original test subset used for tuning before evaluating on the remaining holdout rows.",
    )
    parser.add_argument(
        "--min-decile-score",
        type=float,
        default=None,
        help="Optional minimum decile_score computed from the training split pred_mean4 distribution (for example 0.90).",
    )
    parser.add_argument("--detail-out", default=str(OUT_DIR / "all_companies_dynamic_path_exit_detail.csv"))
    parser.add_argument("--grid-out", default=str(OUT_DIR / "all_companies_dynamic_path_exit_grid.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "all_companies_dynamic_path_exit_summary.csv"))
    args = parser.parse_args()

    candidates = build_candidates()
    scored = load_scored_day1()
    n_train, split_80 = chrono_split_60_20_20(len(scored))
    train_pred = pd.to_numeric(scored.iloc[:n_train]["pred_mean4"], errors="coerce").dropna().to_numpy(dtype=float)
    train_pred_sorted = np.sort(train_pred)
    scored["decile_score"] = percentile_rank(
        train_pred_sorted,
        pd.to_numeric(scored["pred_mean4"], errors="coerce").to_numpy(dtype=float),
    )
    subset = prepare_analysis_subset(
        scored=scored,
        n_train=n_train,
        split_80=split_80,
        analysis_scope=args.analysis_scope,
        test_subset_tune_fraction=float(args.test_subset_tune_fraction),
    )
    subset = subset.dropna(subset=["buy_datetime", "ret_pct"]).copy()
    subset["path_horizon"] = args.path_horizon
    total_analysis_rows = int(len(subset))
    if args.min_decile_score is not None:
        subset = subset[pd.to_numeric(subset["decile_score"], errors="coerce") >= float(args.min_decile_score)].copy()

    cache_dir = Path(args.cache_dir)
    detail_rows = []
    missing_rows = 0
    for _, row in subset.iterrows():
        out = compute_row(row, cache_dir, candidates)
        if out is None:
            missing_rows += 1
            continue
        detail_rows.append(out)

    detail = pd.DataFrame(detail_rows)
    val_detail = detail[detail["split"] == "val"].copy()
    test_detail = detail[detail["split"] == "test"].copy()

    grid_rows = []
    for candidate in candidates:
        val_metrics = summarize_candidate(val_detail, candidate)
        test_metrics = summarize_candidate(test_detail, candidate)
        grid_rows.append(
            {
                "candidate": candidate.name,
                "arm_gain_pct": candidate.arm_gain_pct,
                "trail_drawdown_pct": candidate.trail_drawdown_pct,
                "reentry_mode": candidate.reentry_mode,
                "val_n_trades": val_metrics["n_trades"],
                "val_n_stopped": val_metrics["n_stopped"],
                "val_stop_rate_pct": val_metrics["stop_rate_pct"],
                "val_n_reentered": val_metrics["n_reentered"],
                "val_reentry_rate_among_stopped_pct": fmt(val_metrics["reentry_rate_among_stopped_pct"]),
                "val_baseline_mean_ret_pct": val_metrics["baseline_mean_ret_pct"],
                "val_strategy_mean_ret_pct": val_metrics["strategy_mean_ret_pct"],
                "val_delta_mean_ret_pct": val_metrics["delta_mean_ret_pct"],
                "val_baseline_win_rate_pct": val_metrics["baseline_win_rate_pct"],
                "val_strategy_win_rate_pct": val_metrics["strategy_win_rate_pct"],
                "val_delta_win_rate_pct": val_metrics["delta_win_rate_pct"],
                "test_n_trades": test_metrics["n_trades"],
                "test_n_stopped": test_metrics["n_stopped"],
                "test_stop_rate_pct": test_metrics["stop_rate_pct"],
                "test_n_reentered": test_metrics["n_reentered"],
                "test_reentry_rate_among_stopped_pct": fmt(test_metrics["reentry_rate_among_stopped_pct"]),
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
            "test_n_stopped": 0,
            "test_stop_rate_pct": 0.0,
            "test_n_reentered": 0,
            "test_reentry_rate_among_stopped_pct": np.nan,
        }
    else:
        chosen_candidate = str(best["candidate"])
        best_row = best

    detail_selected = apply_selected(detail, chosen_candidate)

    summary_rows = [
        {"metric": "path_horizon", "value": args.path_horizon},
        {"metric": "analysis_scope", "value": args.analysis_scope},
        {"metric": "test_subset_tune_fraction", "value": float(args.test_subset_tune_fraction)},
        {"metric": "min_decile_score", "value": fmt(args.min_decile_score)},
        {"metric": "analysis_rows_before_filter", "value": total_analysis_rows},
        {"metric": "analysis_rows_after_filter", "value": int(len(subset))},
        {"metric": "rows_with_usable_same_day_minutes", "value": int(len(detail))},
        {"metric": "usable_row_pct", "value": float(len(detail) / len(subset) * 100.0) if len(subset) else 0.0},
        {"metric": "val_rows", "value": int(len(val_detail))},
        {"metric": "test_rows", "value": int(len(test_detail))},
        {"metric": "candidate_count", "value": int(len(candidates))},
        {"metric": "selected_candidate", "value": chosen_candidate},
        {"metric": "val_baseline_mean_ret_pct", "value": float(best_row["val_baseline_mean_ret_pct"])},
        {"metric": "val_strategy_mean_ret_pct", "value": float(best_row["val_strategy_mean_ret_pct"])},
        {"metric": "val_delta_mean_ret_pct", "value": float(best_row["val_delta_mean_ret_pct"])},
        {"metric": "test_baseline_mean_ret_pct", "value": float(best_row["test_baseline_mean_ret_pct"])},
        {"metric": "test_strategy_mean_ret_pct", "value": float(best_row["test_strategy_mean_ret_pct"])},
        {"metric": "test_delta_mean_ret_pct", "value": float(best_row["test_delta_mean_ret_pct"])},
        {"metric": "test_n_stopped", "value": int(best_row["test_n_stopped"])},
        {"metric": "test_stop_rate_pct", "value": float(best_row["test_stop_rate_pct"])},
        {"metric": "test_n_reentered", "value": int(best_row["test_n_reentered"])},
        {"metric": "test_reentry_rate_among_stopped_pct", "value": fmt(best_row["test_reentry_rate_among_stopped_pct"])},
        {"metric": "missing_or_unusable_rows", "value": int(missing_rows)},
    ]

    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail_selected.to_csv(args.detail_out, index=False)
    grid.to_csv(args.grid_out, index=False)
    pd.DataFrame(summary_rows).to_csv(args.summary_out, index=False)

    print(f"path_horizon={args.path_horizon}")
    print(f"analysis_scope={args.analysis_scope}")
    print(f"analysis_rows={len(subset)}")
    print(f"usable_rows={len(detail)}")
    print(f"usable_row_pct={len(detail) / len(subset) * 100.0 if len(subset) else 0.0:.2f}")
    print(f"candidate_count={len(candidates)}")
    print(f"selected_candidate={chosen_candidate}")
    print(f"val_delta_mean_ret_pct={float(best_row['val_delta_mean_ret_pct']):.4f}")
    print(f"test_delta_mean_ret_pct={float(best_row['test_delta_mean_ret_pct']):.4f}")
    print(f"detail_out={args.detail_out}")
    print(f"grid_out={args.grid_out}")
    print(f"summary_out={args.summary_out}")


if __name__ == "__main__":
    main()
