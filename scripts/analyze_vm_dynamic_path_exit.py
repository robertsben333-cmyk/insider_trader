from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_vm_recommendations import (  # noqa: E402
    ET,
    MARKET_CLOSE,
    MARKET_OPEN,
    UTC,
    fetch_minute_bars,
)
from live_trading.strategy_settings import LIVE_PATHS


@dataclass(frozen=True)
class Candidate:
    arm_gain_pct: float
    trail_drawdown_pct: float
    reentry_mode: str

    @property
    def name(self) -> str:
        return f"trail_after_{self.arm_gain_pct:g}_dd_{self.trail_drawdown_pct:g}_{self.reentry_mode}"


HISTORICAL_SELECTED = Candidate(arm_gain_pct=2.0, trail_drawdown_pct=0.5, reentry_mode="stay_out")


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


def bar_dt_et(bar: dict) -> datetime | None:
    ts = bar.get("t")
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts) / 1000, tz=UTC).astimezone(ET)


def is_regular_session_bar(bar: dict) -> bool:
    dt = bar_dt_et(bar)
    if dt is None:
        return False
    if dt.weekday() >= 5:
        return False
    return MARKET_OPEN <= dt.time() < MARKET_CLOSE


def find_entry_bar_index(minute_bars: list[dict], entry_dt_et: datetime) -> int | None:
    target_ms = int(entry_dt_et.timestamp() * 1000)
    for idx, bar in enumerate(minute_bars):
        ts = bar.get("t")
        if ts is not None and int(ts) >= target_ms:
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


def implied_exit_price(entry_price: float, ret_pct: float) -> float | None:
    if not math.isfinite(entry_price) or entry_price <= 0:
        return None
    if not math.isfinite(ret_pct):
        return None
    return float(entry_price * (1.0 + ret_pct / 100.0))


def total_return_pct(entry_price: float, first_exit_price: float, final_exit_price: float, reentry_price: float | None) -> float:
    if (
        not math.isfinite(entry_price)
        or entry_price <= 0
        or not math.isfinite(first_exit_price)
        or first_exit_price <= 0
        or not math.isfinite(final_exit_price)
        or final_exit_price <= 0
    ):
        return float("nan")
    if reentry_price is None:
        return (first_exit_price / entry_price - 1.0) * 100.0
    if not math.isfinite(reentry_price) or reentry_price <= 0:
        return float("nan")
    gross = (first_exit_price / entry_price) * (final_exit_price / reentry_price)
    return (gross - 1.0) * 100.0


def first_reentry_price(
    closes: list[float],
    times: list[str],
    start_idx: int,
    end_idx: int,
    threshold_price: float,
    fallback_close: bool,
) -> tuple[float | None, str]:
    for idx in range(start_idx, end_idx + 1):
        px = closes[idx]
        if px >= threshold_price:
            return px, times[idx]
    if fallback_close:
        return closes[end_idx], times[end_idx]
    return None, ""


def simulate_candidate(
    closes: list[float],
    times: list[str],
    entry_idx: int,
    close_idx: int,
    entry_price: float,
    final_exit_price: float,
    baseline_ret_pct: float,
    candidate: Candidate,
) -> dict:
    armed = False
    peak_price = entry_price
    stop_idx = None

    for idx in range(entry_idx + 1, close_idx + 1):
        px = closes[idx]
        if px > peak_price:
            peak_price = px
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
            "stop_price": float("nan"),
            "reentry_dt": "",
            "reentry_price": float("nan"),
        }

    stop_price = closes[stop_idx]
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

    return {
        "strategy_ret_pct": total_return_pct(entry_price, stop_price, final_exit_price, reentry_price),
        "stopped": True,
        "reentered": reentry_price is not None,
        "stop_dt": times[stop_idx],
        "stop_price": stop_price,
        "reentry_dt": reentry_dt,
        "reentry_price": reentry_price if reentry_price is not None else float("nan"),
    }


def summarize(detail_df: pd.DataFrame, strategy_col: str, stopped_col: str, reentered_col: str) -> dict:
    baseline = pd.to_numeric(detail_df["baseline_ret_pct"], errors="coerce")
    strategy = pd.to_numeric(detail_df[strategy_col], errors="coerce")
    stopped = detail_df[stopped_col].fillna(False).astype(bool)
    reentered = detail_df[reentered_col].fillna(False).astype(bool)
    return {
        "n_trades": int(len(detail_df)),
        "n_stopped": int(stopped.sum()),
        "stop_rate_pct": float(stopped.mean() * 100.0) if len(detail_df) else 0.0,
        "n_reentered": int((stopped & reentered).sum()),
        "reentry_rate_among_stopped_pct": float(((stopped & reentered).sum() / stopped.sum()) * 100.0) if stopped.sum() else float("nan"),
        "baseline_mean_ret_pct": float(baseline.mean()),
        "strategy_mean_ret_pct": float(strategy.mean()),
        "delta_mean_ret_pct": float(strategy.mean() - baseline.mean()),
        "baseline_win_rate_pct": float((baseline > 0).mean() * 100.0),
        "strategy_win_rate_pct": float((strategy > 0).mean() * 100.0),
    }


def fmt(value: float) -> float | str:
    return "" if pd.isna(value) else value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic path-based exits on the recent VM recommendation backtest set."
    )
    parser.add_argument("--input", default=LIVE_PATHS.vm_backtest_detail_file)
    parser.add_argument("--cache-dir", default=LIVE_PATHS.cache_dir)
    parser.add_argument("--summary-out", default=LIVE_PATHS.vm_dynamic_path_summary_file)
    parser.add_argument("--grid-out", default=LIVE_PATHS.vm_dynamic_path_grid_file)
    parser.add_argument("--detail-out", default=LIVE_PATHS.vm_dynamic_path_detail_file)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    src = pd.read_csv(args.input)
    src["ret_1d_pct"] = pd.to_numeric(src["ret_1d_pct"], errors="coerce")
    src["close_1d"] = pd.to_numeric(src["close_1d"], errors="coerce")
    src["entry_price"] = pd.to_numeric(src["entry_price"], errors="coerce")
    src["entry_dt_et"] = pd.to_datetime(src["entry_dt_et"], errors="coerce", utc=True).dt.tz_convert(ET)
    src = src.dropna(subset=["ret_1d_pct", "entry_dt_et", "entry_price"]).copy()

    client = RESTClient(api_key=api_key, retries=3)
    cache_dir = Path(args.cache_dir)
    minute_cache: dict[tuple[str, str], list[dict]] = {}
    candidates = build_candidates()
    detail_rows: list[dict] = []

    for _, row in src.iterrows():
        ticker = str(row["ticker"])
        entry_dt_et = row["entry_dt_et"].to_pydatetime()
        day_key = (ticker, f"{entry_dt_et.date():%Y-%m-%d}")
        if day_key not in minute_cache:
            minute_cache[day_key] = fetch_minute_bars(client, cache_dir, ticker, entry_dt_et.date())
        minute_bars = [bar for bar in minute_cache[day_key] if is_regular_session_bar(bar)]
        minute_bars.sort(key=lambda b: int(b["t"]))
        if not minute_bars:
            continue

        entry_idx = find_entry_bar_index(minute_bars, entry_dt_et)
        if entry_idx is None:
            continue
        close_idx = find_same_day_close_index(minute_bars, entry_idx)
        if close_idx is None or close_idx <= entry_idx:
            continue

        entry_price = float(row["entry_price"])
        baseline_ret_pct = float(row["ret_1d_pct"])
        final_exit_price = float(row["close_1d"]) if math.isfinite(float(row["close_1d"])) and float(row["close_1d"]) > 0 else implied_exit_price(entry_price, baseline_ret_pct)
        if final_exit_price is None or not math.isfinite(final_exit_price) or final_exit_price <= 0:
            continue

        closes = [float(bar["c"]) for bar in minute_bars]
        times = [bar_dt_et(bar).strftime("%Y-%m-%d %H:%M:%S%z") for bar in minute_bars]
        out = {
            "scored_at": row["scored_at"],
            "scored_at_et": row["scored_at_et"],
            "ticker": ticker,
            "event_key": row["event_key"],
            "trade_date": row["trade_date"],
            "score_1d": row["score_1d"],
            "score_3d": row["score_3d"],
            "entry_dt_et": entry_dt_et.strftime("%Y-%m-%d %H:%M:%S%z"),
            "entry_price": entry_price,
            "baseline_ret_pct": baseline_ret_pct,
            "same_day_close_dt_et": times[close_idx],
            "same_day_close_ret_pct": (closes[close_idx] / entry_price - 1.0) * 100.0,
        }

        for candidate in candidates:
            result = simulate_candidate(
                closes=closes,
                times=times,
                entry_idx=entry_idx,
                close_idx=close_idx,
                entry_price=entry_price,
                final_exit_price=final_exit_price,
                baseline_ret_pct=baseline_ret_pct,
                candidate=candidate,
            )
            prefix = candidate.name
            out[f"{prefix}_strategy_ret_pct"] = result["strategy_ret_pct"]
            out[f"{prefix}_stopped"] = result["stopped"]
            out[f"{prefix}_reentered"] = result["reentered"]
            out[f"{prefix}_stop_dt"] = result["stop_dt"]
            out[f"{prefix}_stop_price"] = result["stop_price"]
            out[f"{prefix}_reentry_dt"] = result["reentry_dt"]
            out[f"{prefix}_reentry_price"] = result["reentry_price"]

        detail_rows.append(out)

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise RuntimeError("No usable completed VM trades found for dynamic-path evaluation.")

    grid_rows = []
    for candidate in candidates:
        prefix = candidate.name
        metrics = summarize(
            detail,
            strategy_col=f"{prefix}_strategy_ret_pct",
            stopped_col=f"{prefix}_stopped",
            reentered_col=f"{prefix}_reentered",
        )
        grid_rows.append(
            {
                "candidate": prefix,
                "arm_gain_pct": candidate.arm_gain_pct,
                "trail_drawdown_pct": candidate.trail_drawdown_pct,
                "reentry_mode": candidate.reentry_mode,
                **{k: fmt(v) for k, v in metrics.items()},
            }
        )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["strategy_mean_ret_pct", "delta_mean_ret_pct", "strategy_win_rate_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    historical_prefix = HISTORICAL_SELECTED.name
    hist_metrics = summarize(
        detail,
        strategy_col=f"{historical_prefix}_strategy_ret_pct",
        stopped_col=f"{historical_prefix}_stopped",
        reentered_col=f"{historical_prefix}_reentered",
    )

    vm_best = grid.iloc[0]
    summary_rows = [
        {"metric": "usable_completed_vm_trades", "value": int(len(detail))},
        {"metric": "historical_selected_candidate", "value": historical_prefix},
        {"metric": "historical_baseline_mean_ret_pct", "value": hist_metrics["baseline_mean_ret_pct"]},
        {"metric": "historical_strategy_mean_ret_pct", "value": hist_metrics["strategy_mean_ret_pct"]},
        {"metric": "historical_delta_mean_ret_pct", "value": hist_metrics["delta_mean_ret_pct"]},
        {"metric": "historical_n_stopped", "value": hist_metrics["n_stopped"]},
        {"metric": "historical_stop_rate_pct", "value": hist_metrics["stop_rate_pct"]},
        {"metric": "historical_n_reentered", "value": hist_metrics["n_reentered"]},
        {"metric": "historical_reentry_rate_among_stopped_pct", "value": fmt(hist_metrics["reentry_rate_among_stopped_pct"])},
        {"metric": "vm_best_candidate_in_sample", "value": str(vm_best["candidate"])},
        {"metric": "vm_best_strategy_mean_ret_pct", "value": float(vm_best["strategy_mean_ret_pct"])},
        {"metric": "vm_best_delta_mean_ret_pct", "value": float(vm_best["delta_mean_ret_pct"])},
        {"metric": "vm_best_n_stopped", "value": int(vm_best["n_stopped"])},
        {"metric": "vm_best_stop_rate_pct", "value": float(vm_best["stop_rate_pct"])},
        {"metric": "vm_best_n_reentered", "value": int(vm_best["n_reentered"])},
    ]

    historical_detail = detail.copy()
    historical_detail["selected_candidate"] = historical_prefix
    historical_detail["selected_strategy_ret_pct"] = historical_detail[f"{historical_prefix}_strategy_ret_pct"]
    historical_detail["selected_improvement_pct"] = historical_detail["selected_strategy_ret_pct"] - historical_detail["baseline_ret_pct"]
    historical_detail["selected_stopped"] = historical_detail[f"{historical_prefix}_stopped"]
    historical_detail["selected_reentered"] = historical_detail[f"{historical_prefix}_reentered"]
    historical_detail["selected_stop_dt"] = historical_detail[f"{historical_prefix}_stop_dt"]
    historical_detail["selected_reentry_dt"] = historical_detail[f"{historical_prefix}_reentry_dt"]

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.summary_out, index=False)
    grid.to_csv(args.grid_out, index=False)
    historical_detail.to_csv(args.detail_out, index=False)

    print(f"usable_completed_vm_trades={len(detail)}")
    print(f"historical_selected_candidate={historical_prefix}")
    print(f"historical_delta_mean_ret_pct={hist_metrics['delta_mean_ret_pct']:.4f}")
    print(f"historical_stop_rate_pct={hist_metrics['stop_rate_pct']:.2f}")
    print(f"vm_best_candidate_in_sample={vm_best['candidate']}")
    print(f"vm_best_delta_mean_ret_pct={float(vm_best['delta_mean_ret_pct']):.4f}")
    print(f"summary_out={args.summary_out}")
    print(f"grid_out={args.grid_out}")
    print(f"detail_out={args.detail_out}")


if __name__ == "__main__":
    main()
