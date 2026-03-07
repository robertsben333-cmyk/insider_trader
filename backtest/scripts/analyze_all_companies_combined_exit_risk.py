from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
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

PROFIT_ARM_GAIN_PCT = 2.0
PROFIT_TRAIL_DD_PCT = 0.5

if str(BASE) not in os.sys.path:
    os.sys.path.insert(0, str(BASE))

from backtest.scripts.evaluate_investable_risk_rules import (  # noqa: E402
    chrono_split_60_20_20,
    load_scored_day1,
)


@dataclass(frozen=True)
class Candidate:
    family: str
    loss_mode: str
    loss_param: float | None

    @property
    def name(self) -> str:
        if self.loss_mode == "none":
            return "trail_only"
        if self.loss_mode == "static":
            return f"trail_plus_static_loss_{self.loss_param:g}"
        if self.loss_mode == "vol":
            return f"trail_plus_vol_loss_{self.loss_param:g}x"
        raise ValueError(f"Unknown loss_mode: {self.loss_mode}")


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


def implied_exit_price_from_ret(entry_price: float, ret_pct: float) -> float | None:
    if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(ret_pct):
        return float(entry_price * (1.0 + float(ret_pct) / 100.0))
    return None


def risk_stats(returns: np.ndarray) -> dict:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    mean_ret = float(np.mean(arr)) if n else np.nan
    std_ret = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    median_ret = float(np.median(arr)) if n else np.nan
    win_rate = float((arr > 0).mean() * 100.0) if n else np.nan
    risk_adj = float(mean_ret / std_ret) if n > 1 and np.isfinite(std_ret) and std_ret > 0 else np.nan
    return {
        "n_trades": n,
        "mean_ret_pct": mean_ret,
        "std_ret_pct": std_ret,
        "median_ret_pct": median_ret,
        "win_rate_pct": win_rate,
        "risk_adj_mean_over_std": risk_adj,
    }


def choose_loss_threshold(candidate: Candidate, daily_vol_pct: float) -> float | None:
    if candidate.loss_mode == "none":
        return None
    if candidate.loss_mode == "static":
        return -float(candidate.loss_param)
    if candidate.loss_mode == "vol":
        if np.isfinite(daily_vol_pct) and daily_vol_pct > 0:
            return -float(candidate.loss_param) * float(daily_vol_pct)
        return None
    raise ValueError(f"Unknown loss_mode: {candidate.loss_mode}")


def simulate_candidate(
    closes: np.ndarray,
    times: list[str],
    entry_idx: int,
    close_idx: int,
    entry_price: float,
    baseline_ret_pct: float,
    daily_vol_pct: float,
    candidate: Candidate,
) -> dict:
    peak_price = entry_price
    profit_armed = False
    loss_threshold_pct = choose_loss_threshold(candidate, daily_vol_pct)
    stop_idx = None
    stop_reason = ""

    for idx in range(entry_idx + 1, close_idx + 1):
        px = float(closes[idx])
        if px > peak_price:
            peak_price = px

        gain_pct = (peak_price / entry_price - 1.0) * 100.0
        if gain_pct >= PROFIT_ARM_GAIN_PCT:
            profit_armed = True

        ret_from_entry_pct = (px / entry_price - 1.0) * 100.0
        drawdown_from_peak_pct = (px / peak_price - 1.0) * 100.0

        trail_trigger = profit_armed and drawdown_from_peak_pct <= -PROFIT_TRAIL_DD_PCT
        loss_trigger = loss_threshold_pct is not None and ret_from_entry_pct <= loss_threshold_pct

        if trail_trigger or loss_trigger:
            stop_idx = idx
            stop_reason = "loss_stop" if loss_trigger and not trail_trigger else "profit_trail"
            if loss_trigger and trail_trigger:
                stop_reason = "loss_stop"
            break

    if stop_idx is None:
        return {
            "strategy_ret_pct": baseline_ret_pct,
            "stopped": False,
            "stop_reason": "",
            "stop_dt": "",
            "stop_price": np.nan,
        }

    stop_price = float(closes[stop_idx])
    strategy_ret_pct = (stop_price / entry_price - 1.0) * 100.0
    return {
        "strategy_ret_pct": strategy_ret_pct,
        "stopped": True,
        "stop_reason": stop_reason,
        "stop_dt": times[stop_idx],
        "stop_price": stop_price,
    }


def compute_row(row: pd.Series, cache_dir: Path, candidates: list[Candidate]) -> dict | None:
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

    close_idx = find_same_day_close_index(bars, entry_idx)
    if close_idx is None or close_idx <= entry_idx:
        return None

    entry_price = float(bars[entry_idx]["o"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    baseline_ret_pct = float(row["ret_pct"])
    close_1d_price = implied_exit_price_from_ret(entry_price, baseline_ret_pct)
    if close_1d_price is None or not np.isfinite(close_1d_price) or close_1d_price <= 0:
        return None

    daily_vol_pct = np.nan
    prior_30d_vol = pd.to_numeric(pd.Series([row.get("prior_30d_vol")]), errors="coerce").iloc[0]
    if np.isfinite(prior_30d_vol) and prior_30d_vol >= 0:
        daily_vol_pct = float(prior_30d_vol) / np.sqrt(252.0) * 100.0

    closes = np.array([float(b["c"]) for b in bars], dtype=float)
    times = [bar_dt_et(b).strftime("%Y-%m-%d %H:%M:%S") for b in bars]

    out = {
        "split": row["split"],
        "ticker": ticker,
        "buy_datetime": pd.Timestamp(row["buy_datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_ret_pct": baseline_ret_pct,
        "entry_price": entry_price,
        "close_1d_price": float(close_1d_price),
        "daily_vol_pct": daily_vol_pct,
        "entry_bar_time": times[entry_idx],
        "same_day_close_time": times[close_idx],
        "same_day_close_ret_pct": (closes[close_idx] / entry_price - 1.0) * 100.0,
    }

    for candidate in candidates:
        result = simulate_candidate(
            closes=closes,
            times=times,
            entry_idx=entry_idx,
            close_idx=close_idx,
            entry_price=entry_price,
            baseline_ret_pct=baseline_ret_pct,
            daily_vol_pct=daily_vol_pct,
            candidate=candidate,
        )
        prefix = candidate.name
        out[f"{prefix}_strategy_ret_pct"] = result["strategy_ret_pct"]
        out[f"{prefix}_stopped"] = result["stopped"]
        out[f"{prefix}_stop_reason"] = result["stop_reason"]
        out[f"{prefix}_stop_dt"] = result["stop_dt"]
        out[f"{prefix}_stop_price"] = result["stop_price"]

    return out


def summarize_candidate(detail: pd.DataFrame, candidate: Candidate) -> dict:
    prefix = candidate.name
    eligible = detail[detail[f"{prefix}_strategy_ret_pct"].notna()].copy()
    strategy = pd.to_numeric(eligible[f"{prefix}_strategy_ret_pct"], errors="coerce").to_numpy(dtype=float)
    stopped = eligible[f"{prefix}_stopped"].to_numpy(dtype=bool)
    stop_reason = eligible[f"{prefix}_stop_reason"].astype(str)

    stats = risk_stats(strategy)
    stats.update(
        {
            "n_stopped": int(stopped.sum()),
            "stop_rate_pct": float(stopped.mean() * 100.0) if len(stopped) else 0.0,
            "n_profit_trail_stops": int((stop_reason == "profit_trail").sum()),
            "n_loss_stops": int((stop_reason == "loss_stop").sum()),
        }
    )
    return stats


def build_candidates() -> list[Candidate]:
    candidates = [Candidate(family="trail_only", loss_mode="none", loss_param=None)]
    candidates.extend(
        Candidate(family="static_loss", loss_mode="static", loss_param=param)
        for param in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    )
    candidates.extend(
        Candidate(family="dynamic_loss", loss_mode="vol", loss_param=param)
        for param in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    )
    return candidates


def apply_selected(detail: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    out = detail.copy()
    if candidate_name == "baseline":
        out["selected_candidate"] = "baseline"
        out["selected_strategy_ret_pct"] = out["baseline_ret_pct"]
        out["selected_improvement_pct"] = 0.0
        out["selected_stopped"] = False
        out["selected_stop_reason"] = ""
        return out
    out["selected_candidate"] = candidate_name
    out["selected_strategy_ret_pct"] = out[f"{candidate_name}_strategy_ret_pct"]
    out["selected_improvement_pct"] = out["selected_strategy_ret_pct"] - out["baseline_ret_pct"]
    out["selected_stopped"] = out[f"{candidate_name}_stopped"]
    out["selected_stop_reason"] = out[f"{candidate_name}_stop_reason"]
    return out


def select_best(train_grid: pd.DataFrame, family: str) -> pd.Series:
    family_grid = train_grid[train_grid["family"] == family].copy()
    family_grid = family_grid.sort_values(
        ["train_risk_adj_mean_over_std", "train_mean_ret_pct", "train_std_ret_pct"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return family_grid.iloc[0]


def fmt(value) -> str | float:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine the selected positive trailing rule with static or dynamic downside stops and evaluate mean/std risk on train/val/test."
    )
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--detail-out", default=str(OUT_DIR / "all_companies_combined_exit_risk_detail.csv"))
    parser.add_argument("--grid-out", default=str(OUT_DIR / "all_companies_combined_exit_risk_grid.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "all_companies_combined_exit_risk_summary.csv"))
    args = parser.parse_args()

    candidates = build_candidates()
    scored = load_scored_day1().copy()
    n_train, split_80 = chrono_split_60_20_20(len(scored))
    scored["split"] = "train"
    scored.loc[n_train:split_80 - 1, "split"] = "val"
    scored.loc[split_80:, "split"] = "test"
    scored = scored.dropna(subset=["buy_datetime", "ret_pct"]).copy()

    cache_dir = Path(args.cache_dir)
    detail_rows = []
    missing_rows = 0
    for _, row in scored.iterrows():
        out = compute_row(row, cache_dir, candidates)
        if out is None:
            missing_rows += 1
            continue
        detail_rows.append(out)

    detail = pd.DataFrame(detail_rows)
    split_frames = {name: detail[detail["split"] == name].copy() for name in ["train", "val", "test"]}

    baseline_rows = {}
    for split_name, split_df in split_frames.items():
        stats = risk_stats(pd.to_numeric(split_df["baseline_ret_pct"], errors="coerce").to_numpy(dtype=float))
        baseline_rows[split_name] = stats

    grid_rows = []
    for candidate in candidates:
        row = {
            "candidate": candidate.name,
            "family": candidate.family,
            "loss_mode": candidate.loss_mode,
            "loss_param": "" if candidate.loss_param is None else candidate.loss_param,
        }
        for split_name, split_df in split_frames.items():
            metrics = summarize_candidate(split_df, candidate)
            for key, value in metrics.items():
                row[f"{split_name}_{key}"] = fmt(value)
            row[f"{split_name}_delta_mean_ret_pct"] = fmt(metrics["mean_ret_pct"] - baseline_rows[split_name]["mean_ret_pct"])
            if np.isfinite(metrics["std_ret_pct"]) and np.isfinite(baseline_rows[split_name]["std_ret_pct"]):
                row[f"{split_name}_delta_std_ret_pct"] = fmt(metrics["std_ret_pct"] - baseline_rows[split_name]["std_ret_pct"])
            else:
                row[f"{split_name}_delta_std_ret_pct"] = ""
        grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)
    best_trail_only = select_best(grid, "trail_only")
    best_static = select_best(grid, "static_loss")
    best_dynamic = select_best(grid, "dynamic_loss")

    comparison_candidates = [
        ("baseline", None),
        (str(best_trail_only["candidate"]), best_trail_only),
        (str(best_static["candidate"]), best_static),
        (str(best_dynamic["candidate"]), best_dynamic),
    ]

    summary_rows = [
        {"metric": "profit_trail_rule", "value": f"arm_after_{PROFIT_ARM_GAIN_PCT:g}_trail_dd_{PROFIT_TRAIL_DD_PCT:g}"},
        {"metric": "all_rows_before_filter", "value": int(len(scored))},
        {"metric": "rows_with_usable_same_day_minutes", "value": int(len(detail))},
        {"metric": "usable_row_pct", "value": float(len(detail) / len(scored) * 100.0) if len(scored) else 0.0},
        {"metric": "missing_or_unusable_rows", "value": int(missing_rows)},
        {"metric": "baseline_train_mean_ret_pct", "value": baseline_rows["train"]["mean_ret_pct"]},
        {"metric": "baseline_train_std_ret_pct", "value": baseline_rows["train"]["std_ret_pct"]},
        {"metric": "baseline_val_mean_ret_pct", "value": baseline_rows["val"]["mean_ret_pct"]},
        {"metric": "baseline_val_std_ret_pct", "value": baseline_rows["val"]["std_ret_pct"]},
        {"metric": "baseline_test_mean_ret_pct", "value": baseline_rows["test"]["mean_ret_pct"]},
        {"metric": "baseline_test_std_ret_pct", "value": baseline_rows["test"]["std_ret_pct"]},
        {"metric": "best_trail_only_candidate", "value": str(best_trail_only["candidate"])},
        {"metric": "best_static_candidate", "value": str(best_static["candidate"])},
        {"metric": "best_dynamic_candidate", "value": str(best_dynamic["candidate"])},
    ]

    for label, row in comparison_candidates[1:]:
        for split_name in ["train", "val", "test"]:
            summary_rows.append({"metric": f"{label}_{split_name}_mean_ret_pct", "value": row[f"{split_name}_mean_ret_pct"]})
            summary_rows.append({"metric": f"{label}_{split_name}_std_ret_pct", "value": row[f"{split_name}_std_ret_pct"]})
            summary_rows.append({"metric": f"{label}_{split_name}_risk_adj_mean_over_std", "value": row[f"{split_name}_risk_adj_mean_over_std"]})
            summary_rows.append({"metric": f"{label}_{split_name}_delta_mean_ret_pct", "value": row[f"{split_name}_delta_mean_ret_pct"]})
            summary_rows.append({"metric": f"{label}_{split_name}_delta_std_ret_pct", "value": row[f"{split_name}_delta_std_ret_pct"]})
            summary_rows.append({"metric": f"{label}_{split_name}_stop_rate_pct", "value": row[f"{split_name}_stop_rate_pct"]})
            summary_rows.append({"metric": f"{label}_{split_name}_n_loss_stops", "value": row[f"{split_name}_n_loss_stops"]})
            summary_rows.append({"metric": f"{label}_{split_name}_n_profit_trail_stops", "value": row[f"{split_name}_n_profit_trail_stops"]})

    grid = grid.sort_values(
        ["train_risk_adj_mean_over_std", "train_mean_ret_pct", "val_risk_adj_mean_over_std"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    detail_selected = apply_selected(detail, str(best_dynamic["candidate"]))

    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail_selected.to_csv(args.detail_out, index=False)
    grid.to_csv(args.grid_out, index=False)
    pd.DataFrame(summary_rows).to_csv(args.summary_out, index=False)

    print(f"rows_total={len(scored)}")
    print(f"rows_usable={len(detail)}")
    print(f"usable_row_pct={len(detail) / len(scored) * 100.0 if len(scored) else 0.0:.2f}")
    print(f"best_trail_only={best_trail_only['candidate']}")
    print(f"best_static={best_static['candidate']}")
    print(f"best_dynamic={best_dynamic['candidate']}")
    print(f"summary_out={args.summary_out}")
    print(f"grid_out={args.grid_out}")
    print(f"detail_out={args.detail_out}")


if __name__ == "__main__":
    main()
