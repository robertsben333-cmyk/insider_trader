from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

import train_models
from backtest.scripts.evaluate_investable_risk_rules import chrono_split_60_20_20, load_scored_day1

ET = train_models.ET
DEFAULT_OPEN_FILL = time(9, 45)
MARKET_CLOSE = time(16, 0)
DEFAULT_RAW_PRED_CUTOFF = 1.1033359764462474


@dataclass(frozen=True)
class ExitScenario:
    name: str
    offset_days: int


SCENARIOS = [
    ExitScenario("same_day_close", 0),
    ExitScenario("next_day_close", 1),
    ExitScenario("second_day_close", 2),
]


def next_business_day(d: date) -> date:
    out = d + timedelta(days=1)
    while out.weekday() >= 5:
        out += timedelta(days=1)
    return out


def advance_business_days(d: date, days: int) -> date:
    out = d
    for _ in range(days):
        out = next_business_day(out)
    return out


def classify_intraday_time_bucket(ts: pd.Timestamp) -> str:
    t = ts.timetz().replace(tzinfo=None)
    if t < time(11, 0):
        return "09:46-10:59"
    if t < time(13, 0):
        return "11:00-12:59"
    return "13:00-15:59"


def summarize_returns(frame: pd.DataFrame, ret_col: str) -> dict[str, float | int | str]:
    vals = pd.to_numeric(frame[ret_col], errors="coerce").dropna()
    n = int(len(vals))
    if n == 0:
        return {
            "n_trades": 0,
            "mean_ret_pct": np.nan,
            "median_ret_pct": np.nan,
            "win_rate_pct": np.nan,
        }
    arr = vals.to_numpy(dtype=float)
    return {
        "n_trades": n,
        "mean_ret_pct": float(np.mean(arr)),
        "median_ret_pct": float(np.median(arr)),
        "win_rate_pct": float(np.mean(arr > 0) * 100.0),
    }


def load_day_close(cache_dir: Path, ticker: str, target_date: date) -> float | None:
    from_d = target_date
    to_d = target_date
    bars = train_models.fetch_day_bars(cache_dir, ticker, from_d, to_d)
    for bar in bars:
        bar_date = pd.Timestamp(bar["t"], unit="ms", tz="UTC").tz_convert(ET).date()
        if bar_date == target_date:
            close = bar.get("c")
            if close is not None:
                return float(close)
    return None


def build_detail(scored: pd.DataFrame, cache_dir: Path, pred_cutoff: float) -> pd.DataFrame:
    n_train, split_80 = chrono_split_60_20_20(len(scored))
    subset = scored.iloc[n_train:].copy()
    subset["split"] = np.where(np.arange(len(subset)) < (split_80 - n_train), "val", "test")

    subset["buy_datetime"] = pd.to_datetime(subset["buy_datetime"], errors="coerce")
    subset["buy_price"] = pd.to_numeric(subset["buy_price"], errors="coerce")
    subset["pred_mean4"] = pd.to_numeric(subset["pred_mean4"], errors="coerce")
    subset = subset.dropna(subset=["buy_datetime", "buy_price", "pred_mean4"]).copy()
    subset = subset[subset["pred_mean4"] >= float(pred_cutoff)].copy()

    buy_time = subset["buy_datetime"].dt.time
    subset = subset[
        (subset["buy_datetime"].dt.weekday < 5)
        & (buy_time > DEFAULT_OPEN_FILL)
        & (buy_time < MARKET_CLOSE)
    ].copy()
    subset["entry_time_bucket"] = subset["buy_datetime"].apply(classify_intraday_time_bucket)

    rows: list[dict] = []
    close_cache: dict[tuple[str, date], float | None] = {}

    for row in subset.itertuples(index=False):
        entry_date = pd.Timestamp(row.buy_datetime).date()
        entry_price = float(row.buy_price)
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        out = {
            "split": row.split,
            "ticker": str(row.ticker),
            "trade_date": pd.Timestamp(row.trade_date).strftime("%Y-%m-%d"),
            "buy_datetime": pd.Timestamp(row.buy_datetime).strftime("%Y-%m-%d %H:%M:%S"),
            "buy_price": entry_price,
            "pred_mean4": float(row.pred_mean4),
            "entry_time_bucket": row.entry_time_bucket,
        }

        missing = False
        for scenario in SCENARIOS:
            exit_date = advance_business_days(entry_date, scenario.offset_days)
            cache_key = (str(row.ticker), exit_date)
            if cache_key not in close_cache:
                close_cache[cache_key] = load_day_close(cache_dir, str(row.ticker), exit_date)
            exit_close = close_cache[cache_key]
            if exit_close is None or exit_close <= 0:
                missing = True
                break
            out[f"{scenario.name}_exit_date"] = exit_date.isoformat()
            out[f"{scenario.name}_close"] = exit_close
            out[f"{scenario.name}_ret_pct"] = (exit_close / entry_price - 1.0) * 100.0

        if missing:
            continue

        out["delta_second_vs_next_pct"] = out["second_day_close_ret_pct"] - out["next_day_close_ret_pct"]
        out["delta_next_vs_same_pct"] = out["next_day_close_ret_pct"] - out["same_day_close_ret_pct"]
        rows.append(out)

    return pd.DataFrame(rows)


def build_split_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for split_name in ["val", "test"]:
        split_df = detail[detail["split"] == split_name].copy()
        for scenario in SCENARIOS:
            metrics = summarize_returns(split_df, f"{scenario.name}_ret_pct")
            rows.append(
                {
                    "split": split_name,
                    "scenario": scenario.name,
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["split", "mean_ret_pct"], ascending=[True, False]).reset_index(drop=True)


def build_bucket_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for split_name in ["val", "test"]:
        split_df = detail[detail["split"] == split_name].copy()
        for bucket_name in ["09:46-10:59", "11:00-12:59", "13:00-15:59"]:
            bucket_df = split_df[split_df["entry_time_bucket"] == bucket_name].copy()
            if bucket_df.empty:
                continue
            base_row = {
                "split": split_name,
                "entry_time_bucket": bucket_name,
                "n_trades": int(len(bucket_df)),
                "mean_delta_second_vs_next_pct": float(bucket_df["delta_second_vs_next_pct"].mean()),
                "median_delta_second_vs_next_pct": float(bucket_df["delta_second_vs_next_pct"].median()),
                "pct_second_beats_next": float((bucket_df["delta_second_vs_next_pct"] > 0).mean() * 100.0),
                "mean_delta_next_vs_same_pct": float(bucket_df["delta_next_vs_same_pct"].mean()),
                "pct_next_beats_same": float((bucket_df["delta_next_vs_same_pct"] > 0).mean() * 100.0),
            }
            for scenario in SCENARIOS:
                metrics = summarize_returns(bucket_df, f"{scenario.name}_ret_pct")
                rows.append(
                    {
                        **base_row,
                        "scenario": scenario.name,
                        "scenario_mean_ret_pct": metrics["mean_ret_pct"],
                        "scenario_median_ret_pct": metrics["median_ret_pct"],
                        "scenario_win_rate_pct": metrics["win_rate_pct"],
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["split", "entry_time_bucket", "scenario_mean_ret_pct"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_best_summary(split_summary: pd.DataFrame, bucket_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for split_name in ["val", "test"]:
        split_df = split_summary[split_summary["split"] == split_name].copy()
        if not split_df.empty:
            best = split_df.sort_values("mean_ret_pct", ascending=False).iloc[0]
            rows.append(
                {
                    "scope": "split",
                    "split": split_name,
                    "entry_time_bucket": "",
                    "best_scenario": best["scenario"],
                    "best_mean_ret_pct": float(best["mean_ret_pct"]),
                    "runner_up_mean_ret_pct": float(split_df.sort_values("mean_ret_pct", ascending=False).iloc[1]["mean_ret_pct"]),
                }
            )
        for bucket_name in ["09:46-10:59", "11:00-12:59", "13:00-15:59"]:
            bucket_df = bucket_summary[
                (bucket_summary["split"] == split_name) & (bucket_summary["entry_time_bucket"] == bucket_name)
            ].copy()
            if bucket_df.empty:
                continue
            best = bucket_df.sort_values("scenario_mean_ret_pct", ascending=False).iloc[0]
            sorted_bucket = bucket_df.sort_values("scenario_mean_ret_pct", ascending=False).reset_index(drop=True)
            runner_up = float(sorted_bucket.iloc[1]["scenario_mean_ret_pct"]) if len(sorted_bucket) > 1 else np.nan
            rows.append(
                {
                    "scope": "bucket",
                    "split": split_name,
                    "entry_time_bucket": bucket_name,
                    "best_scenario": best["scenario"],
                    "best_mean_ret_pct": float(best["scenario_mean_ret_pct"]),
                    "runner_up_mean_ret_pct": runner_up,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether next-day close is the best fixed exit for true intraday entries on validation/test."
    )
    parser.add_argument("--cache-dir", default="backtest/data/price_cache")
    parser.add_argument("--pred-cutoff", type=float, default=DEFAULT_RAW_PRED_CUTOFF)
    parser.add_argument("--detail-out", default="backtest/out/intraday_entry_timing_sensitivity_detail.csv")
    parser.add_argument("--split-summary-out", default="backtest/out/intraday_entry_timing_sensitivity_split_summary.csv")
    parser.add_argument("--bucket-summary-out", default="backtest/out/intraday_entry_timing_sensitivity_bucket_summary.csv")
    parser.add_argument("--best-out", default="backtest/out/intraday_entry_timing_sensitivity_best.csv")
    args = parser.parse_args()

    scored = load_scored_day1()
    detail = build_detail(scored=scored, cache_dir=Path(args.cache_dir), pred_cutoff=float(args.pred_cutoff))
    if detail.empty:
        raise RuntimeError("No usable intraday investable rows found.")

    split_summary = build_split_summary(detail)
    bucket_summary = build_bucket_summary(detail)
    best_summary = build_best_summary(split_summary, bucket_summary)

    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_out, index=False)
    split_summary.to_csv(args.split_summary_out, index=False)
    bucket_summary.to_csv(args.bucket_summary_out, index=False)
    best_summary.to_csv(args.best_out, index=False)

    print(f"detail_rows={len(detail)}")
    print(f"detail_out={args.detail_out}")
    print(f"split_summary_out={args.split_summary_out}")
    print(f"bucket_summary_out={args.bucket_summary_out}")
    print(f"best_out={args.best_out}")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
