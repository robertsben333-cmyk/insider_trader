from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "backtest" / "out"
COMPARE_SCRIPT = BASE / "research" / "scripts" / "compare_exit_recycling_strategies.py"
DEPLOY_SCRIPT = BASE / "research" / "scripts" / "deploy_tplus2_open_day1_live.py"
DEFAULT_CUTOFF_CSV = BASE / "backtest" / "out" / "investable_decile_score_sweep_0005_tplus2_open.csv"
MODEL_DIR = BASE / "models" / "prod4"

START_BUDGET_EUR = 10_000.0
TARGET_COL = "return_2d_open_pct"
ACTUAL_RETURN_COL = "stock_only_return_2d_open_pct_raw"
EXIT_DATETIME_COL = "exit_2d_open_datetime"


@dataclass
class Position:
    ticker: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    invested_eur: float
    ret_pct: float


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_raw_cutoff(cutoff_csv: Path, decile_score_threshold: float) -> float:
    table = pd.read_csv(cutoff_csv)
    table["decile_score_threshold"] = pd.to_numeric(table["decile_score_threshold"], errors="coerce")
    table["raw_pred_mean4_cutoff"] = pd.to_numeric(table["raw_pred_mean4_cutoff"], errors="coerce")
    table = table.dropna(subset=["decile_score_threshold", "raw_pred_mean4_cutoff"]).copy()
    if table.empty:
        raise RuntimeError(f"No valid cutoffs found in {cutoff_csv}")
    table["distance"] = (table["decile_score_threshold"] - float(decile_score_threshold)).abs()
    best = table.sort_values(["distance", "decile_score_threshold"]).iloc[0]
    return float(best["raw_pred_mean4_cutoff"])


def classify_time_bucket(ts: pd.Timestamp) -> str:
    t = ts.timetz().replace(tzinfo=None)
    if t < time(11, 0):
        return "09:30-10:59"
    if t < time(13, 0):
        return "11:00-12:59"
    return "13:00-16:00"


def build_candidates(raw_cutoff: float) -> pd.DataFrame:
    import joblib

    deploy = load_module("deploy_tplus2_open_day1_live", DEPLOY_SCRIPT)
    split = deploy.prepare_split()
    test_start = len(split.X_tr) + len(split.X_va)
    scored_test = split.sub.iloc[test_start:].copy().reset_index(drop=True)
    if EXIT_DATETIME_COL not in scored_test.columns and "exit_2d_open_date" in scored_test.columns:
        valid_exit_rows = scored_test["exit_2d_open_date"].apply(lambda d: pd.notna(d))
        scored_test[EXIT_DATETIME_COL] = pd.NaT
        scored_test.loc[valid_exit_rows, EXIT_DATETIME_COL] = (
            pd.to_datetime(scored_test.loc[valid_exit_rows, "exit_2d_open_date"].astype(str), errors="coerce")
            .dt.tz_localize(deploy.ET, ambiguous="NaT", nonexistent="shift_forward")
            + pd.Timedelta(hours=9, minutes=30)
        )

    models: dict[str, object] = {}
    for model_name in deploy.MODEL_NAMES:
        model_path = MODEL_DIR / f"model_1d_{model_name}.pkl"
        models[model_name] = joblib.load(model_path)
    scored_test["pred_mean4"] = deploy.predict_with_models(models, split.X_te)

    picks = scored_test[
        (pd.to_numeric(scored_test["pred_mean4"], errors="coerce") > float(raw_cutoff))
        & pd.to_numeric(scored_test[ACTUAL_RETURN_COL], errors="coerce").notna()
        & scored_test[EXIT_DATETIME_COL].notna()
    ].copy()
    if picks.empty:
        raise RuntimeError("No test-set picks passed the requested cutoff.")

    picks["buy_datetime"] = pd.to_datetime(picks["buy_datetime"], errors="coerce")
    picks["exit_datetime"] = pd.to_datetime(picks[EXIT_DATETIME_COL], errors="coerce")
    picks["ret_pct"] = pd.to_numeric(picks[ACTUAL_RETURN_COL], errors="coerce")
    picks = picks.dropna(subset=["buy_datetime", "exit_datetime", "ret_pct", "pred_mean4"]).copy()
    picks["trade_day"] = picks["buy_datetime"].dt.strftime("%Y-%m-%d")
    picks["entry_time_bucket"] = picks["buy_datetime"].apply(classify_time_bucket)
    picks["raw_cutoff"] = float(raw_cutoff)
    picks["test_rows"] = int(len(scored_test))
    return picks.sort_values(["buy_datetime", "pred_mean4"], ascending=[True, False]).reset_index(drop=True)


def close_due_positions(
    positions: list[Position],
    cash: float,
    ts: pd.Timestamp,
    trade_log: list[dict],
    curve: list[dict],
) -> tuple[list[Position], float]:
    open_positions = list(positions)
    local_cash = float(cash)
    while True:
        due = [p for p in open_positions if p.exit_time <= ts]
        if not due:
            break
        next_exit = min(p.exit_time for p in due)
        closing = [p for p in open_positions if p.exit_time == next_exit]
        open_positions = [p for p in open_positions if p.exit_time != next_exit]
        for pos in closing:
            proceeds = pos.invested_eur * (1.0 + pos.ret_pct / 100.0)
            local_cash += proceeds
            trade_log.append(
                {
                    "ticker": pos.ticker,
                    "entry_time": pos.entry_time,
                    "exit_time": pos.exit_time,
                    "invested_eur": pos.invested_eur,
                    "ret_pct": pos.ret_pct,
                    "proceeds_eur": proceeds,
                    "pnl_eur": proceeds - pos.invested_eur,
                }
            )
        invested = float(sum(p.invested_eur for p in open_positions))
        curve.append(
            {
                "timestamp": next_exit,
                "cash_eur": local_cash,
                "invested_eur": invested,
                "budget_eur": local_cash + invested,
            }
        )
    return open_positions, local_cash


def reserve_cap_for_time(
    day_budget: float,
    reserve_fraction: float,
    reserve_release_time: time | None,
    ts: pd.Timestamp,
) -> float:
    if reserve_release_time is None or reserve_fraction <= 0:
        return float(day_budget)
    current_time = ts.timetz().replace(tzinfo=None)
    if current_time < reserve_release_time:
        return float(day_budget) * (1.0 - float(reserve_fraction))
    return float(day_budget)


def summarize_curve(curve_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict[str, float | int]:
    if curve_df.empty:
        return {
            "end_budget_eur": np.nan,
            "total_return_pct": np.nan,
            "annualized_return_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "trades_executed": 0,
            "win_rate_pct": np.nan,
            "mean_trade_ret_pct": np.nan,
        }

    start_budget = float(curve_df["budget_eur"].iloc[0])
    end_budget = float(curve_df["budget_eur"].iloc[-1])
    start_time = pd.to_datetime(curve_df["timestamp"].iloc[0])
    end_time = pd.to_datetime(curve_df["timestamp"].iloc[-1])
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1.0)
    annualized = ((end_budget / start_budget) ** (365.25 / elapsed_days) - 1.0) * 100.0
    running_peak = curve_df["budget_eur"].cummax()
    drawdown = ((curve_df["budget_eur"] / running_peak) - 1.0) * 100.0

    rets = pd.to_numeric(trades_df.get("ret_pct"), errors="coerce").dropna()
    return {
        "end_budget_eur": end_budget,
        "total_return_pct": (end_budget / start_budget - 1.0) * 100.0,
        "annualized_return_pct": float(annualized),
        "max_drawdown_pct": float(drawdown.min()),
        "trades_executed": int(len(trades_df)),
        "win_rate_pct": float((rets > 0).mean() * 100.0) if not rets.empty else np.nan,
        "mean_trade_ret_pct": float(rets.mean()) if not rets.empty else np.nan,
    }


def simulate_two_day_rotation(
    picks: pd.DataFrame,
    sleeve_fraction: float,
    max_picks_per_day: int,
    reserve_fraction: float,
    reserve_release_time: time | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    cash = float(START_BUDGET_EUR)
    open_positions: list[Position] = []
    trade_log: list[dict] = []
    curve: list[dict] = []
    day_rows: list[dict] = []

    start_ts = pd.to_datetime(picks["buy_datetime"].min()) - pd.Timedelta(minutes=1)
    curve.append({"timestamp": start_ts, "cash_eur": cash, "invested_eur": 0.0, "budget_eur": cash})

    for trade_day, day_batch in picks.groupby("trade_day", sort=True):
        day_batch = day_batch.sort_values(["buy_datetime", "pred_mean4"], ascending=[True, False]).reset_index(drop=True)
        day_start = pd.to_datetime(day_batch["buy_datetime"].min()).normalize() + pd.Timedelta(hours=9, minutes=30)
        open_positions, cash = close_due_positions(open_positions, cash, day_start, trade_log, curve)

        invested = float(sum(p.invested_eur for p in open_positions))
        total_equity = cash + invested
        day_budget = float(total_equity) * float(sleeve_fraction)
        slot_size = day_budget / float(max_picks_per_day)
        used_today = 0.0
        taken_today = 0
        offered_today = int(len(day_batch))

        for _, row in day_batch.iterrows():
            if taken_today >= max_picks_per_day:
                break
            ts = pd.to_datetime(row["buy_datetime"])
            open_positions, cash = close_due_positions(open_positions, cash, ts, trade_log, curve)

            allowed_budget = reserve_cap_for_time(
                day_budget=day_budget,
                reserve_fraction=reserve_fraction,
                reserve_release_time=reserve_release_time,
                ts=ts,
            )
            remaining_budget = max(0.0, allowed_budget - used_today)
            alloc = min(cash, slot_size, remaining_budget)
            if alloc <= 0:
                continue

            cash -= alloc
            used_today += alloc
            taken_today += 1
            open_positions.append(
                Position(
                    ticker=str(row["ticker"]),
                    entry_time=ts,
                    exit_time=pd.to_datetime(row["exit_datetime"]),
                    invested_eur=float(alloc),
                    ret_pct=float(row["ret_pct"]),
                )
            )
            invested = float(sum(p.invested_eur for p in open_positions))
            curve.append(
                {
                    "timestamp": ts,
                    "cash_eur": cash,
                    "invested_eur": invested,
                    "budget_eur": cash + invested,
                }
            )

        day_rows.append(
            {
                "trade_day": trade_day,
                "offered_picks": offered_today,
                "taken_picks": taken_today,
                "day_budget_eur": day_budget,
                "used_budget_eur": used_today,
                "utilization_pct": (used_today / day_budget * 100.0) if day_budget > 0 else np.nan,
            }
        )

    if open_positions:
        final_ts = max(p.exit_time for p in open_positions) + pd.Timedelta(seconds=1)
        open_positions, cash = close_due_positions(open_positions, cash, final_ts, trade_log, curve)

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    trades_df = pd.DataFrame(trade_log).sort_values("entry_time").reset_index(drop=True)
    day_df = pd.DataFrame(day_rows)
    metrics = summarize_curve(curve_df, trades_df)
    metrics["avg_day_budget_utilization_pct"] = float(day_df["utilization_pct"].mean()) if not day_df.empty else np.nan
    metrics["days_with_any_trade"] = int((day_df["taken_picks"] > 0).sum()) if not day_df.empty else 0
    metrics["days_with_unused_signals"] = int(((day_df["offered_picks"] - day_df["taken_picks"]) > 0).sum()) if not day_df.empty else 0
    return curve_df, trades_df, metrics


def describe_candidates(picks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_summary = (
        picks.groupby("trade_day")
        .agg(
            n_picks=("ticker", "count"),
            mean_pred_mean4=("pred_mean4", "mean"),
            mean_ret_pct=("ret_pct", "mean"),
        )
        .reset_index()
        .sort_values("trade_day")
    )

    bucket_summary = (
        picks.groupby("entry_time_bucket")
        .agg(
            n_picks=("ticker", "count"),
            mean_ret_pct=("ret_pct", "mean"),
            median_ret_pct=("ret_pct", "median"),
            win_rate_pct=("ret_pct", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100.0)),
            mean_pred_mean4=("pred_mean4", "mean"),
        )
        .reset_index()
        .sort_values("entry_time_bucket")
    )
    return day_summary, bucket_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a 2-day rotating sleeve allocation on the T+2-open test set.")
    parser.add_argument("--decile-score-threshold", type=float, default=0.87)
    parser.add_argument("--cutoff-csv", default=str(DEFAULT_CUTOFF_CSV))
    parser.add_argument("--sleeve-fraction", type=float, default=0.5)
    parser.add_argument("--max-picks", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--reserve-fractions", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--reserve-release-time", default="13:00")
    parser.add_argument("--grid-out", default=str(OUT_DIR / "two_day_rotation_allocation_grid.csv"))
    parser.add_argument("--candidate-day-out", default=str(OUT_DIR / "two_day_rotation_candidate_day_summary.csv"))
    parser.add_argument("--candidate-bucket-out", default=str(OUT_DIR / "two_day_rotation_candidate_bucket_summary.csv"))
    parser.add_argument("--summary-json", default=str(OUT_DIR / "two_day_rotation_allocation_summary.json"))
    return parser.parse_args()


def parse_release_time(raw: str) -> time | None:
    txt = str(raw).strip().lower()
    if txt in {"", "none", "null"}:
        return None
    return pd.Timestamp(txt).time()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_cutoff = load_raw_cutoff(Path(args.cutoff_csv), float(args.decile_score_threshold))
    picks = build_candidates(raw_cutoff=raw_cutoff)
    day_summary, bucket_summary = describe_candidates(picks)

    release_time = parse_release_time(args.reserve_release_time)
    grid_rows: list[dict] = []
    for max_picks_per_day in args.max_picks:
        for reserve_fraction in args.reserve_fractions:
            curve_df, trades_df, metrics = simulate_two_day_rotation(
                picks=picks,
                sleeve_fraction=float(args.sleeve_fraction),
                max_picks_per_day=int(max_picks_per_day),
                reserve_fraction=float(reserve_fraction),
                reserve_release_time=release_time,
            )
            strategy_name = f"sleeve{int(round(args.sleeve_fraction * 100)):02d}_top{max_picks_per_day}_reserve{int(round(reserve_fraction * 100)):02d}"
            curve_path = OUT_DIR / f"{strategy_name}_curve.csv"
            trades_path = OUT_DIR / f"{strategy_name}_trades.csv"
            curve_df.to_csv(curve_path, index=False)
            trades_df.to_csv(trades_path, index=False)
            grid_rows.append(
                {
                    "strategy": strategy_name,
                    "decile_score_threshold": float(args.decile_score_threshold),
                    "raw_pred_cutoff": raw_cutoff,
                    "candidate_rows": int(len(picks)),
                    "candidate_days": int(day_summary["trade_day"].nunique()),
                    "sleeve_fraction": float(args.sleeve_fraction),
                    "max_picks_per_day": int(max_picks_per_day),
                    "reserve_fraction": float(reserve_fraction),
                    "reserve_release_time": "" if release_time is None else release_time.strftime("%H:%M"),
                    "curve_csv": str(curve_path),
                    "trades_csv": str(trades_path),
                    **metrics,
                }
            )

    grid = pd.DataFrame(grid_rows).sort_values(
        ["end_budget_eur", "max_drawdown_pct"],
        ascending=[False, False],
    ).reset_index(drop=True)
    grid.to_csv(args.grid_out, index=False)
    day_summary.to_csv(args.candidate_day_out, index=False)
    bucket_summary.to_csv(args.candidate_bucket_out, index=False)

    payload = {
        "meta": {
            "decile_score_threshold": float(args.decile_score_threshold),
            "raw_pred_cutoff": raw_cutoff,
            "candidate_rows": int(len(picks)),
            "candidate_days": int(day_summary["trade_day"].nunique()),
            "candidate_date_range": [
                str(day_summary["trade_day"].min()) if not day_summary.empty else "",
                str(day_summary["trade_day"].max()) if not day_summary.empty else "",
            ],
            "reserve_release_time": "" if release_time is None else release_time.strftime("%H:%M"),
        },
        "top_strategies": grid.head(10).to_dict(orient="records"),
        "candidate_bucket_summary": bucket_summary.to_dict(orient="records"),
    }
    Path(args.summary_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"candidate_rows={len(picks)} raw_cutoff={raw_cutoff:.6f}")
    print(f"grid_out={args.grid_out}")
    print(f"summary_json={args.summary_json}")
    if not grid.empty:
        best = grid.iloc[0]
        print(
            "best_strategy={strategy} end_budget={end_budget_eur:.2f} total_return_pct={total_return_pct:.2f} "
            "max_drawdown_pct={max_drawdown_pct:.2f} trades={trades_executed}".format(**best.to_dict())
        )


if __name__ == "__main__":
    main()
