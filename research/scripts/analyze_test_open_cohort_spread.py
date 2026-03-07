from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

DEPLOY_SCRIPT = BASE / "research" / "scripts" / "deploy_tplus2_open_day1_live.py"
MODEL_DIR = BASE / "models" / "prod4"
OUT_DIR = BASE / "backtest" / "out"
DEFAULT_CUTOFF_CSV = BASE / "backtest" / "out" / "investable_decile_score_sweep_0005_tplus2_open.csv"
START_BUDGET_EUR = 10_000.0
DEFAULT_SLEEVE_FRACTION = 0.5


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


def load_cutoff_curve(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["decile_score_threshold"] = pd.to_numeric(table["decile_score_threshold"], errors="coerce")
    table["raw_pred_mean4_cutoff"] = pd.to_numeric(table["raw_pred_mean4_cutoff"], errors="coerce")
    table = table.dropna(subset=["decile_score_threshold", "raw_pred_mean4_cutoff"]).copy()
    table = table.sort_values(["raw_pred_mean4_cutoff", "decile_score_threshold"]).reset_index(drop=True)
    if table.empty:
        raise RuntimeError(f"No valid cutoffs found in {path}")
    return table


def raw_cutoff_for_decile(curve: pd.DataFrame, decile_score_threshold: float) -> float:
    work = curve.copy()
    work["distance"] = (work["decile_score_threshold"] - float(decile_score_threshold)).abs()
    best = work.sort_values(["distance", "decile_score_threshold"]).iloc[0]
    return float(best["raw_pred_mean4_cutoff"])


def estimate_decile_score(raw_prediction: float, curve: pd.DataFrame) -> float:
    deciles = curve["decile_score_threshold"].to_numpy(dtype=float)
    raws = curve["raw_pred_mean4_cutoff"].to_numpy(dtype=float)
    if raw_prediction <= raws[0]:
        return float(deciles[0])
    if raw_prediction >= raws[-1]:
        return float(deciles[-1])
    for idx in range(1, len(curve)):
        x0, x1 = raws[idx - 1], raws[idx]
        if raw_prediction <= x1:
            y0, y1 = deciles[idx - 1], deciles[idx]
            if x1 == x0:
                return float(y1)
            frac = (raw_prediction - x0) / (x1 - x0)
            return float(y0 + frac * (y1 - y0))
    return float(deciles[-1])


def advice_weight(estimated_decile: float, decile_threshold: float, base_alloc: float, bonus_alloc: float) -> float:
    denom = max(1e-9, 1.0 - float(decile_threshold))
    strength = min(max((float(estimated_decile) - float(decile_threshold)) / denom, 0.0), 1.0)
    return min(max(base_alloc + bonus_alloc * strength, base_alloc), base_alloc + bonus_alloc)


def build_candidates(
    curve: pd.DataFrame,
    raw_cutoff: float,
    entry_time: time,
    decile_score_threshold: float,
    base_alloc: float,
    bonus_alloc: float,
) -> pd.DataFrame:
    deploy = load_module("deploy_tplus2_open_day1_live", DEPLOY_SCRIPT)
    split = deploy.prepare_split()

    test_start = len(split.X_tr) + len(split.X_va)
    scored_test = split.sub.iloc[test_start:].copy().reset_index(drop=True)
    if "exit_2d_open_datetime" not in scored_test.columns and "exit_2d_open_date" in scored_test.columns:
        valid = scored_test["exit_2d_open_date"].apply(lambda d: pd.notna(d))
        scored_test["exit_2d_open_datetime"] = pd.NaT
        scored_test.loc[valid, "exit_2d_open_datetime"] = (
            pd.to_datetime(scored_test.loc[valid, "exit_2d_open_date"].astype(str), errors="coerce")
            .dt.tz_localize(deploy.ET, ambiguous="NaT", nonexistent="shift_forward")
            + pd.Timedelta(hours=9, minutes=30)
        )

    models: dict[str, object] = {}
    for model_name in deploy.MODEL_NAMES:
        models[model_name] = joblib.load(MODEL_DIR / f"model_1d_{model_name}.pkl")
    scored_test["pred_mean4"] = deploy.predict_with_models(models, split.X_te)

    scored_test["buy_datetime"] = pd.to_datetime(scored_test["buy_datetime"], errors="coerce")
    scored_test["exit_datetime"] = pd.to_datetime(scored_test["exit_2d_open_datetime"], errors="coerce")
    scored_test["ret_pct"] = pd.to_numeric(scored_test["stock_only_return_2d_open_pct_raw"], errors="coerce")
    scored_test = scored_test.dropna(subset=["buy_datetime", "exit_datetime", "ret_pct", "pred_mean4"]).copy()

    picks = scored_test[scored_test["pred_mean4"] > float(raw_cutoff)].copy()
    picks["buy_time_hhmm"] = picks["buy_datetime"].dt.strftime("%H:%M")
    picks = picks[picks["buy_datetime"].dt.time == entry_time].copy()
    if picks.empty:
        raise RuntimeError("No qualifying opening-cohort test rows found.")

    picks["trade_day"] = picks["buy_datetime"].dt.strftime("%Y-%m-%d")
    picks["estimated_decile_score"] = picks["pred_mean4"].apply(lambda x: estimate_decile_score(float(x), curve))
    picks["equal_weight_raw"] = 1.0
    picks["score_weight_raw"] = np.maximum(pd.to_numeric(picks["pred_mean4"], errors="coerce") - float(raw_cutoff), 1e-9)
    picks["advice_weight_raw"] = picks["estimated_decile_score"].apply(
        lambda x: advice_weight(float(x), decile_score_threshold, base_alloc, bonus_alloc)
    )
    return picks.sort_values(["buy_datetime", "pred_mean4"], ascending=[True, False]).reset_index(drop=True)


def normalize_weights(group: pd.DataFrame, raw_col: str) -> pd.Series:
    vals = pd.to_numeric(group[raw_col], errors="coerce").fillna(0.0).astype(float)
    total = float(vals.sum())
    if total <= 0:
        return pd.Series(np.repeat(1.0 / len(group), len(group)), index=group.index)
    return vals / total


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


def summarize_curve(curve_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict[str, float | int]:
    start_budget = float(curve_df["budget_eur"].iloc[0])
    end_budget = float(curve_df["budget_eur"].iloc[-1])
    start_time = pd.to_datetime(curve_df["timestamp"].iloc[0])
    end_time = pd.to_datetime(curve_df["timestamp"].iloc[-1])
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1.0)
    annualized = ((end_budget / start_budget) ** (365.25 / elapsed_days) - 1.0) * 100.0
    running_peak = curve_df["budget_eur"].cummax()
    drawdown = ((curve_df["budget_eur"] / running_peak) - 1.0) * 100.0
    rets = pd.to_numeric(trades_df["ret_pct"], errors="coerce").dropna()
    return {
        "end_budget_eur": end_budget,
        "total_return_pct": (end_budget / start_budget - 1.0) * 100.0,
        "annualized_return_pct": float(annualized),
        "max_drawdown_pct": float(drawdown.min()),
        "trades_executed": int(len(trades_df)),
        "win_rate_pct": float((rets > 0).mean() * 100.0) if not rets.empty else np.nan,
        "mean_trade_ret_pct": float(rets.mean()) if not rets.empty else np.nan,
    }


def simulate_spread(detail: pd.DataFrame, weight_col: str, sleeve_fraction: float):
    cash = float(START_BUDGET_EUR)
    open_positions: list[Position] = []
    trade_log: list[dict] = []
    curve: list[dict] = []
    daily_rows: list[dict] = []

    start_ts = pd.to_datetime(detail["buy_datetime"].min()) - pd.Timedelta(minutes=1)
    curve.append({"timestamp": start_ts, "cash_eur": cash, "invested_eur": 0.0, "budget_eur": cash})

    for trade_day, group in detail.groupby("trade_day", sort=True):
        entry_ts = pd.to_datetime(group["buy_datetime"].iloc[0])
        open_positions, cash = close_due_positions(open_positions, cash, entry_ts, trade_log, curve)
        invested = float(sum(p.invested_eur for p in open_positions))
        total_equity = cash + invested
        day_budget = total_equity * float(sleeve_fraction)
        weights = normalize_weights(group, weight_col)
        allocations = day_budget * weights
        used_budget = 0.0

        for idx, row in group.iterrows():
            alloc = float(allocations.loc[idx])
            if alloc <= 0 or cash <= 0:
                continue
            alloc = min(alloc, cash)
            cash -= alloc
            used_budget += alloc
            open_positions.append(
                Position(
                    ticker=str(row["ticker"]),
                    entry_time=pd.to_datetime(row["buy_datetime"]),
                    exit_time=pd.to_datetime(row["exit_datetime"]),
                    invested_eur=alloc,
                    ret_pct=float(row["ret_pct"]),
                )
            )
        invested = float(sum(p.invested_eur for p in open_positions))
        curve.append(
            {
                "timestamp": entry_ts,
                "cash_eur": cash,
                "invested_eur": invested,
                "budget_eur": cash + invested,
            }
        )
        daily_rows.append(
            {
                "trade_day": trade_day,
                "n_names": int(len(group)),
                "day_budget_eur": float(day_budget),
                "used_budget_eur": float(used_budget),
                "utilization_pct": float((used_budget / day_budget) * 100.0) if day_budget > 0 else np.nan,
            }
        )

    if open_positions:
        final_ts = max(p.exit_time for p in open_positions) + pd.Timedelta(seconds=1)
        open_positions, cash = close_due_positions(open_positions, cash, final_ts, trade_log, curve)

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    trades_df = pd.DataFrame(trade_log).sort_values("entry_time").reset_index(drop=True)
    daily_df = pd.DataFrame(daily_rows).sort_values("trade_day").reset_index(drop=True)
    metrics = summarize_curve(curve_df, trades_df)
    metrics["candidate_rows"] = int(len(detail))
    metrics["candidate_days"] = int(detail["trade_day"].nunique())
    metrics["avg_names_per_day"] = float(daily_df["n_names"].mean()) if not daily_df.empty else np.nan
    metrics["median_names_per_day"] = float(daily_df["n_names"].median()) if not daily_df.empty else np.nan
    metrics["max_names_per_day"] = int(daily_df["n_names"].max()) if not daily_df.empty else 0
    return curve_df, trades_df, daily_df, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spread each opening-cohort test-set sleeve across all available names.")
    parser.add_argument("--cutoff-csv", default=str(DEFAULT_CUTOFF_CSV))
    parser.add_argument("--decile-score-threshold", type=float, default=0.87)
    parser.add_argument("--entry-time", default="09:45")
    parser.add_argument("--sleeve-fraction", type=float, default=DEFAULT_SLEEVE_FRACTION)
    parser.add_argument("--base-alloc-fraction", type=float, default=0.25)
    parser.add_argument("--bonus-fraction", type=float, default=0.25)
    parser.add_argument("--detail-out", default=str(OUT_DIR / "test_open_cohort_spread_detail.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "test_open_cohort_spread_summary.csv"))
    parser.add_argument("--daily-out", default=str(OUT_DIR / "test_open_cohort_spread_daily.csv"))
    parser.add_argument("--summary-json", default=str(OUT_DIR / "test_open_cohort_spread_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curve = load_cutoff_curve(Path(args.cutoff_csv))
    raw_cutoff = raw_cutoff_for_decile(curve, float(args.decile_score_threshold))
    entry_time = pd.Timestamp(args.entry_time).time()
    detail = build_candidates(
        curve=curve,
        raw_cutoff=raw_cutoff,
        entry_time=entry_time,
        decile_score_threshold=float(args.decile_score_threshold),
        base_alloc=float(args.base_alloc_fraction),
        bonus_alloc=float(args.bonus_fraction),
    )

    strategy_specs = [
        ("equal_weight", "equal_weight_raw"),
        ("score_weight", "score_weight_raw"),
        ("advice_weight", "advice_weight_raw"),
    ]
    summary_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []

    for strategy_name, weight_col in strategy_specs:
        curve_df, trades_df, daily_df, metrics = simulate_spread(
            detail=detail,
            weight_col=weight_col,
            sleeve_fraction=float(args.sleeve_fraction),
        )
        curve_path = Path(str(args.summary_out).replace("_summary.csv", f"_{strategy_name}_curve.csv"))
        trades_path = Path(str(args.summary_out).replace("_summary.csv", f"_{strategy_name}_trades.csv"))
        curve_df.to_csv(curve_path, index=False)
        trades_df.to_csv(trades_path, index=False)
        daily_df = daily_df.copy()
        daily_df["strategy"] = strategy_name
        daily_frames.append(daily_df)
        summary_rows.append(
            {
                "strategy": strategy_name,
                "decile_score_threshold": float(args.decile_score_threshold),
                "raw_cutoff": raw_cutoff,
                "entry_time": entry_time.strftime("%H:%M"),
                "sleeve_fraction": float(args.sleeve_fraction),
                "curve_csv": str(curve_path),
                "trades_csv": str(trades_path),
                **metrics,
            }
        )

    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_out, index=False)
    pd.DataFrame(summary_rows).sort_values("end_budget_eur", ascending=False).to_csv(args.summary_out, index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(args.daily_out, index=False)
    Path(args.summary_json).write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"detail_out={args.detail_out}")
    print(f"summary_out={args.summary_out}")
    for row in summary_rows:
        print(
            "strategy={strategy} candidate_rows={candidate_rows} candidate_days={candidate_days} "
            "end_budget={end_budget_eur:.2f} total_return_pct={total_return_pct:.2f} "
            "max_drawdown_pct={max_drawdown_pct:.2f} avg_names_per_day={avg_names_per_day:.2f}".format(**row)
        )


if __name__ == "__main__":
    main()
