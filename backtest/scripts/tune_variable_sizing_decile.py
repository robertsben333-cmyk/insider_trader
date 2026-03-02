"""
Tune variable-sizing strategy for day-1 trades with decile gating.

This script sweeps:
- Signal blend weights (score vs price-edge vs risk-quality)
- Risk component weights (model disagreement / SD / volatility / momentum)
- Allocation bounds (including max allocation > 0.20)

Selection policy:
- Primary: maximize objective = return_pct + 0.5 * max_drawdown_pct
- Constraint for "safer best": max_drawdown_pct >= DRAWDOWN_FLOOR
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from backtest.scripts import evaluate_investable_risk_rules as base_eval

OUT_DIR = Path("backtest/out")
DRAWDOWN_FLOOR = -10.0


@dataclass(frozen=True)
class StrategyParams:
    name: str
    min_decile_score: float
    min_alloc_fraction: float
    max_alloc_fraction: float
    max_ticker_weight: float
    start_budget_eur: float
    w_score: float
    w_price_edge: float
    w_risk_quality: float
    w_range_risk: float
    w_sd_risk: float
    w_vol_risk: float
    w_mom_risk: float


@dataclass
class Position:
    ticker: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    invested: float
    ret_pct: float
    alloc_fraction: float
    sizing_signal: float


def parse_float_money(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("$", "").replace(",", "")
    if not s:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def scale_between(x: pd.Series, lo: float, hi: float) -> np.ndarray:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(len(xv), 0.5, dtype=float)
    return clamp01((xv - lo) / (hi - lo))


def normalize_weights(vals: tuple[float, ...], default: tuple[float, ...]) -> tuple[float, ...]:
    arr = np.asarray(vals, dtype=float)
    if arr.shape != np.asarray(default, dtype=float).shape:
        raise ValueError("Weight shape mismatch.")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0:
        return default
    arr = arr / s
    return tuple(float(x) for x in arr)


def percentile_rank(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_reference.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    idx = np.searchsorted(sorted_reference, values, side="right")
    return idx / float(sorted_reference.size)


def compute_threshold_and_deciles_from_reference(
    df: pd.DataFrame, reference_df: pd.DataFrame, min_decile_score: float
) -> tuple[float, pd.DataFrame]:
    pred_cal = pd.to_numeric(reference_df["pred_mean4"], errors="coerce").dropna().to_numpy(dtype=float)
    pred_cal_sorted = np.sort(pred_cal)
    raw_cutoff = float(np.quantile(pred_cal_sorted, min_decile_score))

    out = df.copy()
    pred_all = pd.to_numeric(out["pred_mean4"], errors="coerce").to_numpy(dtype=float)
    out["decile_score"] = percentile_rank(pred_cal_sorted, pred_all)
    return raw_cutoff, out


def add_sizing_features(df: pd.DataFrame, calibration_df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = df.copy()

    sig_w = normalize_weights(
        (params.w_score, params.w_price_edge, params.w_risk_quality),
        (0.5, 0.2, 0.3),
    )
    risk_w = normalize_weights(
        (params.w_range_risk, params.w_sd_risk, params.w_vol_risk, params.w_mom_risk),
        (0.3, 0.2, 0.3, 0.2),
    )

    out["insider_price"] = out["last_price"].apply(parse_float_money)
    out["buy_price_num"] = pd.to_numeric(out["buy_price"], errors="coerce")
    out["price_edge_pct"] = (
        (out["insider_price"] - out["buy_price_num"]) / out["buy_price_num"].replace(0.0, np.nan)
    ) * 100.0

    pred_cols = ["pred_HGBR", "pred_XGBoost", "pred_ElasticNet", "pred_SplineElasticNet"]
    pred_mat = out[pred_cols].apply(pd.to_numeric, errors="coerce")
    out["model_range"] = pred_mat.max(axis=1) - pred_mat.min(axis=1)
    out["model_sd"] = pred_mat.std(axis=1, ddof=0)
    out["prior_30d_vol"] = pd.to_numeric(out["prior_30d_vol"], errors="coerce")
    out["prior_30d_pct"] = pd.to_numeric(out["prior_30d_pct"], errors="coerce")
    out["abs_prior_30d_pct"] = out["prior_30d_pct"].abs()

    cal = calibration_df.copy()
    cal["insider_price"] = cal["last_price"].apply(parse_float_money)
    cal["buy_price_num"] = pd.to_numeric(cal["buy_price"], errors="coerce")
    cal["price_edge_pct"] = (
        (cal["insider_price"] - cal["buy_price_num"]) / cal["buy_price_num"].replace(0.0, np.nan)
    ) * 100.0
    cal_preds = cal[pred_cols].apply(pd.to_numeric, errors="coerce")
    cal["model_range"] = cal_preds.max(axis=1) - cal_preds.min(axis=1)
    cal["model_sd"] = cal_preds.std(axis=1, ddof=0)
    cal["prior_30d_vol"] = pd.to_numeric(cal["prior_30d_vol"], errors="coerce")
    cal["abs_prior_30d_pct"] = pd.to_numeric(cal["prior_30d_pct"], errors="coerce").abs()

    edge_lo, edge_hi = cal["price_edge_pct"].quantile([0.1, 0.9]).tolist()
    range_lo, range_hi = cal["model_range"].quantile([0.1, 0.9]).tolist()
    sd_lo, sd_hi = cal["model_sd"].quantile([0.1, 0.9]).tolist()
    vol_lo, vol_hi = cal["prior_30d_vol"].quantile([0.1, 0.9]).tolist()
    mom_lo, mom_hi = cal["abs_prior_30d_pct"].quantile([0.1, 0.9]).tolist()

    denom = max(1e-9, 1.0 - params.min_decile_score)
    score_strength = clamp01((out["decile_score"].to_numpy(dtype=float) - params.min_decile_score) / denom)
    price_edge_strength = scale_between(out["price_edge_pct"], edge_lo, edge_hi)
    range_risk = scale_between(out["model_range"], range_lo, range_hi)
    sd_risk = scale_between(out["model_sd"], sd_lo, sd_hi)
    vol_risk = scale_between(out["prior_30d_vol"], vol_lo, vol_hi)
    mom_risk = scale_between(out["abs_prior_30d_pct"], mom_lo, mom_hi)

    risk_quality = 1.0 - (
        risk_w[0] * range_risk + risk_w[1] * sd_risk + risk_w[2] * vol_risk + risk_w[3] * mom_risk
    )
    risk_quality = clamp01(risk_quality)

    sizing_signal = sig_w[0] * score_strength + sig_w[1] * price_edge_strength + sig_w[2] * risk_quality
    sizing_signal = np.nan_to_num(sizing_signal, nan=0.0, posinf=1.0, neginf=0.0)
    sizing_signal = clamp01(sizing_signal)

    out["score_strength"] = score_strength
    out["price_edge_strength"] = price_edge_strength
    out["risk_quality"] = risk_quality
    out["sizing_signal"] = sizing_signal
    out["alloc_fraction_target"] = (
        params.min_alloc_fraction + (params.max_alloc_fraction - params.min_alloc_fraction) * sizing_signal
    )
    return out


def simulate_variable_sizing(picks: pd.DataFrame, params: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks = picks.dropna(subset=["ret_pct", "buy_datetime", "exit_datetime", "ticker"]).copy()
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame()

    picks = picks.sort_values(["buy_datetime", "sizing_signal", "pred_mean4"], ascending=[True, False, False])
    cash = float(params.start_budget_eur)
    open_positions: list[Position] = []
    trade_log: list[dict] = []
    curve: list[dict] = []

    start_ts = pd.to_datetime(picks["buy_datetime"].min()) - pd.Timedelta(minutes=1)
    curve.append({"timestamp": start_ts, "cash_eur": cash, "invested_eur": 0.0, "budget_eur": cash})

    def mark(ts: pd.Timestamp) -> None:
        invested = float(sum(p.invested for p in open_positions))
        curve.append({"timestamp": ts, "cash_eur": cash, "invested_eur": invested, "budget_eur": cash + invested})

    def close_until(ts: pd.Timestamp) -> None:
        nonlocal cash, open_positions
        while True:
            due = [p for p in open_positions if p.exit_time <= ts]
            if not due:
                break
            next_exit = min(p.exit_time for p in due)
            closing = [p for p in open_positions if p.exit_time == next_exit]
            open_positions = [p for p in open_positions if p.exit_time != next_exit]
            for pos in closing:
                proceeds = pos.invested * (1.0 + pos.ret_pct / 100.0)
                cash += proceeds
                trade_log.append(
                    {
                        "ticker": pos.ticker,
                        "entry_time": pos.entry_time,
                        "exit_time": pos.exit_time,
                        "invested_eur": pos.invested,
                        "ret_pct": pos.ret_pct,
                        "alloc_fraction": pos.alloc_fraction,
                        "sizing_signal": pos.sizing_signal,
                        "proceeds_eur": proceeds,
                        "pnl_eur": proceeds - pos.invested,
                    }
                )
            mark(next_exit)

    for buy_time, batch in picks.groupby("buy_datetime", sort=True):
        buy_time = pd.to_datetime(buy_time)
        close_until(buy_time)
        batch = batch.sort_values(["sizing_signal", "pred_mean4"], ascending=[False, False])
        for _, row in batch.iterrows():
            if cash <= 0:
                break
            invested_total = float(sum(p.invested for p in open_positions))
            budget_now = cash + invested_total
            if budget_now <= 0:
                break

            ticker = str(row["ticker"])
            ticker_exposure = float(sum(p.invested for p in open_positions if p.ticker == ticker))
            ticker_cap = params.max_ticker_weight * budget_now
            cap_remaining = max(0.0, ticker_cap - ticker_exposure)
            if cap_remaining <= 0:
                continue

            alloc_frac = float(
                np.clip(
                    row.get("alloc_fraction_target", 0.0),
                    params.min_alloc_fraction,
                    params.max_alloc_fraction,
                )
            )
            target_alloc = alloc_frac * budget_now
            alloc = min(cash, cap_remaining, target_alloc)
            if alloc <= 0:
                continue

            ret_pct = float(row["ret_pct"])
            if not np.isfinite(ret_pct):
                continue

            cash -= alloc
            open_positions.append(
                Position(
                    ticker=ticker,
                    entry_time=buy_time,
                    exit_time=pd.to_datetime(row["exit_datetime"]),
                    invested=alloc,
                    ret_pct=ret_pct,
                    alloc_fraction=alloc_frac,
                    sizing_signal=float(row.get("sizing_signal", np.nan)),
                )
            )
        mark(buy_time)

    if open_positions:
        last_exit = max(p.exit_time for p in open_positions)
        close_until(last_exit + pd.Timedelta(seconds=1))

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    trades_df = pd.DataFrame(trade_log).sort_values("entry_time").reset_index(drop=True)
    return curve_df, trades_df


def safe_float(val) -> float:
    try:
        out = float(val)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def summarize_run(row_base: dict, metrics: dict, trades: pd.DataFrame) -> dict:
    out = dict(row_base)
    out.update(metrics)
    out["avg_alloc_fraction"] = safe_float(trades["alloc_fraction"].mean()) if not trades.empty else np.nan
    out["p95_alloc_fraction"] = safe_float(trades["alloc_fraction"].quantile(0.95)) if not trades.empty else np.nan
    out["max_alloc_fraction_realized"] = safe_float(trades["alloc_fraction"].max()) if not trades.empty else np.nan
    out["avg_sizing_signal"] = safe_float(trades["sizing_signal"].mean()) if not trades.empty else np.nan
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    min_decile = 0.90
    df = base_eval.load_scored_day1()
    n_train, split_80 = base_eval.chrono_split_60_20_20(len(df))
    train_raw = df.iloc[:n_train].copy()
    val_raw = df.iloc[n_train:split_80].copy()
    test_raw = df.iloc[split_80:].copy()

    raw_cutoff, scored = compute_threshold_and_deciles_from_reference(df, train_raw, min_decile)
    train = scored.iloc[:n_train].copy()
    val = scored.iloc[n_train:split_80].copy()
    test = scored.iloc[split_80:].copy()

    base_val_picks = val[
        (pd.to_numeric(val["pred_mean4"], errors="coerce") > raw_cutoff)
        & (pd.to_numeric(val["decile_score"], errors="coerce") >= min_decile)
    ].copy()
    base_test_picks = test[
        (pd.to_numeric(test["pred_mean4"], errors="coerce") > raw_cutoff)
        & (pd.to_numeric(test["decile_score"], errors="coerce") >= min_decile)
    ].copy()

    fixed_val_curve, fixed_val_trades = base_eval.simulate_budget_curve_from_picks(base_val_picks)
    fixed_val_metrics = base_eval.trade_metrics(fixed_val_curve, fixed_val_trades)
    fixed_test_curve, fixed_test_trades = base_eval.simulate_budget_curve_from_picks(base_test_picks)
    fixed_test_metrics = base_eval.trade_metrics(fixed_test_curve, fixed_test_trades)

    signal_grids = [
        (0.50, 0.20, 0.30),
        (0.60, 0.20, 0.20),
        (0.55, 0.25, 0.20),
        (0.65, 0.20, 0.15),
        (0.50, 0.30, 0.20),
        (0.45, 0.35, 0.20),
        (0.70, 0.15, 0.15),
    ]
    risk_grids = [
        (0.30, 0.20, 0.30, 0.20),
        (0.25, 0.20, 0.40, 0.15),
        (0.35, 0.25, 0.25, 0.15),
    ]
    min_alloc_grids = [0.05, 0.08]
    max_alloc_grids = [0.25, 0.30, 0.35, 0.40]

    rows_val: list[dict] = []

    combos = list(product(signal_grids, risk_grids, min_alloc_grids, max_alloc_grids))
    for idx, ((w_s, w_e, w_r), (w_rg, w_sd, w_vol, w_mom), min_alloc, max_alloc) in enumerate(combos, start=1):
        if min_alloc >= max_alloc:
            continue
        params = StrategyParams(
            name=f"cfg_{idx:03d}",
            min_decile_score=min_decile,
            min_alloc_fraction=min_alloc,
            max_alloc_fraction=max_alloc,
            max_ticker_weight=0.25,
            start_budget_eur=10_000.0,
            w_score=w_s,
            w_price_edge=w_e,
            w_risk_quality=w_r,
            w_range_risk=w_rg,
            w_sd_risk=w_sd,
            w_vol_risk=w_vol,
            w_mom_risk=w_mom,
        )

        val_feat = add_sizing_features(val, train, params)
        picks = val_feat[
            (pd.to_numeric(val_feat["pred_mean4"], errors="coerce") > raw_cutoff)
            & (pd.to_numeric(val_feat["decile_score"], errors="coerce") >= min_decile)
        ].copy()
        curve, trades = simulate_variable_sizing(picks, params)
        metrics = base_eval.trade_metrics(curve, trades)
        if metrics["n_trades"] == 0:
            continue

        row_base = {
            "strategy": params.name,
            "min_decile_score": params.min_decile_score,
            "min_alloc_fraction": params.min_alloc_fraction,
            "max_alloc_fraction": params.max_alloc_fraction,
            "w_score": params.w_score,
            "w_price_edge": params.w_price_edge,
            "w_risk_quality": params.w_risk_quality,
            "w_range_risk": params.w_range_risk,
            "w_sd_risk": params.w_sd_risk,
            "w_vol_risk": params.w_vol_risk,
            "w_mom_risk": params.w_mom_risk,
            "raw_pred_mean4_cutoff": raw_cutoff,
        }
        rows_val.append(summarize_run(row_base, metrics, trades))

    if not rows_val:
        raise RuntimeError("No valid tuning candidates produced trades.")

    val_grid = pd.DataFrame(rows_val).sort_values(
        ["objective_val", "return_pct", "max_drawdown_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    safer = val_grid[val_grid["max_drawdown_pct"] >= DRAWDOWN_FLOOR].copy()
    best = safer.iloc[0] if not safer.empty else val_grid.iloc[0]
    best_name = str(best["strategy"])

    best_params = StrategyParams(
        name=best_name,
        min_decile_score=float(best["min_decile_score"]),
        min_alloc_fraction=float(best["min_alloc_fraction"]),
        max_alloc_fraction=float(best["max_alloc_fraction"]),
        max_ticker_weight=0.25,
        start_budget_eur=10_000.0,
        w_score=float(best["w_score"]),
        w_price_edge=float(best["w_price_edge"]),
        w_risk_quality=float(best["w_risk_quality"]),
        w_range_risk=float(best["w_range_risk"]),
        w_sd_risk=float(best["w_sd_risk"]),
        w_vol_risk=float(best["w_vol_risk"]),
        w_mom_risk=float(best["w_mom_risk"]),
    )

    test_feat = add_sizing_features(test, train, best_params)
    best_test_picks = test_feat[
        (pd.to_numeric(test_feat["pred_mean4"], errors="coerce") > raw_cutoff)
        & (pd.to_numeric(test_feat["decile_score"], errors="coerce") >= min_decile)
    ].copy()
    best_test_curve, best_test_trades = simulate_variable_sizing(best_test_picks, best_params)
    best_test_metrics = base_eval.trade_metrics(best_test_curve, best_test_trades)
    best_test_row = summarize_run(best.to_dict(), best_test_metrics, best_test_trades)

    top_n = 15
    top = val_grid.head(top_n).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))

    compare = pd.DataFrame(
        [
            {
                "strategy": "fixed_cap_allocation",
                "split": "validation",
                **fixed_val_metrics,
            },
            {
                "strategy": "fixed_cap_allocation",
                "split": "test",
                **fixed_test_metrics,
            },
            {
                "strategy": best_name,
                "split": "validation",
                **{k: best[k] for k in fixed_val_metrics.keys()},
            },
            {
                "strategy": best_name,
                "split": "test",
                **{k: best_test_row[k] for k in fixed_test_metrics.keys()},
            },
        ]
    )

    grid_csv = OUT_DIR / "variable_sizing_decile_tuning_grid.csv"
    grid_json = OUT_DIR / "variable_sizing_decile_tuning_grid.json"
    best_json = OUT_DIR / "variable_sizing_decile_tuning_best.json"
    best_csv = OUT_DIR / "variable_sizing_decile_tuning_best.csv"
    top_csv = OUT_DIR / "variable_sizing_decile_tuning_top15.csv"
    best_trades_csv = OUT_DIR / "variable_sizing_decile_tuning_best_trade_log.csv"

    val_grid.to_csv(grid_csv, index=False)
    grid_json.write_text(val_grid.to_json(orient="records", indent=2), encoding="utf-8")
    compare.to_csv(best_csv, index=False)
    top.to_csv(top_csv, index=False)
    best_test_trades.to_csv(best_trades_csv, index=False)

    best_payload = {
        "selection": {
            "selection_split": "validation",
            "drawdown_floor_pct": DRAWDOWN_FLOOR,
            "used_constrained_set": not safer.empty,
        },
        "dataset": {
            "rows_total": len(df),
            "split_train_rows": n_train,
            "split_validation_rows": len(val_raw),
            "split_test_rows": len(test),
            "val_pick_rows": len(base_val_picks),
            "test_pick_rows": len(base_test_picks),
            "raw_pred_mean4_cutoff": raw_cutoff,
        },
        "baseline_fixed_validation": fixed_val_metrics,
        "baseline_fixed_test": fixed_test_metrics,
        "best_candidate_validation": best.to_dict(),
        "best_candidate_test": best_test_row,
        "best_candidate_params": {
            k: v
            for k, v in asdict(best_params).items()
            if k not in {"name", "max_ticker_weight", "start_budget_eur"}
        },
    }
    best_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    print("Variable-sizing tuning complete.")
    print("Selection split: validation. Final evaluation split: test.")
    print(f"Candidates evaluated: {len(val_grid)}")
    print(f"Drawdown floor: {DRAWDOWN_FLOOR:.2f}%  constrained_used={not safer.empty}")
    print("\nBest candidate on validation:")
    print(pd.DataFrame([best]).to_string(index=False))
    print("\nLocked test metrics for selected candidate:")
    print(pd.DataFrame([best_test_row]).to_string(index=False))
    print("\nFixed baseline vs selected candidate:")
    print(compare.to_string(index=False))
    print("\nTop 5 candidates:")
    print(val_grid.head(5).to_string(index=False))
    print(f"\nSaved: {grid_csv}")
    print(f"Saved: {grid_json}")
    print(f"Saved: {best_csv}")
    print(f"Saved: {best_json}")
    print(f"Saved: {top_csv}")
    print(f"Saved: {best_trades_csv}")


if __name__ == "__main__":
    main()
