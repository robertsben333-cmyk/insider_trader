"""
Variable-sizing backtest for day-1 trades with decile-score gating.

Rules:
- Only consider trades with decile_score >= MIN_DECILE_SCORE.
- Size each position using:
  - decile score strength
  - insider-vs-current price gap
  - risk penalties from model disagreement, prior volatility, and prior 30d return
- Keep same budget constraints as the existing backtest:
  - start budget 10,000 EUR
  - max 25% of budget per ticker
"""

from __future__ import annotations

from dataclasses import dataclass
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
START_BUDGET_EUR = 10_000.0
MAX_TICKER_WEIGHT = 0.25
MIN_DECILE_SCORE = 0.90
MIN_ALLOC_FRACTION = 0.05
MAX_ALLOC_FRACTION = 0.25


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


def percentile_rank(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_reference.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    idx = np.searchsorted(sorted_reference, values, side="right")
    return idx / float(sorted_reference.size)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def scale_between(x: pd.Series, lo: float, hi: float) -> np.ndarray:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(len(xv), 0.5, dtype=float)
    return clamp01((xv - lo) / (hi - lo))


def compute_threshold_and_deciles(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    n_train, split_80 = base_eval.chrono_split_60_20_20(len(df))
    cal = df.iloc[:split_80].copy()
    pred_cal = pd.to_numeric(cal["pred_mean4"], errors="coerce").dropna().to_numpy(dtype=float)
    pred_cal_sorted = np.sort(pred_cal)
    raw_cutoff = float(np.quantile(pred_cal_sorted, MIN_DECILE_SCORE))

    out = df.copy()
    pred_all = pd.to_numeric(out["pred_mean4"], errors="coerce").to_numpy(dtype=float)
    out["decile_score"] = percentile_rank(pred_cal_sorted, pred_all)
    return raw_cutoff, out


def add_sizing_features(df: pd.DataFrame, calibration_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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

    score_strength = clamp01((out["decile_score"].to_numpy(dtype=float) - MIN_DECILE_SCORE) / (1.0 - MIN_DECILE_SCORE))
    price_edge_strength = scale_between(out["price_edge_pct"], edge_lo, edge_hi)
    range_risk = scale_between(out["model_range"], range_lo, range_hi)
    sd_risk = scale_between(out["model_sd"], sd_lo, sd_hi)
    vol_risk = scale_between(out["prior_30d_vol"], vol_lo, vol_hi)
    mom_risk = scale_between(out["abs_prior_30d_pct"], mom_lo, mom_hi)

    risk_quality = 1.0 - (0.30 * range_risk + 0.20 * sd_risk + 0.30 * vol_risk + 0.20 * mom_risk)
    risk_quality = clamp01(risk_quality)
    sizing_signal = 0.50 * score_strength + 0.20 * price_edge_strength + 0.30 * risk_quality
    sizing_signal = np.nan_to_num(sizing_signal, nan=0.0, posinf=1.0, neginf=0.0)
    sizing_signal = clamp01(sizing_signal)

    out["score_strength"] = score_strength
    out["price_edge_strength"] = price_edge_strength
    out["risk_quality"] = risk_quality
    out["sizing_signal"] = sizing_signal
    out["alloc_fraction_target"] = MIN_ALLOC_FRACTION + (MAX_ALLOC_FRACTION - MIN_ALLOC_FRACTION) * sizing_signal
    return out


def simulate_variable_sizing(picks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks = picks.dropna(subset=["ret_pct", "buy_datetime", "exit_datetime", "ticker"]).copy()
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame()

    picks = picks.sort_values(["buy_datetime", "sizing_signal", "pred_mean4"], ascending=[True, False, False])
    cash = float(START_BUDGET_EUR)
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
            ticker_cap = MAX_TICKER_WEIGHT * budget_now
            cap_remaining = max(0.0, ticker_cap - ticker_exposure)
            if cap_remaining <= 0:
                continue

            alloc_frac = float(np.clip(row.get("alloc_fraction_target", 0.0), MIN_ALLOC_FRACTION, MAX_ALLOC_FRACTION))
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = base_eval.load_scored_day1()
    n_train, split_80 = base_eval.chrono_split_60_20_20(len(df))

    raw_cutoff, scored = compute_threshold_and_deciles(df)
    cal = scored.iloc[:split_80].copy()
    test = scored.iloc[split_80:].copy()
    test = add_sizing_features(test, cal)

    test_picks = test[
        (pd.to_numeric(test["pred_mean4"], errors="coerce") > raw_cutoff)
        & (pd.to_numeric(test["decile_score"], errors="coerce") >= MIN_DECILE_SCORE)
    ].copy()

    fixed_curve, fixed_trades = base_eval.simulate_budget_curve_from_picks(test_picks)
    fixed_metrics = base_eval.trade_metrics(fixed_curve, fixed_trades)

    var_curve, var_trades = simulate_variable_sizing(test_picks)
    var_metrics = base_eval.trade_metrics(var_curve, var_trades)

    summary = pd.DataFrame(
        [
            {
                "strategy": "fixed_cap_allocation",
                "min_decile_score": MIN_DECILE_SCORE,
                "raw_pred_mean4_cutoff": raw_cutoff,
                **fixed_metrics,
            },
            {
                "strategy": "variable_sizing_score_price_risk",
                "min_decile_score": MIN_DECILE_SCORE,
                "raw_pred_mean4_cutoff": raw_cutoff,
                **var_metrics,
            },
        ]
    )

    test_picks.to_csv(OUT_DIR / "variable_sizing_decile_input_picks.csv", index=False)
    fixed_trades.to_csv(OUT_DIR / "variable_sizing_decile_fixed_trade_log.csv", index=False)
    var_trades.to_csv(OUT_DIR / "variable_sizing_decile_trade_log.csv", index=False)
    summary.to_csv(OUT_DIR / "variable_sizing_decile_summary.csv", index=False)

    payload = {
        "params": {
            "min_decile_score": MIN_DECILE_SCORE,
            "min_alloc_fraction": MIN_ALLOC_FRACTION,
            "max_alloc_fraction": MAX_ALLOC_FRACTION,
            "max_ticker_weight": MAX_TICKER_WEIGHT,
            "start_budget_eur": START_BUDGET_EUR,
            "raw_pred_mean4_cutoff": raw_cutoff,
            "split_train_rows": n_train,
            "split_test_rows": len(test),
            "test_pick_rows": len(test_picks),
        },
        "fixed_metrics": fixed_metrics,
        "variable_metrics": var_metrics,
    }
    (OUT_DIR / "variable_sizing_decile_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Variable sizing backtest complete.")
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'variable_sizing_decile_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'variable_sizing_decile_summary.json'}")
    print(f"Saved: {OUT_DIR / 'variable_sizing_decile_trade_log.csv'}")


if __name__ == "__main__":
    main()
