"""
Evaluate investable-selection rules for day-1 trades:
  baseline: pred_mean4 > fixed threshold (default 0.8)
  risk-aware variants:
    - volatility cap on prior_30d_vol
    - downside-risk cap using train ticker loss rate
    - combined cap

All rules are evaluated with the same budget simulation constraints used in
backtest_testset_budget_curve.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import train_models
from model_ensemble import MODEL_NAMES, to_linear_numeric, to_xgb

MODEL_DIR = BASE / "models" / "prod4"
OUT_DIR = BASE / "backtest" / "out"

START_BUDGET_EUR = 10_000.0
MAX_TICKER_WEIGHT = 0.25
FIXED_THRESHOLD = 0.8


@dataclass
class Position:
    ticker: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    invested: float
    ret_pct: float


def chrono_split_60_20_20(n: int) -> Tuple[int, int]:
    n_train = int(round(n * 0.6))
    n_val = int(round(n * 0.2))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    return n_train, n_train + n_val


def next_business_day_close(ts: pd.Timestamp) -> pd.Timestamp:
    nxt = (ts + pd.offsets.BDay(1)).normalize()
    return nxt + pd.Timedelta(hours=16)


def align_features_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        return X
    cols = [str(c) for c in cols]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Model expects missing feature columns: {missing}")
    return X[cols].copy()


def predict_model_aligned(model_name: str, model, X: pd.DataFrame) -> np.ndarray:
    X_use = align_features_to_model(model, X)
    if model_name == "HGBR":
        return model.predict(X_use)
    if model_name == "XGBoost":
        return model.predict(to_xgb(X_use))
    if model_name == "ElasticNet":
        return model.predict(to_linear_numeric(X_use))
    if model_name == "SplineElasticNet":
        return model.predict(X_use.copy())
    raise ValueError(f"Unknown model name: {model_name}")


def load_scored_day1() -> pd.DataFrame:
    train_models.AGGREGATED_CSV = "backtest/data/backtest_results_aggregated_unfiltered.csv"
    raw = train_models.load_and_merge()
    df, features, _ = train_models.engineer_features(raw)

    # Keep cohorts used in the live system.
    if "is_supported_type" in df.columns:
        df = df[df["is_supported_type"] == 1].copy()

    tgt = "return_1d_pct"
    sub = df.dropna(subset=[tgt]).copy()
    lo, hi = sub[tgt].quantile([0.01, 0.99])
    sub[tgt] = sub[tgt].clip(lo, hi)
    sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
    sub["buy_datetime"] = pd.to_datetime(sub["buy_datetime"], errors="coerce")
    sub = sub.dropna(subset=["trade_date", "buy_datetime"]).sort_values("trade_date").reset_index(drop=True)

    X = sub[features].copy()
    models: Dict[str, object] = {}
    for model_name in MODEL_NAMES:
        path = MODEL_DIR / f"model_1d_{model_name}.pkl"
        models[model_name] = joblib.load(path)

    pred_cols = []
    for model_name in MODEL_NAMES:
        col = f"pred_{model_name}"
        sub[col] = predict_model_aligned(model_name, models[model_name], X)
        pred_cols.append(col)
    sub["pred_mean4"] = sub[pred_cols].mean(axis=1)
    sub["ret_pct"] = pd.to_numeric(sub[tgt], errors="coerce")
    sub["exit_datetime"] = sub["buy_datetime"].apply(next_business_day_close)
    sub["prior_30d_vol"] = pd.to_numeric(sub["prior_30d_vol"], errors="coerce")
    return sub


def simulate_budget_curve_from_picks(picks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    picks = picks.dropna(subset=["ret_pct", "buy_datetime", "exit_datetime", "ticker"]).copy()
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame()

    picks = picks.sort_values(["buy_datetime", "pred_mean4"], ascending=[True, False]).reset_index(drop=True)
    cash = float(START_BUDGET_EUR)
    open_positions: List[Position] = []
    trade_log: List[dict] = []
    curve: List[dict] = []

    start_ts = pd.to_datetime(picks["buy_datetime"].min()) - pd.Timedelta(minutes=1)
    curve.append({"timestamp": start_ts, "cash_eur": cash, "invested_eur": 0.0, "budget_eur": cash})

    def mark(ts: pd.Timestamp):
        invested = float(sum(p.invested for p in open_positions))
        curve.append({"timestamp": ts, "cash_eur": cash, "invested_eur": invested, "budget_eur": cash + invested})

    def close_until(ts: pd.Timestamp):
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
                        "proceeds_eur": proceeds,
                        "pnl_eur": proceeds - pos.invested,
                    }
                )
            mark(next_exit)

    for buy_time, batch in picks.groupby("buy_datetime", sort=True):
        buy_time = pd.to_datetime(buy_time)
        close_until(buy_time)
        batch = batch.sort_values("pred_mean4", ascending=False)
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
            alloc = min(cash, max(0.0, ticker_cap - ticker_exposure))
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
                )
            )
        mark(buy_time)

    if open_positions:
        last_exit = max(p.exit_time for p in open_positions)
        close_until(last_exit + pd.Timedelta(seconds=1))

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    trades_df = pd.DataFrame(trade_log).sort_values("entry_time").reset_index(drop=True)
    return curve_df, trades_df


def trade_metrics(curve_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    if curve_df.empty or trades_df.empty:
        return {
            "n_trades": 0,
            "return_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "win_rate_pct": np.nan,
            "mean_trade_ret_pct": np.nan,
            "std_trade_ret_pct": np.nan,
            "sharpe_trade": np.nan,
            "sortino_trade": np.nan,
            "cvar5_trade_ret_pct": np.nan,
            "objective_val": -np.inf,
        }

    start_budget = float(curve_df["budget_eur"].iloc[0])
    end_budget = float(curve_df["budget_eur"].iloc[-1])
    ret_pct = (end_budget / start_budget - 1.0) * 100.0

    budgets = pd.to_numeric(curve_df["budget_eur"], errors="coerce").to_numpy(dtype=float)
    peak = np.maximum.accumulate(budgets)
    dd = (budgets / np.where(peak == 0, np.nan, peak) - 1.0) * 100.0
    max_dd = float(np.nanmin(dd))

    r = pd.to_numeric(trades_df["ret_pct"], errors="coerce").dropna().to_numpy(dtype=float)
    n = len(r)
    win_rate = float((r > 0).mean() * 100.0) if n else np.nan
    mu = float(np.mean(r)) if n else np.nan
    sd = float(np.std(r, ddof=0)) if n else np.nan
    down = r[r < 0]
    sd_down = float(np.std(down, ddof=0)) if len(down) else np.nan
    sharpe = float(mu / sd) if n and sd > 0 else np.nan
    sortino = float(mu / sd_down) if len(down) and sd_down > 0 else np.nan
    k = max(1, int(np.ceil(0.05 * n)))
    cvar5 = float(np.mean(np.sort(r)[:k])) if n else np.nan

    # Utility: reward return, penalize drawdown.
    objective = float(ret_pct + 0.5 * max_dd)
    return {
        "n_trades": int(n),
        "return_pct": float(ret_pct),
        "max_drawdown_pct": float(max_dd),
        "win_rate_pct": float(win_rate),
        "mean_trade_ret_pct": float(mu),
        "std_trade_ret_pct": float(sd),
        "sharpe_trade": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino_trade": float(sortino) if np.isfinite(sortino) else np.nan,
        "cvar5_trade_ret_pct": float(cvar5),
        "objective_val": objective,
    }


def pick_best_by_validation(candidates: List[dict]) -> dict:
    if not candidates:
        raise ValueError("No candidate strategies.")
    return sorted(
        candidates,
        key=lambda x: (
            x["val"]["objective_val"],
            x["val"]["sortino_trade"] if np.isfinite(x["val"]["sortino_trade"]) else -np.inf,
            x["val"]["return_pct"],
        ),
        reverse=True,
    )[0]


def build_ticker_loss_rate(train_df: pd.DataFrame) -> pd.Series:
    g = train_df.groupby("ticker")["ret_pct"].apply(lambda s: float((pd.to_numeric(s, errors="coerce") < 0).mean()))
    return g


def eval_strategy(split_df: pd.DataFrame, mask: pd.Series) -> dict:
    picks = split_df.loc[mask].copy()
    curve, trades = simulate_budget_curve_from_picks(picks)
    return trade_metrics(curve, trades)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_scored_day1()
    n_train, split_80 = chrono_split_60_20_20(len(df))
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:split_80].copy()
    test = df.iloc[split_80:].copy()

    base_train = train["pred_mean4"] > FIXED_THRESHOLD
    base_val = val["pred_mean4"] > FIXED_THRESHOLD
    base_test = test["pred_mean4"] > FIXED_THRESHOLD

    results = []

    # Baseline
    baseline = {
        "strategy": "baseline_pred_gt_0p8",
        "params": {"pred_threshold": FIXED_THRESHOLD},
        "val": eval_strategy(val, base_val),
        "test": eval_strategy(test, base_test),
    }
    results.append(baseline)

    # Volatility cap candidates
    vol_cands = []
    vol_train = pd.to_numeric(train.loc[base_train, "prior_30d_vol"], errors="coerce").dropna()
    for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        if vol_train.empty:
            continue
        cap = float(vol_train.quantile(q))
        m_val = base_val & (pd.to_numeric(val["prior_30d_vol"], errors="coerce") <= cap)
        m_test = base_test & (pd.to_numeric(test["prior_30d_vol"], errors="coerce") <= cap)
        vol_cands.append(
            {
                "strategy": "vol_cap",
                "params": {"pred_threshold": FIXED_THRESHOLD, "vol_cap_quantile": q, "vol_cap_value": cap},
                "val": eval_strategy(val, m_val),
                "test": eval_strategy(test, m_test),
            }
        )
    best_vol = pick_best_by_validation(vol_cands)
    results.append(best_vol)

    # Downside-risk cap candidates (ticker loss rate on train)
    loss_rate = build_ticker_loss_rate(train)
    global_loss = float((train["ret_pct"] < 0).mean())
    train_lr = train["ticker"].map(loss_rate).fillna(global_loss)
    train_lr_base = train_lr[base_train]
    risk_cands = []
    for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        if train_lr_base.empty:
            continue
        cap = float(train_lr_base.quantile(q))
        val_lr = val["ticker"].map(loss_rate).fillna(global_loss)
        test_lr = test["ticker"].map(loss_rate).fillna(global_loss)
        m_val = base_val & (val_lr <= cap)
        m_test = base_test & (test_lr <= cap)
        risk_cands.append(
            {
                "strategy": "ticker_lossrate_cap",
                "params": {"pred_threshold": FIXED_THRESHOLD, "lossrate_cap_quantile": q, "lossrate_cap_value": cap},
                "val": eval_strategy(val, m_val),
                "test": eval_strategy(test, m_test),
            }
        )
    best_risk = pick_best_by_validation(risk_cands)
    results.append(best_risk)

    # Combined risk cap candidates
    combo_cands = []
    for v in vol_cands:
        for r in risk_cands:
            cap_vol = v["params"]["vol_cap_value"]
            cap_lr = r["params"]["lossrate_cap_value"]
            val_lr = val["ticker"].map(loss_rate).fillna(global_loss)
            test_lr = test["ticker"].map(loss_rate).fillna(global_loss)
            m_val = (
                base_val
                & (pd.to_numeric(val["prior_30d_vol"], errors="coerce") <= cap_vol)
                & (val_lr <= cap_lr)
            )
            m_test = (
                base_test
                & (pd.to_numeric(test["prior_30d_vol"], errors="coerce") <= cap_vol)
                & (test_lr <= cap_lr)
            )
            combo_cands.append(
                {
                    "strategy": "combined_vol_and_lossrate_cap",
                    "params": {
                        "pred_threshold": FIXED_THRESHOLD,
                        "vol_cap_quantile": v["params"]["vol_cap_quantile"],
                        "vol_cap_value": cap_vol,
                        "lossrate_cap_quantile": r["params"]["lossrate_cap_quantile"],
                        "lossrate_cap_value": cap_lr,
                    },
                    "val": eval_strategy(val, m_val),
                    "test": eval_strategy(test, m_test),
                }
            )
    best_combo = pick_best_by_validation(combo_cands)
    results.append(best_combo)

    summary_rows = []
    for r in results:
        row = {
            "strategy": r["strategy"],
            "params": json.dumps(r["params"]),
        }
        for split_name in ["val", "test"]:
            for k, v in r[split_name].items():
                row[f"{split_name}_{k}"] = v
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("test_objective_val", ascending=False).reset_index(drop=True)

    out_csv = OUT_DIR / "investable_risk_rule_comparison.csv"
    out_json = OUT_DIR / "investable_risk_rule_comparison.json"
    summary.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Fixed threshold baseline:", FIXED_THRESHOLD)
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
