"""
Backtest a test-set portfolio budget curve with position sizing constraints.

Rules:
- Start budget: 10,000 EUR
- Candidate trades: test-set rows where prediction > decile-9 lower bound from training set
- Position sizing: invest as much cash as possible, but never exceed 25% of budget per ticker
- Horizon: 1-day realized return (`return_1d_pct`)

Outputs:
- backtest/out/testset_budget_curve.csv
- backtest/out/testset_budget_curve.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from model_ensemble import MODEL_NAMES, predict_model, to_linear_numeric, to_xgb
from train_models import FEATURES, engineer_features, load_and_merge

MODEL_DIR = BASE / "models" / "prod4"
OUT_DIR = BASE / "backtest" / "out"

START_BUDGET_EUR = 10_000.0
MAX_TICKER_WEIGHT = 0.25
HORIZON_DAYS = 1


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


def decile9_lower_bound(signal: np.ndarray) -> float:
    """Lower bound of decile 9 using the same fixed-bin decile logic as model scripts."""
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if signal.size == 0:
        raise ValueError("No finite training predictions available for threshold computation.")

    order = np.argsort(signal)
    n = len(order)
    bin_size = max(1, n // 10)
    start = 8 * bin_size
    end = min(n, 9 * bin_size)
    idx = order[start:end]
    if len(idx) == 0:
        return float(np.quantile(signal, 0.8))
    return float(np.min(signal[idx]))


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
    return predict_model(model_name, model, X_use)


def load_and_score_day1() -> Tuple[pd.DataFrame, float, pd.Timestamp]:
    df_raw = load_and_merge()
    df, features, _ = engineer_features(df_raw)
    if features != FEATURES:
        raise ValueError("Feature list mismatch with canonical training FEATURES.")

    target_col = f"return_{HORIZON_DAYS}d_pct"
    sub = df.dropna(subset=[target_col]).copy()
    lo, hi = sub[target_col].quantile([0.01, 0.99])
    sub[target_col] = sub[target_col].clip(lo, hi)
    sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
    sub["buy_datetime"] = pd.to_datetime(sub["buy_datetime"], errors="coerce")
    sub = sub.dropna(subset=["trade_date", "buy_datetime"]).sort_values("trade_date").reset_index(drop=True)

    X = sub[features].copy()
    n_train, split_80 = chrono_split_60_20_20(len(sub))
    _ = n_train  # keep explicit for readability

    models: Dict[str, object] = {}
    for model_name in MODEL_NAMES:
        model_path = MODEL_DIR / f"model_{HORIZON_DAYS}d_{model_name}.pkl"
        models[model_name] = joblib.load(model_path)

    pred_cols = []
    for model_name in MODEL_NAMES:
        col = f"pred_{model_name}"
        sub[col] = predict_model_aligned(model_name, models[model_name], X)
        pred_cols.append(col)
    sub["pred_mean4"] = sub[pred_cols].mean(axis=1)

    train_signal = sub.loc[: split_80 - 1, "pred_mean4"].to_numpy(dtype=float)
    threshold = decile9_lower_bound(train_signal)
    test_start = sub.loc[split_80, "buy_datetime"]

    test = sub.iloc[split_80:].copy()
    test["exit_datetime"] = test["buy_datetime"].apply(next_business_day_close)
    return test, threshold, test_start


def simulate_budget_curve(test_df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    picks = test_df[
        (pd.to_numeric(test_df["pred_mean4"], errors="coerce") > float(threshold))
        & pd.to_numeric(test_df["return_1d_pct"], errors="coerce").notna()
    ].copy()
    if picks.empty:
        raise ValueError("No test trades passed the threshold.")

    picks["ret_pct"] = pd.to_numeric(picks["return_1d_pct"], errors="coerce")
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


def save_outputs(curve_df: pd.DataFrame, threshold: float, test_start: pd.Timestamp, trades_df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_csv = OUT_DIR / "testset_budget_curve.csv"
    trades_csv = OUT_DIR / "testset_trade_log.csv"
    fig_path = OUT_DIR / "testset_budget_curve.png"

    curve_df.to_csv(curve_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(curve_df["timestamp"], curve_df["budget_eur"], linewidth=2)
    ax.set_title("Test-set Portfolio Budget Over Time (1d horizon)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Budget (EUR)")
    ax.grid(alpha=0.25)
    text = (
        f"Start budget: EUR {START_BUDGET_EUR:,.0f}\n"
        f"Threshold (train decile-9 lower bound): {threshold:.4f}%\n"
        f"Test start: {pd.to_datetime(test_start).date()}\n"
        f"Max per ticker: {MAX_TICKER_WEIGHT:.0%}"
    )
    ax.text(
        0.01,
        0.99,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    start_budget = float(curve_df["budget_eur"].iloc[0])
    end_budget = float(curve_df["budget_eur"].iloc[-1])
    ret_pct = (end_budget / start_budget - 1.0) * 100.0
    print(f"Threshold used (train decile-9 lower bound): {threshold:.6f}%")
    print(f"Trades executed: {len(trades_df):,}")
    print(f"Final budget: EUR {end_budget:,.2f} ({ret_pct:+.2f}%)")
    print(f"Curve CSV: {curve_csv}")
    print(f"Trades CSV: {trades_csv}")
    print(f"Figure: {fig_path}")


def main():
    test_df, threshold, test_start = load_and_score_day1()
    curve_df, trades_df = simulate_budget_curve(test_df, threshold)
    save_outputs(curve_df, threshold, test_start, trades_df)


if __name__ == "__main__":
    main()
