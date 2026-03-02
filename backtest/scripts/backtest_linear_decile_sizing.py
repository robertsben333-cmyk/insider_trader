"""
Backtest linear decile-overweight sizing for day-1 trades.

Sizing logic:
- Baseline allocation per trade: 25% of current budget
- Overweight bonus: increases linearly with decile score above MIN_DECILE_SCORE
- Final target allocation:
    alloc = BASE_ALLOC_FRACTION + TOP_BONUS_FRACTION * score_strength
  where score_strength is in [0, 1]
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
MIN_DECILE_SCORE = 0.90

# New sizing policy
BASE_ALLOC_FRACTION = 0.25
TOP_BONUS_FRACTION = 0.25
MAX_ALLOC_FRACTION = BASE_ALLOC_FRACTION + TOP_BONUS_FRACTION
MAX_TICKER_WEIGHT_BASELINE = 0.25
MAX_TICKER_WEIGHT_LINEAR = MAX_ALLOC_FRACTION


@dataclass
class Position:
    ticker: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    invested: float
    ret_pct: float
    alloc_fraction: float


def percentile_rank(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_reference.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    idx = np.searchsorted(sorted_reference, values, side="right")
    return idx / float(sorted_reference.size)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def compute_threshold_and_deciles_from_train(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    n_train, _ = base_eval.chrono_split_60_20_20(len(df))
    train = df.iloc[:n_train].copy()
    pred_train = pd.to_numeric(train["pred_mean4"], errors="coerce").dropna().to_numpy(dtype=float)
    pred_train_sorted = np.sort(pred_train)
    raw_cutoff = float(np.quantile(pred_train_sorted, MIN_DECILE_SCORE))

    out = df.copy()
    pred_all = pd.to_numeric(out["pred_mean4"], errors="coerce").to_numpy(dtype=float)
    out["decile_score"] = percentile_rank(pred_train_sorted, pred_all)
    return raw_cutoff, out


def add_linear_sizing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    denom = max(1e-9, 1.0 - MIN_DECILE_SCORE)
    score_strength = clamp01((pd.to_numeric(out["decile_score"], errors="coerce").to_numpy(dtype=float) - MIN_DECILE_SCORE) / denom)
    alloc_target = BASE_ALLOC_FRACTION + TOP_BONUS_FRACTION * score_strength
    out["score_strength"] = score_strength
    out["alloc_fraction_target"] = np.clip(alloc_target, BASE_ALLOC_FRACTION, MAX_ALLOC_FRACTION)
    return out


def simulate_budget_curve(
    picks: pd.DataFrame,
    max_ticker_weight: float,
    use_alloc_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks = picks.dropna(subset=["ret_pct", "buy_datetime", "exit_datetime", "ticker"]).copy()
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame()

    sort_cols = ["buy_datetime", "pred_mean4"]
    asc = [True, False]
    if use_alloc_target:
        picks = picks.sort_values(["buy_datetime", "alloc_fraction_target", "pred_mean4"], ascending=[True, False, False])
    else:
        picks = picks.sort_values(sort_cols, ascending=asc)

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
                        "proceeds_eur": proceeds,
                        "pnl_eur": proceeds - pos.invested,
                    }
                )
            mark(next_exit)

    for buy_time, batch in picks.groupby("buy_datetime", sort=True):
        buy_time = pd.to_datetime(buy_time)
        close_until(buy_time)

        if use_alloc_target:
            batch = batch.sort_values(["alloc_fraction_target", "pred_mean4"], ascending=[False, False])
        else:
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
            ticker_cap = max_ticker_weight * budget_now
            cap_remaining = max(0.0, ticker_cap - ticker_exposure)
            if cap_remaining <= 0:
                continue

            if use_alloc_target:
                alloc_frac = float(np.clip(row.get("alloc_fraction_target", BASE_ALLOC_FRACTION), BASE_ALLOC_FRACTION, MAX_ALLOC_FRACTION))
                alloc = min(cash, cap_remaining, alloc_frac * budget_now)
            else:
                alloc_frac = BASE_ALLOC_FRACTION
                alloc = min(cash, cap_remaining)

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

    raw_cutoff, scored = compute_threshold_and_deciles_from_train(df)
    test = scored.iloc[split_80:].copy()
    test = add_linear_sizing(test)

    picks = test[
        (pd.to_numeric(test["pred_mean4"], errors="coerce") > raw_cutoff)
        & (pd.to_numeric(test["decile_score"], errors="coerce") >= MIN_DECILE_SCORE)
    ].copy()

    fixed_curve, fixed_trades = simulate_budget_curve(
        picks,
        max_ticker_weight=MAX_TICKER_WEIGHT_BASELINE,
        use_alloc_target=False,
    )
    linear_curve, linear_trades = simulate_budget_curve(
        picks,
        max_ticker_weight=MAX_TICKER_WEIGHT_LINEAR,
        use_alloc_target=True,
    )

    fixed_metrics = base_eval.trade_metrics(fixed_curve, fixed_trades)
    linear_metrics = base_eval.trade_metrics(linear_curve, linear_trades)

    summary = pd.DataFrame(
        [
            {
                "strategy": "fixed_25pct_cap",
                "min_decile_score": MIN_DECILE_SCORE,
                "raw_pred_mean4_cutoff": raw_cutoff,
                **fixed_metrics,
            },
            {
                "strategy": "linear_decile_overweight",
                "min_decile_score": MIN_DECILE_SCORE,
                "raw_pred_mean4_cutoff": raw_cutoff,
                **linear_metrics,
                "base_alloc_fraction": BASE_ALLOC_FRACTION,
                "top_bonus_fraction": TOP_BONUS_FRACTION,
                "max_alloc_fraction": MAX_ALLOC_FRACTION,
            },
        ]
    )

    picks.to_csv(OUT_DIR / "linear_decile_sizing_input_picks.csv", index=False)
    fixed_trades.to_csv(OUT_DIR / "linear_decile_sizing_fixed_trade_log.csv", index=False)
    linear_trades.to_csv(OUT_DIR / "linear_decile_sizing_trade_log.csv", index=False)
    summary.to_csv(OUT_DIR / "linear_decile_sizing_summary.csv", index=False)

    payload = {
        "params": {
            "min_decile_score": MIN_DECILE_SCORE,
            "base_alloc_fraction": BASE_ALLOC_FRACTION,
            "top_bonus_fraction": TOP_BONUS_FRACTION,
            "max_alloc_fraction": MAX_ALLOC_FRACTION,
            "max_ticker_weight_baseline": MAX_TICKER_WEIGHT_BASELINE,
            "max_ticker_weight_linear": MAX_TICKER_WEIGHT_LINEAR,
            "start_budget_eur": START_BUDGET_EUR,
            "raw_pred_mean4_cutoff": raw_cutoff,
            "split_train_rows": n_train,
            "split_test_rows": len(test),
            "test_pick_rows": len(picks),
        },
        "fixed_25pct_metrics": fixed_metrics,
        "linear_decile_overweight_metrics": linear_metrics,
        "linear_alloc_realized": {
            "avg_alloc_fraction": float(pd.to_numeric(linear_trades["alloc_fraction"], errors="coerce").mean()) if not linear_trades.empty else np.nan,
            "p95_alloc_fraction": float(pd.to_numeric(linear_trades["alloc_fraction"], errors="coerce").quantile(0.95)) if not linear_trades.empty else np.nan,
            "max_alloc_fraction": float(pd.to_numeric(linear_trades["alloc_fraction"], errors="coerce").max()) if not linear_trades.empty else np.nan,
        },
    }
    (OUT_DIR / "linear_decile_sizing_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Linear decile sizing backtest complete.")
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'linear_decile_sizing_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'linear_decile_sizing_summary.json'}")
    print(f"Saved: {OUT_DIR / 'linear_decile_sizing_trade_log.csv'}")


if __name__ == "__main__":
    main()
