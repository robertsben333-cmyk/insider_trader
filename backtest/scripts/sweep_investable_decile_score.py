"""Sweep decile_score_threshold for day-1 investable trade selection.

For each threshold in [0.005, 0.010, ..., 1.000] (step 0.005), computes:
  - raw_pred_mean4_cutoff: quantile of pred_mean4 from the training split
  - val/test budget-simulation metrics at that threshold

Output: backtest/out/investable_decile_score_sweep_0005.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from backtest.scripts import evaluate_investable_risk_rules as base_eval

OUT_DIR = BASE / "backtest" / "out"
STEP = 0.005
THRESHOLDS = np.round(np.arange(STEP, 1.0 + STEP / 2, STEP), 3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = base_eval.load_scored_day1()
    n_train, split_80 = base_eval.chrono_split_60_20_20(len(df))
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:split_80].copy()
    test = df.iloc[split_80:].copy()

    pred_train = pd.to_numeric(train["pred_mean4"], errors="coerce").dropna().to_numpy(dtype=float)
    pred_train_sorted = np.sort(pred_train)

    print(f"Train rows: {n_train}  Val rows: {len(val)}  Test rows: {len(test)}")
    print(f"Sweeping {len(THRESHOLDS)} thresholds from {THRESHOLDS[0]:.3f} to {THRESHOLDS[-1]:.3f}\n")

    rows: list[dict] = []
    for threshold in THRESHOLDS:
        raw_cutoff = float(np.quantile(pred_train_sorted, float(threshold)))

        val_picks = val[pd.to_numeric(val["pred_mean4"], errors="coerce") > raw_cutoff].copy()
        test_picks = test[pd.to_numeric(test["pred_mean4"], errors="coerce") > raw_cutoff].copy()

        val_curve, val_trades = base_eval.simulate_budget_curve_from_picks(val_picks)
        test_curve, test_trades = base_eval.simulate_budget_curve_from_picks(test_picks)

        val_metrics = base_eval.trade_metrics(val_curve, val_trades)
        test_metrics = base_eval.trade_metrics(test_curve, test_trades)

        row: dict = {
            "decile_score_threshold": float(threshold),
            "raw_pred_mean4_cutoff": raw_cutoff,
        }
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        for k, v in test_metrics.items():
            row[f"test_{k}"] = v
        rows.append(row)

        print(
            f"threshold={threshold:.3f}  raw_cutoff={raw_cutoff:.5f}"
            f"  val_n={val_metrics['n_trades']}  test_n={test_metrics['n_trades']}"
        )

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "investable_decile_score_sweep_0005.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
