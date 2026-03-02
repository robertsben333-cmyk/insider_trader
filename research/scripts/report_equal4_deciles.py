# -*- coding: utf-8 -*-
"""
report_equal4_deciles.py
========================
Decile-level predicted vs actual returns for the selected equal-weight size-4 ensemble:
  HGBR + XGBoost + ElasticNet + SplineElasticNet

Split:
  chronological (oldest 80% train, newest 20% test)

Outputs:
  - models/equal4_deciles_time_split.json
  - models/equal4_deciles_time_split.csv
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from compare_models import to_linear_numeric, to_xgb, train_elasticnet, train_spline_elasticnet, train_xgb
from train_models import HORIZONS, MODEL_PARAMS, engineer_features, load_and_merge

MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.2
ENSEMBLE_MODELS = ["HGBR", "XGBoost", "ElasticNet", "SplineElasticNet"]


def chronological_train_test_split(X, y, idx, test_size: float):
    n = len(X)
    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 1)
    split = n - n_test
    return (
        X.iloc[:split].copy(),
        X.iloc[split:].copy(),
        y[:split].copy(),
        y[split:].copy(),
        idx[:split].copy(),
        idx[split:].copy(),
    )


def decile_table(signal: np.ndarray, actuals: np.ndarray, n_bins: int = 10):
    order = np.argsort(signal)
    n = len(order)
    bin_size = max(1, n // n_bins)
    rows = []
    for d in range(n_bins):
        start = d * bin_size
        end = n if d == n_bins - 1 else min(n, (d + 1) * bin_size)
        idx = order[start:end]
        p = signal[idx]
        a = actuals[idx]
        rows.append(
            {
                "decile": d + 1,
                "n": int(len(idx)),
                "mean_pred": float(np.mean(p)) if len(p) else np.nan,
                "mean_actual": float(np.mean(a)) if len(a) else np.nan,
                "pct_pos_actual": float((a > 0).mean() * 100) if len(a) else np.nan,
            }
        )
    return rows


def fit_predict_models(X_tr, y_tr, X_te):
    hgbr = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr.fit(X_tr, y_tr)
    p_hgbr = hgbr.predict(X_te)

    xgb_m = train_xgb(X_tr, y_tr)
    p_xgb = xgb_m.predict(to_xgb(X_te))

    en_m = train_elasticnet(X_tr, y_tr)
    p_en = en_m.predict(to_linear_numeric(X_te))

    sp_m = train_spline_elasticnet(X_tr, y_tr)
    p_sp = sp_m.predict(X_te.copy())

    return {
        "HGBR": p_hgbr,
        "XGBoost": p_xgb,
        "ElasticNet": p_en,
        "SplineElasticNet": p_sp,
    }


def run():
    print("Loading and engineering data...")
    df, features, _ = engineer_features(load_and_merge())
    print(f"  rows={len(df):,} | features={len(features)}")

    out = {"meta": {"split": "time", "test_size": TEST_SIZE, "ensemble_models": ENSEMBLE_MODELS}, "horizons": {}}
    csv_rows = []

    for w in HORIZONS:
        print(f"\n{'=' * 80}\nHorizon: {w}d\n{'=' * 80}")
        tgt = f"return_{w}d_pct"
        sub = df.dropna(subset=[tgt]).copy()
        lo, hi = sub[tgt].quantile([0.01, 0.99])
        sub[tgt] = sub[tgt].clip(lo, hi)
        sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
        sub = sub.dropna(subset=["trade_date"]).sort_values("trade_date").copy()

        X = sub[features].copy()
        y = sub[tgt].values
        idx = sub.index.values

        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = chronological_train_test_split(X, y, idx, TEST_SIZE)

        tr_start = pd.to_datetime(sub.loc[idx_tr, "trade_date"]).min()
        tr_end = pd.to_datetime(sub.loc[idx_tr, "trade_date"]).max()
        te_start = pd.to_datetime(sub.loc[idx_te, "trade_date"]).min()
        te_end = pd.to_datetime(sub.loc[idx_te, "trade_date"]).max()
        print(
            f"  Time split: train {len(X_tr):,} [{tr_start.date()} -> {tr_end.date()}] | "
            f"test {len(X_te):,} [{te_start.date()} -> {te_end.date()}]"
        )

        pred = fit_predict_models(X_tr, y_tr, X_te)
        signal = np.mean(np.column_stack([pred[m] for m in ENSEMBLE_MODELS]), axis=1)
        rows = decile_table(signal, y_te, n_bins=10)

        out["horizons"][str(w)] = {
            "target": tgt,
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "deciles": rows,
        }

        for r in rows:
            csv_rows.append(
                {
                    "horizon_days": int(w),
                    "target": tgt,
                    **r,
                }
            )

        print("  Deciles (mean_pred -> mean_actual):")
        for r in rows:
            print(
                f"    D{r['decile']:>2}: pred={r['mean_pred']:+.3f}% "
                f"actual={r['mean_actual']:+.3f}% n={r['n']}"
            )

    out_json = MODEL_DIR / "equal4_deciles_time_split.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    out_csv = MODEL_DIR / "equal4_deciles_time_split.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    run()

