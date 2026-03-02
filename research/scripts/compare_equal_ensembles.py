# -*- coding: utf-8 -*-
"""
compare_equal_ensembles.py
==========================
Evaluate all equal-weight ensembles of size 3/4/5 from:
  HGBR, XGBoost, HMM-XGBoost, ElasticNet, SplineElasticNet

Split:
  chronological (oldest 80% train, newest 20% test)

Outputs:
  - models/equal_ensemble_grid_time_split.json
"""

import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from compare_models import (
    hmm_regime_predict,
    to_linear_numeric,
    to_xgb,
    train_elasticnet,
    train_spline_elasticnet,
    train_xgb,
)
from train_models import HORIZONS, MODEL_PARAMS, engineer_features, load_and_merge

MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.2
MODELS = ["HGBR", "XGBoost", "HMM-XGBoost", "ElasticNet", "SplineElasticNet"]


def decile_stats(signal: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> dict:
    order = np.argsort(signal)
    n = len(order)
    bin_size = max(1, n // n_bins)

    rows = []
    for d in range(n_bins):
        start = d * bin_size
        end = n if d == n_bins - 1 else min(n, (d + 1) * bin_size)
        idx = order[start:end]
        vals = actuals[idx]
        rows.append(
            {
                "decile": d + 1,
                "n": int(len(vals)),
                "pct_pos": float((vals > 0).mean() * 100) if len(vals) else np.nan,
                "mean_actual": float(vals.mean()) if len(vals) else np.nan,
            }
        )

    top = rows[-1]["mean_actual"]
    bottom = rows[0]["mean_actual"]
    return {
        "rows": rows,
        "top_decile_mean": float(top),
        "bottom_decile_mean": float(bottom),
        "decile_spread": float(top - bottom),
    }


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


def fit_predict_all_models(X_tr, y_tr, X_te, sub, idx_tr):
    hgbr = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr.fit(X_tr, y_tr)
    p_hgbr = hgbr.predict(X_te)

    xgb_m = train_xgb(X_tr, y_tr)
    p_xgb = xgb_m.predict(to_xgb(X_te))

    p_hmm, _ = hmm_regime_predict(X_tr, y_tr, X_te, sub, idx_tr, xgb_m)

    en_m = train_elasticnet(X_tr, y_tr)
    p_en = en_m.predict(to_linear_numeric(X_te))

    sp_m = train_spline_elasticnet(X_tr, y_tr)
    p_sp = sp_m.predict(X_te.copy())

    return {
        "HGBR": p_hgbr,
        "XGBoost": p_xgb,
        "HMM-XGBoost": p_hmm,
        "ElasticNet": p_en,
        "SplineElasticNet": p_sp,
    }


def combo_key(combo):
    return " + ".join(combo)


def run():
    print("Loading and engineering data...")
    df, features, _ = engineer_features(load_and_merge())
    print(f"  rows={len(df):,} | features={len(features)}")

    all_results = {}
    all_combo_stats = {}

    combo_grid = []
    for k in [3, 4, 5]:
        combo_grid.extend(list(combinations(MODELS, k)))

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

        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = chronological_train_test_split(
            X, y, idx, TEST_SIZE
        )
        tr_start = pd.to_datetime(sub.loc[idx_tr, "trade_date"]).min()
        tr_end = pd.to_datetime(sub.loc[idx_tr, "trade_date"]).max()
        te_start = pd.to_datetime(sub.loc[idx_te, "trade_date"]).min()
        te_end = pd.to_datetime(sub.loc[idx_te, "trade_date"]).max()
        print(
            f"  Time split: train {len(X_tr):,} [{tr_start.date()} -> {tr_end.date()}] | "
            f"test {len(X_te):,} [{te_start.date()} -> {te_end.date()}]"
        )

        print("  Training base models and predicting test...")
        pred = fit_predict_all_models(X_tr, y_tr, X_te, sub, idx_tr)

        horizon_rows = []
        for combo in combo_grid:
            mat = np.column_stack([pred[m] for m in combo])
            signal = mat.mean(axis=1)
            ds = decile_stats(signal, y_te)
            key = combo_key(combo)
            row = {
                "combo": list(combo),
                "combo_key": key,
                "n_models": len(combo),
                "top_decile_mean": ds["top_decile_mean"],
                "bottom_decile_mean": ds["bottom_decile_mean"],
                "decile_spread": ds["decile_spread"],
            }
            horizon_rows.append(row)
            all_combo_stats.setdefault(key, []).append(row)

        horizon_rows = sorted(
            horizon_rows,
            key=lambda r: (r["top_decile_mean"], r["decile_spread"]),
            reverse=True,
        )
        all_results[str(w)] = horizon_rows

        print("  Best equal-weight combos:")
        for r in horizon_rows[:5]:
            print(
                f"    [{r['n_models']}] {r['combo_key']} | "
                f"top={r['top_decile_mean']:+.3f}% spread={r['decile_spread']:+.3f}pp"
            )

    summary = []
    for key, rows in all_combo_stats.items():
        rows = sorted(rows, key=lambda r: r["n_models"])
        top_vals = np.array([r["top_decile_mean"] for r in rows], dtype=float)
        spr_vals = np.array([r["decile_spread"] for r in rows], dtype=float)
        n_models = int(rows[0]["n_models"])
        summary.append(
            {
                "combo_key": key,
                "combo": rows[0]["combo"],
                "n_models": n_models,
                "avg_top_decile_mean": float(top_vals.mean()),
                "min_top_decile_mean": float(top_vals.min()),
                "avg_decile_spread": float(spr_vals.mean()),
                "min_decile_spread": float(spr_vals.min()),
            }
        )

    summary = sorted(
        summary,
        key=lambda r: (r["avg_top_decile_mean"], r["min_top_decile_mean"], r["avg_decile_spread"]),
        reverse=True,
    )

    out = {
        "meta": {
            "split": "time",
            "test_size": TEST_SIZE,
            "models": MODELS,
            "combination_sizes": [3, 4, 5],
            "ranking_metric": "avg_top_decile_mean (tie-break: min_top_decile_mean, avg_decile_spread)",
        },
        "by_horizon": all_results,
        "overall_ranking": summary,
    }

    out_path = MODEL_DIR / "equal_ensemble_grid_time_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\nOverall top 10 equal-weight ensembles:")
    for r in summary[:10]:
        print(
            f"  [{r['n_models']}] {r['combo_key']} | "
            f"avg_top={r['avg_top_decile_mean']:+.3f}% min_top={r['min_top_decile_mean']:+.3f}% "
            f"avg_spread={r['avg_decile_spread']:+.3f}pp"
        )


if __name__ == "__main__":
    run()

