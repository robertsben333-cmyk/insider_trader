# -*- coding: utf-8 -*-
"""
compare_weighted_vs_equal_trainval.py
=====================================
Assess whether weighted ensembles are better than equal-weight ensembles
for 3/4/5-model subsets using a chronological train/validation/test split.

Workflow per horizon:
  1) Train base models on TRAIN only
  2) Optimize non-negative simplex weights on VALIDATION only
  3) Compare weighted vs equal on VALIDATION and holdout TEST

Outputs:
  - models/weighted_vs_equal_trainval_time_split.json
"""

import json
import os
import sys
import warnings
from itertools import combinations, product
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

MODELS = ["HGBR", "XGBoost", "HMM-XGBoost", "ElasticNet", "SplineElasticNet"]
GRID_STEP = 0.05

# 60/20/20 chronological split
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2


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
        "top_decile_mean": float(top),
        "bottom_decile_mean": float(bottom),
        "decile_spread": float(top - bottom),
    }


def simplex_weights(n_models: int, step: float = 0.05):
    ticks = int(round(1.0 / step))
    for parts in product(range(ticks + 1), repeat=n_models):
        if sum(parts) == ticks:
            yield np.array(parts, dtype=float) * step


def optimize_weights(preds_val: np.ndarray, y_val: np.ndarray, step: float = 0.05):
    best_w = None
    best_score = -np.inf
    best_spread = -np.inf
    for w in simplex_weights(preds_val.shape[1], step=step):
        ds = decile_stats(preds_val @ w, y_val)
        score = ds["top_decile_mean"]
        spread = ds["decile_spread"]
        if (score > best_score) or (np.isclose(score, best_score) and spread > best_spread):
            best_score = score
            best_spread = spread
            best_w = w
    return best_w, float(best_score), float(best_spread)


def chrono_train_val_test_split(X, y, idx):
    n = len(X)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    # keep last block as test
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_val - 1)
    s1 = n_train
    s2 = n_train + n_val
    return (
        X.iloc[:s1].copy(),
        X.iloc[s1:s2].copy(),
        X.iloc[s2:].copy(),
        y[:s1].copy(),
        y[s1:s2].copy(),
        y[s2:].copy(),
        idx[:s1].copy(),
        idx[s1:s2].copy(),
        idx[s2:].copy(),
    )


def fit_predict_all_models(X_tr, y_tr, X_va, X_te, sub, idx_tr):
    hgbr = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr.fit(X_tr, y_tr)
    p_hgbr_va = hgbr.predict(X_va)
    p_hgbr_te = hgbr.predict(X_te)

    xgb_m = train_xgb(X_tr, y_tr)
    p_xgb_va = xgb_m.predict(to_xgb(X_va))
    p_xgb_te = xgb_m.predict(to_xgb(X_te))

    p_hmm_va, _ = hmm_regime_predict(X_tr, y_tr, X_va, sub, idx_tr, xgb_m)
    p_hmm_te, _ = hmm_regime_predict(X_tr, y_tr, X_te, sub, idx_tr, xgb_m)

    en_m = train_elasticnet(X_tr, y_tr)
    p_en_va = en_m.predict(to_linear_numeric(X_va))
    p_en_te = en_m.predict(to_linear_numeric(X_te))

    sp_m = train_spline_elasticnet(X_tr, y_tr)
    p_sp_va = sp_m.predict(X_va.copy())
    p_sp_te = sp_m.predict(X_te.copy())

    pred_va = {
        "HGBR": p_hgbr_va,
        "XGBoost": p_xgb_va,
        "HMM-XGBoost": p_hmm_va,
        "ElasticNet": p_en_va,
        "SplineElasticNet": p_sp_va,
    }
    pred_te = {
        "HGBR": p_hgbr_te,
        "XGBoost": p_xgb_te,
        "HMM-XGBoost": p_hmm_te,
        "ElasticNet": p_en_te,
        "SplineElasticNet": p_sp_te,
    }
    return pred_va, pred_te


def key(combo):
    return " + ".join(combo)


def run():
    print("Loading and engineering data...")
    df, features, _ = engineer_features(load_and_merge())
    print(f"  rows={len(df):,} | features={len(features)}")

    combo_grid = []
    for k in [3, 4, 5]:
        combo_grid.extend(list(combinations(MODELS, k)))

    out = {
        "meta": {
            "split": "time_train_val_test",
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
            "weight_grid_step": GRID_STEP,
            "models": MODELS,
            "combo_sizes": [3, 4, 5],
            "objective": "maximize validation top-decile mean, tie-break by validation decile spread",
        },
        "horizons": {},
        "summary": {},
    }

    summary_rows = []

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

        (
            X_tr,
            X_va,
            X_te,
            y_tr,
            y_va,
            y_te,
            idx_tr,
            idx_va,
            idx_te,
        ) = chrono_train_val_test_split(X, y, idx)

        d_tr0 = pd.to_datetime(sub.loc[idx_tr, "trade_date"]).min().date()
        d_tr1 = pd.to_datetime(sub.loc[idx_tr, "trade_date"]).max().date()
        d_va0 = pd.to_datetime(sub.loc[idx_va, "trade_date"]).min().date()
        d_va1 = pd.to_datetime(sub.loc[idx_va, "trade_date"]).max().date()
        d_te0 = pd.to_datetime(sub.loc[idx_te, "trade_date"]).min().date()
        d_te1 = pd.to_datetime(sub.loc[idx_te, "trade_date"]).max().date()
        print(
            f"  Split: train {len(X_tr):,} [{d_tr0} -> {d_tr1}] | "
            f"val {len(X_va):,} [{d_va0} -> {d_va1}] | "
            f"test {len(X_te):,} [{d_te0} -> {d_te1}]"
        )

        print("  Training base models...")
        pred_va, pred_te = fit_predict_all_models(X_tr, y_tr, X_va, X_te, sub, idx_tr)

        horizon_rows = []
        for combo in combo_grid:
            combo = list(combo)
            mat_va = np.column_stack([pred_va[m] for m in combo])
            mat_te = np.column_stack([pred_te[m] for m in combo])

            w_eq = np.full(len(combo), 1.0 / len(combo), dtype=float)
            ds_eq_va = decile_stats(mat_va @ w_eq, y_va)
            ds_eq_te = decile_stats(mat_te @ w_eq, y_te)

            w_opt, opt_val_top, opt_val_spread = optimize_weights(mat_va, y_va, step=GRID_STEP)
            ds_w_va = decile_stats(mat_va @ w_opt, y_va)
            ds_w_te = decile_stats(mat_te @ w_opt, y_te)

            row = {
                "combo": combo,
                "combo_key": key(combo),
                "n_models": len(combo),
                "equal_weights": {m: float(v) for m, v in zip(combo, w_eq)},
                "weighted_weights": {m: float(v) for m, v in zip(combo, w_opt)},
                "validation": {
                    "equal": ds_eq_va,
                    "weighted": ds_w_va,
                    "delta_top_decile_pp_weighted_minus_equal": float(
                        ds_w_va["top_decile_mean"] - ds_eq_va["top_decile_mean"]
                    ),
                },
                "test": {
                    "equal": ds_eq_te,
                    "weighted": ds_w_te,
                    "delta_top_decile_pp_weighted_minus_equal": float(
                        ds_w_te["top_decile_mean"] - ds_eq_te["top_decile_mean"]
                    ),
                },
                "opt_objective_validation": {
                    "top_decile_mean": opt_val_top,
                    "decile_spread": opt_val_spread,
                },
            }
            horizon_rows.append(row)
            summary_rows.append(
                {
                    "horizon": int(w),
                    "combo_key": row["combo_key"],
                    "n_models": row["n_models"],
                    "val_delta": row["validation"]["delta_top_decile_pp_weighted_minus_equal"],
                    "test_delta": row["test"]["delta_top_decile_pp_weighted_minus_equal"],
                    "val_weighted_top": row["validation"]["weighted"]["top_decile_mean"],
                    "test_weighted_top": row["test"]["weighted"]["top_decile_mean"],
                }
            )

        horizon_rows.sort(
            key=lambda r: (
                r["test"]["delta_top_decile_pp_weighted_minus_equal"],
                r["test"]["weighted"]["top_decile_mean"],
            ),
            reverse=True,
        )
        out["horizons"][str(w)] = horizon_rows

        best = horizon_rows[0]
        print(
            f"  Best test uplift vs equal: [{best['n_models']}] {best['combo_key']} | "
            f"delta={best['test']['delta_top_decile_pp_weighted_minus_equal']:+.3f}pp "
            f"(weighted top={best['test']['weighted']['top_decile_mean']:+.3f}%)"
        )

    # Aggregate by combo across horizons
    by_combo = {}
    for r in summary_rows:
        by_combo.setdefault(r["combo_key"], []).append(r)

    summary = []
    for ck, rows in by_combo.items():
        val_delta = np.array([x["val_delta"] for x in rows], dtype=float)
        test_delta = np.array([x["test_delta"] for x in rows], dtype=float)
        n_models = int(rows[0]["n_models"])
        summary.append(
            {
                "combo_key": ck,
                "n_models": n_models,
                "avg_val_delta_top_decile_pp": float(val_delta.mean()),
                "avg_test_delta_top_decile_pp": float(test_delta.mean()),
                "min_test_delta_top_decile_pp": float(test_delta.min()),
                "max_test_delta_top_decile_pp": float(test_delta.max()),
                "test_weighted_beats_equal_in_n_horizons": int((test_delta > 0).sum()),
            }
        )

    summary.sort(
        key=lambda r: (
            r["avg_test_delta_top_decile_pp"],
            r["test_weighted_beats_equal_in_n_horizons"],
            r["min_test_delta_top_decile_pp"],
        ),
        reverse=True,
    )

    out["summary"] = {
        "combo_ranking_by_avg_test_delta": summary,
        "global": {
            "n_combo_horizon_cases": len(summary_rows),
            "weighted_beats_equal_on_test_cases": int(sum(1 for r in summary_rows if r["test_delta"] > 0)),
            "weighted_beats_equal_on_validation_cases": int(sum(1 for r in summary_rows if r["val_delta"] > 0)),
            "avg_test_delta_top_decile_pp_all_cases": float(np.mean([r["test_delta"] for r in summary_rows])),
        },
    }

    out_path = MODEL_DIR / "weighted_vs_equal_trainval_time_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    g = out["summary"]["global"]
    print(
        "Global result: weighted > equal on "
        f"{g['weighted_beats_equal_on_test_cases']}/{g['n_combo_horizon_cases']} test cases | "
        f"avg test delta={g['avg_test_delta_top_decile_pp_all_cases']:+.3f}pp"
    )


if __name__ == "__main__":
    run()

