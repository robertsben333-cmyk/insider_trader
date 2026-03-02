# -*- coding: utf-8 -*-
"""
ensemble_models.py
==================
Time-based evaluation of individual models and ensemble schemes.

Comparisons per horizon:
  - Individual models: HGBR, XGBoost, HMM-XGBoost, ElasticNet
  - Equal 3-model (with HMM): HGBR + XGBoost + HMM-XGBoost
  - Equal 3-model (with ElasticNet): HGBR + XGBoost + ElasticNet
  - Equal 4-model: HGBR + XGBoost + HMM-XGBoost + ElasticNet
  - Weighted 4-model: OOF-optimized non-negative weights summing to 1

Evaluation split:
  chronological split (oldest train / newest test).

OOF optimization objective:
  maximize top-decile mean actual return on train OOF predictions.
"""

import json
import os
import sys
import warnings
from itertools import product
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from compare_models import hmm_regime_predict, to_xgb, train_elasticnet, train_spline_elasticnet, train_xgb, to_linear_numeric
from train_models import FEATURES, HORIZONS, MODEL_PARAMS, engineer_features, load_and_merge

MODEL_DIR = BASE / "models"
CHART_DIR = BASE / "data" / "charts"
CHART_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
OOF_SPLITS = 5
GRID_STEP = 0.05

MODEL_NAMES_4 = ["HGBR", "XGBoost", "HMM-XGBoost", "ElasticNet"]
MODEL_NAMES_INDIV = ["HGBR", "XGBoost", "HMM-XGBoost", "ElasticNet", "SplineElasticNet"]
MODEL_NAMES_3_WITH_HMM = ["HGBR", "XGBoost", "HMM-XGBoost"]
MODEL_NAMES_3_NO_HMM = ["HGBR", "XGBoost", "ElasticNet"]


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


def simplex_weights(n_models: int, step: float = 0.05):
    ticks = int(round(1.0 / step))
    for parts in product(range(ticks + 1), repeat=n_models):
        if sum(parts) == ticks:
            yield np.array(parts, dtype=float) * step


def optimize_weights(preds_oof: np.ndarray, y_oof: np.ndarray, step: float = 0.05):
    best_w = None
    best_score = -np.inf
    best_spread = -np.inf

    for w in simplex_weights(preds_oof.shape[1], step=step):
        signal = preds_oof @ w
        ds = decile_stats(signal, y_oof)
        score = ds["top_decile_mean"]
        spread = ds["decile_spread"]
        if (score > best_score) or (np.isclose(score, best_score) and spread > best_spread):
            best_score = score
            best_spread = spread
            best_w = w

    return best_w, float(best_score), float(best_spread)


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


def run():
    print("Loading and engineering data...")
    df, features, _ = engineer_features(load_and_merge())
    print(f"  rows={len(df):,} | features={len(features)}")
    if len(features) != 16:
        print(f"WARNING: expected 16 features, found {len(features)}")

    report = {}

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

        print("  Fitting full-train models for test predictions...")
        pred_test = fit_predict_all_models(X_tr, y_tr, X_te, sub, idx_tr)
        ind_stats = {m: decile_stats(pred_test[m], y_te) for m in MODEL_NAMES_INDIV}

        print("  Building OOF predictions for weight optimization...")
        tscv = TimeSeriesSplit(n_splits=OOF_SPLITS)
        oof = {m: np.full(len(X_tr), np.nan) for m in MODEL_NAMES_4}

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tr), start=1):
            print(f"    Fold {fold}/{OOF_SPLITS}")
            X_f_tr = X_tr.iloc[tr_idx]
            y_f_tr = y_tr[tr_idx]
            X_f_va = X_tr.iloc[val_idx]
            idx_f_tr = idx_tr[tr_idx]

            fold_preds = fit_predict_all_models(X_f_tr, y_f_tr, X_f_va, sub, idx_f_tr)
            for m in MODEL_NAMES_4:
                oof[m][val_idx] = fold_preds[m]

        oof_mat_4 = np.column_stack([oof[m] for m in MODEL_NAMES_4])
        valid_oof = ~np.isnan(oof_mat_4).any(axis=1)
        if valid_oof.sum() < 100:
            raise RuntimeError(f"Insufficient valid OOF rows for horizon {w}d: {valid_oof.sum()}")

        w_opt, oof_top, oof_spread = optimize_weights(oof_mat_4[valid_oof], y_tr[valid_oof], step=GRID_STEP)

        test_mat_3_hmm = np.column_stack([pred_test[m] for m in MODEL_NAMES_3_WITH_HMM])
        test_mat_3_no_hmm = np.column_stack([pred_test[m] for m in MODEL_NAMES_3_NO_HMM])
        test_mat_4 = np.column_stack([pred_test[m] for m in MODEL_NAMES_4])

        w3_eq = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        w4_eq = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

        sig_3_hmm_eq = test_mat_3_hmm @ w3_eq
        sig_3_no_hmm_eq = test_mat_3_no_hmm @ w3_eq
        sig_4_eq = test_mat_4 @ w4_eq
        sig_4_opt = test_mat_4 @ w_opt

        ds_3_hmm_eq = decile_stats(sig_3_hmm_eq, y_te)
        ds_3_no_hmm_eq = decile_stats(sig_3_no_hmm_eq, y_te)
        ds_4_eq = decile_stats(sig_4_eq, y_te)
        ds_4_opt = decile_stats(sig_4_opt, y_te)

        improves_4_vs_3_hmm = ds_4_eq["top_decile_mean"] - ds_3_hmm_eq["top_decile_mean"]
        improves_3_no_hmm_vs_3_hmm = ds_3_no_hmm_eq["top_decile_mean"] - ds_3_hmm_eq["top_decile_mean"]

        print(f"  Equal-3 (with HMM) top decile mean: {ds_3_hmm_eq['top_decile_mean']:+.3f}%")
        print(f"  Equal-3 (no HMM) top decile mean: {ds_3_no_hmm_eq['top_decile_mean']:+.3f}%")
        print(f"  Equal-4 top decile mean: {ds_4_eq['top_decile_mean']:+.3f}%")
        print(f"  Weighted-4 top decile mean: {ds_4_opt['top_decile_mean']:+.3f}%")
        print(
            "  Individual top-decile means: "
            + ", ".join([f"{m}={ind_stats[m]['top_decile_mean']:+.3f}%" for m in MODEL_NAMES_INDIV])
        )
        print(f"  Equal-3 no-HMM delta (vs with-HMM): {improves_3_no_hmm_vs_3_hmm:+.3f}pp")
        print(f"  ElasticNet delta (Equal-4 - Equal-3 with HMM): {improves_4_vs_3_hmm:+.3f}pp")
        print(f"  OOF-opt weights: {dict(zip(MODEL_NAMES_4, [float(x) for x in w_opt]))}")

        report[w] = {
            "weights": {
                "equal_3_with_hmm": {m: float(v) for m, v in zip(MODEL_NAMES_3_WITH_HMM, w3_eq)},
                "equal_3_no_hmm": {m: float(v) for m, v in zip(MODEL_NAMES_3_NO_HMM, w3_eq)},
                "equal_4": {m: float(v) for m, v in zip(MODEL_NAMES_4, w4_eq)},
                "oof_opt_4": {m: float(v) for m, v in zip(MODEL_NAMES_4, w_opt)},
            },
            "oof_objective": {
                "top_decile_mean": oof_top,
                "decile_spread": oof_spread,
            },
            "test_deciles": {
                "individual": ind_stats,
                "equal_3_with_hmm": ds_3_hmm_eq,
                "equal_3_no_hmm": ds_3_no_hmm_eq,
                "equal_4": ds_4_eq,
                "weighted_4": ds_4_opt,
            },
            "equal_3_no_hmm_beats_equal_3_with_hmm": bool(improves_3_no_hmm_vs_3_hmm > 0),
            "equal_3_no_hmm_top_decile_delta_pp_vs_with_hmm": float(improves_3_no_hmm_vs_3_hmm),
            "elasticnet_improves_top_decile_equal_weight": bool(improves_4_vs_3_hmm > 0),
            "elasticnet_top_decile_delta_pp_equal_weight": float(improves_4_vs_3_hmm),
        }

    out_path = MODEL_DIR / "ensemble_results_time_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in report.items()}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Chart: top-decile mean return comparison per horizon
    x = np.arange(len(HORIZONS))
    width = 0.10
    top_hgbr = [report[w]["test_deciles"]["individual"]["HGBR"]["top_decile_mean"] for w in HORIZONS]
    top_xgb = [report[w]["test_deciles"]["individual"]["XGBoost"]["top_decile_mean"] for w in HORIZONS]
    top_hmm = [report[w]["test_deciles"]["individual"]["HMM-XGBoost"]["top_decile_mean"] for w in HORIZONS]
    top_en = [report[w]["test_deciles"]["individual"]["ElasticNet"]["top_decile_mean"] for w in HORIZONS]
    top_sp = [report[w]["test_deciles"]["individual"]["SplineElasticNet"]["top_decile_mean"] for w in HORIZONS]
    top3_hmm = [report[w]["test_deciles"]["equal_3_with_hmm"]["top_decile_mean"] for w in HORIZONS]
    top3_no_hmm = [report[w]["test_deciles"]["equal_3_no_hmm"]["top_decile_mean"] for w in HORIZONS]
    top4 = [report[w]["test_deciles"]["equal_4"]["top_decile_mean"] for w in HORIZONS]
    top4o = [report[w]["test_deciles"]["weighted_4"]["top_decile_mean"] for w in HORIZONS]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 4.0 * width, top_hgbr, width=width, label="HGBR", color="#4FC3F7", alpha=0.85)
    ax.bar(x - 3.0 * width, top_xgb, width=width, label="XGBoost", color="#FF8A65", alpha=0.85)
    ax.bar(x - 2.0 * width, top_hmm, width=width, label="HMM-XGBoost", color="#A5D6A7", alpha=0.85)
    ax.bar(x - 1.0 * width, top_en, width=width, label="ElasticNet", color="#CE93D8", alpha=0.85)
    ax.bar(x + 0.0 * width, top_sp, width=width, label="SplineElasticNet", color="#90CAF9", alpha=0.90)
    ax.bar(x + 1.0 * width, top3_hmm, width=width, label="Equal-3 (with HMM)", color="#64B5F6", alpha=0.85)
    ax.bar(x + 2.0 * width, top3_no_hmm, width=width, label="Equal-3 (with ElasticNet)", color="#81C784", alpha=0.85)
    ax.bar(x + 3.0 * width, top4, width=width, label="Equal-4", color="#BA68C8", alpha=0.85)
    ax.bar(x + 4.0 * width, top4o, width=width, label="Weighted-4", color="#FFD54F", alpha=0.90)
    ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("Top Decile Mean Return - Time Split", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}d" for w in HORIZONS])
    ax.set_ylabel("Top decile mean actual return (%)")
    ax.legend(ncols=5, fontsize=8)

    for i, vals in enumerate([top_hgbr, top_xgb, top_hmm, top_en, top_sp, top3_hmm, top3_no_hmm, top4, top4o]):
        xoff = (-4.0 * width, -3.0 * width, -2.0 * width, -1.0 * width, 0.0 * width, 1.0 * width, 2.0 * width, 3.0 * width, 4.0 * width)[i]
        for j, v in enumerate(vals):
            ax.text(x[j] + xoff, v + (0.05 if v >= 0 else -0.2), f"{v:+.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p = CHART_DIR / "ensemble_pmean_deciles_time_split.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"Saved: {p}")


if __name__ == "__main__":
    run()
