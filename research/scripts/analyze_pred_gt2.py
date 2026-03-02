# -*- coding: utf-8 -*-
"""
analyze_pred_gt2.py
===================
Deep-dive comparison of HGBR vs XGBoost vs HMM-XGBoost
using pred > 2% as the primary selection threshold.

Outputs:
  - Detailed stats table
  - Histogram of actual returns for pred>2% trades (per model/window)
  - Cumulative win-rate chart
  - Summary bar chart
"""

import sys
import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)

import xgboost as xgb
from hmmlearn import hmm as hmmlib
from train_models import load_and_merge, engineer_features, FEATURES, HORIZONS
from compare_models import (
    to_xgb, eval_all, train_xgb, fit_hmm, assign_regimes,
    hmm_regime_predict, N_REGIMES, HMM_COLS
)

MODEL_DIR = BASE / "models"
CHART_DIR = BASE / "data" / "charts"
CHART_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
THRESHOLD    = 2.0      # primary threshold

MODELS = ["HGBR", "XGBoost", "HMM-XGBoost"]
C = {"HGBR": "#4FC3F7", "XGBoost": "#FF8A65", "HMM-XGBoost": "#A5D6A7"}

# ── Load data (same pipeline) ─────────────────────────────────────────────────
print("Loading data...")
df, features, caps = engineer_features(load_and_merge())
print(f"  {len(df):,} rows\n")

# ── Collect predictions for all windows/models ────────────────────────────────
all_preds   = {}   # {w: {model: np.array of predictions}}
all_actuals = {}   # {w: np.array of actual returns}

for w in HORIZONS:
    tgt = f"return_{w}d_pct"
    sub = df.dropna(subset=[tgt]).copy()
    lo, hi = sub[tgt].quantile([0.01, 0.99])
    sub[tgt] = sub[tgt].clip(lo, hi)

    X = sub[features].copy()
    y = sub[tgt].values

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, sub.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    all_actuals[w] = y_te

    preds_w = {}

    # HGBR
    hgbr = joblib.load(MODEL_DIR / f"model_{w}d.pkl")
    preds_w["HGBR"] = hgbr.predict(X_te)

    # XGBoost
    xgb_mdl = joblib.load(MODEL_DIR / f"xgb_model_{w}d.pkl")
    preds_w["XGBoost"] = xgb_mdl.predict(to_xgb(X_te))

    # HMM-XGBoost  (retrain — fast because iters are low)
    print(f"  {w}d: rebuilding HMM-XGBoost...")
    hmm_pred, _ = hmm_regime_predict(X_tr, y_tr, X_te, sub, idx_tr, xgb_mdl)
    preds_w["HMM-XGBoost"] = hmm_pred

    all_preds[w] = preds_w


# ═══════════════════════════════════════════════════════════════════════════════
# DETAILED STATS TABLE  (pred > 2%)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*85)
print(f"  DETAILED COMPARISON  —  pred > {THRESHOLD}%  as selection filter")
print("="*85)

stats = {}   # {w: {model: dict}}

for w in HORIZONS:
    stats[w] = {}
    print(f"\n  {w}-day window  (n_test = {len(all_actuals[w]):,})")
    hdr = (f"  {'Model':<15} {'N flagged':>10} {'% of test':>10} "
           f"{'% act>0':>9} {'% act>2':>9} {'% act>5':>9} "
           f"{'Mean':>8} {'Median':>8} {'Std':>8} {'Sharpe*':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for mn in MODELS:
        preds   = all_preds[w][mn]
        actuals = all_actuals[w]
        mask    = preds > THRESHOLD
        n       = mask.sum()
        n_total = len(preds)

        if n == 0:
            print(f"  {mn:<15} {'—':>10}")
            stats[w][mn] = {}
            continue

        sel = actuals[mask]
        pct_of_test = n / n_total * 100
        pct_pos     = (sel > 0).mean()  * 100
        pct_gt2     = (sel > 2).mean()  * 100
        pct_gt5     = (sel > 5).mean()  * 100
        mean_r      = sel.mean()
        med_r       = np.median(sel)
        std_r       = sel.std()
        sharpe      = mean_r / std_r if std_r > 0 else 0.0   # daily, no Rf

        print(f"  {mn:<15} {n:>10,} {pct_of_test:>9.1f}% "
              f"{pct_pos:>8.1f}% {pct_gt2:>8.1f}% {pct_gt5:>8.1f}% "
              f"{mean_r:>+7.2f}% {med_r:>+7.2f}% {std_r:>7.2f}% {sharpe:>+8.3f}")

        stats[w][mn] = dict(
            n=int(n), pct_of_test=float(pct_of_test),
            pct_pos=float(pct_pos), pct_gt2=float(pct_gt2),
            pct_gt5=float(pct_gt5),
            mean=float(mean_r), median=float(med_r),
            std=float(std_r), sharpe=float(sharpe),
        )

print("\n* Sharpe = mean / std of selected subset (no risk-free rate, indicative only)")

# Save stats
with open(MODEL_DIR / "pred_gt2_stats.json", "w") as f:
    json.dump({str(w): v for w, v in stats.items()}, f, indent=2)
print("Stats saved -> models/pred_gt2_stats.json")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1: Actual return distribution for pred>2% trades (histogram grid)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating charts...")
plt.style.use("dark_background")

fig, axes = plt.subplots(len(HORIZONS), len(MODELS),
                         figsize=(18, 14), sharex="row")
fig.suptitle(f"Actual Return Distribution  |  pred > {THRESHOLD}%  (test set)",
             fontsize=15, color="white", y=1.01)

for row, w in enumerate(HORIZONS):
    actuals = all_actuals[w]
    # shared x-range for this row
    all_selected = np.concatenate([
        actuals[all_preds[w][mn] > THRESHOLD]
        for mn in MODELS if (all_preds[w][mn] > THRESHOLD).sum() > 0
    ])
    clip_v = np.percentile(np.abs(all_selected), 97) * 1.2
    bins   = np.linspace(-clip_v, clip_v, 55)

    for col, mn in enumerate(MODELS):
        ax   = axes[row, col]
        preds = all_preds[w][mn]
        mask  = preds > THRESHOLD
        sel   = actuals[mask]
        n     = len(sel)

        if n == 0:
            ax.text(0.5, 0.5, "no trades", ha="center", va="center",
                    transform=ax.transAxes, color="white")
            continue

        pct_pos = (sel > 0).mean() * 100

        ax.hist(sel, bins=bins, color=C[mn], alpha=0.80, edgecolor="none", density=True)
        ax.axvline(0,          color="white", lw=0.9, ls="--", alpha=0.7)
        ax.axvline(sel.mean(), color="yellow", lw=1.4, ls=":",
                   label=f"mean {sel.mean():+.2f}%")
        ax.axvline(THRESHOLD,  color="lime",   lw=1.0, ls="--", alpha=0.5,
                   label=f">{THRESHOLD}% line")

        title = (f"{mn}  [{w}d]\n"
                 f"n={n}  {pct_pos:.0f}% pos  mean={sel.mean():+.1f}%")
        ax.set_title(title, color="white", fontsize=8.5)
        ax.set_xlabel("Actual return (%)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.legend(fontsize=6, framealpha=0.2)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

plt.tight_layout()
p1 = CHART_DIR / "pred_gt2_return_distributions.png"
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="#0e1117")
plt.close()
print(f"  Saved: {p1}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2: Summary bar chart — key metrics at pred>2% per window
# ═══════════════════════════════════════════════════════════════════════════════
metrics_to_plot = [
    ("pct_pos",  "% Actual > 0%",        "%",  (40, 80)),
    ("pct_gt2",  "% Actual > 2%",        "%",  (0,  70)),
    ("mean",     "Mean Actual Return",   "%",  None),
    ("sharpe",   "Subset Sharpe*",       "",   None),
]

fig, axes = plt.subplots(len(metrics_to_plot), len(HORIZONS),
                         figsize=(20, 14))
fig.suptitle(f"Key Metrics at pred > {THRESHOLD}%  —  All Windows",
             fontsize=14, color="white", y=1.01)

x      = np.arange(len(MODELS))
colors = [C[m] for m in MODELS]

for row, (key, label, unit, ylim) in enumerate(metrics_to_plot):
    for col, w in enumerate(HORIZONS):
        ax  = axes[row, col]
        vals = [stats[w].get(mn, {}).get(key, 0.0) for mn in MODELS]

        bar_colors = []
        for v, col_c in zip(vals, colors):
            bar_colors.append(col_c)

        bars = ax.bar(x, vals, color=bar_colors, width=0.6,
                      alpha=0.85, edgecolor="none")

        # Reference line
        if key in ("pct_pos", "pct_gt2"):
            ax.axhline(50, color="white", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(0, color="white", lw=0.5, ls="-", alpha=0.3)

        if ylim:
            ax.set_ylim(*ylim)

        if row == 0:
            ax.set_title(f"{w}-day", color="white", fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{label} ({unit})" if unit else label,
                          color="white", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            ["HGBR", "XGB", "HMM-XGB"], fontsize=7, color="white", rotation=10
        )
        ax.tick_params(colors="white", labelsize=7)

        for bar, v in zip(bars, vals):
            fmt = f"{v:+.1f}" if key in ("mean", "sharpe") else f"{v:.0f}{unit}"
            h   = bar.get_height()
            ypos = h + abs(h) * 0.03 + 0.3
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    fmt, ha="center", va="bottom", fontsize=7, color="white")

        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

plt.tight_layout()
p2 = CHART_DIR / "pred_gt2_summary_bars.png"
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="#0e1117")
plt.close()
print(f"  Saved: {p2}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3: Cumulative actual return sorted by predicted return (precision curve)
# Shows how precision changes as you raise the threshold from 0 -> 10%
# ═══════════════════════════════════════════════════════════════════════════════
thresholds = np.linspace(0, 8, 50)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Precision Curve: % trades actually > 0%  as prediction threshold rises",
             fontsize=13, color="white")

for ax, w in zip(axes.flat, HORIZONS):
    actuals = all_actuals[w]
    for mn in MODELS:
        preds  = all_preds[w][mn]
        precs, ns = [], []
        for thr in thresholds:
            mask = preds > thr
            n    = mask.sum()
            prec = (actuals[mask] > 0).mean() * 100 if n >= 10 else np.nan
            precs.append(prec)
            ns.append(n)
        precs = np.array(precs, dtype=float)
        valid = ~np.isnan(precs)
        ax.plot(thresholds[valid], precs[valid],
                color=C[mn], lw=2.0, label=mn)

    ax.axhline(50, color="white", lw=0.8, ls="--", alpha=0.5, label="50% baseline")
    ax.axvline(THRESHOLD, color="yellow", lw=1.0, ls=":", alpha=0.7,
               label=f">{THRESHOLD}% threshold")
    ax.set_title(f"{w}-day return", color="white", fontsize=12)
    ax.set_xlabel("Prediction threshold (%)", color="white")
    ax.set_ylabel("% actual > 0% (precision)", color="white")
    ax.set_ylim(30, 100)
    ax.legend(fontsize=8, framealpha=0.25)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

plt.tight_layout()
p3 = CHART_DIR / "pred_gt2_precision_curve.png"
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor="#0e1117")
plt.close()
print(f"  Saved: {p3}")

print("\nDone.")
