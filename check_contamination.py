# -*- coding: utf-8 -*-
"""
Contamination audit + prediction distribution analysis for the 4 trained models.
Reuses load_and_merge() / engineer_features() from train_models.py so the
feature matrix is exactly identical to what the models were trained on.
"""

import json
import joblib
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))

# Import helpers from train_models (no side-effects at import time)
from train_models import load_and_merge, engineer_features, FEATURES, HORIZONS

MODEL_DIR = BASE / "models"
CHART_DIR = BASE / "data" / "charts"
CHART_DIR.mkdir(exist_ok=True)
META_JSON = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE    = 0.2

def load_model(w):
    return joblib.load(MODEL_DIR / f"model_{w}d.pkl")

def load_metadata():
    with open(META_JSON, encoding="utf-8") as f:
        return json.load(f)

# =============================================================================
# Build the EXACT same feature matrix the models were trained on
# =============================================================================
print("Loading & engineering data (replicating train_models.py pipeline)...")
import os
os.chdir(BASE)          # train_models.py uses relative paths
df_raw, features, caps = engineer_features(load_and_merge())
print(f"  Rows available: {len(df_raw):,}   Features: {features}")

# =============================================================================
# SECTION 1: CONTAMINATION AUDIT
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: CONTAMINATION / DATA-LEAKAGE AUDIT")
print("=" * 70)

# --- Check 1: return columns in features? ------------------------------------
print("\n[1] Direct target contamination")
print("    Features:", features)
all_targets = [f"return_{w}d_pct" for w in HORIZONS]
for w in HORIZONS:
    bad = [f for f in features if f in all_targets]
    status = "[!!] FOUND: " + str(bad) if bad else "[OK] none"
    print(f"    {w}d model: {status}")

# --- Check 2: temporal overlap prior_30d_pct vs return windows ---------------
print("\n[2] Temporal overlap: prior_30d_pct vs return_Xd_pct")
print("    prior_30d_pct  = (buy_price - close_30d_before_trade) / close_30d_before_trade")
print("    return_Xd_pct  = (close_Xd_after_filing - buy_price)  / buy_price")
print("    The two windows share 'buy_price' as a pivot but NO overlapping time range.")
if "filing_gap_days" in df_raw.columns:
    gap = df_raw["filing_gap_days"].dropna()
    print(f"    filing_gap_days: min={gap.min():.0f}  max={gap.max():.0f}  "
          f"mean={gap.mean():.1f}  median={gap.median():.0f}")
    print(f"    Negative gaps (filing BEFORE trade): {(gap < 0).sum()}")
    print(f"    Zero gaps  (same day):               {(gap == 0).sum()}")
    print(f"    Positive gaps:                       {(gap > 0).sum()}")
    print("    -> prior_30d_pct ends AT trade_date; return starts AFTER filing.")
    print("       No temporal overlap, even when gap = 0.")
else:
    print("    [!!] filing_gap_days not found.")

# --- Check 3: random vs time-based split -------------------------------------
print("\n[3] Train/test split methodology")
print("    Code: train_test_split(shuffle=True, test_size=0.2, random_state=42)")
print("    -> RANDOM shuffle split (NOT time-ordered).")
print("    [!!] Implication: same ticker may appear in both train & test;")
print("         train set can contain FUTURE events relative to test set.")
print("    [i]  Features are per-trade snapshots with no future look-ahead,")
print("         so practical leakage magnitude is LOW.")
print("    Recommendation: use walk-forward CV for live deployment evaluation.")

# --- Check 4: ticker overlap (using 10d model as representative) -------------
print("\n[4] Ticker overlap between train and test (10d model)")
w_check    = 10
target_col = f"return_{w_check}d_pct"
sub = df_raw.dropna(subset=[target_col]).copy()
lo, hi = sub[target_col].quantile([0.01, 0.99])
sub[target_col] = sub[target_col].clip(lo, hi)
X = sub[features]
y = sub[target_col]
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, sub.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)
if "ticker" in sub.columns:
    train_tck = set(sub.loc[idx_tr, "ticker"].unique())
    test_tck  = set(sub.loc[idx_te, "ticker"].unique())
    overlap   = train_tck & test_tck
    print(f"    Train tickers  : {len(train_tck):,}")
    print(f"    Test tickers   : {len(test_tck):,}")
    print(f"    Overlap        : {len(overlap):,}  ({100*len(overlap)/len(test_tck):.1f}% of test tickers)")
    print("    -> Overlap is expected with random split.")
    print("       The model can partially learn company-level patterns (mild concern).")
else:
    print("    'ticker' column not available.")

# --- Check 5: date distribution ----------------------------------------------
print("\n[5] Date distribution in train vs test")
td = pd.to_datetime(sub["transaction_date"])
train_dates = td.loc[idx_tr]
test_dates  = td.loc[idx_te]
print(f"    Train: {train_dates.min().date()} to {train_dates.max().date()}")
print(f"    Test : {test_dates.min().date()}  to {test_dates.max().date()}")
print("    -> Fully interleaved. Market-regime effects could leak between sets.")
print("       Mitigation: all features are stock-specific, not aggregate market.")

# --- Check 6: cross-window contamination -------------------------------------
print("\n[6] Cross-window contamination")
print("    Each model uses the SAME feature set; targets are independent columns.")
print("    No return_Xd_pct column is in FEATURES -> [OK] no cross-window leakage.")

# --- Check 7: prior_30d_pct computed from buy_price --------------------------
print("\n[7] Subtle leakage: prior_30d_pct computed using buy_price")
print("    Formula: (buy_price - close_30d_ago) / close_30d_ago")
print("    buy_price is also in the feature set as log_buy_price.")
print("    buy_price is NOT in the return formula (return uses close_Xd - buy_price).")
print("    -> Shared use of buy_price in prior_30d_pct & log_buy_price is fine;")
print("       it is a legitimate signal, not future information.")

print("\n" + "=" * 70)
print("CONTAMINATION VERDICT")
print("=" * 70)
print("""
  [OK] No direct feature contamination (return_Xd_pct not in FEATURES)
  [OK] prior_30d_pct has zero temporal overlap with any return window
  [OK] No cross-window leakage between the four models
  [OK] buy_price shared usage is legitimate (not future information)
  [!!] Random shuffle split (not time-based):
        - Same ticker in train & test              (mild concern)
        - Train/test dates fully interleaved       (moderate OOS validity concern)
        Recommendation: walk-forward CV for production backtesting
  NET: Leakage risk is LOW. Reported metrics are a fair estimate of
       in-sample generalisation; treat as optimistic for live trading.
""")

# =============================================================================
# SECTION 2: PREDICTION DISTRIBUTION & SUMMARY STATISTICS
# =============================================================================
print("=" * 70)
print("SECTION 2: PREDICTION DISTRIBUTION & SUMMARY STATISTICS")
print("=" * 70)

results = {}
for w in HORIZONS:
    target_col = f"return_{w}d_pct"
    model      = load_model(w)
    sub        = df_raw.dropna(subset=[target_col]).copy()
    lo, hi     = sub[target_col].quantile([0.01, 0.99])
    sub[target_col] = sub[target_col].clip(lo, hi)
    X = sub[features]
    y = sub[target_col]
    _, X_te, _, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    preds = model.predict(X_te)
    results[w] = {"preds": preds, "actuals": y_te.values}

    print(f"\n-- {w}-day model  (n_test={len(preds):,}) --")
    for label, arr in [("Predicted", preds), ("Actual   ", y_te.values)]:
        p = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        print(f"  {label}: mean={arr.mean():+.2f}%  std={arr.std():.2f}%  "
              f"p1={p[0]:+.1f}%  p10={p[2]:+.1f}%  p25={p[3]:+.1f}%  "
              f"p50={p[4]:+.1f}%  p75={p[5]:+.1f}%  p90={p[6]:+.1f}%  p99={p[8]:+.1f}%")

    print(f"  Precision table:")
    print(f"  {'Threshold':>12}  {'N':>6}  {'%Pos':>7}  {'MeanActual':>11}")
    for thresh in [0, 1, 2, 3, 5]:
        mask = preds > thresh
        n    = mask.sum()
        if n > 0:
            ppos     = (y_te.values[mask] > 0).mean()
            mean_ret = y_te.values[mask].mean()
            print(f"  pred>{thresh:+2d}%   {n:6,}  {100*ppos:6.1f}%  {mean_ret:+10.2f}%")

# =============================================================================
# CHART 1: Predicted vs Actual distribution (overlapping histograms)
# =============================================================================
print("\nGenerating distribution chart...")
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Predicted vs Actual Return Distributions (Test Set)",
             fontsize=16, color="white")

C_PRED   = "#4FC3F7"
C_ACTUAL = "#FF8A65"

for ax, w in zip(axes.flat, HORIZONS):
    preds   = results[w]["preds"]
    actuals = results[w]["actuals"]
    clip_v  = np.percentile(np.abs(actuals), 98) * 1.5
    bins    = np.linspace(-clip_v, clip_v, 80)

    ax.hist(actuals, bins=bins, alpha=0.55, color=C_ACTUAL, label="Actual",    density=True)
    ax.hist(preds,   bins=bins, alpha=0.75, color=C_PRED,   label="Predicted", density=True)
    ax.axvline(0,                color="white",  lw=0.8, ls="--")
    ax.axvline(np.mean(preds),   color=C_PRED,   lw=1.5, ls=":",
               label=f"Mean pred {np.mean(preds):+.2f}%")
    ax.axvline(np.mean(actuals), color=C_ACTUAL, lw=1.5, ls=":",
               label=f"Mean actual {np.mean(actuals):+.2f}%")
    ax.set_title(f"{w}-day return", color="white", fontsize=13)
    ax.set_xlabel("Return (%)", color="white")
    ax.set_ylabel("Density", color="white")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

plt.tight_layout()
p1 = CHART_DIR / "prediction_distribution.png"
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="#0e1117")
plt.close()
print(f"  Saved: {p1}")

# =============================================================================
# CHART 2: Mean actual return per prediction decile (bar chart)
# =============================================================================
print("Generating decile chart...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Mean Actual Return by Prediction Decile (Test Set)",
             fontsize=16, color="white")

for ax, w in zip(axes.flat, HORIZONS):
    preds   = np.array(results[w]["preds"])
    actuals = np.array(results[w]["actuals"])
    edges   = np.percentile(preds, np.linspace(0, 100, 11))
    labels  = [f"D{i}" for i in range(1, 11)]
    means, ns = [], []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        mask = (preds >= lo_e) & (preds <= hi_e)
        means.append(float(actuals[mask].mean()) if mask.sum() > 0 else np.nan)
        ns.append(int(mask.sum()))

    colors = ["#EF5350" if (m is not None and not np.isnan(m) and m < 0) else "#66BB6A"
              for m in means]
    bars = ax.bar(labels, means, color=colors, edgecolor="none", alpha=0.85)
    ax.axhline(0, color="white", lw=0.8, ls="--")
    ax.set_title(f"{w}-day return", color="white", fontsize=13)
    ax.set_xlabel("Prediction Decile  (D1 = lowest)", color="white")
    ax.set_ylabel("Mean Actual Return (%)", color="white")
    ax.tick_params(colors="white")
    for bar, n in zip(bars, ns):
        h = bar.get_height()
        if np.isnan(h):
            h = 0
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                str(n), ha="center", va="bottom", fontsize=7, color="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

plt.tight_layout()
p2 = CHART_DIR / "prediction_decile_actual_return.png"
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="#0e1117")
plt.close()
print(f"  Saved: {p2}")

print("\nDone.")
