# -*- coding: utf-8 -*-
"""
compare_models.py
=================
Compare four models on the shared 16-feature pipeline:
  1) HGBR (pretrained baseline from models/model_{w}d.pkl)
  2) XGBoost
  3) HMM-XGBoost (regime-routed XGBoost)
  4) ElasticNet (linear low-variance anchor)

Outputs:
  - models/model_comparison.json
  - charts under data/charts/
"""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)


def _pip(pkg):
    print(f"  pip install {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


try:
    import xgboost as xgb
    print(f"xgboost {xgb.__version__}")
except ImportError:
    _pip("xgboost")
    import xgboost as xgb

try:
    from hmmlearn import hmm as hmmlib
    import hmmlearn
    print(f"hmmlearn {hmmlearn.__version__}")
except ImportError:
    _pip("hmmlearn")
    from hmmlearn import hmm as hmmlib

from train_models import FEATURES, HORIZONS, engineer_features, load_and_merge

MODEL_DIR = BASE / "models"
CHART_DIR = BASE / "data" / "charts"
CHART_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_REGIMES = 4
HMM_COLS = ["prior_30d_pct", "log_buy_price"]

XGB_BASE = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    early_stopping_rounds=20,
    enable_categorical=True,
    tree_method="hist",
)

MODELS = ["HGBR", "XGBoost", "HMM-XGBoost", "ElasticNet"]


# Helpers

def to_xgb(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "officer_type_enc" in X.columns:
        X["officer_type_enc"] = X["officer_type_enc"].astype("category")
    if "sector_enc" in X.columns:
        X["sector_enc"] = X["sector_enc"].astype("category")
    return X


def to_linear_numeric(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for c in Xn.columns:
        if pd.api.types.is_categorical_dtype(Xn[c]):
            codes = Xn[c].cat.codes.astype(float)
            codes[codes < 0] = np.nan
            Xn[c] = codes
    return Xn.astype(float)


def eval_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    diracc = float((np.sign(y_pred) == np.sign(y_true)).mean())

    order = np.argsort(y_pred)[::-1]
    n10 = max(1, len(order) // 10)
    top_d = float(y_true[order[:n10]].mean())
    bot_d = float(y_true[order[-n10:]].mean())

    prec = {}
    for thr in [0, 2, 5]:
        mask = y_pred > thr
        n = int(mask.sum())
        prec[thr] = dict(
            n=n,
            pct_pos=float((y_true[mask] > 0).mean()) if n else 0.0,
            mean_actual=float(y_true[mask].mean()) if n else 0.0,
        )
    return dict(
        r2=r2,
        mae=mae,
        rmse=rmse,
        diracc=diracc,
        top_dec=top_d,
        bot_dec=bot_d,
        dec_spread=float(top_d - bot_d),
        prec=prec,
    )


def train_xgb(X_tr: pd.DataFrame, y_tr: np.ndarray, params: dict = XGB_BASE) -> xgb.XGBRegressor:
    if len(X_tr) < 60:
        p2 = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        p2["n_estimators"] = 100
        m = xgb.XGBRegressor(**p2)
        m.fit(to_xgb(X_tr), y_tr, verbose=False)
        return m

    X_t2, X_v, y_t2, y_v = train_test_split(
        to_xgb(X_tr), y_tr, test_size=0.1, random_state=RANDOM_STATE
    )
    m = xgb.XGBRegressor(**params)
    m.fit(X_t2, y_t2, eval_set=[(X_v, y_v)], verbose=False)
    return m


def train_elasticnet(X_tr: pd.DataFrame, y_tr: np.ndarray) -> Pipeline:
    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                    alphas=np.logspace(-4, 1, 40),
                    cv=5,
                    max_iter=20000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(to_linear_numeric(X_tr), y_tr)
    return model


def train_spline_elasticnet(X_tr: pd.DataFrame, y_tr: np.ndarray) -> Pipeline:
    """
    Smooth parametric nonlinear model:
    cubic spline basis on continuous features + one-hot on categorical encodings,
    then ElasticNetCV as additive smoother/selector.
    """
    Xn = X_tr.copy()
    cat_cols = [c for c in ["officer_type_enc", "sector_enc"] if c in Xn.columns]
    cont_cols = [c for c in Xn.columns if c not in cat_cols]

    for c in cont_cols:
        if pd.api.types.is_categorical_dtype(Xn[c]):
            codes = Xn[c].cat.codes.astype(float)
            codes[codes < 0] = np.nan
            Xn[c] = codes

    pre = ColumnTransformer(
        transformers=[
            (
                "cont_spline",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
                    ]
                ),
                cont_cols,
            ),
            (
                "cat_onehot",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", pre),
            ("scale", StandardScaler(with_mean=False)),
            (
                "enet",
                ElasticNetCV(
                    l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
                    alphas=np.logspace(-5, 1, 50),
                    cv=5,
                    max_iter=30000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(Xn, y_tr)
    return model


def fit_hmm(X_seq: np.ndarray):
    valid = ~np.isnan(X_seq).any(axis=1)
    Xv = X_seq[valid]
    mu = Xv.mean(0)
    std = Xv.std(0) + 1e-8
    Xs = (Xv - mu) / std

    model = hmmlib.GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(Xs)
    return model, mu, std


def assign_regimes(model, mu: np.ndarray, std: np.ndarray, X_obs: np.ndarray, fallback: int) -> np.ndarray:
    states = np.full(len(X_obs), fallback, dtype=int)
    valid = ~np.isnan(X_obs).any(axis=1)
    if valid.sum() == 0:
        return states

    Xs = (X_obs[valid] - mu) / std
    log_em = np.stack(
        [
            multivariate_normal.logpdf(
                Xs,
                mean=model.means_[k],
                cov=model.covars_[k],
                allow_singular=True,
            )
            for k in range(N_REGIMES)
        ],
        axis=1,
    )
    states[valid] = log_em.argmax(axis=1)
    return states


def hmm_regime_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    sub: pd.DataFrame,
    idx_train,
    global_xgb: xgb.XGBRegressor,
) -> tuple:
    tr_dates = pd.to_datetime(sub.loc[idx_train, "trade_date"]).values
    sort_order = np.argsort(tr_dates)
    unsort = np.argsort(sort_order)

    X_train_sorted = X_train[HMM_COLS].values[sort_order]
    hmm_model, hmm_mu, hmm_std = fit_hmm(X_train_sorted)

    valid_mask = ~np.isnan(X_train_sorted).any(axis=1)
    valid_preds = assign_regimes(hmm_model, hmm_mu, hmm_std, X_train_sorted[valid_mask], 0)
    modal_state = int(np.bincount(valid_preds, minlength=N_REGIMES).argmax())

    tr_states_sorted = assign_regimes(hmm_model, hmm_mu, hmm_std, X_train_sorted, modal_state)
    tr_states = tr_states_sorted[unsort]
    te_states = assign_regimes(hmm_model, hmm_mu, hmm_std, X_test[HMM_COLS].values, modal_state)

    regime_models = {}
    regime_info = {}
    for s in range(N_REGIMES):
        mask = tr_states == s
        n = int(mask.sum())
        if n < 50:
            regime_models[s] = global_xgb
            regime_info[s] = {"n": n, "iters": None, "fallback": True}
        else:
            rm = train_xgb(X_train.iloc[mask], y_train[mask])
            regime_models[s] = rm
            iters = getattr(rm, "best_iteration", None)
            regime_info[s] = {"n": n, "iters": iters, "fallback": False}

    preds = np.zeros(len(X_test))
    for s in range(N_REGIMES):
        mask = te_states == s
        if mask.sum() == 0:
            continue
        preds[mask] = regime_models[s].predict(to_xgb(X_test.iloc[mask]))

    return preds, regime_info


def json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def run():
    print("Loading and engineering data...")
    df, features, _ = engineer_features(load_and_merge())
    print(f"  rows={len(df):,} | features={len(features)}")

    if len(features) != 16:
        print(f"WARNING: expected 16 features, found {len(features)}")

    all_results = {}

    for w in HORIZONS:
        tgt = f"return_{w}d_pct"
        print(f"\n{'=' * 70}\n{w}-day window ({tgt})\n{'=' * 70}")

        sub = df.dropna(subset=[tgt]).copy()
        lo, hi = sub[tgt].quantile([0.01, 0.99])
        sub[tgt] = sub[tgt].clip(lo, hi)

        X = sub[features].copy()
        y = sub[tgt].values

        X_tr, X_te, y_tr, y_te, idx_tr, _ = train_test_split(
            X,
            y,
            sub.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        wres = {}

        print("  [1/4] HGBR")
        hgbr = joblib.load(MODEL_DIR / f"model_{w}d.pkl")
        p_hgbr = hgbr.predict(X_te)
        m1 = eval_all(y_te, p_hgbr)
        wres["HGBR"] = m1
        print(f"        R2={m1['r2']:+.4f} dir={m1['diracc']*100:.1f}% spread={m1['dec_spread']:+.2f}pp")

        print("  [2/4] XGBoost")
        xgb_model = train_xgb(X_tr, y_tr)
        p_xgb = xgb_model.predict(to_xgb(X_te))
        m2 = eval_all(y_te, p_xgb)
        wres["XGBoost"] = m2
        print(f"        R2={m2['r2']:+.4f} dir={m2['diracc']*100:.1f}% spread={m2['dec_spread']:+.2f}pp")

        print("  [3/4] HMM-XGBoost")
        p_hmm, _ = hmm_regime_predict(X_tr, y_tr, X_te, sub, idx_tr, xgb_model)
        m3 = eval_all(y_te, p_hmm)
        wres["HMM-XGBoost"] = m3
        print(f"        R2={m3['r2']:+.4f} dir={m3['diracc']*100:.1f}% spread={m3['dec_spread']:+.2f}pp")

        print("  [4/4] ElasticNet")
        en_model = train_elasticnet(X_tr, y_tr)
        p_en = en_model.predict(to_linear_numeric(X_te))
        m4 = eval_all(y_te, p_en)
        wres["ElasticNet"] = m4
        en_core = en_model.named_steps["enet"]
        print(
            f"        R2={m4['r2']:+.4f} dir={m4['diracc']*100:.1f}% spread={m4['dec_spread']:+.2f}pp "
            f"alpha={en_core.alpha_:.5f} l1={en_core.l1_ratio_:.2f}"
        )

        all_results[w] = wres

        joblib.dump(xgb_model, MODEL_DIR / f"xgb_model_{w}d.pkl")
        joblib.dump(en_model, MODEL_DIR / f"elasticnet_model_{w}d.pkl")

    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON")
    print("=" * 90)
    for w in HORIZONS:
        print(f"\n{w}-day window")
        hdr = (
            f"{'Model':<15} {'R2':>7} {'MAE':>7} {'DirAcc':>8} {'DecSprd':>9} "
            f"{'p>0 n':>7} {'p>0 +':>7} {'p>2 n':>7} {'p>2 +':>7} {'p>2 mu':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for mn in MODELS:
            m = all_results[w][mn]
            p0 = m["prec"][0]
            p2 = m["prec"][2]
            print(
                f"{mn:<15} {m['r2']:>+7.4f} {m['mae']:>7.2f} {m['diracc']*100:>7.1f}% {m['dec_spread']:>+9.2f} "
                f"{p0['n']:>7} {p0['pct_pos']*100:>6.1f}% {p2['n']:>7} {p2['pct_pos']*100:>6.1f}% {p2['mean_actual']:>+7.2f}%"
            )

    with open(MODEL_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump({str(w): v for w, v in all_results.items()}, f, indent=2, default=json_safe)
    print("\nSaved models/model_comparison.json")

    print("Generating charts...")
    plt.style.use("dark_background")
    C = {
        "HGBR": "#4FC3F7",
        "XGBoost": "#FF8A65",
        "HMM-XGBoost": "#A5D6A7",
        "ElasticNet": "#BA68C8",
    }
    x = np.arange(len(MODELS))

    fig, axes = plt.subplots(2, 4, figsize=(24, 9))
    fig.suptitle("Model Comparison - Directional Accuracy and Decile Spread", fontsize=14, color="white")
    for col, w in enumerate(HORIZONS):
        ax0 = axes[0, col]
        vals = [all_results[w][mn]["diracc"] * 100 for mn in MODELS]
        bars = ax0.bar(x, vals, width=0.65, color=[C[m] for m in MODELS], alpha=0.85, edgecolor="none")
        ax0.axhline(50, color="white", lw=0.8, ls="--", alpha=0.5)
        ax0.set_title(f"{w}d - Dir Accuracy (%)", fontsize=10)
        ax0.set_xticks(x)
        ax0.set_xticklabels(MODELS, fontsize=7, rotation=12)
        ax0.set_ylim(48, 58)
        for bar, v in zip(bars, vals):
            ax0.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

        ax1 = axes[1, col]
        vals2 = [all_results[w][mn]["dec_spread"] for mn in MODELS]
        bars2 = ax1.bar(x, vals2, width=0.65, color=[C[m] for m in MODELS], alpha=0.85, edgecolor="none")
        ax1.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
        ax1.set_title(f"{w}d - Decile Spread (pp)", fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(MODELS, fontsize=7, rotation=12)
        for bar, v in zip(bars2, vals2):
            ypos = v + 0.05 if v >= 0 else v - 0.15
            ax1.text(bar.get_x() + bar.get_width() / 2, ypos, f"{v:+.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p1 = CHART_DIR / "model_comparison_accuracy.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(24, 9))
    fig.suptitle("Model Comparison - Positive Precision (pred>2% and pred>5%)", fontsize=14, color="white")
    for col, w in enumerate(HORIZONS):
        for row, thr in enumerate([2, 5]):
            ax = axes[row, col]
            pcts = [all_results[w][mn]["prec"][thr]["pct_pos"] * 100 for mn in MODELS]
            ns = [all_results[w][mn]["prec"][thr]["n"] for mn in MODELS]
            mus = [all_results[w][mn]["prec"][thr]["mean_actual"] for mn in MODELS]
            bars = ax.bar(x, pcts, width=0.65, color=[C[m] for m in MODELS], alpha=0.85, edgecolor="none")
            ax.axhline(50, color="white", lw=0.8, ls="--", alpha=0.5)
            ax.set_title(f"{w}d - Prec when pred>{thr}%", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=7, rotation=12)
            ax.set_ylim(0, 90)
            for bar, pct, n, mu in zip(bars, pcts, ns, mus):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    pct + 1,
                    f"{pct:.0f}%\n(n={n})\nmu={mu:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.tight_layout()
    p2 = CHART_DIR / "model_comparison_precision.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Comparison - R2 and MAE across windows", fontsize=14, color="white")

    window_labels = [f"{w}d" for w in HORIZONS]
    xw = np.arange(len(HORIZONS))
    bw = 0.18

    for idx, mn in enumerate(MODELS):
        r2s = [all_results[w][mn]["r2"] for w in HORIZONS]
        maes = [all_results[w][mn]["mae"] for w in HORIZONS]
        axes[0].bar(xw + idx * bw, r2s, width=bw, label=mn, color=C[mn], alpha=0.85, edgecolor="none")
        axes[1].bar(xw + idx * bw, maes, width=bw, label=mn, color=C[mn], alpha=0.85, edgecolor="none")

    for ax, title, ylabel in zip(axes, ["R2 Score", "MAE (pp)"], ["R2", "Mean Absolute Error (%)"]):
        ax.axhline(0, color="white", lw=0.6, ls="--", alpha=0.4)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(xw + 1.5 * bw)
        ax.set_xticklabels(window_labels)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, framealpha=0.3)

    plt.tight_layout()
    p3 = CHART_DIR / "model_comparison_r2_mae.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")


if __name__ == "__main__":
    run()
