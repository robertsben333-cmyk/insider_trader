# -*- coding: utf-8 -*-
"""
Production model + ensemble pipeline (4 models).

Models:
  - HGBR
  - XGBoost
  - ElasticNet
  - SplineElasticNet

Weight policy by horizon:
  - 1d, 3d: equal weights
  - 5d, 10d: optimize on validation (chronological split)

Outputs:
  - models/prod4/model_{h}d_{model}.pkl
  - models/prod4/ensemble_policy.json
  - models/prod4/eval_time_split.json
"""

import json
import os
import sys
import argparse
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit("xgboost is required. Install it with: py -m pip install xgboost") from exc

import train_models
from train_models import FEATURES, HORIZONS, MODEL_PARAMS, engineer_features, load_and_merge

MODEL_DIR = BASE / "models" / "prod4"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
WEIGHT_GRID_STEP = 0.05

MODEL_NAMES = ["HGBR", "XGBoost", "ElasticNet", "SplineElasticNet"]
OPT_HORIZONS = {5, 10}

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.1,
    random_state=42,
    early_stopping_rounds=20,
    enable_categorical=True,
    tree_method="hist",
)


def to_xgb(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    if "officer_type_enc" in Xc.columns:
        Xc["officer_type_enc"] = Xc["officer_type_enc"].astype("category")
    if "market_type_enc" in Xc.columns:
        Xc["market_type_enc"] = Xc["market_type_enc"].astype("category")
    if "sector_enc" in Xc.columns:
        Xc["sector_enc"] = Xc["sector_enc"].astype("category")
    return Xc


def to_linear_numeric(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for c in Xn.columns:
        if pd.api.types.is_categorical_dtype(Xn[c]):
            codes = Xn[c].cat.codes.astype(float)
            codes[codes < 0] = np.nan
            Xn[c] = codes
    return Xn.astype(float)


def train_xgb(X_tr: pd.DataFrame, y_tr: np.ndarray) -> xgb.XGBRegressor:
    if len(X_tr) < 60:
        p = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
        p["n_estimators"] = 100
        m = xgb.XGBRegressor(**p)
        m.fit(to_xgb(X_tr), y_tr, verbose=False)
        return m

    cut = int(round(len(X_tr) * 0.9))
    cut = max(1, min(cut, len(X_tr) - 1))
    X_t2, X_v = to_xgb(X_tr.iloc[:cut]), to_xgb(X_tr.iloc[cut:])
    y_t2, y_v = y_tr[:cut], y_tr[cut:]
    m = xgb.XGBRegressor(**XGB_PARAMS)
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
    Xn = X_tr.copy()
    cat_cols = [c for c in ["officer_type_enc", "market_type_enc", "sector_enc"] if c in Xn.columns]
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
                "mean_actual": float(vals.mean()) if len(vals) else np.nan,
                "pct_pos": float((vals > 0).mean() * 100) if len(vals) else np.nan,
            }
        )
    top = rows[-1]["mean_actual"]
    bottom = rows[0]["mean_actual"]
    return {
        "top_decile_mean": float(top),
        "bottom_decile_mean": float(bottom),
        "decile_spread": float(top - bottom),
        "rows": rows,
    }


def simplex_weights(n_models: int, step: float = 0.05):
    ticks = int(round(1.0 / step))
    for parts in product(range(ticks + 1), repeat=n_models):
        if sum(parts) == ticks:
            yield np.array(parts, dtype=float) * step


def optimize_weights(preds: np.ndarray, y: np.ndarray, step: float = 0.05):
    best_w = None
    best_top = -np.inf
    best_spread = -np.inf
    for w in simplex_weights(preds.shape[1], step=step):
        ds = decile_stats(preds @ w, y)
        top, spread = ds["top_decile_mean"], ds["decile_spread"]
        if (top > best_top) or (np.isclose(top, best_top) and spread > best_spread):
            best_top = top
            best_spread = spread
            best_w = w
    return best_w, float(best_top), float(best_spread)


def chrono_train_val_test_split(X, y, idx):
    n = len(X)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
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


def fit_models(X_train: pd.DataFrame, y_train: np.ndarray):
    models = {}
    hgbr = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr.fit(X_train, y_train)
    models["HGBR"] = hgbr

    xgb_m = train_xgb(X_train, y_train)
    models["XGBoost"] = xgb_m

    en_m = train_elasticnet(X_train, y_train)
    models["ElasticNet"] = en_m

    sp_m = train_spline_elasticnet(X_train, y_train)
    models["SplineElasticNet"] = sp_m
    return models


def predict_model(model_name: str, model, X: pd.DataFrame) -> np.ndarray:
    if model_name == "HGBR":
        return model.predict(X)
    if model_name == "XGBoost":
        return model.predict(to_xgb(X))
    if model_name == "ElasticNet":
        return model.predict(to_linear_numeric(X))
    if model_name == "SplineElasticNet":
        return model.predict(X.copy())
    raise ValueError(f"Unknown model: {model_name}")


def parse_horizons(h_txt: str) -> list[int]:
    vals = []
    for part in str(h_txt).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("At least one horizon is required.")
    unknown = [h for h in vals if h not in HORIZONS]
    if unknown:
        raise ValueError(f"Unsupported horizons: {unknown}; valid={HORIZONS}")
    return vals


def run(model_dir: Path = MODEL_DIR, horizons: list[int] | None = None, aggregated_file: str | None = None):
    if horizons is None:
        horizons = list(HORIZONS)
    if aggregated_file:
        train_models.AGGREGATED_CSV = aggregated_file

    model_dir.mkdir(parents=True, exist_ok=True)
    print("Loading and engineering data...")
    df, features, _ = engineer_features(load_and_merge())
    if "is_supported_type" in df.columns:
        before = len(df)
        df = df[df["is_supported_type"] == 1].copy()
        print(f"  type filter: kept {len(df):,} / {before:,} rows (Officer/Director/10%)")
    if features != FEATURES:
        print(f"WARNING: expected canonical FEATURES list; got {len(features)} features.")
    print(f"  rows={len(df):,} | features={len(features)}")

    eval_report = {}

    if set(horizons) == {1, 3}:
        policy_rule = "equal(1d,3d)"
    else:
        policy_rule = "equal(1d,3d), validation-optimized(5d,10d)"

    policy = {
        "weights_by_horizon": {},
        "rule": policy_rule,
        "multi_day_targets_are_daily": True,
        "target_return_mode": train_models.TARGET_RETURN_MODE,
        "benchmark_ticker": train_models.BENCHMARK_TICKER,
    }

    for h in horizons:
        print(f"\n{'=' * 78}\nHorizon {h}d\n{'=' * 78}")
        tgt = f"return_{h}d_pct"
        sub = df.dropna(subset=[tgt]).copy()
        lo, hi = sub[tgt].quantile([0.01, 0.99])
        sub[tgt] = sub[tgt].clip(lo, hi)
        sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
        sub = sub.dropna(subset=["trade_date"]).sort_values("trade_date").copy()

        X = sub[features].copy()
        y = sub[tgt].values
        idx = sub.index.values

        X_tr, X_va, X_te, y_tr, y_va, y_te, idx_tr, idx_va, idx_te = chrono_train_val_test_split(X, y, idx)
        print(f"  split sizes: train={len(X_tr):,}, val={len(X_va):,}, test={len(X_te):,}")

        models = fit_models(X_tr, y_tr)
        pred_va = {m: predict_model(m, mdl, X_va) for m, mdl in models.items()}
        pred_te = {m: predict_model(m, mdl, X_te) for m, mdl in models.items()}

        mat_va = np.column_stack([pred_va[m] for m in MODEL_NAMES])
        mat_te = np.column_stack([pred_te[m] for m in MODEL_NAMES])
        w_eq = np.full(len(MODEL_NAMES), 1.0 / len(MODEL_NAMES), dtype=float)

        if h in OPT_HORIZONS:
            w_use, opt_top, opt_spread = optimize_weights(mat_va, y_va, WEIGHT_GRID_STEP)
            mode = "optimized_on_validation"
        else:
            w_use = w_eq.copy()
            ds_tmp = decile_stats(mat_va @ w_use, y_va)
            opt_top, opt_spread = ds_tmp["top_decile_mean"], ds_tmp["decile_spread"]
            mode = "equal"

        ds_val_eq = decile_stats(mat_va @ w_eq, y_va)
        ds_val_use = decile_stats(mat_va @ w_use, y_va)
        ds_te_eq = decile_stats(mat_te @ w_eq, y_te)
        ds_te_use = decile_stats(mat_te @ w_use, y_te)

        print(
            f"  mode={mode} | test top-decile equal={ds_te_eq['top_decile_mean']:+.3f}% "
            f"use={ds_te_use['top_decile_mean']:+.3f}%"
        )
        print(f"  weights={dict(zip(MODEL_NAMES, [float(x) for x in w_use]))}")

        policy["weights_by_horizon"][str(h)] = {
            "mode": mode,
            "weights": {m: float(v) for m, v in zip(MODEL_NAMES, w_use)},
        }

        eval_report[str(h)] = {
            "target": tgt,
            "target_return_mode": train_models.TARGET_RETURN_MODE,
            "benchmark_ticker": train_models.BENCHMARK_TICKER,
            "split_sizes": {"train": int(len(X_tr)), "val": int(len(X_va)), "test": int(len(X_te))},
            "weights": {
                "equal": {m: float(v) for m, v in zip(MODEL_NAMES, w_eq)},
                "used": {m: float(v) for m, v in zip(MODEL_NAMES, w_use)},
            },
            "validation": {
                "equal": ds_val_eq,
                "used": ds_val_use,
                "used_minus_equal_top_decile_pp": float(ds_val_use["top_decile_mean"] - ds_val_eq["top_decile_mean"]),
                "opt_objective_top_decile": float(opt_top),
                "opt_objective_spread": float(opt_spread),
            },
            "test": {
                "equal": ds_te_eq,
                "used": ds_te_use,
                "used_minus_equal_top_decile_pp": float(ds_te_use["top_decile_mean"] - ds_te_eq["top_decile_mean"]),
            },
        }

        # Train final models on train+val for production scoring.
        X_fit = pd.concat([X_tr, X_va], axis=0)
        y_fit = np.concatenate([y_tr, y_va], axis=0)
        final_models = fit_models(X_fit, y_fit)
        for m, mdl in final_models.items():
            path = model_dir / f"model_{h}d_{m}.pkl"
            joblib.dump(mdl, path)

    with open(model_dir / "ensemble_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    with open(model_dir / "eval_time_split.json", "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nSaved: {model_dir / 'ensemble_policy.json'}")
    print(f"Saved: {model_dir / 'eval_time_split.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train prod4 ensemble models.")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--horizons", default="1,3,5,10", help="Comma-separated horizons (e.g. 1,3).")
    ap.add_argument("--aggregated-file", default="", help="Optional override for train_models.AGGREGATED_CSV.")
    args = ap.parse_args()

    run(
        model_dir=Path(args.model_dir),
        horizons=parse_horizons(args.horizons),
        aggregated_file=args.aggregated_file or None,
    )
