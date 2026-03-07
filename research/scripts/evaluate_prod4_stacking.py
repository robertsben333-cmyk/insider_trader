# -*- coding: utf-8 -*-
"""
evaluate_prod4_stacking.py
==========================
Research evaluation for a second-stage model on top of the current prod4 stack.

Workflow per horizon:
  1) Chronological split: first 80% development, last 20% test
  2) Build time-series OOF predictions for the 4 prod4 base models on development
  3) Train meta-models on OOF rows only
  4) Refit base models on full development and score holdout test
  5) Compare:
       - EqualWeight4
       - OOFWeighted4
       - StackElasticNetPredOnly
       - StackHGBRPredOnly
       - StackElasticNetPredPlusAllRaw
       - StackHGBRPredPlusAllRaw

Outputs:
  - research/outcomes/models/prod4_stacking_eval.json
  - research/outcomes/models/prod4_stacking_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)

import train_models
from model_ensemble import (
    MODEL_NAMES,
    MODEL_PARAMS,
    fit_models,
    predict_model,
    to_linear_numeric,
    train_elasticnet,
)
from train_models import FEATURES, HORIZONS, engineer_features, load_and_merge

OUT_DIR = BASE / "research" / "outcomes" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
OOF_SPLITS = 5


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
                "pct_pos": float((vals > 0).mean() * 100.0) if len(vals) else np.nan,
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


def regression_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return {
        "r2": float(r2_score(actual, pred)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(root_mean_squared_error(actual, pred)),
        "dir_acc": float((np.sign(pred) == np.sign(actual)).mean()),
    }


def summarize_signal(name: str, pred: np.ndarray, actual: np.ndarray) -> dict:
    metrics = regression_metrics(pred, actual)
    deciles = decile_stats(pred, actual)
    return {
        "name": name,
        "metrics": metrics,
        "deciles": deciles,
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
        top = ds["top_decile_mean"]
        spread = ds["decile_spread"]
        if (top > best_top) or (np.isclose(top, best_top) and spread > best_spread):
            best_w = w
            best_top = top
            best_spread = spread
    return best_w, float(best_top), float(best_spread)


def chronological_dev_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    idx: np.ndarray,
    test_size: float,
):
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


def build_oof_predictions(X_dev: pd.DataFrame, y_dev: np.ndarray, n_splits: int) -> dict[str, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = {m: np.full(len(X_dev), np.nan, dtype=float) for m in MODEL_NAMES}

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_dev), start=1):
        print(f"    OOF fold {fold}/{n_splits}")
        X_tr = X_dev.iloc[tr_idx]
        y_tr = y_dev[tr_idx]
        X_va = X_dev.iloc[val_idx]

        models = fit_models(X_tr, y_tr)
        for model_name in MODEL_NAMES:
            oof[model_name][val_idx] = predict_model(model_name, models[model_name], X_va)

    return oof


def make_pred_frame(pred_map: dict[str, np.ndarray], valid_mask: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({f"pred_{name}": pred_map[name] for name in MODEL_NAMES})
    if valid_mask is not None:
        frame = frame.loc[valid_mask].reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    return frame


def concat_meta_features(
    pred_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    valid_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    if valid_mask is not None:
        raw = raw_frame.loc[valid_mask].reset_index(drop=True)
    else:
        raw = raw_frame.reset_index(drop=True)
    return pd.concat([pred_frame.reset_index(drop=True), raw], axis=1)


def train_meta_models(
    pred_train: pd.DataFrame,
    pred_plus_raw_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, object]:
    models: dict[str, object] = {}

    models["StackElasticNetPredOnly"] = train_elasticnet(pred_train, y_train)

    hgbr_pred_only = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr_pred_only.fit(pred_train, y_train)
    models["StackHGBRPredOnly"] = hgbr_pred_only

    models["StackElasticNetPredPlusAllRaw"] = train_elasticnet(pred_plus_raw_train, y_train)

    hgbr_pred_raw = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr_pred_raw.fit(pred_plus_raw_train, y_train)
    models["StackHGBRPredPlusAllRaw"] = hgbr_pred_raw

    return models


def predict_meta_models(
    meta_models: dict[str, object],
    pred_test: pd.DataFrame,
    pred_plus_raw_test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, model in meta_models.items():
        if name.endswith("PredOnly"):
            out[name] = model.predict(to_linear_numeric(pred_test) if "ElasticNet" in name else pred_test)
        else:
            feats = pred_plus_raw_test
            out[name] = model.predict(to_linear_numeric(feats) if "ElasticNet" in name else feats)
    return out


def run(aggregated_file: str | None = None, horizons: list[int] | None = None) -> tuple[Path, Path]:
    if horizons is None:
        horizons = list(HORIZONS)
    if aggregated_file:
        train_models.AGGREGATED_CSV = aggregated_file

    print("Loading and engineering data...")
    df, features, _ = engineer_features(load_and_merge())
    if "is_supported_type" in df.columns:
        before = len(df)
        df = df[df["is_supported_type"] == 1].copy()
        print(f"  type filter: kept {len(df):,} / {before:,} rows")
    if features != FEATURES:
        print(f"WARNING: expected canonical FEATURES list; got {len(features)} features.")
    print(f"  rows={len(df):,} | features={len(features)}")

    report: dict[str, object] = {
        "meta": {
            "split": "chrono_dev_test",
            "test_size": TEST_SIZE,
            "oof_splits": OOF_SPLITS,
            "base_models": MODEL_NAMES,
            "feature_set": FEATURES,
            "stacker_candidates": [
                "EqualWeight4",
                "OOFWeighted4",
                "StackElasticNetPredOnly",
                "StackHGBRPredOnly",
                "StackElasticNetPredPlusAllRaw",
                "StackHGBRPredPlusAllRaw",
            ],
        },
        "horizons": {},
    }
    summary_rows: list[dict[str, object]] = []

    for horizon in horizons:
        print(f"\n{'=' * 80}\nHorizon {horizon}d\n{'=' * 80}")
        target_col = f"return_{horizon}d_pct"
        sub = df.dropna(subset=[target_col]).copy()
        lo, hi = sub[target_col].quantile([0.01, 0.99])
        sub[target_col] = sub[target_col].clip(lo, hi)
        sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
        sub = sub.dropna(subset=["trade_date"]).sort_values("trade_date").copy()

        X = sub[features].copy()
        y = sub[target_col].to_numpy(dtype=float)
        idx = sub.index.to_numpy()

        X_dev, X_test, y_dev, y_test, idx_dev, idx_test = chronological_dev_test_split(X, y, idx, TEST_SIZE)

        dev_start = pd.to_datetime(sub.loc[idx_dev, "trade_date"]).min().date()
        dev_end = pd.to_datetime(sub.loc[idx_dev, "trade_date"]).max().date()
        test_start = pd.to_datetime(sub.loc[idx_test, "trade_date"]).min().date()
        test_end = pd.to_datetime(sub.loc[idx_test, "trade_date"]).max().date()
        print(
            f"  Split: dev {len(X_dev):,} [{dev_start} -> {dev_end}] | "
            f"test {len(X_test):,} [{test_start} -> {test_end}]"
        )

        print("  Building OOF base predictions...")
        oof_preds = build_oof_predictions(X_dev, y_dev, OOF_SPLITS)
        pred_oof_mat = np.column_stack([oof_preds[name] for name in MODEL_NAMES])
        valid_oof = np.isfinite(pred_oof_mat).all(axis=1)
        valid_rows = int(valid_oof.sum())
        if valid_rows < 100:
            raise RuntimeError(f"Insufficient valid OOF rows for horizon {horizon}d: {valid_rows}")
        print(f"  Valid OOF rows for meta-training: {valid_rows:,}")

        pred_oof_frame = make_pred_frame(oof_preds, valid_mask=valid_oof)
        pred_raw_oof_frame = concat_meta_features(pred_oof_frame, X_dev, valid_mask=valid_oof)
        y_oof = y_dev[valid_oof]

        print("  Training final base models on full development window...")
        final_base_models = fit_models(X_dev, y_dev)
        test_base_preds = {
            model_name: predict_model(model_name, model, X_test)
            for model_name, model in final_base_models.items()
        }
        pred_test_frame = make_pred_frame(test_base_preds)
        pred_raw_test_frame = concat_meta_features(pred_test_frame, X_test)

        equal_w = np.full(len(MODEL_NAMES), 1.0 / len(MODEL_NAMES), dtype=float)
        opt_w, opt_top, opt_spread = optimize_weights(pred_oof_frame.to_numpy(dtype=float), y_oof)

        test_signals: dict[str, np.ndarray] = {
            "EqualWeight4": pred_test_frame.to_numpy(dtype=float) @ equal_w,
            "OOFWeighted4": pred_test_frame.to_numpy(dtype=float) @ opt_w,
        }

        print("  Training meta-models...")
        meta_models = train_meta_models(pred_oof_frame, pred_raw_oof_frame, y_oof)
        test_signals.update(predict_meta_models(meta_models, pred_test_frame, pred_raw_test_frame))

        horizon_result = {
            "target": target_col,
            "rows": {
                "full": int(len(sub)),
                "development": int(len(X_dev)),
                "test": int(len(X_test)),
                "oof_meta_train": valid_rows,
            },
            "date_ranges": {
                "development_start": str(dev_start),
                "development_end": str(dev_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
            },
            "oof_weight_optimization": {
                "weights": {name: float(v) for name, v in zip(MODEL_NAMES, opt_w)},
                "objective_top_decile_mean": opt_top,
                "objective_decile_spread": opt_spread,
            },
            "test": {},
        }

        for signal_name, signal in test_signals.items():
            result = summarize_signal(signal_name, signal, y_test)
            horizon_result["test"][signal_name] = result
            summary_rows.append(
                {
                    "horizon": horizon,
                    "model": signal_name,
                    "r2": result["metrics"]["r2"],
                    "mae": result["metrics"]["mae"],
                    "rmse": result["metrics"]["rmse"],
                    "dir_acc": result["metrics"]["dir_acc"],
                    "top_decile_mean": result["deciles"]["top_decile_mean"],
                    "bottom_decile_mean": result["deciles"]["bottom_decile_mean"],
                    "decile_spread": result["deciles"]["decile_spread"],
                }
            )

        top_model = max(
            horizon_result["test"].values(),
            key=lambda item: (
                item["deciles"]["top_decile_mean"],
                item["deciles"]["decile_spread"],
            ),
        )
        print(
            f"  Best test stack: {top_model['name']} | "
            f"top_decile_mean={top_model['deciles']['top_decile_mean']:+.3f}% | "
            f"spread={top_model['deciles']['decile_spread']:+.3f}pp"
        )

        report["horizons"][str(horizon)] = horizon_result

    json_path = OUT_DIR / "prod4_stacking_eval.json"
    csv_path = OUT_DIR / "prod4_stacking_summary.csv"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    pd.DataFrame(summary_rows).sort_values(
        ["horizon", "top_decile_mean", "decile_spread"],
        ascending=[True, False, False],
    ).to_csv(csv_path, index=False)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return json_path, csv_path


def parse_horizons(text: str) -> list[int]:
    vals = []
    for part in str(text).split(","):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate second-stage stackers for the prod4 ensemble.")
    parser.add_argument("--aggregated-file", default="", help="Optional override for train_models.AGGREGATED_CSV.")
    parser.add_argument("--horizons", default="1,3,5,10", help="Comma-separated horizons (e.g. 1,3).")
    args = parser.parse_args()

    run(
        aggregated_file=args.aggregated_file or None,
        horizons=parse_horizons(args.horizons),
    )
