"""
Analyze whether missing ownership-percentage data hurts model performance.

This script reuses the production feature engineering path from train_models.py
and compares:
1. Baseline: current feature set including owned_pct_num with NaNs left in place.
2. No ownership feature: remove owned_pct_num entirely.
3. Complete case: keep owned_pct_num but train only on rows where it is present.

Outputs:
- research/outcomes/models/ownership_missingness_summary.csv
- research/outcomes/models/ownership_missingness_details.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_models as tm


OUT_DIR = Path("research/outcomes/models")
SUMMARY_CSV = OUT_DIR / "ownership_missingness_summary.csv"
DETAILS_JSON = OUT_DIR / "ownership_missingness_details.json"


def _evaluate_window(df: pd.DataFrame, features: list[str], horizon: int) -> dict:
    target_col = f"return_{horizon}d_pct"
    selected_cols = list(dict.fromkeys(features + [target_col, "owned_pct_num"]))
    sub = df[selected_cols].dropna(subset=[target_col]).copy()

    lo = sub[target_col].quantile(0.01)
    hi = sub[target_col].quantile(0.99)
    sub[target_col] = sub[target_col].clip(lo, hi)

    X = sub[features]
    y = sub[target_col]
    own_missing = sub["owned_pct_num"].isna()

    X_train, X_test, y_train, y_test, own_missing_train, own_missing_test = train_test_split(
        X,
        y,
        own_missing,
        test_size=tm.TEST_SIZE,
        random_state=tm.RANDOM_STATE,
        shuffle=True,
    )

    model = HistGradientBoostingRegressor(**tm.MODEL_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    order = np.argsort(y_pred)[::-1]
    n10 = max(1, len(order) // 10)
    top_dec_actual = y_test.values[order[:n10]].mean()
    bot_dec_actual = y_test.values[order[-n10:]].mean()

    pred_pos_mask = y_pred > 0
    pred_gt2_mask = y_pred > 2.0

    own_missing_test_arr = np.asarray(own_missing_test, dtype=bool).reshape(-1)

    result = {
        "horizon": horizon,
        "n_rows": int(len(sub)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target_clip_lo": float(lo),
        "target_clip_hi": float(hi),
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "dir_acc": float((np.sign(y_pred) == np.sign(y_test.values)).mean()),
        "dec_spread": float(top_dec_actual - bot_dec_actual),
        "top_dec_mean": float(top_dec_actual),
        "bot_dec_mean": float(bot_dec_actual),
        "prec_pos": float((y_test.values[pred_pos_mask] > 0).mean()) if pred_pos_mask.any() else 0.0,
        "n_pred_pos": int(pred_pos_mask.sum()),
        "prec_gt2_pos": float((y_test.values[pred_gt2_mask] > 0).mean()) if pred_gt2_mask.any() else 0.0,
        "n_pred_gt2": int(pred_gt2_mask.sum()),
        "test_owned_missing_rate": float(own_missing_test_arr.mean()),
    }

    subgroup = {}
    for label, mask in {
        "missing": own_missing_test_arr,
        "present": ~own_missing_test_arr,
    }.items():
        n = int(mask.sum())
        if n == 0:
            subgroup[label] = {"n": 0}
            continue
        actual = y_test.values[mask]
        pred = y_pred[mask]
        pred_pos = pred > 0
        order_local = np.argsort(pred)[::-1]
        n10_local = max(1, len(order_local) // 10)
        top_local = actual[order_local[:n10_local]].mean()
        bot_local = actual[order_local[-n10_local:]].mean()
        subgroup[label] = {
            "n": n,
            "r2": float(r2_score(actual, pred)) if n > 1 else None,
            "mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(root_mean_squared_error(actual, pred)),
            "dir_acc": float((np.sign(pred) == np.sign(actual)).mean()),
            "dec_spread": float(top_local - bot_local),
            "mean_actual": float(actual.mean()),
            "mean_pred": float(pred.mean()),
            "prec_pos": float((actual[pred_pos] > 0).mean()) if pred_pos.any() else 0.0,
            "n_pred_pos": int(pred_pos.sum()),
        }

    result["test_subgroups"] = subgroup

    return result


def _scenario_frame(df: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, list[str]]:
    features = list(tm.FEATURES)
    if scenario == "baseline":
        return df.copy(), features
    if scenario == "no_ownership_feature":
        return df.copy(), [f for f in features if f != "owned_pct_num"]
    if scenario == "complete_case":
        return df[df["owned_pct_num"].notna()].copy(), features
    raise ValueError(f"Unknown scenario: {scenario}")


def _missingness_profile(df: pd.DataFrame) -> dict:
    target_cols = [f"return_{h}d_pct" for h in tm.HORIZONS]
    per_horizon = {}
    for h, col in zip(tm.HORIZONS, target_cols):
        sub = df[df[col].notna()].copy()
        missing_mask = sub["owned_pct_num"].isna()
        present_mask = ~missing_mask
        per_horizon[f"{h}d"] = {
            "n_rows": int(len(sub)),
            "owned_missing_n": int(missing_mask.sum()),
            "owned_missing_rate": float(missing_mask.mean()),
            "mean_target_missing": float(sub.loc[missing_mask, col].mean()) if missing_mask.any() else None,
            "mean_target_present": float(sub.loc[present_mask, col].mean()) if present_mask.any() else None,
            "win_rate_missing": float((sub.loc[missing_mask, col] > 0).mean()) if missing_mask.any() else None,
            "win_rate_present": float((sub.loc[present_mask, col] > 0).mean()) if present_mask.any() else None,
        }
    return {
        "n_rows_total": int(len(df)),
        "owned_missing_n_total": int(df["owned_pct_num"].isna().sum()),
        "owned_missing_rate_total": float(df["owned_pct_num"].isna().mean()),
        "per_horizon": per_horizon,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ownership_missingness_models_") as tmp_model_dir:
        tm.MODEL_DIR = tmp_model_dir
        df = tm.load_and_merge()
        df, _, _ = tm.engineer_features(df)

        profile = _missingness_profile(df)
        scenarios = ["baseline", "no_ownership_feature", "complete_case"]

        details = {
            "analysis_date": pd.Timestamp.utcnow().isoformat(),
            "missingness_profile": profile,
            "scenarios": {},
        }
        rows = []

        for scenario in scenarios:
            scenario_df, scenario_features = _scenario_frame(df, scenario)
            scenario_results = {}
            for horizon in tm.HORIZONS:
                metrics = _evaluate_window(scenario_df, scenario_features, horizon)
                scenario_results[f"{horizon}d"] = metrics
                rows.append(
                    {
                        "scenario": scenario,
                        "horizon": f"{horizon}d",
                        "n_rows": metrics["n_rows"],
                        "n_train": metrics["n_train"],
                        "n_test": metrics["n_test"],
                        "r2": metrics["r2"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "dir_acc": metrics["dir_acc"],
                        "dec_spread": metrics["dec_spread"],
                        "prec_pos": metrics["prec_pos"],
                        "prec_gt2_pos": metrics["prec_gt2_pos"],
                        "test_owned_missing_rate": metrics["test_owned_missing_rate"],
                        "missing_test_rmse": metrics["test_subgroups"].get("missing", {}).get("rmse"),
                        "present_test_rmse": metrics["test_subgroups"].get("present", {}).get("rmse"),
                        "missing_test_dir_acc": metrics["test_subgroups"].get("missing", {}).get("dir_acc"),
                        "present_test_dir_acc": metrics["test_subgroups"].get("present", {}).get("dir_acc"),
                    }
                )
            details["scenarios"][scenario] = scenario_results

    summary = pd.DataFrame(rows).sort_values(["horizon", "scenario"]).reset_index(drop=True)

    baseline = (
        summary[summary["scenario"] == "baseline"][
            ["horizon", "r2", "mae", "rmse", "dir_acc", "dec_spread", "prec_pos", "prec_gt2_pos"]
        ]
        .rename(columns=lambda c: c if c == "horizon" else f"baseline_{c}")
    )
    summary = summary.merge(baseline, on="horizon", how="left")
    for metric in ["r2", "mae", "rmse", "dir_acc", "dec_spread", "prec_pos", "prec_gt2_pos"]:
        summary[f"delta_vs_baseline_{metric}"] = summary[metric] - summary[f"baseline_{metric}"]

    summary.to_csv(SUMMARY_CSV, index=False)
    with open(DETAILS_JSON, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print(f"Saved summary to {SUMMARY_CSV}")
    print(f"Saved details to {DETAILS_JSON}")
    print()
    print("Ownership missingness profile:")
    print(json.dumps(profile, indent=2))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
