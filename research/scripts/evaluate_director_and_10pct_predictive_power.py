"""
Evaluate predictive power of the prod4 model stack on specific insider groups.

Groups:
  - director-like titles
  - 10%+ owner titles

For each group and horizon, this script:
  1) re-estimates the same 4 base models (HGBR, XGBoost, ElasticNet, SplineElasticNet)
  2) uses the same chronological train/val/test split as model_ensemble.py
  3) recalculates decile metrics on the group-specific test set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))

from model_ensemble import (
    MODEL_NAMES,
    OPT_HORIZONS,
    WEIGHT_GRID_STEP,
    chrono_train_val_test_split,
    decile_stats,
    fit_models,
    optimize_weights,
    predict_model,
)
from train_models import HORIZONS, engineer_features, load_and_merge

DIRECTOR_RE = re.compile(r"\bdirector\b|\bdir\b", re.IGNORECASE)
TEN_PCT_RE = re.compile(r"10%|10 percent|10pct|ten percent", re.IGNORECASE)


def _subset(df: pd.DataFrame, group: str) -> pd.DataFrame:
    title = df["title"].astype(str)
    if group == "director":
        mask = title.str.contains(DIRECTOR_RE, na=False)
    elif group == "ten_pct_owner":
        mask = title.str.contains(TEN_PCT_RE, na=False)
    else:
        raise ValueError(f"Unknown group: {group}")
    return df.loc[mask].copy()


def _evaluate_group(group_df: pd.DataFrame, features: list[str]) -> dict:
    out: dict[str, dict] = {}

    for h in HORIZONS:
        tgt = f"return_{h}d_pct"
        sub = group_df.dropna(subset=[tgt]).copy()
        sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
        sub = sub.dropna(subset=["trade_date"]).sort_values("trade_date").copy()

        if len(sub) < 30:
            out[str(h)] = {"status": "insufficient_rows", "rows": int(len(sub))}
            continue

        lo, hi = sub[tgt].quantile([0.01, 0.99])
        sub[tgt] = sub[tgt].clip(lo, hi)

        X = sub[features].copy()
        y = sub[tgt].values
        idx = sub.index.values

        X_tr, X_va, X_te, y_tr, y_va, y_te, idx_tr, idx_va, idx_te = chrono_train_val_test_split(X, y, idx)

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

        ds_te_eq = decile_stats(mat_te @ w_eq, y_te)
        ds_te_use = decile_stats(mat_te @ w_use, y_te)

        out[str(h)] = {
            "status": "ok",
            "split_sizes": {"train": int(len(X_tr)), "val": int(len(X_va)), "test": int(len(X_te))},
            "date_ranges": {
                "train": [
                    str(pd.to_datetime(sub.loc[idx_tr, "trade_date"]).min().date()),
                    str(pd.to_datetime(sub.loc[idx_tr, "trade_date"]).max().date()),
                ],
                "val": [
                    str(pd.to_datetime(sub.loc[idx_va, "trade_date"]).min().date()),
                    str(pd.to_datetime(sub.loc[idx_va, "trade_date"]).max().date()),
                ],
                "test": [
                    str(pd.to_datetime(sub.loc[idx_te, "trade_date"]).min().date()),
                    str(pd.to_datetime(sub.loc[idx_te, "trade_date"]).max().date()),
                ],
            },
            "weights": {
                "mode": mode,
                "equal": {m: float(v) for m, v in zip(MODEL_NAMES, w_eq)},
                "used": {m: float(v) for m, v in zip(MODEL_NAMES, w_use)},
            },
            "validation_opt_objective": {
                "top_decile_mean": float(opt_top),
                "decile_spread": float(opt_spread),
            },
            "test": {
                "equal": ds_te_eq,
                "used": ds_te_use,
                "used_minus_equal_top_decile_pp": float(
                    ds_te_use["top_decile_mean"] - ds_te_eq["top_decile_mean"]
                ),
                "predictive_power_flag": bool(
                    (ds_te_use["top_decile_mean"] > 0.0) and (ds_te_use["decile_spread"] > 0.0)
                ),
            },
        }

    return out


def _build_summary(report: dict) -> pd.DataFrame:
    rows = []
    for group, g in report["groups"].items():
        for h, d in g["horizons"].items():
            if d.get("status") != "ok":
                rows.append(
                    {
                        "group": group,
                        "horizon_days": int(h),
                        "status": d.get("status", "unknown"),
                        "rows_train": np.nan,
                        "rows_val": np.nan,
                        "rows_test": np.nan,
                        "test_top_decile_used": np.nan,
                        "test_bottom_decile_used": np.nan,
                        "test_spread_used": np.nan,
                        "predictive_power_flag": False,
                    }
                )
                continue

            te = d["test"]["used"]
            rows.append(
                {
                    "group": group,
                    "horizon_days": int(h),
                    "status": "ok",
                    "rows_train": d["split_sizes"]["train"],
                    "rows_val": d["split_sizes"]["val"],
                    "rows_test": d["split_sizes"]["test"],
                    "test_top_decile_used": te["top_decile_mean"],
                    "test_bottom_decile_used": te["bottom_decile_mean"],
                    "test_spread_used": te["decile_spread"],
                    "predictive_power_flag": d["test"]["predictive_power_flag"],
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "horizon_days"]).reset_index(drop=True)


def run(output_json: Path, output_csv: Path) -> None:
    base_df = load_and_merge()

    overlap = (
        base_df["title"].astype(str).str.contains(DIRECTOR_RE, na=False)
        & base_df["title"].astype(str).str.contains(TEN_PCT_RE, na=False)
    )
    report = {
        "note": "Groups are evaluated independently and can overlap.",
        "n_rows_total_input": int(len(base_df)),
        "n_rows_overlap_director_and_ten_pct": int(overlap.sum()),
        "groups": {},
    }

    for group in ["director", "ten_pct_owner"]:
        g_raw = _subset(base_df, group)
        if g_raw.empty:
            report["groups"][group] = {"n_rows_raw": 0, "horizons": {}}
            continue

        g_feat, features, _ = engineer_features(g_raw)
        report["groups"][group] = {
            "n_rows_raw": int(len(g_raw)),
            "n_rows_featured": int(len(g_feat)),
            "horizons": _evaluate_group(g_feat, features),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = _build_summary(report)
    summary.to_csv(output_csv, index=False)

    print(f"Saved: {output_json}")
    print(f"Saved: {output_csv}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-estimate prod4 models on director and 10%+ owner groups."
    )
    parser.add_argument(
        "--output-json",
        default="research/outcomes/models/director_10pct_predictive_power_time_split.json",
    )
    parser.add_argument(
        "--output-csv",
        default="research/outcomes/models/director_10pct_predictive_power_time_split.csv",
    )
    parser.add_argument(
        "--aggregated-file",
        default="backtest/data/backtest_results_aggregated.csv",
        help="Aggregated event-level backtest file to use for load_and_merge().",
    )
    args = parser.parse_args()

    import train_models

    train_models.AGGREGATED_CSV = args.aggregated_file
    run(Path(args.output_json), Path(args.output_csv))


if __name__ == "__main__":
    main()
