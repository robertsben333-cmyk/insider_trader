from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score


BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
os.chdir(BASE)

import train_models
from model_ensemble import MODEL_NAMES, chrono_train_val_test_split, decile_stats, to_linear_numeric, to_xgb
from research.scripts import build_unfiltered_aggregated_backtest as aggregate_builder
from research.scripts.deploy_tplus2_open_day1_live import CUSTOM_TARGET, compute_custom_target, fit_models
from train_models import _bar_date_et, engineer_features, fetch_day_bars, load_and_merge


DEFAULT_RAW_CSV = BASE / "backtest" / "data" / "insider_purchases.csv"
DEFAULT_AGG_CSV = BASE / "backtest" / "data" / "backtest_results_aggregated_live_entry.csv"
DEFAULT_CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUTPUT_DIR = BASE / "research" / "outcomes" / "models"
SUMMARY_CSV = OUTPUT_DIR / "live_gap_feature_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "live_gap_feature_summary.json"
TEST_PREDICTIONS_CSV = OUTPUT_DIR / "live_gap_feature_test_predictions.csv"
BUCKET_SUMMARY_CSV = OUTPUT_DIR / "live_gap_feature_bucket_summary.csv"
IMPORTANCE_CSV = OUTPUT_DIR / "live_gap_feature_importance.csv"
COVERAGE_CSV = OUTPUT_DIR / "live_gap_feature_coverage.csv"
CURRENT_PROD_EVAL_JSON = BASE / "models" / "prod4" / "eval_time_split_day1_tplus2_open.json"
CURRENT_BASELINE_AGG_CSV = BASE / "backtest" / "data" / "backtest_results_aggregated.csv"
LEGACY_GAP_RECS_CSV = BASE / "research" / "outcomes" / "feature_screen_recommendations.csv"
PERMUTATION_REPEATS = 5
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a live-aligned previous-close-to-entry gap feature against the active day-1 target."
    )
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--aggregate-csv", type=Path, default=DEFAULT_AGG_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument(
        "--rebuild-aggregate",
        action="store_true",
        help="Rebuild the live-policy aggregate before running the experiment.",
    )
    parser.add_argument(
        "--cache-only-day",
        action="store_true",
        help="Do not call Polygon for missing daily bars while building the live aggregate.",
    )
    return parser.parse_args()


def ensure_live_aggregate(args: argparse.Namespace) -> None:
    if args.aggregate_csv.exists() and not args.rebuild_aggregate:
        return
    logger = aggregate_builder.setup_logger()
    aggregate_builder.build_dataset(
        input_csv=args.raw_csv,
        output_csv=args.aggregate_csv,
        cache_dir=args.cache_dir,
        max_workers=int(args.max_workers),
        entry_policy="live",
        supported_titles_only=True,
        allow_last_price_fallback=False,
        cache_only_day=bool(args.cache_only_day),
        logger=logger,
    )


def load_current_prod_reference() -> dict[str, Any]:
    if not CURRENT_PROD_EVAL_JSON.exists():
        return {}
    try:
        return json.loads(CURRENT_PROD_EVAL_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_legacy_gap_reference() -> list[dict[str, Any]]:
    if not LEGACY_GAP_RECS_CSV.exists():
        return []
    try:
        recs = pd.read_csv(LEGACY_GAP_RECS_CSV)
    except Exception:
        return []
    keep = recs[recs["feature"].isin(["overnight_gap_to_entry", "open_to_buy_move"])].copy()
    return keep.to_dict("records")


def load_engineered_frame(aggregate_csv: Path, raw_csv: Path, cache_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    original_agg = train_models.AGGREGATED_CSV
    original_raw = train_models.ORIGINAL_CSV
    original_cache = train_models.CACHE_DIR
    try:
        train_models.AGGREGATED_CSV = str(aggregate_csv)
        train_models.ORIGINAL_CSV = str(raw_csv)
        train_models.CACHE_DIR = str(cache_dir)
        merged = load_and_merge()
        feat_df, base_features, _ = engineer_features(merged)
    finally:
        train_models.AGGREGATED_CSV = original_agg
        train_models.ORIGINAL_CSV = original_raw
        train_models.CACHE_DIR = original_cache
    return feat_df, base_features


def _event_key_series(df: pd.DataFrame) -> pd.Series:
    trade_date = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df["ticker"].astype(str) + "|" + trade_date.fillna("")


def restrict_to_current_event_universe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if not CURRENT_BASELINE_AGG_CSV.exists():
        return df.copy(), {"current_event_keys": 0, "live_event_keys": int(len(df)), "matched_event_keys": int(len(df))}
    current = pd.read_csv(CURRENT_BASELINE_AGG_CSV, usecols=["ticker", "trade_date"])
    current_keys = set(_event_key_series(current).tolist())
    out = df.copy()
    out["_event_key"] = _event_key_series(out)
    filtered = out[out["_event_key"].isin(current_keys)].copy()
    matched_event_keys = int(filtered["_event_key"].nunique())
    filtered.drop(columns=["_event_key"], inplace=True, errors="ignore")
    return filtered, {
        "current_event_keys": int(len(current_keys)),
        "live_event_keys": int(out["_event_key"].nunique()),
        "matched_event_keys": matched_event_keys,
    }


def _coerce_live_entry_bucket(df: pd.DataFrame) -> pd.Series:
    if "entry_bucket" in df.columns:
        series = df["entry_bucket"].fillna("").astype(str).str.strip()
        if bool(series.ne("").any()):
            return series
    txn = pd.to_datetime(df.get("transaction_date"), errors="coerce")
    return txn.apply(lambda ts: aggregate_builder.compute_entry_bucket(ts, "live") if pd.notna(ts) else "")


def add_live_gap_feature(df: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out["buy_datetime"] = pd.to_datetime(out["buy_datetime"], errors="coerce")
    out["buy_price"] = pd.to_numeric(out["buy_price"], errors="coerce")
    out["live_entry_bucket"] = _coerce_live_entry_bucket(out)
    out["live_gap_feature_source"] = out.get("buy_price_source", pd.Series("", index=out.index)).fillna("").astype(str)
    out["prev_regular_close"] = np.nan
    out["live_gap_to_entry_pct"] = np.nan

    valid = out["ticker"].notna() & out["buy_datetime"].notna() & (out["buy_price"] > 0)
    work = out.loc[valid, ["ticker", "buy_datetime", "buy_price"]].copy()
    work["buy_date"] = work["buy_datetime"].dt.date

    for ticker, grp in work.groupby("ticker", sort=True):
        min_day = min(grp["buy_date"]) - pd.Timedelta(days=7)
        max_day = max(grp["buy_date"]) + pd.Timedelta(days=1)
        bars = fetch_day_bars(cache_dir, str(ticker), min_day, max_day)
        dated = [(_bar_date_et(bar), bar) for bar in bars]
        dated = [(d, bar) for d, bar in dated if d is not None and bar.get("c") is not None]
        if len(dated) < 2:
            continue
        trading_days = [d for d, _bar in dated]
        close_by_day = {d: float(bar["c"]) for d, bar in dated}

        for row_idx, row in grp.iterrows():
            buy_day = row["buy_date"]
            pos = bisect.bisect_left(trading_days, buy_day)
            if pos <= 0:
                continue
            prev_day = trading_days[pos - 1]
            prev_close = close_by_day.get(prev_day)
            buy_price = float(row["buy_price"])
            if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0 or not np.isfinite(buy_price) or buy_price <= 0:
                continue
            out.at[row_idx, "prev_regular_close"] = prev_close
            out.at[row_idx, "live_gap_to_entry_pct"] = ((buy_price / prev_close) - 1.0) * 100.0

    gap = pd.to_numeric(out["live_gap_to_entry_pct"], errors="coerce")
    if bool(gap.notna().any()):
        lo, hi = gap.quantile([0.005, 0.995])
        out["live_gap_to_entry_pct"] = gap.clip(lo, hi)

    coverage = {
        "rows_total": int(len(out)),
        "rows_with_gap": int(out["live_gap_to_entry_pct"].notna().sum()),
        "coverage_ratio": float(out["live_gap_to_entry_pct"].notna().mean()) if len(out) else 0.0,
        "by_bucket": (
            out.assign(_nonnull=out["live_gap_to_entry_pct"].notna().astype(int))
            .groupby("live_entry_bucket", dropna=False)
            .agg(rows=("ticker", "count"), rows_with_gap=("_nonnull", "sum"))
            .assign(coverage_ratio=lambda x: x["rows_with_gap"] / x["rows"].where(x["rows"] > 0))
            .reset_index()
            .to_dict("records")
        ),
        "by_source": (
            out.assign(_nonnull=out["live_gap_to_entry_pct"].notna().astype(int))
            .groupby("live_gap_feature_source", dropna=False)
            .agg(rows=("ticker", "count"), rows_with_gap=("_nonnull", "sum"))
            .assign(coverage_ratio=lambda x: x["rows_with_gap"] / x["rows"].where(x["rows"] > 0))
            .reset_index()
            .to_dict("records")
        ),
    }
    return out, coverage


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = compute_custom_target(df)
    out = out.dropna(subset=[CUSTOM_TARGET]).copy()
    lo, hi = out[CUSTOM_TARGET].quantile([0.01, 0.99])
    out[CUSTOM_TARGET] = out[CUSTOM_TARGET].clip(lo, hi)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["buy_datetime"] = pd.to_datetime(out["buy_datetime"], errors="coerce")
    out = out.dropna(subset=["trade_date", "buy_datetime"]).sort_values("trade_date").reset_index(drop=True)
    return out


def predict_all_models(models: dict[str, object], X: pd.DataFrame) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for model_name in MODEL_NAMES:
        model = models[model_name]
        if model_name == "HGBR":
            preds[model_name] = model.predict(X)
        elif model_name == "XGBoost":
            preds[model_name] = model.predict(to_xgb(X))
        elif model_name == "ElasticNet":
            preds[model_name] = model.predict(to_linear_numeric(X))
        elif model_name == "SplineElasticNet":
            preds[model_name] = model.predict(X.copy())
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    preds["pred_mean4"] = np.mean(np.column_stack([preds[name] for name in MODEL_NAMES]), axis=1)
    return preds


def top_decile_positive_rate(signal: np.ndarray, actuals: np.ndarray) -> float:
    order = np.argsort(signal)
    n = max(1, len(order) // 10)
    top_idx = order[-n:]
    return float((actuals[top_idx] > 0).mean()) if len(top_idx) else float("nan")


def summarize_metrics(signal: np.ndarray, actuals: np.ndarray) -> dict[str, Any]:
    deciles = decile_stats(signal, actuals)
    return {
        "r2": float(r2_score(actuals, signal)),
        "mae": float(mean_absolute_error(actuals, signal)),
        "top_decile_mean": float(deciles["top_decile_mean"]),
        "bottom_decile_mean": float(deciles["bottom_decile_mean"]),
        "decile_spread": float(deciles["decile_spread"]),
        "top_decile_positive_rate": top_decile_positive_rate(signal, actuals),
        "deciles": deciles["rows"],
    }


def scenario_test_frame(test_rows: pd.DataFrame, preds: dict[str, np.ndarray], scenario: str) -> pd.DataFrame:
    out = test_rows.copy()
    out["scenario"] = scenario
    out["pred_HGBR"] = preds["HGBR"]
    out["pred_XGBoost"] = preds["XGBoost"]
    out["pred_ElasticNet"] = preds["ElasticNet"]
    out["pred_SplineElasticNet"] = preds["SplineElasticNet"]
    out["pred_mean4"] = preds["pred_mean4"]
    return out


def summarize_buckets(test_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, bucket), grp in test_frame.groupby(["scenario", "live_entry_bucket"], dropna=False, sort=True):
        actuals = grp[CUSTOM_TARGET].to_numpy(dtype=float)
        signal = grp["pred_mean4"].to_numpy(dtype=float)
        if len(grp) == 0:
            continue
        metrics = summarize_metrics(signal, actuals)
        rows.append(
            {
                "scenario": scenario,
                "live_entry_bucket": bucket,
                "rows": int(len(grp)),
                "feature_coverage": float(pd.to_numeric(grp["live_gap_to_entry_pct"], errors="coerce").notna().mean()),
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "top_decile_mean": metrics["top_decile_mean"],
                "decile_spread": metrics["decile_spread"],
                "top_decile_positive_rate": metrics["top_decile_positive_rate"],
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_rows(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for model_name in ("HGBR", "XGBoost"):
        model = models[model_name]
        X_eval = X_test if model_name == "HGBR" else to_xgb(X_test)
        perm = permutation_importance(
            model,
            X_eval,
            y_test,
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
        )
        imp = pd.DataFrame(
            {
                "model": model_name,
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values(["importance_mean", "feature"], ascending=[False, True]).reset_index(drop=True)
        imp["rank"] = np.arange(1, len(imp) + 1)
        rows.extend(imp.to_dict("records"))
        gap_row = imp.loc[imp["feature"] == "live_gap_to_entry_pct"]
        if not gap_row.empty:
            rec = gap_row.iloc[0]
            meta[model_name] = {
                "candidate_rank": int(rec["rank"]),
                "candidate_importance_mean": float(rec["importance_mean"]),
                "candidate_importance_std": float(rec["importance_std"]),
            }
    return pd.DataFrame(rows), meta


def run_scenario(
    sub: pd.DataFrame,
    feature_cols: list[str],
    scenario: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, object], pd.DataFrame, np.ndarray]:
    idx = np.arange(len(sub))
    X = sub[feature_cols].copy()
    y = sub[CUSTOM_TARGET].to_numpy(dtype=float)

    split = chrono_train_val_test_split(X, y, idx)
    X_tr, X_va, X_te, y_tr, y_va, y_te, _idx_tr, _idx_va, idx_te = split
    X_fit = pd.concat([X_tr, X_va], axis=0)
    y_fit = np.concatenate([y_tr, y_va], axis=0)
    models = fit_models(X_fit, y_fit)
    preds = predict_all_models(models, X_te)
    metrics = summarize_metrics(preds["pred_mean4"], y_te)
    metrics.update(
        {
            "scenario": scenario,
            "feature_count": int(len(feature_cols)),
            "split_train": int(len(X_tr)),
            "split_val": int(len(X_va)),
            "split_test": int(len(X_te)),
        }
    )

    cols = [
        "transaction_date",
        "trade_date",
        "ticker",
        "company_name",
        "owner_name",
        "title",
        "buy_datetime",
        "buy_price",
        "buy_price_source",
        "entry_bucket",
        "live_entry_bucket",
        "prev_regular_close",
        "live_gap_feature_source",
        "live_gap_to_entry_pct",
        CUSTOM_TARGET,
    ]
    test_rows = sub.iloc[idx_te][cols].reset_index(drop=True)
    pred_frame = scenario_test_frame(test_rows, preds, scenario)
    return metrics, pred_frame, models, X_te, y_te


def main() -> None:
    args = parse_args()
    ensure_live_aggregate(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feat_df, base_features = load_engineered_frame(args.aggregate_csv, args.raw_csv, args.cache_dir)
    feat_df, event_universe = restrict_to_current_event_universe(feat_df)
    feat_df, coverage = add_live_gap_feature(feat_df, args.cache_dir)
    sub = prepare_model_frame(feat_df)
    modeled_coverage = {
        "rows_total": int(len(sub)),
        "rows_with_gap": int(pd.to_numeric(sub["live_gap_to_entry_pct"], errors="coerce").notna().sum()),
        "coverage_ratio": float(pd.to_numeric(sub["live_gap_to_entry_pct"], errors="coerce").notna().mean()) if len(sub) else 0.0,
    }

    base_metrics, base_preds, _base_models, _base_X_te, _base_y_te = run_scenario(sub, base_features, "baseline")
    candidate_features = base_features + ["live_gap_to_entry_pct"]
    cand_metrics, cand_preds, cand_models, cand_X_te, cand_y_te = run_scenario(sub, candidate_features, "candidate")
    bucket_summary = summarize_buckets(pd.concat([base_preds, cand_preds], ignore_index=True))
    importance_df, importance_meta = permutation_importance_rows(cand_models, cand_X_te, cand_y_te, candidate_features)

    summary_rows = []
    for metrics in (base_metrics, cand_metrics):
        row = {k: v for k, v in metrics.items() if k != "deciles"}
        row["feature_coverage"] = float(modeled_coverage["coverage_ratio"])
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    coverage_rows = [
        {
            "section": "overall_engineered",
            "rows_total": coverage["rows_total"],
            "rows_with_gap": coverage["rows_with_gap"],
            "coverage_ratio": coverage["coverage_ratio"],
        },
        {"section": "overall_modeled", **modeled_coverage},
    ]
    coverage_rows.extend({"section": "by_bucket", **row} for row in coverage["by_bucket"])
    coverage_rows.extend({"section": "by_source", **row} for row in coverage["by_source"])
    coverage_df = pd.DataFrame(coverage_rows)
    deltas = {
        "delta_r2": float(cand_metrics["r2"] - base_metrics["r2"]),
        "delta_mae": float(cand_metrics["mae"] - base_metrics["mae"]),
        "delta_top_decile_mean": float(cand_metrics["top_decile_mean"] - base_metrics["top_decile_mean"]),
        "delta_decile_spread": float(cand_metrics["decile_spread"] - base_metrics["decile_spread"]),
        "delta_top_decile_positive_rate": float(
            cand_metrics["top_decile_positive_rate"] - base_metrics["top_decile_positive_rate"]
        ),
    }

    summary_payload = {
        "meta": {
            "aggregate_csv": str(args.aggregate_csv),
            "raw_csv": str(args.raw_csv),
            "cache_dir": str(args.cache_dir),
            "entry_policy": "live",
            "supported_titles_only": True,
            "event_universe": "intersection_with_current_backtest_results_aggregated",
            "event_universe_counts": event_universe,
            "target": CUSTOM_TARGET,
            "current_prod_reference": load_current_prod_reference(),
            "legacy_gap_reference": load_legacy_gap_reference(),
        },
        "coverage": {"engineered": coverage, "modeled": modeled_coverage},
        "scenarios": {
            "baseline": base_metrics,
            "candidate": cand_metrics,
        },
        "deltas": deltas,
        "candidate_importance": importance_meta,
    }

    summary_df.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    pd.concat([base_preds, cand_preds], ignore_index=True).to_csv(TEST_PREDICTIONS_CSV, index=False)
    bucket_summary.to_csv(BUCKET_SUMMARY_CSV, index=False)
    importance_df.to_csv(IMPORTANCE_CSV, index=False)
    coverage_df.to_csv(COVERAGE_CSV, index=False)

    print(f"Rows modeled: {len(sub)}")
    print(
        "Baseline vs candidate: "
        f"top_decile_mean {base_metrics['top_decile_mean']:+.4f}% -> {cand_metrics['top_decile_mean']:+.4f}% | "
        f"decile_spread {base_metrics['decile_spread']:+.4f} -> {cand_metrics['decile_spread']:+.4f} | "
        f"top_decile_positive_rate {base_metrics['top_decile_positive_rate']:.4f} -> {cand_metrics['top_decile_positive_rate']:.4f}"
    )
    if importance_meta:
        print(f"Candidate importance: {json.dumps(importance_meta, indent=2)}")
    print(f"Saved summary csv: {SUMMARY_CSV}")
    print(f"Saved summary json: {SUMMARY_JSON}")
    print(f"Saved test predictions: {TEST_PREDICTIONS_CSV}")
    print(f"Saved bucket summary: {BUCKET_SUMMARY_CSV}")
    print(f"Saved importance csv: {IMPORTANCE_CSV}")
    print(f"Saved coverage csv: {COVERAGE_CSV}")


if __name__ == "__main__":
    main()
