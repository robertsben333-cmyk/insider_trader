from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from model_ensemble import (  # noqa: E402
    MODEL_NAMES,
    MODEL_PARAMS,
    to_linear_numeric,
    to_xgb,
    train_elasticnet,
    train_spline_elasticnet,
    train_xgb,
)
from train_models import engineer_features, load_and_merge  # noqa: E402

CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUT_DIR = BASE / "research" / "outcomes" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

TEST_SIZE = 0.2
ARM_GAIN_PCT = 2.0
TRAIL_DRAWDOWN_PCT = 0.5
TARGET_COL = "return_1d_profit_trail_2p0_dd_0p5_pct"
TARGET_LABEL = "next_day_close_with_same_day_profit_trail_2.0pct_arm_0.5pct_dd_stay_out"


def chrono_train_test_split(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, test_size: float):
    n = len(X)
    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 1)
    split = n - n_test
    return (
        X.iloc[:split].copy(),
        X.iloc[split:].copy(),
        y[:split].copy(),
        y[split:].copy(),
        meta.iloc[:split].copy(),
        meta.iloc[split:].copy(),
    )


def cache_path(ticker: str, trade_date: str) -> Path:
    return CACHE_DIR / f"{ticker}_min_{trade_date}_{trade_date}.json"


def load_cached_minute_bars(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def bar_dt_et(bar: dict) -> datetime:
    return datetime.fromtimestamp(int(bar["t"]) / 1000, tz=UTC).astimezone(ET)


def is_regular_session_bar(bar: dict) -> bool:
    dt = bar_dt_et(bar)
    return dt.weekday() < 5 and MARKET_OPEN <= dt.time() < MARKET_CLOSE


def find_entry_bar_index(minute_bars: list[dict], entry_dt: datetime) -> int | None:
    entry_ts = int(entry_dt.timestamp() * 1000)
    for idx, bar in enumerate(minute_bars):
        if int(bar["t"]) >= entry_ts:
            return idx
    return None


def find_same_day_close_index(minute_bars: list[dict], entry_idx: int) -> int | None:
    entry_date = bar_dt_et(minute_bars[entry_idx]).date()
    last_idx = None
    for idx in range(entry_idx, len(minute_bars)):
        if bar_dt_et(minute_bars[idx]).date() != entry_date:
            break
        last_idx = idx
    return last_idx


def implied_exit_price_from_ret(entry_price: float, ret_pct: float) -> float | None:
    if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(ret_pct):
        return float(entry_price * (1.0 + float(ret_pct) / 100.0))
    return None


def percentile_rank(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_reference.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    idx = np.searchsorted(sorted_reference, values, side="right")
    return idx / float(sorted_reference.size)


def simulate_profit_trail_target(
    minute_bars: list[dict],
    entry_dt: datetime,
    entry_price: float,
    baseline_ret_pct: float,
) -> tuple[float | None, bool, str]:
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(baseline_ret_pct):
        return None, False, "invalid_input"

    entry_idx = find_entry_bar_index(minute_bars, entry_dt)
    if entry_idx is None:
        return None, False, "missing_entry_bar"

    close_idx = find_same_day_close_index(minute_bars, entry_idx)
    if close_idx is None or close_idx <= entry_idx:
        return None, False, "missing_close_bar"

    close_1d_price = implied_exit_price_from_ret(entry_price, baseline_ret_pct)
    if close_1d_price is None or not np.isfinite(close_1d_price) or close_1d_price <= 0:
        return None, False, "invalid_baseline_exit"

    peak_price = float(entry_price)
    stop_idx = None
    armed = False

    for idx in range(entry_idx + 1, close_idx + 1):
        px = float(minute_bars[idx]["c"])
        if px > peak_price:
            peak_price = px

        gain_pct = (peak_price / float(entry_price) - 1.0) * 100.0
        if gain_pct >= ARM_GAIN_PCT:
            armed = True

        if not armed:
            continue

        drawdown_pct = (px / peak_price - 1.0) * 100.0
        if drawdown_pct <= -TRAIL_DRAWDOWN_PCT:
            stop_idx = idx
            break

    if stop_idx is None:
        return float(baseline_ret_pct), False, "baseline_next_day_close"

    stop_price = float(minute_bars[stop_idx]["c"])
    strategy_ret_pct = (stop_price / float(entry_price) - 1.0) * 100.0
    return strategy_ret_pct, True, "same_day_profit_trail"


def build_profit_trail_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sub = df.copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
    sub["buy_datetime"] = pd.to_datetime(sub["buy_datetime"], errors="coerce")
    sub["buy_price"] = pd.to_numeric(sub["buy_price"], errors="coerce")
    sub["return_1d_pct"] = pd.to_numeric(sub["return_1d_pct"], errors="coerce")
    sub = sub.dropna(subset=["trade_date", "buy_datetime", "buy_price", "return_1d_pct", "ticker"]).copy()
    sub = sub.sort_values("trade_date").reset_index(drop=True)

    results: list[dict] = []
    minute_cache: dict[tuple[str, str], list[dict]] = {}
    reason_counts: dict[str, int] = {}

    for row in sub.itertuples(index=False):
        ticker = str(row.ticker)
        trade_day = pd.Timestamp(row.buy_datetime).strftime("%Y-%m-%d")
        key = (ticker, trade_day)
        if key not in minute_cache:
            bars = load_cached_minute_bars(cache_path(ticker, trade_day))
            bars = [b for b in bars if is_regular_session_bar(b)]
            bars.sort(key=lambda b: int(b["t"]))
            minute_cache[key] = bars

        target_ret, stopped, reason = simulate_profit_trail_target(
            minute_bars=minute_cache[key],
            entry_dt=pd.Timestamp(row.buy_datetime).to_pydatetime().replace(tzinfo=ET),
            entry_price=float(row.buy_price),
            baseline_ret_pct=float(row.return_1d_pct),
        )
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        result = {
            "transaction_date": row.transaction_date,
            "trade_date": pd.Timestamp(row.trade_date).strftime("%Y-%m-%d"),
            "ticker": ticker,
            "company_name": row.company_name,
            "owner_name": row.owner_name,
            "title": row.title,
            "buy_datetime": pd.Timestamp(row.buy_datetime).strftime("%Y-%m-%d %H:%M:%S"),
            "buy_price": float(row.buy_price),
            "return_1d_pct": float(row.return_1d_pct),
            TARGET_COL: target_ret,
            "profit_trail_triggered": bool(stopped) if target_ret is not None else False,
            "target_reason": reason,
        }
        results.append(result)

    target_df = pd.DataFrame(results)
    coverage = float(target_df[TARGET_COL].notna().mean() * 100.0) if len(target_df) else 0.0
    triggered = int(target_df["profit_trail_triggered"].sum()) if len(target_df) else 0
    summary = {
        "input_rows": int(len(sub)),
        "target_rows": int(len(target_df)),
        "target_non_null_rows": int(target_df[TARGET_COL].notna().sum()),
        "target_coverage_pct": coverage,
        "profit_trail_triggered_rows": triggered,
        "profit_trail_triggered_pct": float((triggered / len(target_df)) * 100.0) if len(target_df) else 0.0,
        "reason_counts": reason_counts,
    }
    return target_df, summary


def fit_predict_models(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame) -> dict[str, np.ndarray]:
    hgbr = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr.fit(X_tr, y_tr)

    xgb_m = train_xgb(X_tr, y_tr)
    en_m = train_elasticnet(X_tr, y_tr)
    sp_m = train_spline_elasticnet(X_tr, y_tr)

    return {
        "HGBR": hgbr.predict(X_te),
        "XGBoost": xgb_m.predict(to_xgb(X_te)),
        "ElasticNet": en_m.predict(to_linear_numeric(X_te)),
        "SplineElasticNet": sp_m.predict(X_te.copy()),
    }, {
        "HGBR": hgbr,
        "XGBoost": xgb_m,
        "ElasticNet": en_m,
        "SplineElasticNet": sp_m,
    }


def predict_with_fitted(models: dict[str, object], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "HGBR": models["HGBR"].predict(X),
        "XGBoost": models["XGBoost"].predict(to_xgb(X)),
        "ElasticNet": models["ElasticNet"].predict(to_linear_numeric(X)),
        "SplineElasticNet": models["SplineElasticNet"].predict(X.copy()),
    }


def decile_bucket_table(signal: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> list[dict]:
    order = np.argsort(signal)
    n = len(order)
    bin_size = max(1, n // n_bins)
    rows: list[dict] = []
    for d in range(n_bins):
        start = d * bin_size
        end = n if d == n_bins - 1 else min(n, (d + 1) * bin_size)
        idx = order[start:end]
        s = signal[idx]
        a = actuals[idx]
        rows.append(
            {
                "decile": d + 1,
                "n": int(len(idx)),
                "mean_pred": float(np.mean(s)) if len(s) else np.nan,
                "mean_actual": float(np.mean(a)) if len(a) else np.nan,
                "median_actual": float(np.median(a)) if len(a) else np.nan,
                "std_actual": float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
                "pct_pos_actual": float((a > 0).mean() * 100.0) if len(a) else np.nan,
            }
        )
    return rows


def decile_threshold_table(decile_scores: np.ndarray, actuals: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for cut in np.arange(0.0, 1.0, 0.1):
        mask = decile_scores >= cut
        vals = actuals[mask]
        rows.append(
            {
                "min_decile_score": round(float(cut), 1),
                "n": int(mask.sum()),
                "mean_actual": float(np.mean(vals)) if len(vals) else np.nan,
                "median_actual": float(np.median(vals)) if len(vals) else np.nan,
                "std_actual": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "pct_pos_actual": float((vals > 0).mean() * 100.0) if len(vals) else np.nan,
            }
        )
    return rows


def model_decile_metrics(signal: np.ndarray, actuals: np.ndarray) -> dict:
    rows = decile_bucket_table(signal, actuals, n_bins=10)
    return {
        "top_decile_mean": float(rows[-1]["mean_actual"]),
        "bottom_decile_mean": float(rows[0]["mean_actual"]),
        "decile_spread": float(rows[-1]["mean_actual"] - rows[0]["mean_actual"]),
    }


def run() -> None:
    print("Loading and engineering research dataset...")
    base_df = load_and_merge()
    feature_df, features, _ = engineer_features(base_df)

    print("Building next-day-close target with 2.0 / 0.5 same-day trailing-profit overlay...")
    target_df, target_summary = build_profit_trail_target(feature_df)
    merged = feature_df.merge(
        target_df[
            [
                "transaction_date",
                "ticker",
                "owner_name",
                TARGET_COL,
                "profit_trail_triggered",
                "target_reason",
            ]
        ],
        on=["transaction_date", "ticker", "owner_name"],
        how="left",
    )

    sub = merged.dropna(subset=[TARGET_COL]).copy()
    lo, hi = sub[TARGET_COL].quantile([0.01, 0.99])
    sub[TARGET_COL] = sub[TARGET_COL].clip(lo, hi)
    sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
    sub = sub.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    X = sub[features].copy()
    y = sub[TARGET_COL].to_numpy(dtype=float)
    meta = sub[
        [
            "transaction_date",
            "trade_date",
            "ticker",
            "owner_name",
            TARGET_COL,
            "return_1d_pct",
            "profit_trail_triggered",
            "target_reason",
        ]
    ].copy()

    X_tr, X_te, y_tr, y_te, meta_tr, meta_te = chrono_train_test_split(X, y, meta, TEST_SIZE)

    tr_start = pd.to_datetime(meta_tr["trade_date"]).min().date()
    tr_end = pd.to_datetime(meta_tr["trade_date"]).max().date()
    te_start = pd.to_datetime(meta_te["trade_date"]).min().date()
    te_end = pd.to_datetime(meta_te["trade_date"]).max().date()

    print(
        f"Chronological split: train {len(X_tr):,} [{tr_start} -> {tr_end}] | "
        f"test {len(X_te):,} [{te_start} -> {te_end}]"
    )

    pred, fitted = fit_predict_models(X_tr, y_tr, X_te)
    train_pred = predict_with_fitted(fitted, X_tr)

    ensemble_test = np.mean(np.column_stack([pred[m] for m in MODEL_NAMES]), axis=1)
    ensemble_train = np.mean(np.column_stack([train_pred[m] for m in MODEL_NAMES]), axis=1)
    train_signal_sorted = np.sort(ensemble_train.astype(float))
    test_decile_score = percentile_rank(train_signal_sorted, ensemble_test.astype(float))

    test_pred = meta_te.copy()
    for model_name in MODEL_NAMES:
        test_pred[f"pred_{model_name}"] = pred[model_name]
    test_pred["pred_mean4"] = ensemble_test
    test_pred["decile_score"] = test_decile_score

    bucket_rows = decile_bucket_table(ensemble_test, y_te, n_bins=10)
    threshold_rows = decile_threshold_table(test_decile_score, y_te)

    model_rows = []
    for model_name in [*MODEL_NAMES, "EqualWeight4"]:
        signal = ensemble_test if model_name == "EqualWeight4" else pred[model_name]
        metrics = model_decile_metrics(signal, y_te)
        model_rows.append({"model": model_name, **metrics})

    out_prefix = "nextday_close_profit_trail_2p0_dd_0p5"
    test_pred.to_csv(OUT_DIR / f"{out_prefix}_test_predictions.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(OUT_DIR / f"{out_prefix}_decile_buckets.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT_DIR / f"{out_prefix}_decile_thresholds.csv", index=False)
    pd.DataFrame(model_rows).to_csv(OUT_DIR / f"{out_prefix}_model_spread.csv", index=False)

    model_dir = OUT_DIR / f"{out_prefix}_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in fitted.items():
        joblib.dump(model, model_dir / f"model_1d_{model_name}.pkl")

    summary = {
        "target": TARGET_LABEL,
        "target_column": TARGET_COL,
        "target_summary": target_summary,
        "split": {
            "type": "chronological_train_test",
            "test_size": TEST_SIZE,
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "train_start": str(tr_start),
            "train_end": str(tr_end),
            "test_start": str(te_start),
            "test_end": str(te_end),
        },
        "target_clip": {"lo": float(lo), "hi": float(hi)},
        "model_spread": model_rows,
        "ensemble_decile_buckets": bucket_rows,
        "ensemble_decile_thresholds": threshold_rows,
    }
    with open(OUT_DIR / f"{out_prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved summary to {OUT_DIR / f'{out_prefix}_summary.json'}")
    print(f"Saved predictions to {OUT_DIR / f'{out_prefix}_test_predictions.csv'}")


if __name__ == "__main__":
    run()
