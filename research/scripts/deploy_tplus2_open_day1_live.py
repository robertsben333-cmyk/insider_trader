from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

BASE = Path("C:/Users/XavierFriesen/insider_trades_predictor")
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from model_ensemble import to_linear_numeric, to_xgb, train_elasticnet, train_spline_elasticnet, train_xgb
from train_models import (
    BENCHMARK_TICKER,
    CACHE_DIR,
    ET,
    MODEL_PARAMS,
    TARGET_RETURN_MODE,
    _bar_date_et,
    _per_day_compound_pct,
    engineer_features,
    fetch_day_bars,
    fetch_minute_bars,
    find_last_close,
    find_price_at_or_after,
    load_and_merge,
)

MODEL_DIR = BASE / "models" / "prod4"
BACKUP_DIR = BASE / "models_r_day1"
OUT_MODELS_DIR = BASE / "research" / "outcomes" / "models"
OUT_BACKTEST_DIR = BASE / "backtest" / "out"

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
STEP = 0.005
THRESHOLDS = np.round(np.arange(STEP, 1.0 + STEP / 2, STEP), 3)
MODEL_NAMES = ["HGBR", "XGBoost", "ElasticNet", "SplineElasticNet"]
CUSTOM_TARGET = "return_2d_open_pct"
CUSTOM_HORIZON_DAYS = 2
DEFAULT_DECILE_SCORE_THRESHOLD = 0.93


@dataclass
class SplitData:
    sub: pd.DataFrame
    features: list[str]
    X_tr: pd.DataFrame
    X_va: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: np.ndarray
    y_va: np.ndarray
    y_te: np.ndarray


def chrono_train_val_test_split(X, y):
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
    )


def normalize_buy_datetime(series: pd.Series) -> pd.Series:
    buy_dt = pd.to_datetime(series, errors="coerce")
    if getattr(buy_dt.dt, "tz", None) is None:
        return buy_dt.dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward")
    return buy_dt.dt.tz_convert(ET)


def fit_models(X_train: pd.DataFrame, y_train: np.ndarray):
    models = {}
    hgbr = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr.fit(X_train, y_train)
    models["HGBR"] = hgbr
    models["XGBoost"] = train_xgb(X_train, y_train)
    models["ElasticNet"] = train_elasticnet(X_train, y_train)
    models["SplineElasticNet"] = train_spline_elasticnet(X_train, y_train)
    return models


def predict_with_models(models: dict[str, object], X: pd.DataFrame) -> np.ndarray:
    preds = []
    for model_name in MODEL_NAMES:
        model = models[model_name]
        if model_name == "HGBR":
            preds.append(model.predict(X))
        elif model_name == "XGBoost":
            preds.append(model.predict(to_xgb(X)))
        elif model_name == "ElasticNet":
            preds.append(model.predict(to_linear_numeric(X)))
        elif model_name == "SplineElasticNet":
            preds.append(model.predict(X.copy()))
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    return np.mean(np.column_stack(preds), axis=1)


def decile_table(signal: np.ndarray, actuals: np.ndarray, n_bins: int = 10):
    order = np.argsort(signal)
    n = len(order)
    bin_size = max(1, n // n_bins)
    rows = []
    for d in range(n_bins):
        start = d * bin_size
        end = n if d == n_bins - 1 else min(n, (d + 1) * bin_size)
        idx = order[start:end]
        p = signal[idx]
        a = actuals[idx]
        rows.append(
            {
                "decile": d + 1,
                "n": int(len(idx)),
                "mean_pred": float(np.mean(p)) if len(p) else np.nan,
                "mean_actual": float(np.mean(a)) if len(a) else np.nan,
                "pct_pos_actual": float((a > 0).mean() * 100) if len(a) else np.nan,
            }
        )
    return rows


def compute_custom_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["buy_datetime"] = normalize_buy_datetime(out["buy_datetime"])
    out["buy_price"] = pd.to_numeric(out["buy_price"], errors="coerce")

    stock_raw = pd.Series(np.nan, index=out.index, dtype=float)
    stock_exit_open = pd.Series(np.nan, index=out.index, dtype=float)
    exit_dates = pd.Series(pd.NaT, index=out.index, dtype="object")

    cache_dir = Path(CACHE_DIR)
    valid = out["ticker"].notna() & out["buy_datetime"].notna() & (out["buy_price"] > 0)
    work = out.loc[valid, ["ticker", "buy_datetime", "buy_price"]].copy()
    work["buy_date"] = work["buy_datetime"].dt.date

    for ticker, grp in work.groupby("ticker", sort=True):
        min_buy = min(grp["buy_date"])
        max_buy = max(grp["buy_date"])
        day_bars = fetch_day_bars(cache_dir, str(ticker), min_buy, max_buy + pd.Timedelta(days=10))
        dated_bars = [(_bar_date_et(bar), bar) for bar in day_bars]
        dated_bars = [(d, bar) for d, bar in dated_bars if d is not None]
        if not dated_bars:
            continue

        trading_dates = [d for d, _bar in dated_bars]
        date_to_idx = {d: idx for idx, d in enumerate(trading_dates)}

        for row_idx, row in grp.iterrows():
            entry_date = row["buy_date"]
            entry_idx = date_to_idx.get(entry_date)
            if entry_idx is None:
                continue
            exit_idx = entry_idx + CUSTOM_HORIZON_DAYS
            if exit_idx >= len(dated_bars):
                continue
            exit_date, exit_bar = dated_bars[exit_idx]
            exit_open = float(exit_bar.get("o")) if exit_bar.get("o") is not None else np.nan
            entry_price = float(row["buy_price"])
            if not np.isfinite(exit_open) or exit_open <= 0 or not np.isfinite(entry_price) or entry_price <= 0:
                continue
            stock_exit_open.loc[row_idx] = exit_open
            exit_dates.loc[row_idx] = exit_date
            stock_raw.loc[row_idx] = ((exit_open / entry_price) - 1.0) * 100.0

    out["exit_2d_open_date"] = exit_dates
    out["exit_2d_open_price"] = stock_exit_open
    out["stock_only_return_2d_open_pct_raw"] = stock_raw
    out["stock_only_return_2d_open_pct"] = _per_day_compound_pct(stock_raw, CUSTOM_HORIZON_DAYS)

    bench_entry = pd.Series(np.nan, index=out.index, dtype=float)
    bench_exit_open = pd.Series(np.nan, index=out.index, dtype=float)

    exit_date_values = [d for d in out["exit_2d_open_date"].tolist() if isinstance(d, date) and pd.notna(d)]
    if exit_date_values:
        spy_day_bars = fetch_day_bars(Path(CACHE_DIR), BENCHMARK_TICKER, min(exit_date_values), max(exit_date_values))
        spy_open_by_date = {}
        for bar in spy_day_bars:
            bar_date = _bar_date_et(bar)
            if bar_date is None:
                continue
            open_px = bar.get("o")
            if open_px is not None:
                spy_open_by_date[bar_date] = float(open_px)
    else:
        spy_open_by_date = {}

    unique_buy_dates = sorted(set(d for d in work["buy_date"].tolist() if pd.notna(d)))
    spy_minute_by_date: dict[date, list[dict]] = {}
    for buy_date in unique_buy_dates:
        spy_minute_by_date[buy_date] = fetch_minute_bars(Path(CACHE_DIR), BENCHMARK_TICKER, buy_date)

    for row_idx, row in work.iterrows():
        buy_date = row["buy_date"]
        buy_dt = row["buy_datetime"]
        exit_date = out.at[row_idx, "exit_2d_open_date"]
        if not isinstance(exit_date, date) or pd.isna(exit_date):
            continue
        bars = spy_minute_by_date.get(buy_date, [])
        entry_px = find_price_at_or_after(bars, int(buy_dt.timestamp() * 1000))
        if not np.isfinite(entry_px):
            entry_px = find_last_close(bars)
        exit_open = spy_open_by_date.get(exit_date, np.nan)
        if not np.isfinite(entry_px) or entry_px <= 0 or not np.isfinite(exit_open) or exit_open <= 0:
            continue
        bench_entry.loc[row_idx] = float(entry_px)
        bench_exit_open.loc[row_idx] = float(exit_open)

    benchmark_raw = ((bench_exit_open / bench_entry) - 1.0) * 100.0
    out[f"{BENCHMARK_TICKER.lower()}_return_2d_open_pct_raw"] = benchmark_raw
    out[f"{BENCHMARK_TICKER.lower()}_return_2d_open_pct"] = _per_day_compound_pct(benchmark_raw, CUSTOM_HORIZON_DAYS)
    out[CUSTOM_TARGET] = (
        pd.to_numeric(out["stock_only_return_2d_open_pct"], errors="coerce")
        - pd.to_numeric(out[f"{BENCHMARK_TICKER.lower()}_return_2d_open_pct"], errors="coerce")
    )
    out["target_return_mode"] = TARGET_RETURN_MODE
    out["benchmark_ticker"] = BENCHMARK_TICKER
    return out


def prepare_split() -> SplitData:
    df_raw = load_and_merge()
    df, features, _ = engineer_features(df_raw)
    df = compute_custom_target(df)
    sub = df.dropna(subset=[CUSTOM_TARGET]).copy()
    lo, hi = sub[CUSTOM_TARGET].quantile([0.01, 0.99])
    sub[CUSTOM_TARGET] = sub[CUSTOM_TARGET].clip(lo, hi)
    sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
    sub["buy_datetime"] = normalize_buy_datetime(sub["buy_datetime"])
    sub = sub.dropna(subset=["trade_date", "buy_datetime"]).sort_values("trade_date").reset_index(drop=True)

    X = sub[features].copy()
    y = sub[CUSTOM_TARGET].to_numpy(dtype=float)
    X_tr, X_va, X_te, y_tr, y_va, y_te = chrono_train_val_test_split(X, y)
    return SplitData(sub=sub, features=features, X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te)


def ensure_backup() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    prod4_backup = BACKUP_DIR / "prod4"
    if prod4_backup.exists():
        raise RuntimeError(f"Backup target already exists: {prod4_backup}")
    shutil.copytree(MODEL_DIR, prod4_backup)

    extra_files = [
        BASE / "backtest" / "out" / "investable_decile_score_sweep_0005.csv",
        BASE / "research" / "outcomes" / "models" / "equal4_deciles_time_split.csv",
        BASE / "research" / "outcomes" / "models" / "equal4_deciles_time_split.json",
    ]
    for path in extra_files:
        if path.exists():
            shutil.copy2(path, BACKUP_DIR / path.name)


def build_cutoff_csv(train_signal: np.ndarray, out_csv: Path) -> float:
    pred_train_sorted = np.sort(train_signal.astype(float))
    rows = []
    default_raw = float("nan")
    for threshold in THRESHOLDS:
        raw_cutoff = float(np.quantile(pred_train_sorted, float(threshold)))
        rows.append(
            {
                "decile_score_threshold": float(threshold),
                "raw_pred_mean4_cutoff": raw_cutoff,
            }
        )
        if abs(float(threshold) - DEFAULT_DECILE_SCORE_THRESHOLD) < 1e-9:
            default_raw = raw_cutoff
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return default_raw


def build_benchmark_csv(test_signal: np.ndarray, y_te: np.ndarray, out_csv: Path, out_json: Path) -> None:
    rows = decile_table(test_signal, y_te, n_bins=10)
    csv_rows = [{"horizon_days": 1, "target": CUSTOM_TARGET, **r} for r in rows]
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    payload = {
        "meta": {
            "split": "time",
            "test_size": 0.2,
            "ensemble_models": MODEL_NAMES,
            "target_return_mode": TARGET_RETURN_MODE,
            "benchmark_ticker": BENCHMARK_TICKER,
            "target": CUSTOM_TARGET,
            "exit_rule": "sell_at_open_2_trading_days_after_buy",
        },
        "horizons": {
            "1": {
                "target": CUSTOM_TARGET,
                "deciles": rows,
            }
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    print("Backing up current live-scoring models and day-1 artifacts...")
    ensure_backup()

    print("Preparing T+2-open custom target dataset...")
    split = prepare_split()
    print(
        f"rows={len(split.sub):,} train={len(split.X_tr):,} val={len(split.X_va):,} test={len(split.X_te):,}"
    )

    print("Training T+2-open day-1 models...")
    trainval_X = pd.concat([split.X_tr, split.X_va], axis=0)
    trainval_y = np.concatenate([split.y_tr, split.y_va], axis=0)
    final_models = fit_models(trainval_X, trainval_y)
    train_signal_for_cutoff = predict_with_models(final_models, split.X_tr)
    test_signal = predict_with_models(final_models, split.X_te)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, model in final_models.items():
        joblib.dump(model, MODEL_DIR / f"model_1d_{model_name}.pkl")

    cutoff_csv = OUT_BACKTEST_DIR / "investable_decile_score_sweep_0005_tplus2_open.csv"
    OUT_BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    default_raw = build_cutoff_csv(train_signal_for_cutoff, cutoff_csv)

    benchmark_csv = OUT_MODELS_DIR / "equal4_deciles_time_split_tplus2_open_live.csv"
    benchmark_json = OUT_MODELS_DIR / "equal4_deciles_time_split_tplus2_open_live.json"
    OUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    build_benchmark_csv(test_signal, split.y_te, benchmark_csv, benchmark_json)

    eval_payload = {
        "1": {
            "target": CUSTOM_TARGET,
            "target_return_mode": TARGET_RETURN_MODE,
            "benchmark_ticker": BENCHMARK_TICKER,
            "split_sizes": {
                "train": int(len(split.X_tr)),
                "val": int(len(split.X_va)),
                "test": int(len(split.X_te)),
            },
            "threshold_093_raw_pred_mean4_cutoff": float(default_raw),
            "artifacts": {
                "cutoff_csv": str(cutoff_csv),
                "benchmark_csv": str(benchmark_csv),
            },
        }
    }
    (MODEL_DIR / "eval_time_split_day1_tplus2_open.json").write_text(
        json.dumps(eval_payload, indent=2), encoding="utf-8"
    )

    print("Deployment complete.")
    print(f"Backup dir: {BACKUP_DIR}")
    print(f"Live day-1 models written to: {MODEL_DIR}")
    print(f"Cutoff csv: {cutoff_csv}")
    print(f"Benchmark csv: {benchmark_csv}")
    print(f"Default raw threshold for decile_score=0.93: {default_raw:.12f}")


if __name__ == "__main__":
    main()
