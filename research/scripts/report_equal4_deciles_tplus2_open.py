from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import date
from pathlib import Path

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

MODEL_DIR = BASE / "research" / "outcomes" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
ENSEMBLE_MODELS = ["HGBR", "XGBoost", "ElasticNet", "SplineElasticNet"]
CUSTOM_TARGET = "return_2d_open_pct"
CUSTOM_HORIZON_DAYS = 2


def chronological_train_test_split(X, y, idx, test_size: float):
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


def fit_predict_models(X_tr, y_tr, X_te):
    hgbr = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr.fit(X_tr, y_tr)
    p_hgbr = hgbr.predict(X_te)

    xgb_m = train_xgb(X_tr, y_tr)
    p_xgb = xgb_m.predict(to_xgb(X_te))

    en_m = train_elasticnet(X_tr, y_tr)
    p_en = en_m.predict(to_linear_numeric(X_te))

    sp_m = train_spline_elasticnet(X_tr, y_tr)
    p_sp = sp_m.predict(X_te.copy())

    return {
        "HGBR": p_hgbr,
        "XGBoost": p_xgb,
        "ElasticNet": p_en,
        "SplineElasticNet": p_sp,
    }


def normalize_buy_datetime(series: pd.Series) -> pd.Series:
    buy_dt = pd.to_datetime(series, errors="coerce")
    if getattr(buy_dt.dt, "tz", None) is None:
        return buy_dt.dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward")
    return buy_dt.dt.tz_convert(ET)


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
        spy_day_bars = fetch_day_bars(
            cache_dir,
            BENCHMARK_TICKER,
            min(exit_date_values),
            max(exit_date_values),
        )
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
        spy_minute_by_date[buy_date] = fetch_minute_bars(cache_dir, BENCHMARK_TICKER, buy_date)

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
    out["benchmark_entry_price_2d_open"] = bench_entry
    out["benchmark_exit_open_price_2d_open"] = bench_exit_open
    out[f"{BENCHMARK_TICKER.lower()}_return_2d_open_pct_raw"] = benchmark_raw
    out[f"{BENCHMARK_TICKER.lower()}_return_2d_open_pct"] = _per_day_compound_pct(benchmark_raw, CUSTOM_HORIZON_DAYS)
    out[CUSTOM_TARGET] = (
        pd.to_numeric(out["stock_only_return_2d_open_pct"], errors="coerce")
        - pd.to_numeric(out[f"{BENCHMARK_TICKER.lower()}_return_2d_open_pct"], errors="coerce")
    )
    out["target_return_mode"] = TARGET_RETURN_MODE
    out["benchmark_ticker"] = BENCHMARK_TICKER
    return out


def run():
    print("Loading and engineering data...")
    df, features, _ = engineer_features(load_and_merge())
    print(f"  rows={len(df):,} | features={len(features)}")

    print("Computing custom T+2 open target...")
    df = compute_custom_target(df)

    tgt = CUSTOM_TARGET
    sub = df.dropna(subset=[tgt]).copy()
    lo, hi = sub[tgt].quantile([0.01, 0.99])
    sub[tgt] = sub[tgt].clip(lo, hi)
    sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
    sub = sub.dropna(subset=["trade_date"]).sort_values("trade_date").copy()

    X = sub[features].copy()
    y = sub[tgt].values
    idx = sub.index.values

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = chronological_train_test_split(X, y, idx, TEST_SIZE)

    tr_start = pd.to_datetime(sub.loc[idx_tr, "trade_date"]).min()
    tr_end = pd.to_datetime(sub.loc[idx_tr, "trade_date"]).max()
    te_start = pd.to_datetime(sub.loc[idx_te, "trade_date"]).min()
    te_end = pd.to_datetime(sub.loc[idx_te, "trade_date"]).max()

    print(f"\n{'=' * 80}\nCustom horizon: T+2 open\n{'=' * 80}")
    print(
        f"  Time split: train {len(X_tr):,} [{tr_start.date()} -> {tr_end.date()}] | "
        f"test {len(X_te):,} [{te_start.date()} -> {te_end.date()}]"
    )

    pred = fit_predict_models(X_tr, y_tr, X_te)
    signal = np.mean(np.column_stack([pred[m] for m in ENSEMBLE_MODELS]), axis=1)
    rows = decile_table(signal, y_te, n_bins=10)

    out = {
        "meta": {
            "split": "time",
            "test_size": TEST_SIZE,
            "ensemble_models": ENSEMBLE_MODELS,
            "target_return_mode": TARGET_RETURN_MODE,
            "benchmark_ticker": BENCHMARK_TICKER,
            "target": CUSTOM_TARGET,
            "exit_rule": "sell_at_open_2_trading_days_after_buy",
            "horizon_days_for_compounding": CUSTOM_HORIZON_DAYS,
        },
        "target_summary": {
            "usable_rows": int(len(sub)),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "train_start": tr_start.strftime("%Y-%m-%d"),
            "train_end": tr_end.strftime("%Y-%m-%d"),
            "test_start": te_start.strftime("%Y-%m-%d"),
            "test_end": te_end.strftime("%Y-%m-%d"),
        },
        "deciles": rows,
    }

    csv_rows = [{"target": CUSTOM_TARGET, **r} for r in rows]

    print("  Deciles (mean_pred -> mean_actual):")
    for r in rows:
        print(
            f"    D{r['decile']:>2}: pred={r['mean_pred']:+.3f}% "
            f"actual={r['mean_actual']:+.3f}% n={r['n']}"
        )

    out_json = MODEL_DIR / "equal4_deciles_time_split_tplus2_open.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    out_csv = MODEL_DIR / "equal4_deciles_time_split_tplus2_open.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    run()
