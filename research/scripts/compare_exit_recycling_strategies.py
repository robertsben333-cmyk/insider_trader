from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
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

OUT_DIR = BASE / "backtest" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
ENSEMBLE_MODELS = ["HGBR", "XGBoost", "ElasticNet", "SplineElasticNet"]
START_BUDGET_EUR = 10_000.0
MAX_TICKER_WEIGHT = 0.25
CUSTOM_TARGET = "return_2d_open_pct"
CUSTOM_HORIZON_DAYS = 2


@dataclass
class StrategySpec:
    name: str
    signal_target_col: str
    actual_return_col: str
    exit_datetime_col: str
    notes: str


@dataclass
class Position:
    ticker: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    invested: float
    ret_pct: float


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


def decile9_lower_bound(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if signal.size == 0:
        raise ValueError("No finite training predictions available for threshold computation.")

    order = np.argsort(signal)
    n = len(order)
    bin_size = max(1, n // 10)
    start = 8 * bin_size
    end = min(n, 9 * bin_size)
    idx = order[start:end]
    if len(idx) == 0:
        return float(np.quantile(signal, 0.8))
    return float(np.min(signal[idx]))


def fit_models(X_tr, y_tr):
    hgbr = HistGradientBoostingRegressor(**MODEL_PARAMS)
    hgbr.fit(X_tr, y_tr)

    xgb_m = train_xgb(X_tr, y_tr)
    en_m = train_elasticnet(X_tr, y_tr)
    sp_m = train_spline_elasticnet(X_tr, y_tr)

    return {
        "HGBR": hgbr,
        "XGBoost": xgb_m,
        "ElasticNet": en_m,
        "SplineElasticNet": sp_m,
    }


def predict_with_models(models: dict[str, object], X: pd.DataFrame) -> np.ndarray:
    preds = []
    for model_name in ENSEMBLE_MODELS:
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


def next_business_day_close(ts: pd.Timestamp) -> pd.Timestamp:
    nxt = (ts + pd.offsets.BDay(1)).normalize()
    return nxt + pd.Timedelta(hours=16)


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
    out["exit_2d_open_datetime"] = pd.NaT

    valid_exit_rows = out["exit_2d_open_date"].apply(lambda d: isinstance(d, date) and pd.notna(d))
    out.loc[valid_exit_rows, "exit_2d_open_datetime"] = (
        pd.to_datetime(out.loc[valid_exit_rows, "exit_2d_open_date"].astype(str), errors="coerce")
        .dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward")
        + pd.Timedelta(hours=9, minutes=30)
    )

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


def prepare_base_frame() -> tuple[pd.DataFrame, list[str]]:
    df_raw = load_and_merge()
    df, features, _ = engineer_features(df_raw)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["buy_datetime"] = normalize_buy_datetime(df["buy_datetime"])
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
    df["stock_only_return_1d_pct"] = pd.to_numeric(df["stock_only_return_1d_pct"], errors="coerce")
    df["exit_1d_close_datetime"] = df["buy_datetime"].apply(next_business_day_close)
    df = compute_custom_target(df)
    return df, features


def build_scored_test_set(df: pd.DataFrame, features: list[str], signal_target_col: str) -> tuple[pd.DataFrame, float]:
    sub = df.dropna(subset=[signal_target_col, "trade_date", "buy_datetime"]).copy()
    lo, hi = sub[signal_target_col].quantile([0.01, 0.99])
    sub[signal_target_col] = sub[signal_target_col].clip(lo, hi)
    sub = sub.sort_values("trade_date").reset_index(drop=True)

    X = sub[features].copy()
    y = sub[signal_target_col].to_numpy(dtype=float)
    idx = sub.index.to_numpy()
    X_tr, X_te, y_tr, _y_te, idx_tr, idx_te = chronological_train_test_split(X, y, idx, TEST_SIZE)

    models = fit_models(X_tr, y_tr)
    train_signal = predict_with_models(models, X_tr)
    test_signal = predict_with_models(models, X_te)
    threshold = decile9_lower_bound(train_signal)

    out = sub.iloc[idx_te].copy()
    out["pred_mean4"] = test_signal
    out["signal_threshold"] = threshold
    return out, threshold


def simulate_budget_curve(picks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    picks = picks.dropna(subset=["ret_pct", "buy_datetime", "exit_datetime", "ticker", "pred_mean4"]).copy()
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    picks = picks.sort_values(["buy_datetime", "pred_mean4"], ascending=[True, False]).reset_index(drop=True)

    cash = float(START_BUDGET_EUR)
    open_positions: list[Position] = []
    trade_log: list[dict] = []
    curve: list[dict] = []
    recycled_entry_count = 0
    recycled_batch_count = 0

    start_ts = pd.to_datetime(picks["buy_datetime"].min()) - pd.Timedelta(minutes=1)
    curve.append({"timestamp": start_ts, "cash_eur": cash, "invested_eur": 0.0, "budget_eur": cash})

    def mark(ts: pd.Timestamp):
        invested = float(sum(p.invested for p in open_positions))
        curve.append({"timestamp": ts, "cash_eur": cash, "invested_eur": invested, "budget_eur": cash + invested})

    def close_until(ts: pd.Timestamp) -> list[pd.Timestamp]:
        nonlocal cash, open_positions
        closed_exit_times: list[pd.Timestamp] = []
        while True:
            due = [p for p in open_positions if p.exit_time <= ts]
            if not due:
                break
            next_exit = min(p.exit_time for p in due)
            closing = [p for p in open_positions if p.exit_time == next_exit]
            open_positions = [p for p in open_positions if p.exit_time != next_exit]
            for pos in closing:
                proceeds = pos.invested * (1.0 + pos.ret_pct / 100.0)
                cash += proceeds
                closed_exit_times.append(pos.exit_time)
                trade_log.append(
                    {
                        "ticker": pos.ticker,
                        "entry_time": pos.entry_time,
                        "exit_time": pos.exit_time,
                        "invested_eur": pos.invested,
                        "ret_pct": pos.ret_pct,
                        "proceeds_eur": proceeds,
                        "pnl_eur": proceeds - pos.invested,
                    }
                )
            mark(next_exit)
        return closed_exit_times

    for buy_time, batch in picks.groupby("buy_datetime", sort=True):
        buy_time = pd.to_datetime(buy_time)
        closed_exit_times = close_until(buy_time)
        recycled_today = any(pd.Timestamp(ts).date() == buy_time.date() for ts in closed_exit_times)

        batch = batch.sort_values("pred_mean4", ascending=False)
        batch_allocations = 0
        for _, row in batch.iterrows():
            if cash <= 0:
                break

            invested_total = float(sum(p.invested for p in open_positions))
            budget_now = cash + invested_total
            if budget_now <= 0:
                break

            ticker = str(row["ticker"])
            ticker_exposure = float(sum(p.invested for p in open_positions if p.ticker == ticker))
            ticker_cap = MAX_TICKER_WEIGHT * budget_now
            alloc = min(cash, max(0.0, ticker_cap - ticker_exposure))
            if alloc <= 0:
                continue

            ret_pct = float(row["ret_pct"])
            if not np.isfinite(ret_pct):
                continue

            cash -= alloc
            batch_allocations += 1
            open_positions.append(
                Position(
                    ticker=ticker,
                    entry_time=buy_time,
                    exit_time=pd.to_datetime(row["exit_datetime"]),
                    invested=alloc,
                    ret_pct=ret_pct,
                )
            )

        if recycled_today and batch_allocations > 0:
            recycled_batch_count += 1
            recycled_entry_count += batch_allocations
        mark(buy_time)

    if open_positions:
        last_exit = max(p.exit_time for p in open_positions)
        close_until(last_exit + pd.Timedelta(seconds=1))

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    trades_df = pd.DataFrame(trade_log).sort_values("entry_time").reset_index(drop=True)

    if curve_df.empty:
        return curve_df, trades_df, {}

    start_budget = float(curve_df["budget_eur"].iloc[0])
    end_budget = float(curve_df["budget_eur"].iloc[-1])
    start_time = pd.to_datetime(curve_df["timestamp"].iloc[0])
    end_time = pd.to_datetime(curve_df["timestamp"].iloc[-1])
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1.0)
    annualized_return_pct = ((end_budget / start_budget) ** (365.25 / elapsed_days) - 1.0) * 100.0
    running_peak = curve_df["budget_eur"].cummax()
    max_drawdown_pct = (((curve_df["budget_eur"] / running_peak) - 1.0) * 100.0).min()

    summary = {
        "start_budget_eur": start_budget,
        "end_budget_eur": end_budget,
        "total_return_pct": (end_budget / start_budget - 1.0) * 100.0,
        "annualized_return_pct": annualized_return_pct,
        "max_drawdown_pct": float(max_drawdown_pct),
        "trades_executed": int(len(trades_df)),
        "recycled_entry_count": int(recycled_entry_count),
        "recycled_batch_count": int(recycled_batch_count),
    }
    return curve_df, trades_df, summary


def run_strategy(base_df: pd.DataFrame, features: list[str], spec: StrategySpec) -> dict:
    scored_test, threshold = build_scored_test_set(base_df, features, spec.signal_target_col)
    picks = scored_test[
        (pd.to_numeric(scored_test["pred_mean4"], errors="coerce") > float(threshold))
        & pd.to_numeric(scored_test[spec.actual_return_col], errors="coerce").notna()
        & scored_test[spec.exit_datetime_col].notna()
    ].copy()

    picks["ret_pct"] = pd.to_numeric(picks[spec.actual_return_col], errors="coerce")
    picks["exit_datetime"] = pd.to_datetime(picks[spec.exit_datetime_col], errors="coerce")

    curve_df, trades_df, sim_summary = simulate_budget_curve(picks)

    curve_path = OUT_DIR / f"{spec.name}_budget_curve.csv"
    trades_path = OUT_DIR / f"{spec.name}_trade_log.csv"
    if not curve_df.empty:
        curve_df.to_csv(curve_path, index=False)
    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)

    out = {
        "strategy": spec.name,
        "signal_target_col": spec.signal_target_col,
        "actual_return_col": spec.actual_return_col,
        "exit_datetime_col": spec.exit_datetime_col,
        "notes": spec.notes,
        "threshold_pred_mean4": float(threshold),
        "test_rows": int(len(scored_test)),
        "candidate_rows": int(len(picks)),
        "test_start": pd.to_datetime(scored_test["trade_date"].min()).strftime("%Y-%m-%d"),
        "test_end": pd.to_datetime(scored_test["trade_date"].max()).strftime("%Y-%m-%d"),
        "curve_csv": str(curve_path),
        "trades_csv": str(trades_path),
    }
    out.update(sim_summary)
    return out


def main() -> None:
    print("Loading engineered dataset and building custom T+2 open labels...")
    base_df, features = prepare_base_frame()

    strategies = [
        StrategySpec(
            name="portfolio_day1_close_signal_day1_close_exit",
            signal_target_col="return_1d_pct",
            actual_return_col="stock_only_return_1d_pct",
            exit_datetime_col="exit_1d_close_datetime",
            notes="Current-style signal and current-style exit. Portfolio cash uses raw stock returns.",
        ),
        StrategySpec(
            name="portfolio_day1_close_signal_tplus2_open_exit",
            signal_target_col="return_1d_pct",
            actual_return_col="stock_only_return_2d_open_pct_raw",
            exit_datetime_col="exit_2d_open_datetime",
            notes="Current day-1 signal, but hold until T+2 open to isolate exit timing and same-day cash recycling.",
        ),
        StrategySpec(
            name="portfolio_tplus2_open_signal_tplus2_open_exit",
            signal_target_col=CUSTOM_TARGET,
            actual_return_col="stock_only_return_2d_open_pct_raw",
            exit_datetime_col="exit_2d_open_datetime",
            notes="Fully retrained T+2 open signal with T+2 open exit.",
        ),
    ]

    results = []
    for spec in strategies:
        print(f"Running: {spec.name}")
        results.append(run_strategy(base_df, features, spec))

    summary = pd.DataFrame(results).sort_values("end_budget_eur", ascending=False).reset_index(drop=True)
    summary_csv = OUT_DIR / "exit_recycling_strategy_comparison.csv"
    summary_json = OUT_DIR / "exit_recycling_strategy_comparison.json"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nStrategy comparison:")
    for row in results:
        print(
            f"  {row['strategy']}: final EUR {row['end_budget_eur']:,.2f} "
            f"({row['total_return_pct']:+.2f}%), trades={row['trades_executed']}, "
            f"same_day_recycled_entries={row['recycled_entry_count']}"
        )
    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
