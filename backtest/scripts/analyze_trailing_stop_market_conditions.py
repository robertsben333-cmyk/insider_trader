from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient


BASE = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUT_DIR = BASE / "backtest" / "out"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
FEATURES = [
    "benchmark_same_day_ret_pct",
    "benchmark_prev_day_ret_pct",
    "benchmark_prev_5d_ret_pct",
]


def _json_load(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _json_save(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _aggs_to_dicts(aggs: Iterable[object]) -> list[dict]:
    return [
        {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
        for a in aggs
        if getattr(a, "timestamp", None) is not None and getattr(a, "close", None) is not None
    ]


def day_cache_path(cache_dir: Path, ticker: str, from_d: date, to_d: date) -> Path:
    return cache_dir / f"{ticker}_lkbk_{from_d:%Y-%m-%d}_{to_d:%Y-%m-%d}.json"


def fetch_day_bars(client: RESTClient, cache_dir: Path, ticker: str, from_d: date, to_d: date) -> list[dict]:
    path = day_cache_path(cache_dir, ticker, from_d, to_d)
    cached = _json_load(path)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=from_d,
            to=to_d,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
        bars = _aggs_to_dicts(aggs)
    except Exception:
        bars = []
    _json_save(path, bars)
    return bars


def bar_date_et(bar: dict) -> date | None:
    ts = bar.get("t")
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts) / 1000, tz=UTC).astimezone(ET).date()


def load_trade_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "split",
            "ticker",
            "buy_datetime",
            "baseline_ret_pct",
            "selected_candidate",
            "selected_strategy_ret_pct",
            "selected_improvement_pct",
            "selected_stopped",
        ],
    )
    df = df[df["selected_candidate"].fillna("").ne("none")].copy()
    df["buy_datetime"] = pd.to_datetime(df["buy_datetime"], errors="coerce")
    df["buy_date"] = df["buy_datetime"].dt.date
    df["baseline_ret_pct"] = pd.to_numeric(df["baseline_ret_pct"], errors="coerce")
    df["selected_strategy_ret_pct"] = pd.to_numeric(df["selected_strategy_ret_pct"], errors="coerce")
    df["selected_improvement_pct"] = pd.to_numeric(df["selected_improvement_pct"], errors="coerce")
    df["selected_stopped"] = df["selected_stopped"].fillna(False).astype(bool)
    df = df.dropna(subset=["buy_date", "selected_improvement_pct"]).copy()
    return df


def build_benchmark_frame(client: RESTClient, cache_dir: Path, ticker: str, buy_dates: pd.Series) -> pd.DataFrame:
    min_date = min(buy_dates)
    max_date = max(buy_dates)
    bars = fetch_day_bars(
        client=client,
        cache_dir=cache_dir,
        ticker=ticker,
        from_d=min_date - timedelta(days=20),
        to_d=max_date,
    )
    rows: list[dict] = []
    for bar in bars:
        bar_d = bar_date_et(bar)
        close_px = bar.get("c")
        if bar_d is None or close_px is None:
            continue
        rows.append({"buy_date": bar_d, "benchmark_close": float(close_px)})
    bench = pd.DataFrame(rows).drop_duplicates(subset=["buy_date"]).sort_values("buy_date").reset_index(drop=True)
    if bench.empty:
        raise RuntimeError(f"No benchmark bars found for {ticker}.")

    bench["benchmark_same_day_ret_pct"] = (bench["benchmark_close"] / bench["benchmark_close"].shift(1) - 1.0) * 100.0
    bench["benchmark_prev_day_ret_pct"] = (bench["benchmark_close"].shift(1) / bench["benchmark_close"].shift(2) - 1.0) * 100.0
    bench["benchmark_prev_5d_ret_pct"] = (bench["benchmark_close"].shift(1) / bench["benchmark_close"].shift(6) - 1.0) * 100.0
    return bench


def correlation_rows(df: pd.DataFrame, scope: str, level: str, target_col: str) -> list[dict]:
    rows: list[dict] = []
    for feature in FEATURES:
        sample = df[[feature, target_col]].dropna()
        rows.append(
            {
                "section": "correlation",
                "scope": scope,
                "level": level,
                "feature": feature,
                "n_obs": int(len(sample)),
                "pearson_corr": sample[target_col].corr(sample[feature], method="pearson"),
                "spearman_corr": sample[target_col].corr(sample[feature], method="spearman"),
                "mean_target_pct": sample[target_col].mean(),
            }
        )
    return rows


def sign_bucket_rows(df: pd.DataFrame, scope: str, level: str, target_col: str, stop_col: str) -> list[dict]:
    rows: list[dict] = []
    for feature in FEATURES:
        for label, mask in {
            "positive": df[feature] > 0,
            "non_positive": df[feature] <= 0,
        }.items():
            sample = df.loc[mask & df[feature].notna() & df[target_col].notna()].copy()
            rows.append(
                {
                    "section": "sign_bucket",
                    "scope": scope,
                    "level": level,
                    "feature": feature,
                    "bucket": label,
                    "n_obs": int(len(sample)),
                    "mean_improvement_pct": sample[target_col].mean(),
                    "median_improvement_pct": sample[target_col].median(),
                    "stop_rate_pct": sample[stop_col].mean() * 100.0 if len(sample) else float("nan"),
                    "baseline_mean_ret_pct": sample["baseline_ret_pct"].mean(),
                }
            )
    return rows


def date_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "selected_improvement_pct": "mean",
        "selected_stopped": "mean",
        "baseline_ret_pct": "mean",
    }
    for feature in FEATURES:
        aggregations[feature] = "first"
    out = df.groupby("buy_date", as_index=False).agg(aggregations)
    out = out.rename(
        columns={
            "selected_improvement_pct": "date_mean_improvement_pct",
            "selected_stopped": "date_stop_rate",
            "baseline_ret_pct": "date_baseline_mean_ret_pct",
        }
    )
    out["date_stop_rate_pct"] = out["date_stop_rate"] * 100.0
    return out


def summarize_scope(df: pd.DataFrame, scope: str) -> tuple[list[dict], pd.DataFrame]:
    summary_rows = []
    summary_rows.extend(correlation_rows(df, scope=scope, level="trade", target_col="selected_improvement_pct"))
    summary_rows.extend(
        sign_bucket_rows(
            df,
            scope=scope,
            level="trade",
            target_col="selected_improvement_pct",
            stop_col="selected_stopped",
        )
    )

    date_df = date_level_frame(df)
    summary_rows.extend(
        correlation_rows(
            date_df,
            scope=scope,
            level="date",
            target_col="date_mean_improvement_pct",
        )
    )
    summary_rows.extend(
        sign_bucket_rows(
            date_df.rename(
                columns={
                    "date_mean_improvement_pct": "selected_improvement_pct",
                    "date_stop_rate": "selected_stopped",
                    "date_baseline_mean_ret_pct": "baseline_ret_pct",
                }
            ),
            scope=scope,
            level="date",
            target_col="selected_improvement_pct",
            stop_col="selected_stopped",
        )
    )
    date_df["scope"] = scope
    return summary_rows, date_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate trailing-stop usefulness with broad market conditions in old backtest data."
    )
    parser.add_argument("--input", default=str(OUT_DIR / "all_companies_dynamic_path_exit_detail.csv"))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--trade-out", default=str(OUT_DIR / "trailing_stop_market_condition_trade_detail.csv"))
    parser.add_argument("--date-out", default=str(OUT_DIR / "trailing_stop_market_condition_date_summary.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "trailing_stop_market_condition_summary.csv"))
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    trades = load_trade_data(Path(args.input))
    client = RESTClient(api_key=api_key, retries=3)
    benchmark = build_benchmark_frame(
        client=client,
        cache_dir=Path(args.cache_dir),
        ticker=args.benchmark,
        buy_dates=trades["buy_date"],
    )

    merged = trades.merge(benchmark, on="buy_date", how="left")
    merged = merged.dropna(subset=FEATURES).copy()

    summary_rows: list[dict] = []
    date_frames: list[pd.DataFrame] = []

    for scope_name, scope_df in {
        "all": merged,
        "val": merged[merged["split"] == "val"].copy(),
        "test": merged[merged["split"] == "test"].copy(),
    }.items():
        if scope_df.empty:
            continue
        rows, date_df = summarize_scope(scope_df, scope=scope_name)
        summary_rows.extend(rows)
        date_frames.append(date_df)

    trade_out = Path(args.trade_out)
    trade_out.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["buy_datetime", "ticker"]).to_csv(trade_out, index=False)

    date_out = Path(args.date_out)
    pd.concat(date_frames, ignore_index=True).sort_values(["scope", "buy_date"]).to_csv(date_out, index=False)

    summary_out = Path(args.summary_out)
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)

    test_date = pd.concat(date_frames, ignore_index=True)
    test_date = test_date[test_date["scope"] == "test"].copy()
    test_trade = merged[merged["split"] == "test"].copy()

    for feature in FEATURES:
        trade_corr = test_trade["selected_improvement_pct"].corr(test_trade[feature], method="pearson")
        date_corr = test_date["date_mean_improvement_pct"].corr(test_date[feature], method="pearson")
        print(
            f"test {feature}: trade_pearson={trade_corr:.4f} "
            f"date_pearson={date_corr:.4f}"
        )
    print(f"trade_out={trade_out}")
    print(f"date_out={date_out}")
    print(f"summary_out={summary_out}")


if __name__ == "__main__":
    main()
