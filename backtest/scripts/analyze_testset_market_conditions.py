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
OUT_DIR = BASE / "backtest" / "out"
CACHE_DIR = BASE / "backtest" / "data" / "price_cache"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
WINDOWS: list[tuple[str, str, int]] = [
    ("same_day", "benchmark_same_day_ret_pct", 1),
    ("week", "benchmark_trailing_week_ret_pct", 5),
    ("month", "benchmark_trailing_month_ret_pct", 21),
]
REGIME_ORDER = ["bullish", "bearish", "flat"]


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


def classify_regime(value: float) -> str:
    if pd.isna(value):
        return ""
    if value > 0:
        return "bullish"
    if value < 0:
        return "bearish"
    return "flat"


def load_trade_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "entry_time" not in df.columns or "ret_pct" not in df.columns:
        raise ValueError("Expected test trade log columns 'entry_time' and 'ret_pct'.")

    df["entry_datetime"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["buy_date"] = df["entry_datetime"].dt.date
    df["strategy_ret_pct"] = pd.to_numeric(df["ret_pct"], errors="coerce")
    df["invested_eur"] = pd.to_numeric(df.get("invested_eur"), errors="coerce")
    df["pnl_eur"] = pd.to_numeric(df.get("pnl_eur"), errors="coerce")
    return df.dropna(subset=["buy_date", "strategy_ret_pct"]).copy()


def build_benchmark_frame(client: RESTClient, cache_dir: Path, ticker: str, buy_dates: pd.Series) -> pd.DataFrame:
    min_date = min(buy_dates)
    max_date = max(buy_dates)
    max_lookback_days = max(window_days for _window_label, _feature_col, window_days in WINDOWS)
    bars = fetch_day_bars(
        client=client,
        cache_dir=cache_dir,
        ticker=ticker,
        from_d=min_date - timedelta(days=max_lookback_days * 3),
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

    close = bench["benchmark_close"]
    bench["benchmark_same_day_ret_pct"] = (close / close.shift(1) - 1.0) * 100.0
    bench["benchmark_trailing_week_ret_pct"] = (close / close.shift(5) - 1.0) * 100.0
    bench["benchmark_trailing_month_ret_pct"] = (close / close.shift(21) - 1.0) * 100.0

    for window_label, feature_col, _window_days in WINDOWS:
        bench[f"{window_label}_regime"] = bench[feature_col].apply(classify_regime)

    return bench


def summarize_by_regime(df: pd.DataFrame, level: str) -> pd.DataFrame:
    rows: list[dict] = []
    overall_mean = float(df["strategy_ret_pct"].mean()) if len(df) else float("nan")
    overall_weighted = (
        float((df["pnl_eur"].sum() / df["invested_eur"].sum()) * 100.0)
        if {"pnl_eur", "invested_eur"}.issubset(df.columns) and float(df["invested_eur"].fillna(0).sum()) > 0
        else float("nan")
    )

    for window_label, feature_col, _window_days in WINDOWS:
        regime_col = f"{window_label}_regime"
        for regime in REGIME_ORDER:
            sample = df[df[regime_col] == regime].copy()
            if sample.empty:
                continue
            invested_sum = float(sample["invested_eur"].fillna(0).sum()) if "invested_eur" in sample.columns else float("nan")
            pnl_sum = float(sample["pnl_eur"].fillna(0).sum()) if "pnl_eur" in sample.columns else float("nan")
            weighted_ret = (pnl_sum / invested_sum * 100.0) if invested_sum and pd.notna(invested_sum) else float("nan")
            rows.append(
                {
                    "level": level,
                    "window": window_label,
                    "regime": regime,
                    "n_obs": int(len(sample)),
                    "mean_strategy_ret_pct": float(sample["strategy_ret_pct"].mean()),
                    "median_strategy_ret_pct": float(sample["strategy_ret_pct"].median()),
                    "win_rate_pct": float((sample["strategy_ret_pct"] > 0).mean() * 100.0),
                    "mean_benchmark_ret_pct": float(sample[feature_col].mean()),
                    "mean_return_diff_vs_all_pct": float(sample["strategy_ret_pct"].mean() - overall_mean),
                    "invested_eur": invested_sum,
                    "pnl_eur": pnl_sum,
                    "capital_weighted_ret_pct": weighted_ret,
                    "capital_weighted_diff_vs_all_pct": float(weighted_ret - overall_weighted) if pd.notna(weighted_ret) and pd.notna(overall_weighted) else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def date_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for buy_date, sample in df.groupby("buy_date", sort=True):
        row: dict[str, object] = {
            "buy_date": buy_date,
            "strategy_ret_pct": float(sample["strategy_ret_pct"].mean()),
            "invested_eur": float(sample["invested_eur"].fillna(0).sum()),
            "pnl_eur": float(sample["pnl_eur"].fillna(0).sum()),
            "n_obs": int(len(sample)),
        }
        for window_label, feature_col, _window_days in WINDOWS:
            row[feature_col] = float(sample[feature_col].iloc[0])
            row[f"{window_label}_regime"] = str(sample[f"{window_label}_regime"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, benchmark: str) -> None:
    trade_summary = summary[summary["level"] == "trade"].copy()
    if trade_summary.empty:
        print("No trade-level summary rows generated.")
        return

    print(f"benchmark={benchmark}")
    for window_label, _feature_col, _window_days in WINDOWS:
        print(f"{window_label}:")
        window_rows = trade_summary[trade_summary["window"] == window_label].copy()
        for regime in REGIME_ORDER:
            sample = window_rows[window_rows["regime"] == regime]
            if sample.empty:
                continue
            row = sample.iloc[0]
            print(
                f"  {regime}: n={int(row['n_obs'])} "
                f"mean_ret={float(row['mean_strategy_ret_pct']):.4f}% "
                f"win_rate={float(row['win_rate_pct']):.1f}% "
                f"weighted_ret={float(row['capital_weighted_ret_pct']):.4f}% "
                f"benchmark_mean={float(row['mean_benchmark_ret_pct']):.4f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze backtest test-set trade returns by same-day, week, and month market regime."
    )
    parser.add_argument("--input", default=str(OUT_DIR / "testset_trade_log.csv"))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--detail-out", default=str(OUT_DIR / "testset_trade_log_market_regime_detail.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "testset_trade_log_market_regime_summary.csv"))
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
    feature_columns = [feature_col for _window_label, feature_col, _window_days in WINDOWS]
    merged = merged.dropna(subset=feature_columns).copy()

    trade_summary = summarize_by_regime(merged, level="trade")
    date_summary = summarize_by_regime(date_level_frame(merged), level="date")
    summary = pd.concat([trade_summary, date_summary], ignore_index=True)

    detail_out = Path(args.detail_out)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["buy_date", "ticker", "entry_time"]).to_csv(detail_out, index=False)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["level", "window", "regime"]).to_csv(summary_out, index=False)

    print_summary(summary, benchmark=args.benchmark)
    print(f"detail_out={detail_out}")
    print(f"summary_out={summary_out}")


if __name__ == "__main__":
    main()
