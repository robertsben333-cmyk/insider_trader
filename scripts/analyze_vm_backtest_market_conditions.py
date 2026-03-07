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

from live_trading.strategy_settings import LIVE_PATHS


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


def resolve_return_column(columns: list[str], requested: str | None) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Requested return column {requested!r} not found in input.")
        return requested

    preferred = ["ret_2d_open_pct", "ret_2d_pct", "ret_1d_pct", "selected_strategy_ret_pct"]
    for name in preferred:
        if name in columns:
            return name

    candidates = sorted(col for col in columns if col.startswith("ret_") and col.endswith("_pct"))
    if candidates:
        return candidates[0]

    raise ValueError("Could not infer a strategy return column from input.")


def resolve_entry_column(columns: list[str]) -> str:
    for name in ("buy_datetime_et", "entry_dt_et", "buy_datetime", "entry_datetime"):
        if name in columns:
            return name
    raise ValueError("Could not infer an entry datetime column from input.")


def classify_regime(value: float) -> str:
    if pd.isna(value):
        return ""
    if value > 0:
        return "bullish"
    if value < 0:
        return "bearish"
    return "flat"


def load_trade_data(path: Path, requested_return_col: str | None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    return_col = resolve_return_column(list(df.columns), requested_return_col)
    entry_col = resolve_entry_column(list(df.columns))

    df["entry_datetime"] = pd.to_datetime(df[entry_col], errors="coerce")
    df["buy_date"] = df["entry_datetime"].dt.date
    df["strategy_ret_pct"] = pd.to_numeric(df[return_col], errors="coerce")

    numeric_candidates = [
        "score_1d",
        "score_3d",
        "estimated_decile_score",
        "entry_price",
        "exit_open_price",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df.dropna(subset=["buy_date", "strategy_ret_pct"]).copy()
    out["source_return_col"] = return_col
    return out, return_col


def build_benchmark_frame(client: RESTClient, cache_dir: Path, ticker: str, buy_dates: pd.Series) -> pd.DataFrame:
    min_date = min(buy_dates)
    max_date = max(buy_dates)
    max_lookback_days = max(window_days for _label, _col, window_days in WINDOWS)
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


def date_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, str] = {"strategy_ret_pct": "mean", "ticker": "count"}
    for window_label, feature_col, _window_days in WINDOWS:
        aggregations[feature_col] = "first"
        aggregations[f"{window_label}_regime"] = "first"

    out = df.groupby("buy_date", as_index=False).agg(aggregations)
    out = out.rename(columns={"strategy_ret_pct": "strategy_ret_pct", "ticker": "n_trades"})
    return out


def summarize_by_regime(df: pd.DataFrame, level: str) -> pd.DataFrame:
    rows: list[dict] = []
    overall_mean = float(df["strategy_ret_pct"].mean()) if len(df) else float("nan")

    for window_label, feature_col, _window_days in WINDOWS:
        regime_col = f"{window_label}_regime"
        for regime in REGIME_ORDER:
            sample = df[df[regime_col] == regime].copy()
            if sample.empty:
                continue
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
                }
            )

    return pd.DataFrame(rows)


def output_paths(vm_sync_dir: Path) -> tuple[Path, Path]:
    detail_out = vm_sync_dir / "historical_recommended_backtest_market_regime_detail.csv"
    summary_out = vm_sync_dir / "historical_recommended_backtest_market_regime_summary.csv"
    return detail_out, summary_out


def print_trade_level_summary(summary: pd.DataFrame, benchmark: str, return_col: str) -> None:
    trade_summary = summary[summary["level"] == "trade"].copy()
    if trade_summary.empty:
        print("No trade-level market regime summary rows generated.")
        return

    print(f"benchmark={benchmark}")
    print(f"return_col={return_col}")
    for window_label, _feature_col, _window_days in WINDOWS:
        window_rows = trade_summary[trade_summary["window"] == window_label].copy()
        if window_rows.empty:
            continue
        print(f"{window_label}:")
        for regime in REGIME_ORDER:
            sample = window_rows[window_rows["regime"] == regime]
            if sample.empty:
                continue
            row = sample.iloc[0]
            print(
                f"  {regime}: n={int(row['n_obs'])} "
                f"mean_ret={float(row['mean_strategy_ret_pct']):.4f}% "
                f"win_rate={float(row['win_rate_pct']):.1f}% "
                f"benchmark_mean={float(row['mean_benchmark_ret_pct']):.4f}%"
            )


def main() -> None:
    default_detail_out, default_summary_out = output_paths(Path(LIVE_PATHS.vm_sync_dir))

    parser = argparse.ArgumentParser(
        description="Segment VM historical backtest returns by same-day, week, and month market conditions."
    )
    parser.add_argument("--input", default=LIVE_PATHS.vm_backtest_detail_file)
    parser.add_argument("--cache-dir", default=LIVE_PATHS.cache_dir)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--return-col", default=None)
    parser.add_argument("--detail-out", default=str(default_detail_out))
    parser.add_argument("--summary-out", default=str(default_summary_out))
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    trades, return_col = load_trade_data(Path(args.input), args.return_col)
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
    merged.sort_values(["buy_date", "ticker", "event_key"], ascending=[True, True, True]).to_csv(detail_out, index=False)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["level", "window", "regime"]).to_csv(summary_out, index=False)

    print_trade_level_summary(summary, benchmark=args.benchmark, return_col=return_col)
    print(f"detail_out={detail_out}")
    print(f"summary_out={summary_out}")


if __name__ == "__main__":
    main()
