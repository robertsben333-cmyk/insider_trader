from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
os.chdir(BASE)

from research.scripts.deploy_tplus2_open_day1_live import prepare_split
from train_models import CACHE_DIR, _bar_date_et, fetch_day_bars


OUTPUT_CSV = BASE / "research" / "outcomes" / "models" / "testset_open_gap_predictor_detail.csv"


def build_test_frame() -> pd.DataFrame:
    split = prepare_split()
    test_start = len(split.X_tr) + len(split.X_va)
    test_df = split.sub.iloc[test_start:].copy().reset_index(drop=True)

    test_df["trade_date"] = pd.to_datetime(test_df["trade_date"], errors="coerce")
    test_df["buy_datetime"] = pd.to_datetime(test_df["buy_datetime"], errors="coerce")
    test_df["buy_price"] = pd.to_numeric(test_df["buy_price"], errors="coerce")
    test_df["return_2d_open_pct"] = pd.to_numeric(test_df["return_2d_open_pct"], errors="coerce")
    test_df["stock_only_return_2d_open_pct_raw"] = pd.to_numeric(
        test_df["stock_only_return_2d_open_pct_raw"], errors="coerce"
    )
    test_df["spy_return_2d_open_pct_raw"] = pd.to_numeric(
        test_df["spy_return_2d_open_pct_raw"], errors="coerce"
    )
    test_df = test_df.dropna(
        subset=["ticker", "trade_date", "buy_datetime", "buy_price", "return_2d_open_pct", "stock_only_return_2d_open_pct_raw"]
    ).copy()

    cache_dir = Path(CACHE_DIR)
    rows: list[dict] = []
    for ticker, grp in test_df.groupby("ticker", sort=True):
        min_day = grp["trade_date"].min().date()
        max_day = grp["trade_date"].max().date()
        bars = fetch_day_bars(cache_dir, str(ticker), min_day - pd.Timedelta(days=7), max_day + pd.Timedelta(days=7))
        dated = [(_bar_date_et(bar), bar) for bar in bars]
        dated = [(d, bar) for d, bar in dated if d is not None]
        if len(dated) < 2:
            continue
        trading_days = [d for d, _ in dated]
        by_day = {d: bar for d, bar in dated}

        for _, row in grp.iterrows():
            trade_day = row["trade_date"].date()
            if trade_day not in by_day:
                continue
            idx = trading_days.index(trade_day)
            if idx == 0:
                continue
            prev_day = trading_days[idx - 1]
            prev_bar = by_day[prev_day]
            trade_bar = by_day[trade_day]
            prev_close = prev_bar.get("c")
            trade_open = trade_bar.get("o")
            if prev_close is None or trade_open is None:
                continue
            prev_close = float(prev_close)
            trade_open = float(trade_open)
            if prev_close <= 0 or trade_open <= 0:
                continue
            rows.append(
                {
                    "ticker": row["ticker"],
                    "trade_date": trade_day.isoformat(),
                    "transaction_date": row.get("transaction_date"),
                    "buy_datetime": row["buy_datetime"],
                    "buy_price": float(row["buy_price"]),
                    "prev_close_day": prev_day.isoformat(),
                    "prev_close": prev_close,
                    "trade_open": trade_open,
                    "overnight_gap_pct": ((trade_open / prev_close) - 1.0) * 100.0,
                    "buy_open_slippage_pct": ((float(row["buy_price"]) / trade_open) - 1.0) * 100.0,
                    "stock_only_return_2d_open_pct_raw": float(row["stock_only_return_2d_open_pct_raw"]),
                    "spy_return_2d_open_pct_raw": float(row["spy_return_2d_open_pct_raw"])
                    if pd.notna(row.get("spy_return_2d_open_pct_raw"))
                    else None,
                    "return_2d_open_pct": float(row["return_2d_open_pct"]),
                }
            )

    return pd.DataFrame(rows).sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def print_bucket_stats(frame: pd.DataFrame, target_col: str, label: str) -> None:
    work = frame.dropna(subset=["overnight_gap_pct", target_col]).copy()
    pearson = work["overnight_gap_pct"].corr(work[target_col], method="pearson")
    spearman = work["overnight_gap_pct"].corr(work[target_col], method="spearman")
    print(f"{label} rows:      {len(work)}")
    print(f"{label} pearson:  {pearson:.4f}")
    print(f"{label} spearman: {spearman:.4f}")
    split = (
        work.assign(gap_bucket=work["overnight_gap_pct"].apply(lambda x: "gap_down_or_flat" if x <= 0 else "gap_up"))
        .groupby("gap_bucket")
        .agg(
            trades=("ticker", "count"),
            avg_gap_pct=("overnight_gap_pct", "mean"),
            avg_target=("overnight_gap_pct", lambda s: 0.0),
        )
        .reset_index()
    )
    split["avg_target"] = (
        work.assign(gap_bucket=work["overnight_gap_pct"].apply(lambda x: "gap_down_or_flat" if x <= 0 else "gap_up"))
        .groupby("gap_bucket")[target_col]
        .mean()
        .values
    )
    split["median_target"] = (
        work.assign(gap_bucket=work["overnight_gap_pct"].apply(lambda x: "gap_down_or_flat" if x <= 0 else "gap_up"))
        .groupby("gap_bucket")[target_col]
        .median()
        .values
    )
    print(split.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("")


def print_top_examples(frame: pd.DataFrame, target_col: str) -> None:
    cols = ["ticker", "trade_date", "overnight_gap_pct", target_col, "buy_open_slippage_pct"]
    ordered = frame.sort_values("overnight_gap_pct")
    print("Lowest overnight gaps:")
    print(ordered[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("")
    print("Highest overnight gaps:")
    print(ordered[cols].tail(10).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate close-to-open gap predictor on the exact T+2-open test set.")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    load_dotenv(BASE / ".env")
    frame = build_test_frame()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    print(f"Test rows analyzed: {len(frame)}")
    print(f"Test period: {frame['trade_date'].min()} -> {frame['trade_date'].max()}")
    print("")
    print_bucket_stats(frame, "stock_only_return_2d_open_pct_raw", "Stock-only T+2 open return")
    print_bucket_stats(frame, "return_2d_open_pct", "SPY-adjusted T+2 open excess return")
    print_top_examples(frame, "stock_only_return_2d_open_pct_raw")
    print("")
    print(f"Saved detail to: {args.output_csv}")


if __name__ == "__main__":
    main()
