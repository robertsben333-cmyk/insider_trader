from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import train_models
from backtest.scripts.evaluate_investable_risk_rules import chrono_split_60_20_20, load_scored_day1


OUT_DIR = BASE / "backtest" / "out"
FAST_SCORED_CSV = BASE / "backtest" / "data" / "full_scored.csv"
AGG_CSV = BASE / "backtest" / "data" / "backtest_results_aggregated_unfiltered.csv"
STEP = 0.005
THRESHOLDS = np.round(np.arange(STEP, 1.0 + STEP / 2, STEP), 3)


def load_scored_dataset() -> pd.DataFrame:
    if FAST_SCORED_CSV.exists() and AGG_CSV.exists():
        scored = pd.read_csv(FAST_SCORED_CSV)
        agg = pd.read_csv(
            AGG_CSV,
            usecols=["ticker", "transaction_date", "trade_date", "buy_datetime", "return_1d_pct"],
        )
        scored["transaction_date"] = scored["transaction_date"].astype(str)
        agg["transaction_date"] = agg["transaction_date"].astype(str)
        merged = scored.merge(agg, on=["ticker", "transaction_date"], how="left")
        merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
        merged["buy_datetime"] = pd.to_datetime(merged["buy_datetime"], errors="coerce")
        merged["return_1d_pct"] = pd.to_numeric(merged["return_1d_pct"], errors="coerce")
        merged["pred_mean4"] = pd.to_numeric(merged["pred_mean4"], errors="coerce")
        merged = merged.dropna(subset=["trade_date", "buy_datetime", "return_1d_pct", "pred_mean4"]).copy()
        if not merged.empty:
            return merged.sort_values("trade_date").reset_index(drop=True)
    return load_scored_day1().sort_values("trade_date").reset_index(drop=True)


def compute_open_to_open_returns(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["ticker", "trade_date", "pred_mean4"]).copy()

    entry_open = pd.Series(np.nan, index=out.index, dtype=float)
    exit_open = pd.Series(np.nan, index=out.index, dtype=float)
    entry_date = pd.Series(pd.NaT, index=out.index, dtype="object")
    exit_date = pd.Series(pd.NaT, index=out.index, dtype="object")

    for ticker, grp in out.groupby("ticker", sort=True):
        min_trade = grp["trade_date"].min().date()
        max_trade = grp["trade_date"].max().date()
        bars = train_models.fetch_day_bars(cache_dir, str(ticker), min_trade, max_trade + pd.Timedelta(days=10))
        dated = [(train_models._bar_date_et(bar), bar) for bar in bars]
        dated = [(d, bar) for d, bar in dated if d is not None]
        if len(dated) < 3:
            continue

        trading_days = [d for d, _ in dated]
        by_day = {d: bar for d, bar in dated}
        day_to_idx = {d: idx for idx, d in enumerate(trading_days)}

        for row_idx, row in grp.iterrows():
            trade_day = row["trade_date"].date()
            idx = day_to_idx.get(trade_day)
            if idx is None:
                continue
            if idx + 2 >= len(trading_days):
                continue

            ent_day = trading_days[idx + 1]
            ex_day = trading_days[idx + 2]
            ent_open = by_day[ent_day].get("o")
            ex_open = by_day[ex_day].get("o")
            if ent_open is None or ex_open is None:
                continue

            ent_open = float(ent_open)
            ex_open = float(ex_open)
            if not np.isfinite(ent_open) or not np.isfinite(ex_open) or ent_open <= 0 or ex_open <= 0:
                continue

            entry_open.loc[row_idx] = ent_open
            exit_open.loc[row_idx] = ex_open
            entry_date.loc[row_idx] = ent_day
            exit_date.loc[row_idx] = ex_day

    out["entry_tplus1_open_date"] = entry_date
    out["exit_tplus2_open_date"] = exit_date
    out["entry_tplus1_open_price"] = entry_open
    out["exit_tplus2_open_price"] = exit_open
    out["trade_date_open_to_open_ret_pct"] = ((exit_open / entry_open) - 1.0) * 100.0
    return out.dropna(subset=["trade_date_open_to_open_ret_pct"]).copy()


def build_test_deciles(test: pd.DataFrame) -> pd.DataFrame:
    work = test.dropna(subset=["pred_mean4", "trade_date_open_to_open_ret_pct"]).copy()
    work = work.sort_values("pred_mean4").reset_index(drop=True)
    order = np.arange(len(work))
    bin_size = max(1, len(work) // 10)

    rows: list[dict] = []
    for decile in range(10):
        start = decile * bin_size
        end = len(work) if decile == 9 else min(len(work), (decile + 1) * bin_size)
        bucket = work.iloc[start:end].copy()
        if bucket.empty:
            continue
        returns = pd.to_numeric(bucket["trade_date_open_to_open_ret_pct"], errors="coerce").dropna()
        rows.append(
            {
                "decile": decile + 1,
                "n_trades": int(len(bucket)),
                "mean_pred": float(bucket["pred_mean4"].mean()),
                "median_pred": float(bucket["pred_mean4"].median()),
                "mean_return_pct": float(returns.mean()),
                "median_return_pct": float(returns.median()),
                "win_rate_pct": float((returns > 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_sweep(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    pred_train = pd.to_numeric(train["pred_mean4"], errors="coerce").dropna().to_numpy(dtype=float)
    pred_train_sorted = np.sort(pred_train)
    rows: list[dict] = []

    for threshold in THRESHOLDS:
        raw_cutoff = float(np.quantile(pred_train_sorted, float(threshold)))
        picks = test[pd.to_numeric(test["pred_mean4"], errors="coerce") > raw_cutoff].copy()
        returns = pd.to_numeric(picks["trade_date_open_to_open_ret_pct"], errors="coerce").dropna()
        rows.append(
            {
                "decile_score_threshold": float(threshold),
                "raw_pred_mean4_cutoff": raw_cutoff,
                "n_trades": int(len(returns)),
                "mean_return_pct": float(returns.mean()) if not returns.empty else np.nan,
                "median_return_pct": float(returns.median()) if not returns.empty else np.nan,
                "win_rate_pct": float((returns > 0).mean() * 100.0) if not returns.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, float | int | str]:
    returns = pd.to_numeric(frame["trade_date_open_to_open_ret_pct"], errors="coerce").dropna()
    return {
        "n_trades": int(len(returns)),
        "mean_return_pct": float(returns.mean()),
        "median_return_pct": float(returns.median()),
        "win_rate_pct": float((returns > 0).mean() * 100.0),
        "start_trade_date": str(pd.to_datetime(frame["trade_date"]).min().date()),
        "end_trade_date": str(pd.to_datetime(frame["trade_date"]).max().date()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze test-split returns for buy open on T+1 after trade_date and sell open on T+2."
    )
    parser.add_argument("--cache-dir", default="backtest/data/price_cache")
    parser.add_argument(
        "--detail-out",
        default=str(OUT_DIR / "test_trade_date_tplus1_open_tplus2_open_detail.csv"),
    )
    parser.add_argument(
        "--deciles-out",
        default=str(OUT_DIR / "test_trade_date_tplus1_open_tplus2_open_by_decile.csv"),
    )
    parser.add_argument(
        "--sweep-out",
        default=str(OUT_DIR / "test_trade_date_tplus1_open_tplus2_open_threshold_sweep.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored = load_scored_dataset()
    n_train, split_80 = chrono_split_60_20_20(len(scored))
    train = scored.iloc[:n_train].copy()
    test = scored.iloc[split_80:].copy()

    test = compute_open_to_open_returns(test, Path(args.cache_dir))
    detail_summary = summarize(test)
    deciles = build_test_deciles(test)
    sweep = build_threshold_sweep(train, test)

    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(args.detail_out, index=False)
    deciles.to_csv(args.deciles_out, index=False)
    sweep.to_csv(args.sweep_out, index=False)

    print("Test split summary")
    for key, value in detail_summary.items():
        print(f"{key}: {value}")
    print("")
    print("Top deciles by mean return")
    print(deciles.sort_values("mean_return_pct", ascending=False).head(5).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("")
    print("Selected threshold rows")
    focus = sweep[sweep["decile_score_threshold"].isin([0.8, 0.85, 0.9, 0.93, 0.95, 0.98])].copy()
    print(focus.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("")
    print(f"detail_out={args.detail_out}")
    print(f"deciles_out={args.deciles_out}")
    print(f"sweep_out={args.sweep_out}")


if __name__ == "__main__":
    main()
