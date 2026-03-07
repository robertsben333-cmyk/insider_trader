from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "backtest" / "out"
CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
SWEEP_CSV = OUT_DIR / "investable_decile_score_sweep_0005.csv"

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

if str(BASE) not in os.sys.path:
    os.sys.path.insert(0, str(BASE))

from backtest.scripts.evaluate_investable_risk_rules import (  # noqa: E402
    chrono_split_60_20_20,
    load_scored_day1,
    simulate_budget_curve_from_picks,
)


class PriceCache:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, ticker: str, kind: str, f: str, t: str) -> Path:
        return self.dir / f"{ticker}_{kind}_{f}_{t}.json"

    def get(self, ticker: str, kind: str, f: str, t: str) -> Optional[list]:
        p = self._path(ticker, kind, f, t)
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def put(self, ticker: str, kind: str, f: str, t: str, data: list) -> None:
        p = self._path(ticker, kind, f, t)
        with self._lock:
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(data, fh)


def fetch_minute_bars(client: RESTClient, cache: PriceCache, ticker: str, d: date) -> list:
    ds = d.strftime("%Y-%m-%d")
    cached = cache.get(ticker, "min", ds, ds)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=d,
            to=d,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
        bars = [
            {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
            for a in aggs
            if a.timestamp and a.close
        ]
    except Exception:
        bars = []
    cache.put(ticker, "min", ds, ds, bars)
    return bars


def bar_dt_et(bar: dict) -> datetime:
    return datetime.fromtimestamp(int(bar["t"]) / 1000, tz=ZoneInfo("UTC")).astimezone(ET)


def is_regular_session_bar(bar: dict) -> bool:
    dt = bar_dt_et(bar)
    return dt.weekday() < 5 and MARKET_OPEN <= dt.time() < MARKET_CLOSE


def trading_days_between(start: date, end: date) -> list[date]:
    out = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


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
        cur_date = bar_dt_et(minute_bars[idx]).date()
        if cur_date != entry_date:
            break
        last_idx = idx
    return last_idx


def find_next_day_open_index(minute_bars: list[dict], entry_idx: int) -> int | None:
    entry_date = bar_dt_et(minute_bars[entry_idx]).date()
    for idx in range(entry_idx + 1, len(minute_bars)):
        cur_dt = bar_dt_et(minute_bars[idx])
        if cur_dt.date() > entry_date:
            return idx
    return None


def compute_signal_detail(
    client: RESTClient,
    cache: PriceCache,
    row: pd.Series,
    checkpoints: list[int],
) -> dict:
    ticker = str(row["ticker"])
    entry_dt = pd.Timestamp(row["entry_time"]).to_pydatetime().replace(tzinfo=ET)
    exit_dt = pd.Timestamp(row["exit_time"]).to_pydatetime().replace(tzinfo=ET)
    invested_eur = float(row["invested_eur"])
    baseline_ret = float(row["ret_pct"])

    all_bars: list[dict] = []
    for d in trading_days_between(entry_dt.date(), exit_dt.date()):
        all_bars.extend(fetch_minute_bars(client, cache, ticker, d))
    minute_bars = [b for b in all_bars if is_regular_session_bar(b)]
    minute_bars.sort(key=lambda b: int(b["t"]))

    out = {
        "split": row["split"],
        "ticker": ticker,
        "entry_time": pd.Timestamp(row["entry_time"]).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time": pd.Timestamp(row["exit_time"]).strftime("%Y-%m-%d %H:%M:%S"),
        "invested_eur": invested_eur,
        "baseline_ret_pct": baseline_ret,
        "signal_available": 0,
    }

    if not minute_bars:
        return out

    entry_idx = find_entry_bar_index(minute_bars, entry_dt)
    if entry_idx is None:
        return out

    entry_price = float(minute_bars[entry_idx]["o"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return out

    out["signal_available"] = 1
    out["entry_bar_time"] = bar_dt_et(minute_bars[entry_idx]).strftime("%Y-%m-%d %H:%M:%S")
    out["entry_price"] = entry_price

    same_day_close_idx = find_same_day_close_index(minute_bars, entry_idx)
    next_day_open_idx = find_next_day_open_index(minute_bars, entry_idx)

    for checkpoint in checkpoints:
        ret_col = f"ret_{checkpoint}m_pct"
        dt_col = f"exit_{checkpoint}m_dt"
        idx = entry_idx + checkpoint
        if idx < len(minute_bars):
            px = float(minute_bars[idx]["c"])
            out[ret_col] = (px / entry_price - 1.0) * 100.0
            out[dt_col] = bar_dt_et(minute_bars[idx]).strftime("%Y-%m-%d %H:%M:%S")
        else:
            out[ret_col] = np.nan
            out[dt_col] = ""

    if same_day_close_idx is not None:
        px = float(minute_bars[same_day_close_idx]["c"])
        out["ret_same_day_close_pct"] = (px / entry_price - 1.0) * 100.0
        out["exit_same_day_close_dt"] = bar_dt_et(minute_bars[same_day_close_idx]).strftime("%Y-%m-%d %H:%M:%S")
    else:
        out["ret_same_day_close_pct"] = np.nan
        out["exit_same_day_close_dt"] = ""

    if next_day_open_idx is not None:
        px = float(minute_bars[next_day_open_idx]["o"])
        out["ret_next_day_open_pct"] = (px / entry_price - 1.0) * 100.0
        out["exit_next_day_open_dt"] = bar_dt_et(minute_bars[next_day_open_idx]).strftime("%Y-%m-%d %H:%M:%S")
    else:
        out["ret_next_day_open_pct"] = np.nan
        out["exit_next_day_open_dt"] = ""

    return out


def select_threshold_from_validation(sweep_csv: Path) -> float:
    sweep = pd.read_csv(sweep_csv)
    best = sweep.sort_values("val_objective_val", ascending=False).iloc[0]
    return float(best["raw_pred_mean4_cutoff"])


def summarize_rule(detail: pd.DataFrame, signal_col: str, threshold: float) -> dict:
    eligible = detail[detail[signal_col].notna()].copy()
    if eligible.empty:
        return {}

    flagged_mask = pd.to_numeric(eligible[signal_col], errors="coerce") <= float(threshold)
    strategy_ret = np.where(flagged_mask, eligible[signal_col].to_numpy(dtype=float), eligible["baseline_ret_pct"].to_numpy(dtype=float))
    baseline_ret = eligible["baseline_ret_pct"].to_numpy(dtype=float)
    invested = eligible["invested_eur"].to_numpy(dtype=float)

    baseline_pnl = invested * baseline_ret / 100.0
    strategy_pnl = invested * strategy_ret / 100.0
    final_is_loser = baseline_ret < 0
    flagged = flagged_mask.to_numpy(dtype=bool)

    flagged_count = int(flagged.sum())
    flagged_losers = int(np.logical_and(flagged, final_is_loser).sum())
    total_losers = int(final_is_loser.sum())

    return {
        "signal": signal_col,
        "threshold_pct": float(threshold),
        "n_trades": int(len(eligible)),
        "n_flagged": flagged_count,
        "flag_rate_pct": float(flagged_count / len(eligible) * 100.0),
        "loser_precision_pct": float(flagged_losers / flagged_count * 100.0) if flagged_count else np.nan,
        "loser_recall_pct": float(flagged_losers / total_losers * 100.0) if total_losers else np.nan,
        "baseline_mean_ret_pct": float(np.mean(baseline_ret)),
        "strategy_mean_ret_pct": float(np.mean(strategy_ret)),
        "delta_mean_ret_pct": float(np.mean(strategy_ret) - np.mean(baseline_ret)),
        "baseline_total_pnl_eur": float(np.sum(baseline_pnl)),
        "strategy_total_pnl_eur": float(np.sum(strategy_pnl)),
        "delta_total_pnl_eur": float(np.sum(strategy_pnl) - np.sum(baseline_pnl)),
    }


def apply_rule(detail: pd.DataFrame, signal_col: str, threshold: float) -> pd.DataFrame:
    out = detail.copy()
    if signal_col == "none":
        out["selected_signal"] = "none"
        out["selected_threshold_pct"] = ""
        out["flagged"] = False
        out["strategy_ret_pct"] = out["baseline_ret_pct"]
        out["baseline_pnl_eur"] = out["invested_eur"] * out["baseline_ret_pct"] / 100.0
        out["strategy_pnl_eur"] = out["baseline_pnl_eur"]
        out["pnl_delta_eur"] = 0.0
        return out

    signal = pd.to_numeric(out[signal_col], errors="coerce")
    flagged = signal <= float(threshold)
    out["selected_signal"] = signal_col
    out["selected_threshold_pct"] = float(threshold)
    out["flagged"] = flagged.fillna(False)
    out["strategy_ret_pct"] = np.where(out["flagged"], signal, out["baseline_ret_pct"])
    out["baseline_pnl_eur"] = out["invested_eur"] * out["baseline_ret_pct"] / 100.0
    out["strategy_pnl_eur"] = out["invested_eur"] * out["strategy_ret_pct"] / 100.0
    out["pnl_delta_eur"] = out["strategy_pnl_eur"] - out["baseline_pnl_eur"]
    return out


def fmt(value) -> str | float:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate early-exit rules on the original investable validation/test split.")
    parser.add_argument("--sweep-csv", default=str(SWEEP_CSV))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[15, 30, 60, 120])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-0.5, -1.0, -2.0, -3.0, -5.0])
    parser.add_argument("--detail-out", default=str(OUT_DIR / "investable_early_exit_detail.csv"))
    parser.add_argument("--rule-out", default=str(OUT_DIR / "investable_early_exit_rule_grid.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "investable_early_exit_summary.csv"))
    args = parser.parse_args()

    load_dotenv(BASE / ".env")
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    threshold = select_threshold_from_validation(Path(args.sweep_csv))
    scored = load_scored_day1()
    n_train, split_80 = chrono_split_60_20_20(len(scored))
    val_df = scored.iloc[n_train:split_80].copy()
    test_df = scored.iloc[split_80:].copy()

    val_picks = val_df[pd.to_numeric(val_df["pred_mean4"], errors="coerce") > threshold].copy()
    test_picks = test_df[pd.to_numeric(test_df["pred_mean4"], errors="coerce") > threshold].copy()

    _, val_trades = simulate_budget_curve_from_picks(val_picks)
    _, test_trades = simulate_budget_curve_from_picks(test_picks)
    val_trades["split"] = "val"
    test_trades["split"] = "test"
    trades = pd.concat([val_trades, test_trades], ignore_index=True)

    client = RESTClient(api_key=api_key, retries=3)
    cache = PriceCache(Path(args.cache_dir))

    detail_rows = []
    for _, row in trades.iterrows():
        detail_rows.append(compute_signal_detail(client, cache, row, args.checkpoints))
    detail = pd.DataFrame(detail_rows)

    signal_cols = [f"ret_{checkpoint}m_pct" for checkpoint in args.checkpoints]
    signal_cols.extend(["ret_same_day_close_pct", "ret_next_day_open_pct"])

    val_detail = detail[detail["split"] == "val"].copy()
    test_detail = detail[detail["split"] == "test"].copy()

    grid_rows = []
    for signal_col in signal_cols:
        for threshold_candidate in args.thresholds:
            val_metrics = summarize_rule(val_detail, signal_col, threshold_candidate)
            if not val_metrics:
                continue
            test_metrics = summarize_rule(test_detail, signal_col, threshold_candidate)
            row_out = {
                "signal": signal_col,
                "threshold_pct": threshold_candidate,
                "val_n_trades": val_metrics["n_trades"],
                "val_n_flagged": val_metrics["n_flagged"],
                "val_flag_rate_pct": val_metrics["flag_rate_pct"],
                "val_loser_precision_pct": fmt(val_metrics["loser_precision_pct"]),
                "val_loser_recall_pct": fmt(val_metrics["loser_recall_pct"]),
                "val_baseline_mean_ret_pct": val_metrics["baseline_mean_ret_pct"],
                "val_strategy_mean_ret_pct": val_metrics["strategy_mean_ret_pct"],
                "val_delta_mean_ret_pct": val_metrics["delta_mean_ret_pct"],
                "val_baseline_total_pnl_eur": val_metrics["baseline_total_pnl_eur"],
                "val_strategy_total_pnl_eur": val_metrics["strategy_total_pnl_eur"],
                "val_delta_total_pnl_eur": val_metrics["delta_total_pnl_eur"],
                "test_n_trades": test_metrics.get("n_trades", 0),
                "test_n_flagged": test_metrics.get("n_flagged", 0),
                "test_flag_rate_pct": fmt(test_metrics.get("flag_rate_pct")),
                "test_loser_precision_pct": fmt(test_metrics.get("loser_precision_pct")),
                "test_loser_recall_pct": fmt(test_metrics.get("loser_recall_pct")),
                "test_baseline_mean_ret_pct": fmt(test_metrics.get("baseline_mean_ret_pct")),
                "test_strategy_mean_ret_pct": fmt(test_metrics.get("strategy_mean_ret_pct")),
                "test_delta_mean_ret_pct": fmt(test_metrics.get("delta_mean_ret_pct")),
                "test_baseline_total_pnl_eur": fmt(test_metrics.get("baseline_total_pnl_eur")),
                "test_strategy_total_pnl_eur": fmt(test_metrics.get("strategy_total_pnl_eur")),
                "test_delta_total_pnl_eur": fmt(test_metrics.get("delta_total_pnl_eur")),
            }
            grid_rows.append(row_out)

    grid = pd.DataFrame(grid_rows)
    grid = grid.sort_values(
        ["val_strategy_total_pnl_eur", "val_loser_precision_pct", "val_n_flagged"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = grid.iloc[0]
    if float(best["val_delta_total_pnl_eur"]) <= 0:
        chosen_signal = "none"
        chosen_threshold = np.nan
        best_row = {
            "val_baseline_total_pnl_eur": float(detail[detail["split"] == "val"]["invested_eur"].mul(detail[detail["split"] == "val"]["baseline_ret_pct"]).sum() / 100.0),
            "val_strategy_total_pnl_eur": float(detail[detail["split"] == "val"]["invested_eur"].mul(detail[detail["split"] == "val"]["baseline_ret_pct"]).sum() / 100.0),
            "val_delta_total_pnl_eur": 0.0,
            "test_baseline_total_pnl_eur": float(detail[detail["split"] == "test"]["invested_eur"].mul(detail[detail["split"] == "test"]["baseline_ret_pct"]).sum() / 100.0),
            "test_strategy_total_pnl_eur": float(detail[detail["split"] == "test"]["invested_eur"].mul(detail[detail["split"] == "test"]["baseline_ret_pct"]).sum() / 100.0),
            "test_delta_total_pnl_eur": 0.0,
            "test_baseline_mean_ret_pct": float(detail[detail["split"] == "test"]["baseline_ret_pct"].mean()),
            "test_strategy_mean_ret_pct": float(detail[detail["split"] == "test"]["baseline_ret_pct"].mean()),
            "test_delta_mean_ret_pct": 0.0,
            "test_n_flagged": 0,
            "test_flag_rate_pct": 0.0,
            "test_loser_precision_pct": np.nan,
            "test_loser_recall_pct": np.nan,
        }
    else:
        chosen_signal = str(best["signal"])
        chosen_threshold = float(best["threshold_pct"])
        best_row = best

    detailed_with_rule = apply_rule(detail, chosen_signal, chosen_threshold)

    summary_rows = [
        {"metric": "baseline_pred_mean4_threshold", "value": threshold},
        {"metric": "val_executed_trades", "value": int(len(val_detail))},
        {"metric": "test_executed_trades", "value": int(len(test_detail))},
        {"metric": "selected_signal", "value": chosen_signal},
        {"metric": "selected_threshold_pct", "value": "" if not np.isfinite(chosen_threshold) else chosen_threshold},
        {"metric": "val_baseline_total_pnl_eur", "value": float(best_row["val_baseline_total_pnl_eur"])},
        {"metric": "val_strategy_total_pnl_eur", "value": float(best_row["val_strategy_total_pnl_eur"])},
        {"metric": "val_delta_total_pnl_eur", "value": float(best_row["val_delta_total_pnl_eur"])},
        {"metric": "test_baseline_total_pnl_eur", "value": float(best_row["test_baseline_total_pnl_eur"])},
        {"metric": "test_strategy_total_pnl_eur", "value": float(best_row["test_strategy_total_pnl_eur"])},
        {"metric": "test_delta_total_pnl_eur", "value": float(best_row["test_delta_total_pnl_eur"])},
        {"metric": "test_baseline_mean_ret_pct", "value": float(best_row["test_baseline_mean_ret_pct"])},
        {"metric": "test_strategy_mean_ret_pct", "value": float(best_row["test_strategy_mean_ret_pct"])},
        {"metric": "test_delta_mean_ret_pct", "value": float(best_row["test_delta_mean_ret_pct"])},
        {"metric": "test_n_flagged", "value": int(best_row["test_n_flagged"])},
        {"metric": "test_flag_rate_pct", "value": float(best_row["test_flag_rate_pct"])},
        {"metric": "test_loser_precision_pct", "value": fmt(best_row["test_loser_precision_pct"])},
        {"metric": "test_loser_recall_pct", "value": fmt(best_row["test_loser_recall_pct"])},
    ]

    detail_out = Path(args.detail_out)
    rule_out = Path(args.rule_out)
    summary_out = Path(args.summary_out)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    detailed_with_rule.to_csv(detail_out, index=False)
    grid.to_csv(rule_out, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)

    print(f"baseline_threshold={threshold:.6f}")
    print(f"val_executed={len(val_detail)} test_executed={len(test_detail)}")
    if chosen_signal == "none":
        print("selected_rule=none")
    else:
        print(f"selected_rule={chosen_signal} <= {chosen_threshold}")
    print(f"val_delta_total_pnl_eur={float(best_row['val_delta_total_pnl_eur']):.2f}")
    print(f"test_delta_total_pnl_eur={float(best_row['test_delta_total_pnl_eur']):.2f}")
    print(f"detail_out={detail_out}")
    print(f"rule_out={rule_out}")
    print(f"summary_out={summary_out}")


if __name__ == "__main__":
    main()
