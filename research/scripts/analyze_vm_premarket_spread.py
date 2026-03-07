from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from live_trading.strategy_settings import ACTIVE_STRATEGY, LIVE_PATHS
from scripts.backtest_vm_recommendations import (
    ET,
    UTC,
    bar_date_et,
    fetch_day_bars,
    parse_scored_at_utc,
    resolve_entry,
)


MARKET_OPEN = time(9, 30)
DEFAULT_SLEEVE_FRACTION = 0.5
START_BUDGET_EUR = 10_000.0
DEFAULT_MAX_LOOKAHEAD_DAYS = 20
DEFAULT_EXIT_DAYS = int(ACTIVE_STRATEGY.sell_after_trading_days)
DEFAULT_DECILE_SCORE_THRESHOLD = float(ACTIVE_STRATEGY.day1_decile_score_threshold)
DEFAULT_RAW_CUTOFF = float(ACTIVE_STRATEGY.day1_raw_threshold_fallback)
DEFAULT_BASE_ALLOC = float(ACTIVE_STRATEGY.advice_base_alloc_fraction)
DEFAULT_BONUS_ALLOC = float(ACTIVE_STRATEGY.advice_bonus_fraction)


@dataclass
class Position:
    ticker: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    invested_eur: float
    ret_pct: float


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_decile_curve(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["decile_score_threshold"] = pd.to_numeric(table["decile_score_threshold"], errors="coerce")
    table["raw_pred_mean4_cutoff"] = pd.to_numeric(table["raw_pred_mean4_cutoff"], errors="coerce")
    table = table.dropna(subset=["decile_score_threshold", "raw_pred_mean4_cutoff"]).copy()
    table = table.sort_values(["raw_pred_mean4_cutoff", "decile_score_threshold"]).reset_index(drop=True)
    if table.empty:
        raise RuntimeError(f"No valid curve points found in {path}")
    return table


def estimate_decile_score(raw_prediction: float, curve: pd.DataFrame) -> float:
    points = curve[["decile_score_threshold", "raw_pred_mean4_cutoff"]].to_numpy(dtype=float)
    deciles = points[:, 0]
    raws = points[:, 1]
    if raw_prediction <= raws[0]:
        return float(deciles[0])
    if raw_prediction >= raws[-1]:
        return float(deciles[-1])
    for idx in range(1, len(points)):
        x0, x1 = raws[idx - 1], raws[idx]
        if raw_prediction <= x1:
            y0, y1 = deciles[idx - 1], deciles[idx]
            if x1 == x0:
                return float(y1)
            frac = (raw_prediction - x0) / (x1 - x0)
            return float(y0 + frac * (y1 - y0))
    return float(deciles[-1])


def advice_weight(
    estimated_decile: float,
    decile_threshold: float,
    base_alloc_fraction: float,
    bonus_fraction: float,
) -> float:
    denom = max(1e-9, 1.0 - float(decile_threshold))
    strength = min(max((float(estimated_decile) - float(decile_threshold)) / denom, 0.0), 1.0)
    return min(max(base_alloc_fraction + bonus_fraction * strength, base_alloc_fraction), base_alloc_fraction + bonus_fraction)


def compute_exit_open_return(
    client: RESTClient,
    cache_dir: Path,
    ticker: str,
    entry_dt: datetime,
    entry_price: float,
    exit_days: int,
) -> tuple[datetime | None, float | None, float | None]:
    from_d = entry_dt.date()
    to_d = from_d + timedelta(days=max(DEFAULT_MAX_LOOKAHEAD_DAYS, exit_days + 10))
    bars = fetch_day_bars(client, cache_dir, ticker, from_d, to_d)
    dated_bars = [(bar_date_et(bar), bar) for bar in bars]
    dated_bars = [(d, b) for d, b in dated_bars if d is not None and d >= from_d]
    if not dated_bars:
        return None, None, None

    entry_bar_idx = None
    for idx, (bar_d, _bar) in enumerate(dated_bars):
        if bar_d == from_d:
            entry_bar_idx = idx
            break
    if entry_bar_idx is None:
        return None, None, None

    exit_idx = entry_bar_idx + int(exit_days)
    if exit_idx >= len(dated_bars):
        return None, None, None

    exit_d, exit_bar = dated_bars[exit_idx]
    exit_open = exit_bar.get("o")
    if exit_d is None or exit_open is None:
        return None, None, None
    exit_open = float(exit_open)
    if not np.isfinite(exit_open) or exit_open <= 0 or entry_price <= 0:
        return None, None, None

    exit_dt = datetime.combine(exit_d, MARKET_OPEN, tzinfo=ET)
    ret_pct = ((exit_open / entry_price) - 1.0) * 100.0
    return exit_dt, exit_open, ret_pct


def build_detail(
    input_path: Path,
    cache_dir: Path,
    cutoff_curve_path: Path,
    raw_cutoff: float,
    decile_score_threshold: float,
    exit_days: int,
    base_alloc_fraction: float,
    bonus_fraction: float,
) -> pd.DataFrame:
    rows = load_rows(input_path)
    curve = load_decile_curve(cutoff_curve_path)

    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")
    client = RESTClient(api_key=api_key, retries=3)

    detail_rows: list[dict] = []
    for row in rows:
        if str(row.get("horizon_days", "")) != "1":
            continue
        pred_mean4 = pd.to_numeric(row.get("pred_mean4"), errors="coerce")
        if not np.isfinite(pred_mean4) or float(pred_mean4) <= float(raw_cutoff):
            continue

        scored_at_raw = str(row.get("scored_at", ""))
        scored_utc = parse_scored_at_utc(scored_at_raw)
        scored_et = scored_utc.astimezone(ET)
        if scored_et.timetz().replace(tzinfo=None) >= MARKET_OPEN:
            continue

        entry_dt = datetime.combine(scored_et.date(), MARKET_OPEN, tzinfo=ET)
        ticker = str(row["ticker"])
        entry_dt_resolved, entry_price, entry_source = resolve_entry(client, cache_dir, ticker, entry_dt)
        if entry_dt_resolved is None or entry_price is None or entry_price <= 0:
            continue

        exit_dt, exit_open, ret_pct = compute_exit_open_return(
            client=client,
            cache_dir=cache_dir,
            ticker=ticker,
            entry_dt=entry_dt_resolved,
            entry_price=float(entry_price),
            exit_days=exit_days,
        )
        if exit_dt is None or exit_open is None or ret_pct is None:
            continue

        estimated_decile = estimate_decile_score(float(pred_mean4), curve)
        weight_equal = 1.0
        weight_score = max(float(pred_mean4) - float(raw_cutoff), 1e-9)
        weight_advice = advice_weight(
            estimated_decile=estimated_decile,
            decile_threshold=decile_score_threshold,
            base_alloc_fraction=base_alloc_fraction,
            bonus_fraction=bonus_fraction,
        )

        detail_rows.append(
            {
                "event_key": row["event_key"],
                "ticker": ticker,
                "trade_date": row.get("trade_date", ""),
                "transaction_date": row.get("transaction_date", ""),
                "scored_at": scored_at_raw,
                "scored_at_et": scored_et.strftime("%Y-%m-%d %H:%M:%S%z"),
                "entry_day": entry_dt_resolved.strftime("%Y-%m-%d"),
                "entry_dt_et": entry_dt_resolved.strftime("%Y-%m-%d %H:%M:%S%z"),
                "entry_price": float(entry_price),
                "entry_price_source": entry_source,
                "exit_dt_et": exit_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
                "exit_open_price": float(exit_open),
                "ret_2d_open_pct": float(ret_pct),
                "pred_mean4": float(pred_mean4),
                "estimated_decile_score": float(estimated_decile),
                "equal_weight_raw": float(weight_equal),
                "score_weight_raw": float(weight_score),
                "advice_weight_raw": float(weight_advice),
            }
        )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise RuntimeError("No usable premarket VM rows passed the cutoff.")
    detail = detail.sort_values(["entry_dt_et", "pred_mean4"], ascending=[True, False]).reset_index(drop=True)
    return detail


def normalize_weights(group: pd.DataFrame, raw_col: str) -> pd.Series:
    vals = pd.to_numeric(group[raw_col], errors="coerce").fillna(0.0).astype(float)
    total = float(vals.sum())
    if total <= 0:
        return pd.Series(np.repeat(1.0 / len(group), len(group)), index=group.index)
    return vals / total


def close_due_positions(
    positions: list[Position],
    cash: float,
    ts: pd.Timestamp,
    trade_log: list[dict],
    curve: list[dict],
) -> tuple[list[Position], float]:
    open_positions = list(positions)
    local_cash = float(cash)
    while True:
        due = [p for p in open_positions if p.exit_time <= ts]
        if not due:
            break
        next_exit = min(p.exit_time for p in due)
        closing = [p for p in open_positions if p.exit_time == next_exit]
        open_positions = [p for p in open_positions if p.exit_time != next_exit]
        for pos in closing:
            proceeds = pos.invested_eur * (1.0 + pos.ret_pct / 100.0)
            local_cash += proceeds
            trade_log.append(
                {
                    "ticker": pos.ticker,
                    "entry_time": pos.entry_time,
                    "exit_time": pos.exit_time,
                    "invested_eur": pos.invested_eur,
                    "ret_pct": pos.ret_pct,
                    "proceeds_eur": proceeds,
                    "pnl_eur": proceeds - pos.invested_eur,
                }
            )
        invested = float(sum(p.invested_eur for p in open_positions))
        curve.append(
            {
                "timestamp": next_exit,
                "cash_eur": local_cash,
                "invested_eur": invested,
                "budget_eur": local_cash + invested,
            }
        )
    return open_positions, local_cash


def summarize_curve(curve_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict[str, float | int]:
    if curve_df.empty:
        return {
            "end_budget_eur": np.nan,
            "total_return_pct": np.nan,
            "annualized_return_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "trades_executed": 0,
            "win_rate_pct": np.nan,
            "mean_trade_ret_pct": np.nan,
        }
    start_budget = float(curve_df["budget_eur"].iloc[0])
    end_budget = float(curve_df["budget_eur"].iloc[-1])
    start_time = pd.to_datetime(curve_df["timestamp"].iloc[0])
    end_time = pd.to_datetime(curve_df["timestamp"].iloc[-1])
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1.0)
    annualized = ((end_budget / start_budget) ** (365.25 / elapsed_days) - 1.0) * 100.0
    running_peak = curve_df["budget_eur"].cummax()
    drawdown = ((curve_df["budget_eur"] / running_peak) - 1.0) * 100.0
    rets = pd.to_numeric(trades_df.get("ret_pct"), errors="coerce").dropna()
    return {
        "end_budget_eur": end_budget,
        "total_return_pct": (end_budget / start_budget - 1.0) * 100.0,
        "annualized_return_pct": float(annualized),
        "max_drawdown_pct": float(drawdown.min()),
        "trades_executed": int(len(trades_df)),
        "win_rate_pct": float((rets > 0).mean() * 100.0) if not rets.empty else np.nan,
        "mean_trade_ret_pct": float(rets.mean()) if not rets.empty else np.nan,
    }


def simulate_daily_spread(
    detail: pd.DataFrame,
    weight_col: str,
    sleeve_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    cash = float(START_BUDGET_EUR)
    open_positions: list[Position] = []
    trade_log: list[dict] = []
    curve: list[dict] = []
    daily_rows: list[dict] = []

    start_ts = pd.to_datetime(detail["entry_dt_et"].min()) - pd.Timedelta(minutes=1)
    curve.append({"timestamp": start_ts, "cash_eur": cash, "invested_eur": 0.0, "budget_eur": cash})

    for entry_day, group in detail.groupby("entry_day", sort=True):
        entry_ts = pd.to_datetime(group["entry_dt_et"].iloc[0])
        open_positions, cash = close_due_positions(open_positions, cash, entry_ts, trade_log, curve)

        invested = float(sum(p.invested_eur for p in open_positions))
        total_equity = cash + invested
        day_budget = total_equity * float(sleeve_fraction)
        weights = normalize_weights(group, weight_col)
        allocations = day_budget * weights

        used_budget = 0.0
        for idx, row in group.iterrows():
            alloc = float(allocations.loc[idx])
            if alloc <= 0 or cash <= 0:
                continue
            alloc = min(alloc, cash)
            cash -= alloc
            used_budget += alloc
            open_positions.append(
                Position(
                    ticker=str(row["ticker"]),
                    entry_time=pd.to_datetime(row["entry_dt_et"]),
                    exit_time=pd.to_datetime(row["exit_dt_et"]),
                    invested_eur=alloc,
                    ret_pct=float(row["ret_2d_open_pct"]),
                )
            )

        invested = float(sum(p.invested_eur for p in open_positions))
        curve.append(
            {
                "timestamp": entry_ts,
                "cash_eur": cash,
                "invested_eur": invested,
                "budget_eur": cash + invested,
            }
        )
        daily_rows.append(
            {
                "entry_day": entry_day,
                "n_names": int(len(group)),
                "day_budget_eur": float(day_budget),
                "used_budget_eur": float(used_budget),
                "utilization_pct": float((used_budget / day_budget) * 100.0) if day_budget > 0 else np.nan,
            }
        )

    if open_positions:
        final_ts = max(p.exit_time for p in open_positions) + pd.Timedelta(seconds=1)
        open_positions, cash = close_due_positions(open_positions, cash, final_ts, trade_log, curve)

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    trades_df = pd.DataFrame(trade_log).sort_values("entry_time").reset_index(drop=True)
    daily_df = pd.DataFrame(daily_rows).sort_values("entry_day").reset_index(drop=True)
    metrics = summarize_curve(curve_df, trades_df)
    metrics["candidate_days"] = int(detail["entry_day"].nunique())
    metrics["candidate_rows"] = int(len(detail))
    metrics["avg_names_per_day"] = float(daily_df["n_names"].mean()) if not daily_df.empty else np.nan
    metrics["median_names_per_day"] = float(daily_df["n_names"].median()) if not daily_df.empty else np.nan
    metrics["max_names_per_day"] = int(daily_df["n_names"].max()) if not daily_df.empty else 0
    return curve_df, trades_df, daily_df, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest spreading each premarket VM cohort across all available names.")
    parser.add_argument("--input", required=True, help="Path to full VM live_predictions.csv")
    parser.add_argument("--cache-dir", default=LIVE_PATHS.cache_dir)
    parser.add_argument("--cutoff-csv", default=ACTIVE_STRATEGY.day1_decile_cutoff_file)
    parser.add_argument("--decile-score-threshold", type=float, default=DEFAULT_DECILE_SCORE_THRESHOLD)
    parser.add_argument("--raw-cutoff", type=float, default=DEFAULT_RAW_CUTOFF)
    parser.add_argument("--sleeve-fraction", type=float, default=DEFAULT_SLEEVE_FRACTION)
    parser.add_argument("--exit-days", type=int, default=DEFAULT_EXIT_DAYS)
    parser.add_argument("--base-alloc-fraction", type=float, default=DEFAULT_BASE_ALLOC)
    parser.add_argument("--bonus-fraction", type=float, default=DEFAULT_BONUS_ALLOC)
    parser.add_argument("--detail-out", default="backtest/out/vm_premarket_spread_detail.csv")
    parser.add_argument("--summary-out", default="backtest/out/vm_premarket_spread_summary.csv")
    parser.add_argument("--daily-out", default="backtest/out/vm_premarket_spread_daily.csv")
    parser.add_argument("--summary-json", default="backtest/out/vm_premarket_spread_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    cutoff_curve_path = Path(args.cutoff_csv)

    detail = build_detail(
        input_path=input_path,
        cache_dir=cache_dir,
        cutoff_curve_path=cutoff_curve_path,
        raw_cutoff=float(args.raw_cutoff),
        decile_score_threshold=float(args.decile_score_threshold),
        exit_days=int(args.exit_days),
        base_alloc_fraction=float(args.base_alloc_fraction),
        bonus_fraction=float(args.bonus_fraction),
    )

    strategy_specs = [
        ("equal_weight", "equal_weight_raw"),
        ("score_weight", "score_weight_raw"),
        ("advice_weight", "advice_weight_raw"),
    ]
    summary_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []

    for strategy_name, weight_col in strategy_specs:
        curve_df, trades_df, daily_df, metrics = simulate_daily_spread(
            detail=detail,
            weight_col=weight_col,
            sleeve_fraction=float(args.sleeve_fraction),
        )
        curve_path = Path(str(args.summary_out).replace("_summary.csv", f"_{strategy_name}_curve.csv"))
        trades_path = Path(str(args.summary_out).replace("_summary.csv", f"_{strategy_name}_trades.csv"))
        curve_df.to_csv(curve_path, index=False)
        trades_df.to_csv(trades_path, index=False)

        daily_df = daily_df.copy()
        daily_df["strategy"] = strategy_name
        daily_frames.append(daily_df)

        summary_rows.append(
            {
                "strategy": strategy_name,
                "input_file": str(input_path),
                "raw_cutoff": float(args.raw_cutoff),
                "decile_score_threshold": float(args.decile_score_threshold),
                "sleeve_fraction": float(args.sleeve_fraction),
                "exit_days": int(args.exit_days),
                "curve_csv": str(curve_path),
                "trades_csv": str(trades_path),
                **metrics,
            }
        )

    detail_out = Path(args.detail_out)
    summary_out = Path(args.summary_out)
    daily_out = Path(args.daily_out)
    summary_json = Path(args.summary_json)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_out, index=False)
    pd.DataFrame(summary_rows).sort_values("end_budget_eur", ascending=False).to_csv(summary_out, index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(daily_out, index=False)
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"detail_out={detail_out}")
    print(f"summary_out={summary_out}")
    for row in summary_rows:
        print(
            "strategy={strategy} candidate_rows={candidate_rows} candidate_days={candidate_days} "
            "end_budget={end_budget_eur:.2f} total_return_pct={total_return_pct:.2f} "
            "max_drawdown_pct={max_drawdown_pct:.2f} avg_names_per_day={avg_names_per_day:.2f}".format(**row)
        )


if __name__ == "__main__":
    main()
