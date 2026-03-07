from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

OUT_DIR = BASE / "backtest" / "out"
ROTATION_SCRIPT = BASE / "research" / "scripts" / "analyze_two_day_rotation.py"
PRICE_CACHE_DIR = BASE / "backtest" / "data" / "price_cache"

START_BUDGET_EUR = 10_000.0
DEFAULT_SLEEVE_FRACTION = 0.5


@dataclass
class Position:
    ticker: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    score: float
    source: str

    @property
    def invested_eur(self) -> float:
        return float(self.shares * self.entry_price)


def load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_candidates(decile_score_threshold: float) -> tuple[pd.DataFrame, float]:
    rotation = load_module("analyze_two_day_rotation", ROTATION_SCRIPT)
    raw_cutoff = rotation.load_raw_cutoff(rotation.DEFAULT_CUTOFF_CSV, float(decile_score_threshold))
    picks = rotation.build_candidates(raw_cutoff=raw_cutoff).copy()
    picks["buy_datetime"] = pd.to_datetime(picks["buy_datetime"], errors="coerce")
    picks["exit_datetime"] = pd.to_datetime(picks["exit_datetime"], errors="coerce")
    picks["buy_price"] = pd.to_numeric(picks["buy_price"], errors="coerce")
    picks = picks.dropna(subset=["buy_datetime", "exit_datetime", "ret_pct", "pred_mean4", "buy_price"]).copy()
    picks["score_weight_raw"] = np.maximum(pd.to_numeric(picks["pred_mean4"], errors="coerce") - float(raw_cutoff), 1e-9)
    picks["trade_day"] = picks["buy_datetime"].dt.strftime("%Y-%m-%d")
    picks["buy_time_hhmm"] = picks["buy_datetime"].dt.strftime("%H:%M")
    return picks.sort_values(["buy_datetime", "pred_mean4"], ascending=[True, False]).reset_index(drop=True), raw_cutoff


def normalize_weights(group: pd.DataFrame, raw_col: str) -> pd.Series:
    vals = pd.to_numeric(group[raw_col], errors="coerce").fillna(0.0).astype(float)
    total = float(vals.sum())
    if total <= 0:
        return pd.Series(np.repeat(1.0 / len(group), len(group)), index=group.index)
    return vals / total


def summarize_curve(curve_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict[str, float | int]:
    start_budget = float(curve_df["budget_eur"].iloc[0])
    end_budget = float(curve_df["budget_eur"].iloc[-1])
    start_time = pd.to_datetime(curve_df["timestamp"].iloc[0])
    end_time = pd.to_datetime(curve_df["timestamp"].iloc[-1])
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1.0)
    annualized = ((end_budget / start_budget) ** (365.25 / elapsed_days) - 1.0) * 100.0
    running_peak = curve_df["budget_eur"].cummax()
    drawdown = ((curve_df["budget_eur"] / running_peak) - 1.0) * 100.0
    rets = pd.to_numeric(trades_df["ret_pct"], errors="coerce").dropna()
    return {
        "end_budget_eur": end_budget,
        "total_return_pct": (end_budget / start_budget - 1.0) * 100.0,
        "annualized_return_pct": float(annualized),
        "max_drawdown_pct": float(drawdown.min()),
        "trades_executed": int(len(trades_df)),
        "win_rate_pct": float((rets > 0).mean() * 100.0) if not rets.empty else np.nan,
        "mean_trade_ret_pct": float(rets.mean()) if not rets.empty else np.nan,
    }


def minute_price_at_or_after(train_models, price_cache: dict[tuple[str, str], list[dict]], ticker: str, ts: pd.Timestamp) -> float:
    d = ts.date()
    key = (ticker, d.isoformat())
    if key not in price_cache:
        price_cache[key] = train_models.fetch_minute_bars(PRICE_CACHE_DIR, ticker, d)
    bars = price_cache[key]
    px = float(train_models.find_price_at_or_after(bars, int(ts.timestamp() * 1000)))
    if not np.isfinite(px):
        px = float(train_models.find_last_close(bars))
    return px


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
            proceeds = float(pos.shares * pos.exit_price)
            local_cash += proceeds
            cost = float(pos.shares * pos.entry_price)
            trade_log.append(
                {
                    "action": "scheduled_exit",
                    "ticker": pos.ticker,
                    "entry_time": pos.entry_time,
                    "exit_time": pos.exit_time,
                    "invested_eur": cost,
                    "ret_pct": (pos.exit_price / pos.entry_price - 1.0) * 100.0,
                    "proceeds_eur": proceeds,
                    "pnl_eur": proceeds - cost,
                    "score": pos.score,
                    "source": pos.source,
                }
            )
        invested = float(sum(p.invested_eur for p in open_positions))
        curve.append({"timestamp": next_exit, "cash_eur": local_cash, "invested_eur": invested, "budget_eur": local_cash + invested})
    return open_positions, local_cash


def sell_oldest_for_cash(
    train_models,
    price_cache: dict[tuple[str, str], list[dict]],
    positions: list[Position],
    cash: float,
    target_cash: float,
    ts: pd.Timestamp,
    trade_log: list[dict],
    curve: list[dict],
    allow_any: bool,
    batch_score: float,
) -> tuple[list[Position], float, float]:
    open_positions = list(positions)
    local_cash = float(cash)
    sold_eur = 0.0

    while local_cash + 1e-9 < target_cash and open_positions:
        open_positions.sort(key=lambda p: p.entry_time)
        oldest = open_positions[0]
        if (not allow_any) and batch_score <= float(oldest.score):
            break

        sell_px = minute_price_at_or_after(train_models, price_cache, oldest.ticker, ts)
        if not np.isfinite(sell_px) or sell_px <= 0:
            break

        needed = target_cash - local_cash
        oldest_value = float(oldest.shares * sell_px)
        sell_value = min(needed, oldest_value)
        shares_to_sell = sell_value / sell_px
        shares_to_sell = min(shares_to_sell, oldest.shares)
        proceeds = float(shares_to_sell * sell_px)
        cost = float(shares_to_sell * oldest.entry_price)
        local_cash += proceeds
        sold_eur += proceeds
        oldest.shares -= shares_to_sell

        trade_log.append(
            {
                "action": "early_sell_oldest",
                "ticker": oldest.ticker,
                "entry_time": oldest.entry_time,
                "exit_time": ts,
                "invested_eur": cost,
                "ret_pct": (sell_px / oldest.entry_price - 1.0) * 100.0,
                "proceeds_eur": proceeds,
                "pnl_eur": proceeds - cost,
                "score": oldest.score,
                "source": oldest.source,
            }
        )

        if oldest.shares <= 1e-12:
            open_positions.pop(0)
        else:
            open_positions[0] = oldest

        invested = float(sum(p.invested_eur for p in open_positions))
        curve.append({"timestamp": ts, "cash_eur": local_cash, "invested_eur": invested, "budget_eur": local_cash + invested})

    return open_positions, local_cash, sold_eur


def buy_batch(
    batch: pd.DataFrame,
    cash: float,
    batch_budget: float,
    ts: pd.Timestamp,
    positions: list[Position],
    curve: list[dict],
    source: str,
) -> tuple[list[Position], float, float]:
    open_positions = list(positions)
    local_cash = float(cash)
    used = 0.0
    weights = normalize_weights(batch, "score_weight_raw")
    allocs = batch_budget * weights

    for idx, row in batch.iterrows():
        alloc = min(float(allocs.loc[idx]), local_cash)
        if alloc <= 0:
            continue
        entry_price = float(row["buy_price"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        shares = alloc / entry_price
        local_cash -= alloc
        used += alloc
        exit_price = entry_price * (1.0 + float(row["ret_pct"]) / 100.0)
        open_positions.append(
            Position(
                ticker=str(row["ticker"]),
                entry_time=pd.to_datetime(row["buy_datetime"]),
                exit_time=pd.to_datetime(row["exit_datetime"]),
                entry_price=entry_price,
                exit_price=float(exit_price),
                shares=float(shares),
                score=float(row["pred_mean4"]),
                source=source,
            )
        )

    invested = float(sum(p.invested_eur for p in open_positions))
    curve.append({"timestamp": ts, "cash_eur": local_cash, "invested_eur": invested, "budget_eur": local_cash + invested})
    return open_positions, local_cash, used


def simulate_strategy(
    candidates: pd.DataFrame,
    sleeve_fraction: float,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    import train_models

    cash = float(START_BUDGET_EUR)
    open_positions: list[Position] = []
    trade_log: list[dict] = []
    curve: list[dict] = []
    daily_rows: list[dict] = []
    price_cache: dict[tuple[str, str], list[dict]] = {}

    start_ts = pd.to_datetime(candidates["buy_datetime"].min()) - pd.Timedelta(minutes=1)
    curve.append({"timestamp": start_ts, "cash_eur": cash, "invested_eur": 0.0, "budget_eur": cash})

    for trade_day, day_batch in candidates.groupby("trade_day", sort=True):
        day_batch = day_batch.sort_values(["buy_datetime", "pred_mean4"], ascending=[True, False]).reset_index(drop=True)
        day_start = pd.to_datetime(day_batch["buy_datetime"].min()).normalize() + pd.Timedelta(hours=9, minutes=30)
        open_positions, cash = close_due_positions(open_positions, cash, day_start, trade_log, curve)

        invested = float(sum(p.invested_eur for p in open_positions))
        total_equity = cash + invested
        day_budget = float(total_equity) * float(sleeve_fraction)

        open_batch = day_batch[day_batch["buy_time_hhmm"] == "09:45"].copy()
        intraday_batches = [grp.copy() for _, grp in day_batch[day_batch["buy_time_hhmm"] != "09:45"].groupby("buy_datetime", sort=True)]

        used_open = 0.0
        if not open_batch.empty:
            open_positions, cash, used_open = buy_batch(
                batch=open_batch,
                cash=cash,
                batch_budget=min(day_budget, cash),
                ts=pd.to_datetime(open_batch["buy_datetime"].iloc[0]),
                positions=open_positions,
                curve=curve,
                source="open_batch",
            )

        intraday_bought = 0
        intraday_sold = 0.0
        for batch in intraday_batches:
            ts = pd.to_datetime(batch["buy_datetime"].iloc[0])
            open_positions, cash = close_due_positions(open_positions, cash, ts, trade_log, curve)

            if strategy == "open_only_score_weight":
                continue

            invested = float(sum(p.invested_eur for p in open_positions))
            book_equity = cash + invested
            batch_budget = float(book_equity) * float(sleeve_fraction)

            if strategy == "intraday_cash_only_score_weight":
                batch_cap = min(batch_budget, cash)
            elif strategy == "replace_oldest_any_score_weight":
                target_cash = batch_budget
                open_positions, cash, sold_now = sell_oldest_for_cash(
                    train_models=train_models,
                    price_cache=price_cache,
                    positions=open_positions,
                    cash=cash,
                    target_cash=target_cash,
                    ts=ts,
                    trade_log=trade_log,
                    curve=curve,
                    allow_any=True,
                    batch_score=float(pd.to_numeric(batch["pred_mean4"], errors="coerce").max()),
                )
                intraday_sold += sold_now
                batch_cap = min(batch_budget, cash)
            elif strategy == "replace_oldest_if_batch_top_score_higher":
                target_cash = batch_budget
                open_positions, cash, sold_now = sell_oldest_for_cash(
                    train_models=train_models,
                    price_cache=price_cache,
                    positions=open_positions,
                    cash=cash,
                    target_cash=target_cash,
                    ts=ts,
                    trade_log=trade_log,
                    curve=curve,
                    allow_any=False,
                    batch_score=float(pd.to_numeric(batch["pred_mean4"], errors="coerce").max()),
                )
                intraday_sold += sold_now
                batch_cap = min(batch_budget, cash)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if batch_cap <= 0:
                continue

            open_positions, cash, used_intraday = buy_batch(
                batch=batch,
                cash=cash,
                batch_budget=batch_cap,
                ts=ts,
                positions=open_positions,
                curve=curve,
                source="intraday_batch",
            )
            if used_intraday > 0:
                intraday_bought += int(len(batch))

        daily_rows.append(
            {
                "trade_day": trade_day,
                "open_names": int(len(open_batch)),
                "intraday_names": int(sum(len(b) for b in intraday_batches)),
                "day_budget_eur": day_budget,
                "used_open_budget_eur": used_open,
                "intraday_sell_proceeds_eur": intraday_sold,
                "intraday_bought_names": intraday_bought,
            }
        )

    if open_positions:
        final_ts = max(p.exit_time for p in open_positions) + pd.Timedelta(seconds=1)
        open_positions, cash = close_due_positions(open_positions, cash, final_ts, trade_log, curve)

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    trades_df = pd.DataFrame(trade_log).sort_values(["exit_time", "ticker"]).reset_index(drop=True)
    daily_df = pd.DataFrame(daily_rows).sort_values("trade_day").reset_index(drop=True)
    metrics = summarize_curve(curve_df, trades_df)
    metrics["candidate_rows"] = int(len(candidates))
    metrics["candidate_days"] = int(candidates["trade_day"].nunique())
    metrics["open_rows"] = int((candidates["buy_time_hhmm"] == "09:45").sum())
    metrics["intraday_rows"] = int((candidates["buy_time_hhmm"] != "09:45").sum())
    metrics["days_with_intraday_candidates"] = int((daily_df["intraday_names"] > 0).sum()) if not daily_df.empty else 0
    metrics["intraday_sell_proceeds_eur"] = float(daily_df["intraday_sell_proceeds_eur"].sum()) if not daily_df.empty else 0.0
    return curve_df, trades_df, daily_df, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test selling the oldest-running positions to fund new intraday test-set candidates.")
    parser.add_argument("--decile-score-threshold", type=float, default=0.87)
    parser.add_argument("--sleeve-fraction", type=float, default=DEFAULT_SLEEVE_FRACTION)
    parser.add_argument("--summary-out", default=str(OUT_DIR / "test_oldest_replacement_summary.csv"))
    parser.add_argument("--daily-out", default=str(OUT_DIR / "test_oldest_replacement_daily.csv"))
    parser.add_argument("--summary-json", default=str(OUT_DIR / "test_oldest_replacement_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, raw_cutoff = build_candidates(decile_score_threshold=float(args.decile_score_threshold))

    strategies = [
        "open_only_score_weight",
        "intraday_cash_only_score_weight",
        "replace_oldest_any_score_weight",
        "replace_oldest_if_batch_top_score_higher",
    ]
    summary_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []

    for strategy in strategies:
        curve_df, trades_df, daily_df, metrics = simulate_strategy(
            candidates=candidates,
            sleeve_fraction=float(args.sleeve_fraction),
            strategy=strategy,
        )
        prefix = Path(str(args.summary_out).replace("_summary.csv", f"_{strategy}"))
        curve_df.to_csv(Path(f"{prefix}_curve.csv"), index=False)
        trades_df.to_csv(Path(f"{prefix}_trades.csv"), index=False)
        daily_df = daily_df.copy()
        daily_df["strategy"] = strategy
        daily_frames.append(daily_df)
        summary_rows.append(
            {
                "strategy": strategy,
                "decile_score_threshold": float(args.decile_score_threshold),
                "raw_cutoff": raw_cutoff,
                "sleeve_fraction": float(args.sleeve_fraction),
                "curve_csv": str(Path(f"{prefix}_curve.csv")),
                "trades_csv": str(Path(f"{prefix}_trades.csv")),
                **metrics,
            }
        )

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).sort_values("end_budget_eur", ascending=False).to_csv(args.summary_out, index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(args.daily_out, index=False)
    Path(args.summary_json).write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"summary_out={args.summary_out}")
    for row in summary_rows:
        print(
            "strategy={strategy} end_budget={end_budget_eur:.2f} total_return_pct={total_return_pct:.2f} "
            "max_drawdown_pct={max_drawdown_pct:.2f} intraday_rows={intraday_rows} "
            "days_with_intraday_candidates={days_with_intraday_candidates}".format(**row)
        )


if __name__ == "__main__":
    main()
