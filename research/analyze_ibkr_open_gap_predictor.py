from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = REPO_ROOT / "research" / "ibkr_vm_analysis" / "ibkr_paper_trader_state.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "research" / "ibkr_vm_analysis" / "ibkr_open_gap_predictor_analysis.csv"
POLYGON_BASE_URL = "https://api.polygon.io"


@dataclass
class DailyBar:
    date: str
    open: float
    close: float


def _read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weighted_avg_price(fills: list[dict[str, Any]]) -> float | None:
    total_qty = sum(int(fill["quantity"]) for fill in fills)
    if total_qty <= 0:
        return None
    total_notional = sum(float(fill["price"]) * int(fill["quantity"]) for fill in fills)
    return total_notional / float(total_qty)


def _fetch_daily_bars(
    session: requests.Session,
    api_key: str,
    ticker: str,
    start_date: str,
    end_date: str,
) -> list[DailyBar]:
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    response = session.get(
        url,
        params={
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000,
            "apiKey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    bars: list[DailyBar] = []
    for row in results:
        ts = pd.to_datetime(row["t"], unit="ms", utc=True)
        bars.append(
            DailyBar(
                date=ts.tz_convert("America/New_York").date().isoformat(),
                open=float(row["o"]),
                close=float(row["c"]),
            )
        )
    return bars


def _build_daily_bar_lookup(
    session: requests.Session,
    api_key: str,
    trades: pd.DataFrame,
) -> dict[str, list[DailyBar]]:
    lookups: dict[str, list[DailyBar]] = {}
    today_et = datetime.now().astimezone().date().isoformat()
    for ticker, group in trades.groupby("ticker"):
        start = (pd.to_datetime(group["entry_trade_day"]).min() - pd.Timedelta(days=10)).date().isoformat()
        lookups[ticker] = _fetch_daily_bars(session, api_key, ticker, start, today_et)
    return lookups


def _nth_trading_day_after(trade_days: list[str], anchor_day: str, offset: int) -> str | None:
    try:
        idx = trade_days.index(anchor_day)
    except ValueError:
        return None
    target_idx = idx + offset
    if target_idx >= len(trade_days):
        return None
    return trade_days[target_idx]


def _trade_rows_from_state(state: dict[str, Any]) -> pd.DataFrame:
    fills = pd.DataFrame(state.get("fills", []))
    lots = pd.DataFrame(state.get("lots", []))
    candidates = pd.DataFrame(state.get("candidates", []))

    if fills.empty or lots.empty:
        return pd.DataFrame()

    buy_fills = fills[fills["side"].astype(str).str.upper() == "BOT"].copy()
    buy_fills["quantity"] = buy_fills["quantity"].astype(int)
    buy_fills["price"] = buy_fills["price"].astype(float)

    fill_groups = (
        buy_fills.groupby("local_order_id", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "entry_avg_price": _weighted_avg_price(g.to_dict("records")),
                    "entry_fill_qty": int(g["quantity"].sum()),
                    "entry_fill_count": int(len(g)),
                    "entry_first_fill_at": str(pd.to_datetime(g["filled_at"], utc=True).min()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"local_order_id": "entry_order_id"})
    )

    lots = lots.merge(fill_groups, on="entry_order_id", how="left")
    if not candidates.empty:
        candidate_cols = [
            "candidate_id",
            "ticker",
            "signal_score",
            "estimated_decile_score",
            "entry_trade_day",
            "intended_entry_at",
        ]
        lots = lots.merge(candidates[candidate_cols], on=["candidate_id", "ticker"], how="left", suffixes=("", "_candidate"))
        lots["signal_score"] = pd.to_numeric(lots["signal_score"], errors="coerce")
        lots["estimated_decile_score"] = pd.to_numeric(lots["estimated_decile_score"], errors="coerce")

    lots["entry_trade_day"] = lots["entry_trade_day"].fillna(lots["entry_trade_day_candidate"])
    lots["entry_trade_day"] = lots["entry_trade_day"].astype(str)
    lots["entry_avg_price"] = pd.to_numeric(lots["entry_avg_price"], errors="coerce")
    lots["entry_value"] = pd.to_numeric(lots["entry_value"], errors="coerce")
    lots["last_mark_price"] = pd.to_numeric(lots["last_mark_price"], errors="coerce")
    lots["realized_pnl"] = pd.to_numeric(lots["realized_pnl"], errors="coerce")
    return lots


def analyze(state_path: Path, output_path: Path, api_key: str) -> pd.DataFrame:
    state = _read_state(state_path)
    trades = _trade_rows_from_state(state)
    if trades.empty:
        return trades

    session = requests.Session()
    daily_lookup = _build_daily_bar_lookup(session, api_key, trades)
    rows: list[dict[str, Any]] = []

    for _, trade in trades.iterrows():
        ticker = str(trade["ticker"])
        entry_day = str(trade["entry_trade_day"])
        bars = daily_lookup.get(ticker, [])
        if not bars:
            continue

        trade_days = [bar.date for bar in bars]
        by_day = {bar.date: bar for bar in bars}
        buy_day = by_day.get(entry_day)
        if buy_day is None:
            continue

        try:
            buy_idx = trade_days.index(entry_day)
        except ValueError:
            continue
        if buy_idx == 0:
            continue

        prev_day = trade_days[buy_idx - 1]
        prev_close = by_day[prev_day].close
        buy_open = buy_day.open
        overnight_gap_pct = ((buy_open / prev_close) - 1.0) * 100.0

        target_close_day = _nth_trading_day_after(trade_days, entry_day, 2)
        latest_known_day = trade_days[-1]

        if target_close_day is not None:
            outcome_day = target_close_day
            outcome_basis = "tplus2_close"
        else:
            outcome_day = latest_known_day
            outcome_basis = "latest_close_fallback"

        outcome_close = by_day[outcome_day].close
        entry_avg_price = float(trade["entry_avg_price"]) if pd.notna(trade["entry_avg_price"]) else float(buy_open)
        actual_return_pct = ((outcome_close / entry_avg_price) - 1.0) * 100.0

        current_mark_return_pct = None
        if pd.notna(trade["last_mark_price"]) and entry_avg_price > 0:
            current_mark_return_pct = ((float(trade["last_mark_price"]) / entry_avg_price) - 1.0) * 100.0

        realized_return_pct = None
        if pd.notna(trade["realized_pnl"]) and pd.notna(trade["entry_value"]) and float(trade["entry_value"]) != 0:
            realized_return_pct = (float(trade["realized_pnl"]) / float(trade["entry_value"])) * 100.0

        rows.append(
            {
                "ticker": ticker,
                "candidate_id": trade["candidate_id"],
                "lot_id": trade["lot_id"],
                "status": trade["status"],
                "entry_trade_day": entry_day,
                "opened_at": trade["opened_at"],
                "due_exit_at": trade["due_exit_at"],
                "signal_score": trade.get("signal_score"),
                "estimated_decile_score": trade.get("estimated_decile_score"),
                "entry_fill_qty": trade["entry_fill_qty"],
                "entry_fill_count": trade["entry_fill_count"],
                "prev_close_day": prev_day,
                "prev_close": prev_close,
                "buy_day_open": buy_open,
                "overnight_gap_pct": overnight_gap_pct,
                "entry_avg_price": entry_avg_price,
                "outcome_day": outcome_day,
                "outcome_basis": outcome_basis,
                "outcome_close": outcome_close,
                "actual_return_pct": actual_return_pct,
                "current_mark_return_pct": current_mark_return_pct,
                "realized_return_pct": realized_return_pct,
                "last_mark_price": trade["last_mark_price"],
            }
        )

    result = pd.DataFrame(rows).sort_values(["entry_trade_day", "ticker"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def print_summary(result: pd.DataFrame) -> None:
    if result.empty:
        print("No trades available for analysis.")
        return

    valid = result.dropna(subset=["overnight_gap_pct", "actual_return_pct"]).copy()
    print(f"Trades analyzed: {len(result)}")
    print(f"Trades with full predictor/outcome data: {len(valid)}")
    print(f"Open lots: {(result['status'] == 'open').sum()}")
    print(f"Closed lots: {(result['status'] == 'closed').sum()}")
    print("")

    def print_corr_block(frame: pd.DataFrame, label: str) -> None:
        if frame.empty:
            print(f"{label}: no rows")
            print("")
            return
        pearson = frame["overnight_gap_pct"].corr(frame["actual_return_pct"], method="pearson")
        spearman = frame["overnight_gap_pct"].corr(frame["actual_return_pct"], method="spearman")
        print(f"{label} pearson:  {pearson:.4f}")
        print(f"{label} spearman: {spearman:.4f}")
        split = (
            frame.assign(
                gap_bucket=frame["overnight_gap_pct"].apply(
                    lambda x: "gap_down_or_flat" if x <= 0 else "gap_up"
                )
            )
            .groupby("gap_bucket")
            .agg(
                trades=("ticker", "count"),
                avg_gap_pct=("overnight_gap_pct", "mean"),
                median_gap_pct=("overnight_gap_pct", "median"),
                avg_actual_return_pct=("actual_return_pct", "mean"),
                median_actual_return_pct=("actual_return_pct", "median"),
            )
            .reset_index()
        )
        print(split.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        print("")

    print_corr_block(valid, "Lot-level")

    event_level = (
        valid.groupby(["ticker", "entry_trade_day"], as_index=False)
        .agg(
            overnight_gap_pct=("overnight_gap_pct", "first"),
            actual_return_pct=("actual_return_pct", "mean"),
            entry_fill_qty=("entry_fill_qty", "sum"),
        )
    )
    print_corr_block(event_level, "Ticker-day")

    closed_only = valid[valid["outcome_basis"] == "tplus2_close"].copy()
    print_corr_block(closed_only, "Closed-only")

    strongest = valid.sort_values("overnight_gap_pct")
    print("Lowest overnight gaps:")
    print(
        strongest[["ticker", "entry_trade_day", "overnight_gap_pct", "actual_return_pct", "status", "outcome_basis"]]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:,.3f}")
    )
    print("")
    print("Highest overnight gaps:")
    print(
        strongest[["ticker", "entry_trade_day", "overnight_gap_pct", "actual_return_pct", "status", "outcome_basis"]]
        .tail(10)
        .to_string(index=False, float_format=lambda x: f"{x:,.3f}")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether buy-day close-to-open gap predicts trade return.")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not found in environment or .env.")

    result = analyze(args.state_path, args.output_path, api_key)
    print_summary(result)
    print("")
    print(f"Saved analysis to: {args.output_path}")


if __name__ == "__main__":
    main()
