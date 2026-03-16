"""
Simulate live trading from alert_candidate_history.csv using exact trading logic.
Run from repo root:  python -m scripts.simulate_live_trading

Entry: next market open after scored_at (or intraday close-at-or-after for intraday signals).
Exit:  open of entry_date + sell_after_trading_days (default 2).

Calculates per-trade and portfolio returns using Polygon day bars (open prices).
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from polygon import RESTClient

from live_trading.market_calendar import (
    ET,
    UTC,
    add_trading_days,
    intended_entry_from_score,
    is_trading_day,
    next_trading_day,
    parse_scored_at_utc,
    sleeve_id_for_trade_day,
)
from live_trading.strategy_settings import (
    ACTIVE_STRATEGY,
    EXECUTION_POLICY,
    TRADING_BUDGET,
)


# ─── Polygon helpers ──────────────────────────────────────────────────────────

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


def _aggs_to_dicts(aggs: Any) -> list[dict]:
    return [
        {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
        for a in aggs
        if getattr(a, "timestamp", None) is not None
    ]


def fetch_day_bars(
    client: RESTClient, cache_dir: Path, ticker: str, from_d: date, to_d: date
) -> list[dict]:
    path = cache_dir / f"{ticker}_lkbk_{from_d:%Y-%m-%d}_{to_d:%Y-%m-%d}.json"
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


def _bar_et_date(bar: dict) -> date | None:
    ts = bar.get("t")
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts) / 1000, tz=UTC).astimezone(ET).date()


def get_open_on_or_after(bars: list[dict], target_date: date) -> tuple[date, float] | None:
    """Return (actual_date, open_price) for the first bar on or after target_date."""
    for bar in bars:
        d = _bar_et_date(bar)
        if d is None or d < target_date:
            continue
        o = bar.get("o")
        if o is not None:
            return d, float(o)
    return None


# ─── Gate filter ─────────────────────────────────────────────────────────────

def gate_failure_reason(row: dict) -> str | None:
    """Returns rejection reason string, or None if the candidate passes.

    step_up_from_prev_close_pct is checked only when populated; if the VM
    didn't compute it (empty string) the check is skipped so we can still
    simulate trades.
    """
    try:
        decile = float(row.get("estimated_decile_score") or "nan")
    except (ValueError, TypeError):
        decile = float("nan")
    # Use the threshold recorded in the row when it was scored (may differ from current setting)
    try:
        threshold = float(row.get("decile_score_threshold") or ACTIVE_STRATEGY.day1_decile_score_threshold)
    except (ValueError, TypeError):
        threshold = float(ACTIVE_STRATEGY.day1_decile_score_threshold)
    if not math.isfinite(decile) or decile < threshold:
        return "below_decile_threshold"

    step_up_raw = str(row.get("step_up_from_prev_close_pct") or "").strip()
    if step_up_raw:
        try:
            step_up = float(step_up_raw)
        except (ValueError, TypeError):
            step_up = float("nan")
        if math.isfinite(step_up) and step_up > float(ACTIVE_STRATEGY.max_step_up_from_prev_close_pct):
            return "step_up_exceeds_max"
    return None


# ─── Sizing ───────────────────────────────────────────────────────────────────

def normalize_batch_weights(
    batch: list[dict],
    max_allocation_ratio: float,
) -> dict[str, float]:
    """Return {candidate_id: weight} with max_allocation_ratio cap applied."""
    weights: dict[str, float] = {}
    for c in batch:
        w = float(c.get("advised_allocation_fraction") or 0)
        weights[c["_cid"]] = max(w, 1e-9)
    if len(weights) > 1:
        min_w = min(weights.values())
        cap = min_w * max_allocation_ratio
        for k in list(weights.keys()):
            if weights[k] > cap:
                weights[k] = cap
    return weights


def compute_per_candidate_notionals(
    batch: list[dict],
    sleeve_equity: float,
    weights: dict[str, float],
    max_allocation_ratio: float,
    min_order_notional: float,
) -> dict[str, float]:
    """
    Proportional allocation capped by the per-candidate safety cap fraction.
    Mirrors _preview_open_batch_allocations logic.
    """
    n = len(batch)
    if n == 0:
        return {}
    if n == 1:
        cap_frac = TRADING_BUDGET.max_fraction_single_name
    elif n == 2:
        cap_frac = TRADING_BUDGET.max_fraction_two_names
    else:
        cap_frac = TRADING_BUDGET.max_fraction_three_plus_names

    per_cap = sleeve_equity * cap_frac
    available = sleeve_equity
    total_w = sum(weights.values())

    notionals: dict[str, float] = {}
    for c in batch:
        w = weights[c["_cid"]]
        desired = available * w / total_w
        notionals[c["_cid"]] = min(desired, per_cap)
    return notionals


# ─── Portfolio state ──────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    ticker: str
    sleeve_id: str
    entry_date: date
    exit_date: date        # open of exit_date
    shares: int
    entry_price: float
    notional_invested: float
    signal_score: float
    scored_at: str
    event_key: str


@dataclass
class SleeveState:
    sleeve_id: str
    cash: float
    equity: float = field(init=False)
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        self.equity = self.cash

    def update_equity(self, open_positions: list[OpenPosition]) -> None:
        invested = sum(p.notional_invested for p in open_positions if p.sleeve_id == self.sleeve_id)
        self.equity = self.cash + invested


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate live trading from alert_candidate_history.csv."
    )
    parser.add_argument("--input", default="live/data/vm_sync/alert_candidate_history.csv")
    parser.add_argument("--cache-dir", default="live/data/price_cache")
    parser.add_argument("--start-date", default="2026-03-02",
                        help="Only include signals scored_at >= this date (YYYY-MM-DD).")
    parser.add_argument("--detail-out",
                        default="live/data/vm_sync/live_trade_simulation_detail.csv")
    parser.add_argument("--summary-out",
                        default="live/data/vm_sync/live_trade_simulation_summary.csv")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    start_date = date.fromisoformat(args.start_date)
    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    detail_out = Path(args.detail_out)
    summary_out = Path(args.summary_out)

    rows = load_rows(input_path)
    client = RESTClient(api_key=api_key, retries=3)

    anchor_date = TRADING_BUDGET.anchor_date()
    sleeve_count = TRADING_BUDGET.sleeve_count
    initial_budget = TRADING_BUDGET.initial_strategy_budget
    sleeve_cash = initial_budget / sleeve_count

    sleeves: dict[str, SleeveState] = {
        f"sleeve_{i}": SleeveState(sleeve_id=f"sleeve_{i}", cash=sleeve_cash)
        for i in range(sleeve_count)
    }

    # ── Step 1: parse + filter candidates ────────────────────────────────────
    candidates: list[dict] = []
    for i, row in enumerate(rows):
        scored_at_raw = str(row.get("scored_at", "")).strip()
        if not scored_at_raw:
            continue
        try:
            scored_utc = parse_scored_at_utc(scored_at_raw)
        except ValueError:
            continue

        scored_et = scored_utc.astimezone(ET)
        if scored_et.date() < start_date:
            continue

        intended_entry = intended_entry_from_score(scored_utc)
        entry_date = intended_entry.date()
        sleeve_id = sleeve_id_for_trade_day(entry_date, anchor_date, sleeve_count)

        try:
            score_1d = row.get("score_1d", "")
            pred_mean4 = row.get("pred_mean4", "")
            signal_score = float(score_1d) if score_1d and score_1d.strip() else float(pred_mean4)
        except (ValueError, TypeError):
            signal_score = 0.0

        row["_cid"] = f"{row.get('event_key', '')}|{scored_at_raw}"
        row["_entry_date"] = entry_date
        row["_sleeve_id"] = sleeve_id
        row["_signal_score"] = signal_score
        row["_scored_at"] = scored_at_raw
        row["_gate_failure"] = gate_failure_reason(row)
        candidates.append(row)

    print(f"Total candidates from {args.start_date}: {len(candidates)}")
    rejected = [c for c in candidates if c["_gate_failure"]]
    print(f"  Rejected by gate: {len(rejected)}")
    for reason, cnt in sorted(
        {r["_gate_failure"]: sum(1 for x in rejected if x["_gate_failure"] == r["_gate_failure"]) for r in rejected}.items(),
        key=lambda kv: -kv[1],
    ):
        print(f"    {reason}: {cnt}")

    active = [c for c in candidates if not c["_gate_failure"]]
    print(f"  Passed gate: {len(active)}")

    # ── Step 2: dedup — per (ticker, entry_date) keep highest signal_score ───
    dedup: dict[tuple[str, date], dict] = {}
    for c in sorted(active, key=lambda r: float(r["_signal_score"]), reverse=True):
        key = (str(c.get("ticker", "")).upper(), c["_entry_date"])
        if key not in dedup:
            dedup[key] = c
    active_deduped = list(dedup.values())
    print(f"  After dedup: {len(active_deduped)}")

    # ── Step 3: group into (sleeve, entry_date) batches ──────────────────────
    batches: dict[tuple[str, date], list[dict]] = {}
    for c in active_deduped:
        key = (c["_sleeve_id"], c["_entry_date"])
        batches.setdefault(key, []).append(c)

    max_candidates = int(ACTIVE_STRATEGY.max_open_batch_candidates)
    for key in batches:
        batch = sorted(batches[key], key=lambda r: (-r["_signal_score"], r.get("ticker", "")))
        batches[key] = batch[:max_candidates]

    print(f"\nBatches: {len(batches)}")
    for (sleeve_id, entry_date), batch in sorted(batches.items()):
        print(f"  {entry_date}  {sleeve_id}  {len(batch)} candidates: "
              f"{', '.join(c.get('ticker','') for c in batch)}")

    # ── Step 4: simulate in chronological order ───────────────────────────────
    open_positions: list[OpenPosition] = []
    trade_rows: list[dict] = []

    max_alloc_ratio = float(EXECUTION_POLICY.max_allocation_ratio)
    min_notional = float(EXECUTION_POLICY.min_order_notional)

    for (sleeve_id, entry_date) in sorted(batches.keys()):
        batch = batches[(sleeve_id, entry_date)]
        exit_date = add_trading_days(entry_date, ACTIVE_STRATEGY.sell_after_trading_days)
        sleeve = sleeves[sleeve_id]

        # Close any positions from this sleeve due on or before entry_date
        still_open: list[OpenPosition] = []
        for pos in open_positions:
            if pos.sleeve_id != sleeve_id or pos.exit_date > entry_date:
                still_open.append(pos)
                continue
            # Fetch exit price
            bars = fetch_day_bars(
                client, cache_dir, pos.ticker,
                pos.exit_date - timedelta(days=5),
                pos.exit_date + timedelta(days=5),
            )
            exit_result = get_open_on_or_after(bars, pos.exit_date)
            if exit_result is None:
                # No price available — treat as still open
                still_open.append(pos)
                continue
            actual_exit_date, exit_price = exit_result
            pnl = pos.shares * (exit_price - pos.entry_price)
            ret_pct = (exit_price / pos.entry_price - 1) * 100
            sleeve.cash += pos.notional_invested + pnl
            sleeve.realized_pnl += pnl
            # Update trade row with exit info
            for tr in trade_rows:
                if tr["ticker"] == pos.ticker and tr["entry_date"] == pos.entry_date.isoformat():
                    tr["exit_date"] = actual_exit_date.isoformat()
                    tr["exit_price"] = round(exit_price, 4)
                    tr["pnl"] = round(pnl, 2)
                    tr["return_pct"] = round(ret_pct, 4)
                    tr["status"] = "filled"
                    tr["sleeve_cash_after_exit"] = round(sleeve.cash, 2)
                    break

        open_positions = still_open
        sleeve.update_equity(open_positions)

        # Size this batch
        weights = normalize_batch_weights(batch, max_alloc_ratio)
        notionals = compute_per_candidate_notionals(
            batch, sleeve.equity, weights, max_alloc_ratio, min_notional
        )

        print(f"\n  {entry_date}  {sleeve_id}  equity=${sleeve.equity:.2f}  cash=${sleeve.cash:.2f}")

        for c in batch:
            ticker = str(c.get("ticker", "")).upper()
            target_notional = notionals.get(c["_cid"], 0.0)

            # Fetch entry price
            bars = fetch_day_bars(
                client, cache_dir, ticker,
                entry_date - timedelta(days=5),
                exit_date + timedelta(days=5),
            )
            entry_result = get_open_on_or_after(bars, entry_date)
            if entry_result is None:
                print(f"    {ticker}: no entry price")
                trade_rows.append({
                    "scored_at": c["_scored_at"],
                    "ticker": ticker,
                    "event_key": c.get("event_key", ""),
                    "entry_date": entry_date.isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "sleeve_id": sleeve_id,
                    "status": "no_entry_price",
                    "entry_price": "", "exit_price": "", "shares": "",
                    "notional_invested": "", "target_notional": round(target_notional, 2),
                    "pnl": "", "return_pct": "",
                    "signal_score": round(c["_signal_score"], 6),
                    "decile_score": c.get("estimated_decile_score", ""),
                    "advised_alloc_frac": c.get("advised_allocation_fraction", ""),
                    "sleeve_cash_after_exit": "",
                })
                continue

            actual_entry_date, entry_price = entry_result
            shares = int(math.floor(target_notional / entry_price))

            if shares < 1 or shares * entry_price < min_notional:
                print(f"    {ticker}: insufficient cash (target=${target_notional:.2f}, price=${entry_price:.2f})")
                trade_rows.append({
                    "scored_at": c["_scored_at"],
                    "ticker": ticker,
                    "event_key": c.get("event_key", ""),
                    "entry_date": actual_entry_date.isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "sleeve_id": sleeve_id,
                    "status": "insufficient_cash",
                    "entry_price": round(entry_price, 4), "exit_price": "", "shares": 0,
                    "notional_invested": 0, "target_notional": round(target_notional, 2),
                    "pnl": "", "return_pct": "",
                    "signal_score": round(c["_signal_score"], 6),
                    "decile_score": c.get("estimated_decile_score", ""),
                    "advised_alloc_frac": c.get("advised_allocation_fraction", ""),
                    "sleeve_cash_after_exit": "",
                })
                continue

            notional_invested = shares * entry_price
            sleeve.cash -= notional_invested
            sleeve.update_equity(open_positions)

            print(f"    {ticker}: {shares} shares @ ${entry_price:.3f} = ${notional_invested:.2f}  (target=${target_notional:.2f})")

            position = OpenPosition(
                ticker=ticker,
                sleeve_id=sleeve_id,
                entry_date=actual_entry_date,
                exit_date=exit_date,
                shares=shares,
                entry_price=entry_price,
                notional_invested=notional_invested,
                signal_score=c["_signal_score"],
                scored_at=c["_scored_at"],
                event_key=c.get("event_key", ""),
            )
            open_positions.append(position)

            trade_rows.append({
                "scored_at": c["_scored_at"],
                "ticker": ticker,
                "event_key": c.get("event_key", ""),
                "entry_date": actual_entry_date.isoformat(),
                "exit_date": "",          # filled when position closes
                "sleeve_id": sleeve_id,
                "status": "open",
                "entry_price": round(entry_price, 4),
                "exit_price": "",
                "shares": shares,
                "notional_invested": round(notional_invested, 2),
                "target_notional": round(target_notional, 2),
                "pnl": "",
                "return_pct": "",
                "signal_score": round(c["_signal_score"], 6),
                "decile_score": c.get("estimated_decile_score", ""),
                "advised_alloc_frac": c.get("advised_allocation_fraction", ""),
                "sleeve_cash_after_exit": "",
            })

    # Close any remaining open positions (using latest available price)
    for pos in open_positions:
        sleeve = sleeves[pos.sleeve_id]
        bars = fetch_day_bars(
            client, cache_dir, pos.ticker,
            pos.exit_date - timedelta(days=5),
            pos.exit_date + timedelta(days=10),
        )
        exit_result = get_open_on_or_after(bars, pos.exit_date)
        if exit_result is None:
            continue
        actual_exit_date, exit_price = exit_result
        pnl = pos.shares * (exit_price - pos.entry_price)
        ret_pct = (exit_price / pos.entry_price - 1) * 100
        sleeve.cash += pos.notional_invested + pnl
        sleeve.realized_pnl += pnl
        for tr in trade_rows:
            if tr["ticker"] == pos.ticker and tr["entry_date"] == pos.entry_date.isoformat():
                tr["exit_date"] = actual_exit_date.isoformat()
                tr["exit_price"] = round(exit_price, 4)
                tr["pnl"] = round(pnl, 2)
                tr["return_pct"] = round(ret_pct, 4)
                tr["status"] = "filled"
                tr["sleeve_cash_after_exit"] = round(sleeve.cash, 2)
                break

    # ── Write detail CSV ──────────────────────────────────────────────────────
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scored_at", "ticker", "event_key", "entry_date", "exit_date",
        "sleeve_id", "status", "entry_price", "exit_price", "shares",
        "notional_invested", "target_notional", "pnl", "return_pct",
        "signal_score", "decile_score", "advised_alloc_frac", "sleeve_cash_after_exit",
    ]
    with detail_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trade_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    filled = [r for r in trade_rows if r.get("return_pct") not in ("", None)]
    rets = [float(r["return_pct"]) for r in filled]
    pnls_filled = [float(r["pnl"]) for r in filled if r.get("pnl") not in ("", None)]
    total_pnl = sum(p.realized_pnl for p in sleeves.values())
    final_equity = sum(s.cash for s in sleeves.values()) + sum(
        pos.notional_invested for pos in open_positions
    )

    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS — starting {args.start_date}")
    print(f"{'='*60}")
    print(f"Initial budget:   ${initial_budget:,.2f}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Trades entered:   {len([r for r in trade_rows if r['status'] in ('filled','open')])}")
    print(f"Completed trades: {len(filled)}")
    if rets:
        print(f"\nPer-trade returns (equal-weight):")
        print(f"  Mean:    {mean(rets):+.2f}%")
        print(f"  Median:  {median(rets):+.2f}%")
        print(f"  Win rate: {100*sum(r>0 for r in rets)/len(rets):.1f}%")
        print(f"  Best:    {max(rets):+.2f}%")
        print(f"  Worst:   {min(rets):+.2f}%")
        total_invested_sum = sum(float(r.get("notional_invested") or 0) for r in filled)
        if total_invested_sum > 0:
            weighted_ret = sum(
                float(r["pnl"]) / float(r["notional_invested"])
                for r in filled if r.get("notional_invested")
            ) / len(filled) * 100
            print(f"\nDollar-weighted returns:")
            print(f"  Avg dollar-weighted: {weighted_ret:+.2f}%")
    print(f"\nPortfolio:")
    print(f"  Total realized P&L: ${total_pnl:+,.2f}")
    print(f"  Final equity:       ${final_equity:,.2f}")
    print(f"  Total return:       {100*total_pnl/initial_budget:+.2f}%")
    for sid, sl in sorted(sleeves.items()):
        print(f"  {sid}: cash=${sl.cash:,.2f}  realized_pnl=${sl.realized_pnl:+,.2f}")

    print(f"\nDetail: {detail_out}")

    # Write summary CSV
    summary_rows = [
        {"metric": "start_date", "value": args.start_date},
        {"metric": "initial_budget_usd", "value": initial_budget},
        {"metric": "total_candidates", "value": len(candidates)},
        {"metric": "rejected_gate", "value": len(rejected)},
        {"metric": "trades_entered", "value": len([r for r in trade_rows if r["status"] in ("filled", "open")])},
        {"metric": "completed_trades", "value": len(filled)},
        {"metric": "mean_return_pct", "value": round(mean(rets), 4) if rets else ""},
        {"metric": "median_return_pct", "value": round(median(rets), 4) if rets else ""},
        {"metric": "win_rate_pct", "value": round(100 * sum(r > 0 for r in rets) / len(rets), 2) if rets else ""},
        {"metric": "best_return_pct", "value": round(max(rets), 4) if rets else ""},
        {"metric": "worst_return_pct", "value": round(min(rets), 4) if rets else ""},
        {"metric": "total_realized_pnl_usd", "value": round(total_pnl, 2)},
        {"metric": "final_equity_usd", "value": round(final_equity, 2)},
        {"metric": "total_return_pct", "value": round(100 * total_pnl / initial_budget, 4)},
    ]
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary: {summary_out}")


if __name__ == "__main__":
    main()
