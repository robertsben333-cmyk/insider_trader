from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from live_trading.market_calendar import (
    ET,
    candidate_expiry_datetime,
    intended_entry_from_score,
    is_regular_trading_hours,
    parse_scored_at_utc,
    parse_time_hhmm,
    sleeve_id_for_trade_day,
)
from live_trading.strategy_settings import ACTIVE_STRATEGY, ExecutionPolicy, TradingBudgetConfig
from live_trading.trader_state import SignalCandidate


def _to_float(value: object) -> float:
    try:
        return float(pd.to_numeric(value, errors="coerce"))
    except Exception:
        return float("nan")


def load_signal_candidates(
    snapshot_path: Path,
    *,
    budget_config: TradingBudgetConfig,
    execution_policy: ExecutionPolicy,
    now_et: datetime | None = None,
) -> list[SignalCandidate]:
    if not snapshot_path.exists():
        return []

    df = pd.read_csv(snapshot_path)
    if df.empty:
        return []
    if "is_tradable" in df.columns:
        tradable = pd.to_numeric(df["is_tradable"], errors="coerce").fillna(0).astype(int)
        df = df[tradable == 1].copy()
    if df.empty:
        return []

    buy_cutoff = parse_time_hhmm(execution_policy.buy_cutoff_time)
    current_et = now_et.astimezone(ET) if now_et is not None else None
    out: list[SignalCandidate] = []
    for row in df.to_dict("records"):
        ticker = str(row.get("ticker", "") or "").strip().upper()
        event_key = str(row.get("event_key", "") or "").strip()
        scored_at_raw = str(row.get("scored_at", "") or "").strip()
        if not ticker or not event_key or not scored_at_raw:
            continue
        try:
            scored_utc = parse_scored_at_utc(scored_at_raw)
        except ValueError:
            continue
        intended_entry = intended_entry_from_score(scored_utc)
        expires_at = candidate_expiry_datetime(intended_entry, buy_cutoff)
        if current_et is not None and expires_at < current_et:
            continue
        scored_et = scored_utc.astimezone(ET)
        advised_alloc = _to_float(row.get("advised_allocation_fraction"))
        if not pd.notna(advised_alloc):
            advised_alloc = 0.0
        estimated_decile = _to_float(row.get("estimated_decile_score"))
        if pd.notna(estimated_decile) and float(estimated_decile) < float(ACTIVE_STRATEGY.day1_decile_score_threshold):
            continue
        score_1d = _to_float(row.get("score_1d"))
        pred_mean4 = _to_float(row.get("pred_mean4"))
        signal_score = score_1d if pd.notna(score_1d) else pred_mean4
        buy_price_hint = _to_float(row.get("buy_price"))
        prev_regular_close = _to_float(row.get("prev_regular_close"))
        step_up_from_prev_close_pct = _to_float(row.get("step_up_from_prev_close_pct"))
        entry_bucket = "intraday" if is_regular_trading_hours(scored_et) else "open"
        out.append(
            SignalCandidate(
                candidate_id=f"{event_key}|{scored_at_raw}",
                event_key=event_key,
                ticker=ticker,
                scored_at=scored_at_raw,
                intended_entry_at=intended_entry.isoformat(),
                expires_at=expires_at.isoformat(),
                sleeve_id=sleeve_id_for_trade_day(
                    intended_entry.date(),
                    budget_config.anchor_date(),
                    budget_config.sleeve_count,
                ),
                signal_score=float(signal_score) if pd.notna(signal_score) else 0.0,
                estimated_decile_score=float(estimated_decile) if pd.notna(estimated_decile) else 0.0,
                advised_allocation_fraction=max(0.0, float(advised_alloc)),
                score_column=str(row.get("alert_score_column", "score_1d") or "score_1d"),
                entry_bucket=entry_bucket,
                entry_trade_day=intended_entry.date().isoformat(),
                buy_price_hint=float(buy_price_hint) if pd.notna(buy_price_hint) else None,
                prev_regular_close=float(prev_regular_close) if pd.notna(prev_regular_close) else None,
                step_up_from_prev_close_pct=(
                    float(step_up_from_prev_close_pct) if pd.notna(step_up_from_prev_close_pct) else None
                ),
            )
        )
    out.sort(key=lambda row: (row.intended_entry_at, -row.signal_score, row.ticker))
    return out
