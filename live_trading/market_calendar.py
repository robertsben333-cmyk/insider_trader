from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def market_open_datetime(trade_day: date) -> datetime:
    return datetime.combine(trade_day, MARKET_OPEN, tzinfo=ET)


def market_close_datetime(trade_day: date) -> datetime:
    return datetime.combine(trade_day, MARKET_CLOSE, tzinfo=ET)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5


def next_trading_day(day: date) -> date:
    cur = day
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def add_trading_days(day: date, offset: int) -> date:
    if offset < 0:
        raise ValueError("Negative trading-day offsets are not supported.")
    cur = next_trading_day(day)
    moved = 0
    while moved < offset:
        cur += timedelta(days=1)
        if is_trading_day(cur):
            moved += 1
    return cur


def parse_scored_at_utc(raw: str) -> datetime:
    return datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def parse_time_hhmm(raw: str) -> time:
    return datetime.strptime(str(raw), "%H:%M").time()


def parse_iso_datetime(raw: str) -> datetime:
    normalized = str(raw).replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt


def intended_entry_from_score(scored_utc: datetime) -> datetime:
    scored_et = scored_utc.astimezone(ET)
    trade_day = scored_et.date()

    if not is_trading_day(trade_day):
        next_day = next_trading_day(trade_day + timedelta(days=1))
        return market_open_datetime(next_day)

    open_dt = market_open_datetime(trade_day)
    close_dt = market_close_datetime(trade_day)
    if scored_et < open_dt:
        return open_dt
    if scored_et >= close_dt:
        next_day = next_trading_day(trade_day + timedelta(days=1))
        return market_open_datetime(next_day)
    return scored_et


def exit_at_tplus_open(entry_at: datetime, trading_days: int) -> datetime:
    entry_et = entry_at.astimezone(ET)
    exit_day = add_trading_days(entry_et.date(), trading_days)
    return market_open_datetime(exit_day)


def candidate_expiry_datetime(entry_at: datetime, buy_cutoff_time: time) -> datetime:
    entry_et = entry_at.astimezone(ET)
    expiry = datetime.combine(entry_et.date(), buy_cutoff_time, tzinfo=ET)
    if expiry < entry_et:
        return entry_et
    return expiry


def sleeve_id_for_trade_day(trade_day: date, anchor_day: date, sleeve_count: int) -> str:
    if sleeve_count <= 0:
        raise ValueError("sleeve_count must be positive.")
    ordinal = (trade_day - anchor_day).days % sleeve_count
    return f"sleeve_{ordinal}"
