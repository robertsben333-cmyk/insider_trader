from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Iterable
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from polygon import RESTClient
import os

from live_trading.strategy_settings import LIVE_PATHS, RUNTIME_DEFAULTS


ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


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


def minute_cache_path(cache_dir: Path, ticker: str, d: date) -> Path:
    return cache_dir / f"{ticker}_min_{d:%Y-%m-%d}_{d:%Y-%m-%d}.json"


def day_cache_path(cache_dir: Path, ticker: str, from_d: date, to_d: date) -> Path:
    return cache_dir / f"{ticker}_lkbk_{from_d:%Y-%m-%d}_{to_d:%Y-%m-%d}.json"


def fetch_minute_bars(client: RESTClient, cache_dir: Path, ticker: str, target_date: date) -> list[dict]:
    path = minute_cache_path(cache_dir, ticker, target_date)
    cached = _json_load(path)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=target_date,
            to=target_date,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
        bars = _aggs_to_dicts(aggs)
    except Exception:
        bars = []
    _json_save(path, bars)
    return bars


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


def next_weekday(d: date) -> date:
    out = d
    while out.weekday() >= 5:
        out += timedelta(days=1)
    return out


def parse_scored_at_utc(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def intended_entry_from_score(scored_utc: datetime) -> datetime:
    scored_et = scored_utc.astimezone(ET)
    d = scored_et.date()

    if d.weekday() >= 5:
        nd = next_weekday(d + timedelta(days=1))
        return datetime.combine(nd, MARKET_OPEN, tzinfo=ET)

    open_dt = datetime.combine(d, MARKET_OPEN, tzinfo=ET)
    close_dt = datetime.combine(d, MARKET_CLOSE, tzinfo=ET)
    if scored_et < open_dt:
        return open_dt
    if scored_et >= close_dt:
        nd = next_weekday(d + timedelta(days=1))
        return datetime.combine(nd, MARKET_OPEN, tzinfo=ET)
    return scored_et


def find_close_at_or_after(bars: Iterable[dict], target_ts_ms: int) -> float | None:
    for bar in bars:
        if int(bar.get("t", -1)) >= target_ts_ms:
            close = bar.get("c")
            return float(close) if close is not None else None
    return None


def find_last_close(bars: Iterable[dict]) -> float | None:
    last_close = None
    for bar in bars:
        close = bar.get("c")
        if close is not None:
            last_close = float(close)
    return last_close


def resolve_entry(
    client: RESTClient,
    cache_dir: Path,
    ticker: str,
    target_dt_et: datetime,
) -> tuple[datetime | None, float | None, str]:
    entry_dt = target_dt_et
    for _ in range(7):
        entry_date = entry_dt.date()
        bars = fetch_minute_bars(client, cache_dir, ticker, entry_date)
        if bars:
            entry_px = find_close_at_or_after(bars, int(entry_dt.timestamp() * 1000))
            if entry_px is None:
                entry_px = find_last_close(bars)
                if entry_px is not None:
                    return entry_dt, entry_px, "same_day_last_close_fallback"
            else:
                return entry_dt, entry_px, "minute_close_at_or_after"

        next_d = next_weekday(entry_date + timedelta(days=1))
        entry_dt = datetime.combine(next_d, MARKET_OPEN, tzinfo=ET)
    return None, None, "no_entry_price"


def bar_date_et(bar: dict) -> date | None:
    ts = bar.get("t")
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts) / 1000, tz=UTC).astimezone(ET).date()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_float(raw: str) -> float | None:
    try:
        return float(raw)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest VM-recommended predictions using score-time entry.")
    parser.add_argument("--input", default=LIVE_PATHS.vm_predictions_file)
    parser.add_argument("--cache-dir", default=LIVE_PATHS.cache_dir)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(RUNTIME_DEFAULTS.historical_backtest_horizons))
    parser.add_argument("--summary-out", default=LIVE_PATHS.vm_backtest_summary_file)
    parser.add_argument("--detail-out", default=LIVE_PATHS.vm_backtest_detail_file)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    summary_out = Path(args.summary_out)
    detail_out = Path(args.detail_out)
    rows = load_rows(input_path)
    client = RESTClient(api_key=api_key, retries=3)

    detail_rows: list[dict] = []
    day_cache: dict[tuple[str, str], list[dict]] = {}

    for row in rows:
        ticker = str(row["ticker"])
        scored_at_raw = str(row["scored_at"])
        scored_utc = parse_scored_at_utc(scored_at_raw)
        intended_entry = intended_entry_from_score(scored_utc)
        entry_dt, entry_price, entry_source = resolve_entry(client, cache_dir, ticker, intended_entry)
        if entry_dt is None or entry_price is None or entry_price <= 0:
            continue

        from_d = entry_dt.date()
        to_d = from_d + timedelta(days=max(args.horizons) + 20)
        cache_key = (ticker, f"{from_d:%Y-%m-%d}_{to_d:%Y-%m-%d}")
        if cache_key not in day_cache:
            day_cache[cache_key] = fetch_day_bars(client, cache_dir, ticker, from_d, to_d)
        bars = day_cache[cache_key]
        dated_bars = [(bar_date_et(bar), bar) for bar in bars]
        dated_bars = [(d, b) for d, b in dated_bars if d is not None and d >= entry_dt.date()]
        if not dated_bars:
            continue

        entry_bar_idx = None
        for idx, (bar_d, _bar) in enumerate(dated_bars):
            if bar_d == entry_dt.date():
                entry_bar_idx = idx
                break
        if entry_bar_idx is None:
            continue

        out = {
            "scored_at": scored_at_raw,
            "scored_at_et": scored_utc.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S%z"),
            "ticker": ticker,
            "event_key": row["event_key"],
            "trade_date": row.get("trade_date", ""),
            "score_1d": row.get("score_1d", ""),
            "score_3d": row.get("score_3d", ""),
            "target_return_mode": row.get("target_return_mode", ""),
            "benchmark_ticker": row.get("benchmark_ticker", ""),
            "entry_dt_et": entry_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "entry_price": entry_price,
            "entry_price_source": entry_source,
        }

        for horizon in args.horizons:
            idx = entry_bar_idx + horizon
            close_col = f"close_{horizon}d"
            ret_col = f"ret_{horizon}d_pct"
            exit_col = f"exit_date_{horizon}d"
            if idx < len(dated_bars):
                exit_d, exit_bar = dated_bars[idx]
                exit_close = safe_float(str(exit_bar.get("c", "")))
                out[exit_col] = exit_d.isoformat() if exit_d is not None else ""
                out[close_col] = exit_close if exit_close is not None else ""
                out[ret_col] = ((exit_close / entry_price) - 1.0) * 100.0 if exit_close is not None else ""
            else:
                out[exit_col] = ""
                out[close_col] = ""
                out[ret_col] = ""

        detail_rows.append(out)

    detail_rows.sort(key=lambda r: (r["scored_at"], str(r["ticker"])), reverse=True)

    horizons = list(args.horizons)
    summary_rows: list[dict] = []
    for horizon in horizons:
        ret_col = f"ret_{horizon}d_pct"
        vals = [float(r[ret_col]) for r in detail_rows if str(r.get(ret_col, "")) != ""]
        if not vals:
            summary_rows.append({
                "horizon_days": horizon,
                "n_trades": 0,
                "mean_return_pct": "",
                "median_return_pct": "",
                "win_rate_pct": "",
                "min_return_pct": "",
                "max_return_pct": "",
            })
            continue
        wins = sum(v > 0 for v in vals)
        summary_rows.append({
            "horizon_days": horizon,
            "n_trades": len(vals),
            "mean_return_pct": sum(vals) / len(vals),
            "median_return_pct": median(vals),
            "win_rate_pct": 100.0 * wins / len(vals),
            "min_return_pct": min(vals),
            "max_return_pct": max(vals),
        })

    detail_out.parent.mkdir(parents=True, exist_ok=True)
    with detail_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()) if detail_rows else [])
        if detail_rows:
            writer.writeheader()
            writer.writerows(detail_rows)

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "horizon_days",
                "n_trades",
                "mean_return_pct",
                "median_return_pct",
                "win_rate_pct",
                "min_return_pct",
                "max_return_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"detail_rows={len(detail_rows)}")
    print(f"detail_out={detail_out}")
    print(f"summary_out={summary_out}")
    for row in summary_rows:
        print(
            "horizon={horizon_days}d n={n_trades} mean={mean_return_pct} median={median_return_pct} "
            "win_rate={win_rate_pct}".format(**row)
        )


if __name__ == "__main__":
    main()
