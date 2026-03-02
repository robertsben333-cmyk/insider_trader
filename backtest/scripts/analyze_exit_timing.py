"""
Exit timing sensitivity analysis on executed test-set trades.

Uses the existing executed trade log (`backtest/out/testset_trade_log.csv`) as
the fixed entry set, then compares multiple exit schedules for the same trades.

Scenarios:
  - same_day_14_00
  - same_day_15_00
  - same_day_15_30
  - same_day_close
  - next_day_open
  - next_day_10_30
  - next_day_12_00
  - next_day_14_00
  - next_day_15_30
  - next_day_close   (current baseline)
  - second_day_close

Outputs:
  - backtest/out/exit_timing_detail.csv
  - backtest/out/exit_timing_summary.csv
  - backtest/out/exit_timing_paired_vs_next_day_close.csv
"""

from __future__ import annotations

import json
import math
import os
import threading
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

BASE = Path(__file__).resolve().parents[2]
TRADE_LOG = BASE / "backtest" / "out" / "testset_trade_log.csv"
CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUT_DIR = BASE / "backtest" / "out"

ET = ZoneInfo("America/New_York")
MAX_WORKERS = 10
BASELINE_SCENARIO = "next_day_close"


@dataclass(frozen=True)
class ExitScenario:
    name: str
    kind: str  # "close" or "time"
    day_offset: int
    hour: int = 16
    minute: int = 0


SCENARIOS: List[ExitScenario] = [
    ExitScenario("same_day_14_00", "time", 0, 14, 0),
    ExitScenario("same_day_15_00", "time", 0, 15, 0),
    ExitScenario("same_day_15_30", "time", 0, 15, 30),
    ExitScenario("same_day_close", "close", 0, 16, 0),
    ExitScenario("next_day_open", "time", 1, 9, 30),
    ExitScenario("next_day_10_30", "time", 1, 10, 30),
    ExitScenario("next_day_12_00", "time", 1, 12, 0),
    ExitScenario("next_day_14_00", "time", 1, 14, 0),
    ExitScenario("next_day_15_30", "time", 1, 15, 30),
    ExitScenario("next_day_close", "close", 1, 16, 0),
    ExitScenario("second_day_close", "close", 2, 16, 0),
]


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


_thread_local = threading.local()


def _get_client(api_key: str) -> RESTClient:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = RESTClient(api_key=api_key, retries=3)
    return _thread_local.client


def _aggs_to_dicts(aggs) -> list:
    return [
        {"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
        for a in aggs
        if a.timestamp and a.close
    ]


def fetch_daily_bars(
    client: RESTClient,
    cache: PriceCache,
    ticker: str,
    from_d: date,
    to_d: date,
) -> list:
    fs = from_d.strftime("%Y-%m-%d")
    ts = to_d.strftime("%Y-%m-%d")
    cached = cache.get(ticker, "day", fs, ts)
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
    cache.put(ticker, "day", fs, ts, bars)
    return bars


def fetch_minute_bars(
    client: RESTClient,
    cache: PriceCache,
    ticker: str,
    d: date,
) -> list:
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
        bars = _aggs_to_dicts(aggs)
    except Exception:
        bars = []
    cache.put(ticker, "min", ds, ds, bars)
    return bars


def _trading_dates_from_daily(daily_bars: List[dict]) -> List[date]:
    out = []
    for bar in daily_bars:
        d = datetime.fromtimestamp(bar["t"] / 1000, tz=ET).date()
        out.append(d)
    return sorted(set(out))


def _daily_map(daily_bars: List[dict]) -> Dict[date, dict]:
    out: Dict[date, dict] = {}
    for bar in daily_bars:
        d = datetime.fromtimestamp(bar["t"] / 1000, tz=ET).date()
        out[d] = bar
    return out


def _ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _price_at_or_after(minute_bars: List[dict], target_ms: int) -> Optional[float]:
    for bar in minute_bars:
        if bar["t"] >= target_ms:
            return bar["c"]
    return None


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _paired_stats(diff: pd.Series) -> dict:
    n = int(diff.notna().sum())
    if n == 0:
        return {
            "shared_n": 0,
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "z_stat_approx": float("nan"),
            "p_value_approx": float("nan"),
        }

    mean_delta = float(diff.mean())
    std = float(diff.std(ddof=1)) if n > 1 else float("nan")
    se = std / math.sqrt(n) if n > 1 and np.isfinite(std) else float("nan")

    if n > 1 and np.isfinite(se) and se > 0:
        z = mean_delta / se
        p = 2.0 * (1.0 - _normal_cdf(abs(z)))
        ci_low = mean_delta - 1.96 * se
        ci_high = mean_delta + 1.96 * se
    else:
        z = float("nan")
        p = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "shared_n": n,
        "mean_delta": mean_delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "z_stat_approx": z,
        "p_value_approx": p,
    }


def _ret_per_24h_linear(ret_pct: float, hold_hours: float) -> float:
    if not np.isfinite(ret_pct) or not np.isfinite(hold_hours) or hold_hours <= 0:
        return float("nan")
    return float(ret_pct) / (float(hold_hours) / 24.0)


def _ret_per_24h_comp(ret_pct: float, hold_hours: float) -> float:
    if not np.isfinite(ret_pct) or not np.isfinite(hold_hours) or hold_hours <= 0:
        return float("nan")
    gross = 1.0 + float(ret_pct) / 100.0
    if gross <= 0:
        return float("nan")
    hold_days = float(hold_hours) / 24.0
    if hold_days <= 0:
        return float("nan")
    return (gross ** (1.0 / hold_days) - 1.0) * 100.0


def process_ticker(
    ticker: str,
    rows: List[dict],
    api_key: str,
    cache: PriceCache,
) -> List[dict]:
    client = _get_client(api_key)
    entry_dates = [r["entry_dt"].date() for r in rows]
    from_d = min(entry_dates) - timedelta(days=4)
    to_d = max(entry_dates) + timedelta(days=12)

    daily_bars = fetch_daily_bars(client, cache, ticker, from_d, to_d)
    if not daily_bars:
        return []

    trading_dates = _trading_dates_from_daily(daily_bars)
    daily_by_date = _daily_map(daily_bars)
    if not trading_dates:
        return []

    row_info: Dict[int, dict] = {}
    minute_dates_needed: set[date] = set()

    for r in rows:
        trade_id = int(r["trade_id"])
        entry_dt = r["entry_dt"]
        entry_date = entry_dt.date()

        base_idx = bisect_left(trading_dates, entry_date)
        if base_idx >= len(trading_dates):
            continue

        entry_trade_date = trading_dates[base_idx]
        minute_dates_needed.add(entry_trade_date)

        scenario_targets: Dict[str, dict] = {}
        for sc in SCENARIOS:
            td_idx = base_idx + sc.day_offset
            if td_idx >= len(trading_dates):
                continue
            exit_date = trading_dates[td_idx]
            if sc.kind == "time":
                exit_dt = datetime.combine(exit_date, time(sc.hour, sc.minute), tzinfo=ET)
                if sc.day_offset == 0 and exit_dt <= entry_dt:
                    continue
                minute_dates_needed.add(exit_date)
                scenario_targets[sc.name] = {
                    "kind": "time",
                    "exit_date": exit_date,
                    "exit_dt": exit_dt,
                }
            else:
                exit_dt = datetime.combine(exit_date, time(16, 0), tzinfo=ET)
                scenario_targets[sc.name] = {
                    "kind": "close",
                    "exit_date": exit_date,
                    "exit_dt": exit_dt,
                }

        row_info[trade_id] = {
            "entry_dt": entry_dt,
            "entry_trade_date": entry_trade_date,
            "scenario_targets": scenario_targets,
            "baseline_ret_pct": r["baseline_ret_pct"],
        }

    minute_bars_by_date: Dict[date, list] = {}
    for d in sorted(minute_dates_needed):
        minute_bars_by_date[d] = fetch_minute_bars(client, cache, ticker, d)

    out_rows: List[dict] = []
    for r in rows:
        trade_id = int(r["trade_id"])
        info = row_info.get(trade_id)
        if info is None:
            continue

        entry_dt = info["entry_dt"]
        entry_trade_date = info["entry_trade_date"]
        entry_bars = minute_bars_by_date.get(entry_trade_date, [])
        entry_price = _price_at_or_after(entry_bars, _ts_ms(entry_dt))
        if entry_price is None or entry_price <= 0:
            continue

        for sc in SCENARIOS:
            target = info["scenario_targets"].get(sc.name)
            if target is None:
                continue

            exit_dt = target["exit_dt"]
            exit_price: Optional[float]
            if target["kind"] == "close":
                day_bar = daily_by_date.get(target["exit_date"])
                exit_price = day_bar["c"] if day_bar else None
            else:
                mb = minute_bars_by_date.get(target["exit_date"], [])
                exit_price = _price_at_or_after(mb, _ts_ms(exit_dt))

            if exit_price is None or exit_price <= 0:
                continue

            ret_pct = (float(exit_price) / float(entry_price) - 1.0) * 100.0
            hold_hours = (exit_dt - entry_dt).total_seconds() / 3600.0
            out_rows.append(
                {
                    "trade_id": trade_id,
                    "ticker": ticker,
                    "scenario": sc.name,
                    "entry_dt": entry_dt.isoformat(),
                    "exit_dt": exit_dt.isoformat(),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "ret_pct": float(ret_pct),
                    "hold_hours": float(hold_hours),
                    "hold_days": float(hold_hours) / 24.0,
                    "ret_per_24h_linear_pct": _ret_per_24h_linear(float(ret_pct), float(hold_hours)),
                    "ret_per_24h_comp_pct": _ret_per_24h_comp(float(ret_pct), float(hold_hours)),
                    "baseline_ret_pct_from_log": float(info["baseline_ret_pct"]),
                }
            )

    return out_rows


def main() -> None:
    load_dotenv(BASE / ".env")
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    if not TRADE_LOG.exists():
        raise FileNotFoundError(f"Missing trade log: {TRADE_LOG}")

    trades = pd.read_csv(TRADE_LOG, parse_dates=["entry_time", "exit_time"])
    if trades.empty:
        raise RuntimeError("Trade log is empty; nothing to analyze.")

    trades = trades.reset_index(drop=True).copy()
    trades["trade_id"] = trades.index
    trades["ret_pct"] = pd.to_numeric(trades["ret_pct"], errors="coerce")

    grouped: Dict[str, List[dict]] = {}
    for _, row in trades.iterrows():
        ticker = str(row["ticker"])
        entry_dt = pd.to_datetime(row["entry_time"], errors="coerce")
        if pd.isna(entry_dt):
            continue
        py_dt = entry_dt.to_pydatetime()
        if py_dt.tzinfo is None:
            py_dt = py_dt.replace(tzinfo=ET)
        else:
            py_dt = py_dt.astimezone(ET)

        grouped.setdefault(ticker, []).append(
            {
                "trade_id": int(row["trade_id"]),
                "entry_dt": py_dt,
                "baseline_ret_pct": float(row["ret_pct"]) if pd.notna(row["ret_pct"]) else float("nan"),
            }
        )

    cache = PriceCache(CACHE_DIR)

    all_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_ticker, ticker, rows, api_key, cache): ticker
            for ticker, rows in grouped.items()
        }
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                print(f"[WARN] {ticker}: {exc}")
            done += 1
            if done % 50 == 0 or done == total:
                print(f"Processed {done}/{total} tickers")

    if not all_rows:
        raise RuntimeError("No exit-timing rows produced.")

    detail_df = pd.DataFrame(all_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_csv = OUT_DIR / "exit_timing_detail.csv"
    summary_csv = OUT_DIR / "exit_timing_summary.csv"
    paired_csv = OUT_DIR / "exit_timing_paired_vs_next_day_close.csv"

    detail_df.to_csv(detail_csv, index=False)

    summary_rows = []
    for sc in SCENARIOS:
        sub = detail_df[detail_df["scenario"] == sc.name]
        r = pd.to_numeric(sub["ret_pct"], errors="coerce").dropna()
        summary_rows.append(
            {
                "scenario": sc.name,
                "n_trades": int(len(r)),
                "avg_ret_pct": float(r.mean()) if len(r) else float("nan"),
                "median_ret_pct": float(r.median()) if len(r) else float("nan"),
                "win_rate_pct": float((r > 0).mean() * 100.0) if len(r) else float("nan"),
                "std_ret_pct": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
                "sum_ret_pct": float(r.sum()) if len(r) else float("nan"),
                "avg_hold_hours": float(pd.to_numeric(sub["hold_hours"], errors="coerce").mean()) if len(sub) else float("nan"),
                "median_hold_hours": float(pd.to_numeric(sub["hold_hours"], errors="coerce").median()) if len(sub) else float("nan"),
                "avg_hold_days": float(pd.to_numeric(sub["hold_days"], errors="coerce").mean()) if len(sub) else float("nan"),
                "avg_ret_per_24h_linear_pct": float(pd.to_numeric(sub["ret_per_24h_linear_pct"], errors="coerce").mean()) if len(sub) else float("nan"),
                "median_ret_per_24h_linear_pct": float(pd.to_numeric(sub["ret_per_24h_linear_pct"], errors="coerce").median()) if len(sub) else float("nan"),
                "avg_ret_per_24h_comp_pct": float(pd.to_numeric(sub["ret_per_24h_comp_pct"], errors="coerce").mean()) if len(sub) else float("nan"),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)

    base = detail_df[detail_df["scenario"] == BASELINE_SCENARIO][["trade_id", "ret_pct"]].rename(
        columns={"ret_pct": "ret_base"}
    )
    pair_rows = []
    for sc in SCENARIOS:
        if sc.name == BASELINE_SCENARIO:
            continue
        cur = detail_df[detail_df["scenario"] == sc.name][["trade_id", "ret_pct"]].rename(columns={"ret_pct": "ret_alt"})
        merged = base.merge(cur, on="trade_id", how="inner")
        diff = pd.to_numeric(merged["ret_alt"], errors="coerce") - pd.to_numeric(merged["ret_base"], errors="coerce")
        st = _paired_stats(diff.dropna())
        pair_rows.append({"scenario_vs_next_day_close": sc.name, **st})

    paired_df = pd.DataFrame(pair_rows)
    paired_df.to_csv(paired_csv, index=False)

    # Sanity check: recomputed next_day_close vs saved log return.
    check = detail_df[detail_df["scenario"] == BASELINE_SCENARIO].copy()
    check["abs_diff_vs_log"] = (
        pd.to_numeric(check["ret_pct"], errors="coerce")
        - pd.to_numeric(check["baseline_ret_pct_from_log"], errors="coerce")
    ).abs()
    mean_abs_diff = float(check["abs_diff_vs_log"].dropna().mean()) if not check.empty else float("nan")

    best_row = summary_df.sort_values("avg_ret_pct", ascending=False).iloc[0]

    print("\nSaved:")
    print(f"  {detail_csv}")
    print(f"  {summary_csv}")
    print(f"  {paired_csv}")
    print("\nSummary (sorted by avg_ret_pct):")
    print(summary_df.sort_values("avg_ret_pct", ascending=False).to_string(index=False))
    print("\nPaired deltas vs next_day_close:")
    print(paired_df.sort_values("mean_delta", ascending=False).to_string(index=False))
    print(f"\nBaseline sanity check | mean abs diff vs trade log return: {mean_abs_diff:.6f} pp")
    print(f"Top scenario by average return: {best_row['scenario']} ({best_row['avg_ret_pct']:+.4f}%)")


if __name__ == "__main__":
    main()
