"""
Scatter-plot analysis: returns vs 30-day price change before insider trade
==========================================================================
For each insider purchase we compute:
    prior_30d_pct = (insider_price - close_30d_before) / close_30d_before * 100

Where:
  - insider_price   = last_price column (OpenInsider reported trade price)
  - close_30d_before = closing price ~30 calendar days before trade_date
                       (taken from the nearest daily bar on or before that date)

Then scatter-plots for each holding horizon (1d, 3d, 5d, 10d).
Polygon daily bars are cached to data/price_cache/ like the main backtest.
"""

import os
import re
import json
import threading
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from polygon import RESTClient
from tqdm import tqdm

# ── Suppress noise ───────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("momentum")

# ── Config ──────────────────────────────────────────────────────────
BACKTEST_CSV  = "data/backtest_results.csv"
OUT_DIR       = "data/charts"
CACHE_DIR     = "data/price_cache"
HORIZONS      = [1, 3, 5, 10]
MAX_WORKERS   = 10
LOOKBACK_DAYS = 30   # calendar days before trade_date
FETCH_WINDOW  = 50   # fetch 50 days of history to ensure enough trading days

# ── Price cache (reuse existing) ─────────────────────────────────────
class PriceCache:
    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, ticker, kind, f, t):
        return self.dir / f"{ticker}_{kind}_{f}_{t}.json"

    def get(self, ticker, kind, f, t):
        p = self._path(ticker, kind, f, t)
        if p.exists():
            try:
                with open(p) as fh:
                    return json.load(fh)
            except Exception:
                return None
        return None

    def put(self, ticker, kind, f, t, data):
        p = self._path(ticker, kind, f, t)
        with self._lock:
            with open(p, "w") as fh:
                json.dump(data, fh)

# ── Thread-local Polygon client ──────────────────────────────────────
_thread_local = threading.local()

def _get_client(api_key):
    if not hasattr(_thread_local, "client"):
        _thread_local.client = RESTClient(api_key=api_key, retries=3)
    return _thread_local.client

# ── Helpers ──────────────────────────────────────────────────────────
def clean_money(s):
    if not isinstance(s, str):
        try:
            return float(s)
        except Exception:
            return np.nan
    s = s.replace("+", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def fetch_daily_bars(client, cache, ticker, from_d, to_d):
    fs, ts = from_d.strftime("%Y-%m-%d"), to_d.strftime("%Y-%m-%d")
    cached = cache.get(ticker, "lkbk", fs, ts)
    if cached is not None:
        return cached
    try:
        aggs = client.get_aggs(
            ticker=ticker, multiplier=1, timespan="day",
            from_=from_d, to=to_d,
            adjusted=True, sort="asc", limit=50000,
        )
        bars = [{"t": a.timestamp, "c": a.close} for a in aggs if a.timestamp and a.close]
    except Exception:
        bars = []
    cache.put(ticker, "lkbk", fs, ts, bars)
    return bars


def find_close_on_or_before(bars, target_date):
    """Return close price of last bar on or before target_date."""
    ET = ZoneInfo("America/New_York")
    result = None
    for bar in bars:
        bar_date = pd.Timestamp(bar["t"], unit="ms", tz="UTC").tz_convert(ET).date()
        if bar_date <= target_date:
            result = bar["c"]
        else:
            break
    return result


# ── Worker: fetch lookback bars for one (ticker, trade_date) group ───
def process_group(ticker, trade_date, insider_price, api_key, cache):
    from_d = trade_date - timedelta(days=FETCH_WINDOW)
    to_d   = trade_date
    client = _get_client(api_key)
    bars   = fetch_daily_bars(client, cache, ticker, from_d, to_d)

    target_date = trade_date - timedelta(days=LOOKBACK_DAYS)
    close_30d   = find_close_on_or_before(bars, target_date)

    if close_30d is None or close_30d == 0:
        return None

    prior_30d_pct = (insider_price - close_30d) / close_30d * 100
    return prior_30d_pct


# ── Binned-average helper ────────────────────────────────────────────
def binned_avg(x, y, n_bins=40):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= 15:
            centres.append((lo + hi) / 2)
            means.append(y[mask].mean())
    return np.array(centres), np.array(means)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise SystemExit("Set POLYGON_API_KEY in .env")

    cache = PriceCache(CACHE_DIR)

    # Load backtest results
    df = pd.read_csv(BACKTEST_CSV)
    df["trade_date_d"]     = pd.to_datetime(df["trade_date"]).dt.date
    df["last_price_clean"] = df["last_price"].apply(clean_money)
    df = df[df["last_price_clean"].notna() & (df["last_price_clean"] > 0)].copy()
    logger.info(f"Loaded {len(df):,} trades")

    # Unique (ticker, trade_date) pairs
    pairs = (
        df[["ticker", "trade_date_d", "last_price_clean"]]
        .drop_duplicates(["ticker", "trade_date_d"])
        .reset_index(drop=True)
    )
    logger.info(f"Fetching 30-day lookback for {len(pairs):,} unique (ticker, trade_date) pairs...")

    # Concurrent fetch
    results: Dict[Tuple[str, object], Optional[float]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_group,
                row.ticker, row.trade_date_d, row.last_price_clean,
                api_key, cache
            ): (row.ticker, row.trade_date_d)
            for row in pairs.itertuples()
        }
        with tqdm(total=len(futures), desc="Lookback", unit="pair") as pbar:
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = None
                pbar.update(1)

    # Map results back to main df
    df["prior_30d_pct"] = df.apply(
        lambda r: results.get((r["ticker"], r["trade_date_d"])), axis=1
    )

    valid = df["prior_30d_pct"].notna().sum()
    logger.info(f"Prior 30d momentum computed for {valid:,} / {len(df):,} trades")

    # Cap extremes for plotting (keep ±100%)
    df_plot = df[df["prior_30d_pct"].between(-100, 200)].copy()
    logger.info(f"After capping prior_30d_pct to [-100%, +200%]: {len(df_plot):,} trades")

    # ── Plot ──────────────────────────────────────────────────────────
    for h in HORIZONS:
        ret_col = f"return_{h}d_pct"
        sub = df_plot[["prior_30d_pct", ret_col]].dropna()

        ret_lo = sub[ret_col].quantile(0.005)
        ret_hi = sub[ret_col].quantile(0.995)

        fig, ax = plt.subplots(figsize=(13, 7))

        ax.scatter(sub["prior_30d_pct"], sub[ret_col],
                   s=4, alpha=0.12, color="#5584AC", edgecolors="none", rasterized=True)

        bx, by = binned_avg(sub["prior_30d_pct"].values, sub[ret_col].values, n_bins=40)
        ax.plot(bx, by, color="#FF6B35", linewidth=2.5, label="Binned average", zorder=5)

        ax.axhline(0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")
        ax.axvline(0, color="#AAAAAA", linewidth=0.8, alpha=0.5, linestyle=":")

        # Shade: down-trend vs up-trend prior to insider buying
        ax.axvspan(sub["prior_30d_pct"].min(), 0, alpha=0.04, color="#FF4444", label="Stock fell 30d before trade")
        ax.axvspan(0, sub["prior_30d_pct"].max(), alpha=0.04, color="#00C851", label="Stock rose 30d before trade")

        ax.set_xlim(sub["prior_30d_pct"].min(), sub["prior_30d_pct"].max())
        ax.set_ylim(ret_lo, ret_hi)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))

        median_prior = sub["prior_30d_pct"].median()
        ax.set_xlabel("30-Day Price Change Before Insider Trade", fontsize=13)
        ax.set_ylabel(f"{h}-Day Return (%)", fontsize=13)
        ax.set_title(
            f"Insider Officer Purchases — {h}-Day Return vs Prior 30-Day Momentum\n"
            f"({len(sub):,} trades  |  median prior 30d: {median_prior:+.1f}%)",
            fontsize=15, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="upper right")

        # Dark theme
        fig.patch.set_facecolor("#1E1E1E")
        ax.set_facecolor("#2B2B2B")
        ax.tick_params(colors="white", labelsize=11)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.legend(fontsize=11, loc="upper right",
                  facecolor="#2B2B2B", edgecolor="white", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.grid(True, alpha=0.2, color="white")

        out_path = f"{OUT_DIR}/scatter_return_{h}d_vs_prior30d.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved {out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
