"""
Variable Stop-Loss Analysis  (Training -> Validation -> Test)
==============================================================
Finds the optimal per-segment stop-loss multiplier k using only training data,
then validates on the val set, and finally applies to the test set.

Stop rule:
    stop_pct = -k * daily_vol_pct
    where daily_vol_pct = prior_30d_vol / sqrt(252) * 100
    Exit if intraday low < entry_price * (1 + stop_pct/100)

Segments (to keep cells large enough):
    sector_group : Biotech, Finance, Tech, Other
    price_tier   : micro (<$2), small ($2-$10), large (>$10)
    pred_tier    : within top-decile trades, bottom half vs top half of pred_mean4

k sweep: 0.5 ... 4.0 step 0.25  (plus "none" = no stop)

Methodology:
1. Load & score full dataset, train/val/test split 60/20/20 (chronological)
2. Filter to top-decile predictions (same threshold as backtest)
3. Fetch intraday minute bars for training AND validation trades (caching)
4. Sweep k across segments on TRAIN; pick best k per segment by sum-of-returns
5. Evaluate chosen k on VAL set for out-of-sample confirmation
6. Apply per-segment k to TEST set; compare to baseline and best fixed stop

Usage:
    py backtest/scripts/analyze_variable_stoploss.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from model_ensemble import MODEL_NAMES, predict_model, to_linear_numeric, to_xgb
from train_models import FEATURES, engineer_features, load_and_merge

# ─────────────────────── Config ────────────────────────────────────────

MODEL_DIR  = BASE / "models" / "prod4"
CACHE_DIR  = BASE / "backtest" / "data" / "price_cache"
OUT_DIR    = BASE / "backtest" / "out"
ET         = ZoneInfo("America/New_York")

HORIZON_DAYS    = 1
MAX_WORKERS     = 10
K_VALUES        = np.round(np.arange(0.5, 4.25, 0.25), 2).tolist()   # 0.5..4.0

# ─────────────────────── Segment definitions ───────────────────────────

SECTOR_MAP = {
    "Health Care": "Biotech",
    "Financials":  "Finance",
    "Information Technology": "Tech",
    "Communication Services": "Tech",
}
def sector_group(s: str) -> str:
    return SECTOR_MAP.get(s, "Other")

def price_tier(price: float) -> str:
    if price < 2.0:  return "micro"
    if price < 10.0: return "small"
    return "large"

def pred_tier(pred: float, med: float) -> str:
    return "high" if pred >= med else "low"


# ─────────────────────── Price Cache ───────────────────────────────────

class PriceCache:
    def __init__(self, d: Path):
        self.dir = d
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, ticker, kind, f, t): return self.dir / f"{ticker}_{kind}_{f}_{t}.json"

    def get(self, ticker, kind, f, t):
        p = self._path(ticker, kind, f, t)
        if p.exists():
            try:
                with open(p) as fh: return json.load(fh)
            except Exception: return None
        return None

    def put(self, ticker, kind, f, t, data):
        p = self._path(ticker, kind, f, t)
        with self._lock:
            with open(p, "w") as fh: json.dump(data, fh)


_tls = threading.local()
def _get_client(api_key):
    if not hasattr(_tls, "client"):
        _tls.client = RESTClient(api_key)
    return _tls.client


def fetch_minute_bars(api_key: str, cache: PriceCache, ticker: str, d: date) -> list:
    ds = d.strftime("%Y-%m-%d")
    cached = cache.get(ticker, "min", ds, ds)
    if cached is not None:
        return cached
    try:
        client = _get_client(api_key)
        aggs = client.get_aggs(
            ticker=ticker, multiplier=1, timespan="minute",
            from_=d, to=d, adjusted=True, sort="asc", limit=50000,
        )
        bars = [{"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
                for a in aggs if a.timestamp and a.close]
    except Exception as e:
        print(f"  [WARN] {ticker} {ds}: {e}")
        bars = []
    cache.put(ticker, "min", ds, ds, bars)
    return bars


def trading_days_between(start: date, end: date) -> List[date]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5: days.append(cur)
        cur += timedelta(days=1)
    return days


def intraday_low_pct(all_bars: list, entry_ts_ms: int, exit_ts_ms: int) -> Optional[float]:
    """Min intraday low as % from entry bar open, over entire holding period."""
    relevant = [b for b in all_bars if b["t"] >= entry_ts_ms and b["t"] <= exit_ts_ms]
    if not relevant: return None
    ep = relevant[0]["o"]
    if not ep or ep <= 0: return None
    return (min(b["l"] for b in relevant) / ep - 1.0) * 100.0


def fetch_path_for_trade(api_key, cache, ticker, entry_dt, exit_dt) -> Optional[float]:
    hold_dates = trading_days_between(entry_dt.date(), exit_dt.date())
    all_bars: list = []
    for d in hold_dates:
        all_bars.extend(fetch_minute_bars(api_key, cache, ticker, d))
    all_bars.sort(key=lambda b: b["t"])
    entry_ts = int(entry_dt.timestamp() * 1000)
    exit_ts  = int(exit_dt.timestamp() * 1000)
    return intraday_low_pct(all_bars, entry_ts, exit_ts)


# ─────────────────────── Model scoring (same as backtest script) ────────

def align_features(model, X):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None: return X
    return X[[str(c) for c in cols]].copy()

def predict_aligned(name, model, X):
    Xa = align_features(model, X)
    if name == "HGBR":        return model.predict(Xa)
    if name == "XGBoost":     return model.predict(to_xgb(Xa))
    if name == "ElasticNet":  return model.predict(to_linear_numeric(Xa))
    if name == "SplineElasticNet": return model.predict(Xa.copy())
    return predict_model(name, model, Xa)

def chrono_split_60_20_20(n):
    n_train = max(1, min(int(round(n * 0.6)), n - 2))
    n_val   = max(1, min(int(round(n * 0.2)), n - n_train - 1))
    return n_train, n_train + n_val

def decile9_lower_bound(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    order  = np.argsort(signal)
    n = len(order)
    bs = max(1, n // 10)
    idx = order[8 * bs: min(n, 9 * bs)]
    return float(np.min(signal[idx])) if len(idx) else float(np.quantile(signal, 0.8))

def next_business_day_close(ts):
    return (ts + pd.offsets.BDay(1)).normalize() + pd.Timedelta(hours=16)


def load_and_score() -> pd.DataFrame:
    print("Loading and scoring full dataset...")
    df_raw = load_and_merge()
    df, features, _ = engineer_features(df_raw)
    assert features == FEATURES

    target_col = f"return_{HORIZON_DAYS}d_pct"
    sub = df.dropna(subset=[target_col]).copy()
    lo, hi = sub[target_col].quantile([0.01, 0.99])
    sub[target_col] = sub[target_col].clip(lo, hi)
    sub["trade_date"]   = pd.to_datetime(sub["trade_date"],   errors="coerce")
    sub["buy_datetime"] = pd.to_datetime(sub["buy_datetime"], errors="coerce")
    sub = sub.dropna(subset=["trade_date", "buy_datetime"]).sort_values("trade_date").reset_index(drop=True)

    X = sub[features].copy()
    n_train, split_80 = chrono_split_60_20_20(len(sub))

    models = {name: joblib.load(MODEL_DIR / f"model_{HORIZON_DAYS}d_{name}.pkl") for name in MODEL_NAMES}
    pred_cols = []
    for name, mdl in models.items():
        col = f"pred_{name}"
        sub[col] = predict_aligned(name, mdl, X)
        pred_cols.append(col)
    sub["pred_mean4"] = sub[pred_cols].mean(axis=1)

    train_signal = sub.loc[:n_train - 1, "pred_mean4"].to_numpy(dtype=float)
    threshold    = decile9_lower_bound(train_signal)
    print(f"  Decile-9 threshold: {threshold:.4f}%   |  train rows: {n_train}  |  val starts at: {split_80}")

    sub["split"] = "train"
    sub.loc[n_train:split_80 - 1, "split"] = "val"
    sub.loc[split_80:, "split"]            = "test"
    sub["exit_datetime"] = sub["buy_datetime"].apply(next_business_day_close)

    # Keep only top-decile predictions per split
    picks = sub[sub["pred_mean4"] > threshold].copy()
    picks["ret_pct"] = pd.to_numeric(picks[f"return_{HORIZON_DAYS}d_pct"], errors="coerce")
    picks = picks.dropna(subset=["ret_pct"]).reset_index(drop=True)

    # Derive stop-calibration features
    picks["daily_vol_pct"] = picks["prior_30d_vol"].clip(lower=0) / np.sqrt(252) * 100
    picks["entry_price"]   = picks["last_price_clean"].clip(lower=0.01)
    picks["sec_group"]     = picks["sector"].apply(sector_group)
    picks["price_tier"]    = picks["entry_price"].apply(price_tier)

    # Pred tier within top-decile: split at median pred_mean4 of TRAIN picks
    train_med = picks.loc[picks["split"] == "train", "pred_mean4"].median()
    picks["pred_tier"] = picks["pred_mean4"].apply(lambda p: pred_tier(p, train_med))
    picks["segment"]   = picks["sec_group"] + "_" + picks["price_tier"] + "_" + picks["pred_tier"]

    print(f"  Top-decile picks: train={int((picks['split']=='train').sum())}  "
          f"val={int((picks['split']=='val').sum())}  "
          f"test={int((picks['split']=='test').sum())}")
    return picks, threshold


# ─────────────────────── Intraday fetch (threaded) ──────────────────────

def fetch_all_paths(df: pd.DataFrame, api_key: str, cache: PriceCache, label: str) -> pd.Series:
    """Fetch intraday low_pct for every row in df; return as Series indexed like df."""
    tasks = {}
    results = {}

    def _work(idx, ticker, entry_dt, exit_dt):
        lp = fetch_path_for_trade(api_key, cache, ticker, entry_dt, exit_dt)
        return idx, lp

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                _work,
                idx,
                row["ticker"],
                row["buy_datetime"].to_pydatetime().replace(tzinfo=ET),
                row["exit_datetime"].to_pydatetime().replace(tzinfo=ET),
            ): idx
            for idx, row in df.iterrows()
        }
        done = 0
        for fut in as_completed(futures):
            idx, lp = fut.result()
            results[idx] = lp
            done += 1
            if done % 50 == 0 or done == len(futures):
                print(f"  [{label}] fetched {done}/{len(futures)}", end="\r", flush=True)
    print()
    return pd.Series(results, name="low_pct")


# ─────────────────────── Stop simulation ────────────────────────────────

def simulate_stop_k(low_pct: Optional[float], daily_vol_pct: float, k, orig_ret: float) -> Tuple[float, bool]:
    """Return (simulated_ret, was_stopped). k='none' means no stop applied."""
    if k == "none" or low_pct is None or daily_vol_pct <= 0:
        return orig_ret, False
    stop_level = -float(k) * daily_vol_pct
    if low_pct <= stop_level:
        return stop_level, True
    return orig_ret, False


def sweep_k_for_group(rows: pd.DataFrame, k_values: list) -> Tuple[float, dict]:
    """
    Sweep k on the given rows. Return (best_k, stats_per_k) where best_k
    maximises sum of simulated returns.
    """
    best_k, best_sum = "none", float(rows["ret_pct"].sum())
    stats = {"none": dict(sum_ret=best_sum, avg_ret=rows["ret_pct"].mean(),
                          win_rate=(rows["ret_pct"] > 0).mean() * 100,
                          n_stopped=0, n=len(rows))}
    for k in k_values:
        rets, stopped = [], 0
        for _, r in rows.iterrows():
            ret, st = simulate_stop_k(r.get("low_pct"), r["daily_vol_pct"], k, r["ret_pct"])
            rets.append(ret)
            stopped += int(st)
        s = float(np.sum(rets))
        stats[k] = dict(sum_ret=s, avg_ret=float(np.mean(rets)),
                        win_rate=float(np.mean(np.array(rets) > 0) * 100),
                        n_stopped=stopped, n=len(rows))
        if s > best_sum:
            best_sum, best_k = s, k
    return best_k, stats


# ─────────────────────── Main ────────────────────────────────────────────

def main():
    load_dotenv(BASE / ".env")
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set in .env")

    picks, threshold = load_and_score()
    cache = PriceCache(CACHE_DIR)

    train_df = picks[picks["split"] == "train"].copy()
    val_df   = picks[picks["split"] == "val"].copy()
    test_df  = picks[picks["split"] == "test"].copy()

    # ── Fetch intraday paths for train + val ────────────────────────────
    print(f"\nFetching intraday paths for {len(train_df)} TRAIN trades...")
    train_lows = fetch_all_paths(train_df, api_key, cache, "TRAIN")
    train_df["low_pct"] = train_lows

    print(f"\nFetching intraday paths for {len(val_df)} VAL trades...")
    val_lows = fetch_all_paths(val_df, api_key, cache, "VAL")
    val_df["low_pct"] = val_lows

    # ── GLOBAL sweep on training set ────────────────────────────────────
    print("\n" + "="*70)
    print("GLOBAL k SWEEP  (training set)")
    print("="*70)
    best_k_global, global_stats = sweep_k_for_group(train_df, K_VALUES)
    print(f"\n{'k':>8}  {'SumRet':>8}  {'AvgRet':>7}  {'WinRate':>7}  {'Stopped':>7}")
    print("-" * 48)
    for k, s in global_stats.items():
        label = f"{k:.2f}" if k != "none" else "none"
        marker = " <-- best" if k == best_k_global else ""
        print(f"{label:>8}  {s['sum_ret']:>+8.1f}  {s['avg_ret']:>+7.2f}  "
              f"{s['win_rate']:>6.1f}%  {s['n_stopped']:>7}{marker}")
    print(f"\nBest global k on train: {best_k_global}")

    # ── PER-SEGMENT sweep on training set ───────────────────────────────
    print("\n" + "="*70)
    print("PER-SEGMENT k SWEEP  (training set, min 15 trades per segment)")
    print("="*70)

    MIN_SEGMENT_TRADES = 15
    segment_best_k: Dict[str, float] = {}
    seg_train_rows = []

    segments = sorted(train_df["segment"].unique())
    for seg in segments:
        grp = train_df[train_df["segment"] == seg]
        n = len(grp)
        if n < MIN_SEGMENT_TRADES:
            print(f"  {seg:<40}  n={n:3d}  -> too small, will use global k={best_k_global}")
            segment_best_k[seg] = best_k_global
            continue
        bk, sts = sweep_k_for_group(grp, K_VALUES)
        segment_best_k[seg] = bk
        base = sts["none"]["sum_ret"]
        best = sts[bk]["sum_ret"]
        print(f"  {seg:<40}  n={n:3d}  best_k={bk!s:>5}  "
              f"base={base:>+7.1f}  best={best:>+7.1f}  "
              f"delta={best-base:>+6.1f}")
        for k, s in sts.items():
            seg_train_rows.append({
                "segment": seg, "k": k, "n": s["n"],
                "sum_ret": s["sum_ret"], "avg_ret": s["avg_ret"],
                "win_rate": s["win_rate"], "n_stopped": s["n_stopped"],
            })

    # Save segment sweep table
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg_sweep_df = pd.DataFrame(seg_train_rows)
    seg_sweep_df.to_csv(OUT_DIR / "variable_stoploss_segment_sweep.csv", index=False)
    print(f"\nSegment sweep saved -> {OUT_DIR / 'variable_stoploss_segment_sweep.csv'}")

    # ── Apply per-segment k to VAL set ─────────────────────────────────
    print("\n" + "="*70)
    print("VALIDATION SET: per-segment k vs baseline vs best-global k")
    print("="*70)

    def apply_variable_stop(df_in, seg_k_map, global_k, fallback_k=None):
        rets_var, rets_global, rets_base = [], [], []
        for _, r in df_in.iterrows():
            k_seg  = seg_k_map.get(r["segment"], global_k)
            k_fb   = fallback_k if fallback_k is not None else global_k
            rv, _  = simulate_stop_k(r.get("low_pct"), r["daily_vol_pct"], k_seg,  r["ret_pct"])
            rg, _  = simulate_stop_k(r.get("low_pct"), r["daily_vol_pct"], global_k, r["ret_pct"])
            rets_var.append(rv)
            rets_global.append(rg)
            rets_base.append(r["ret_pct"])
        return np.array(rets_base), np.array(rets_global), np.array(rets_var)

    val_base, val_global, val_var = apply_variable_stop(val_df, segment_best_k, best_k_global)

    def print_stats(label, arr):
        print(f"  {label:<25}  sum={arr.sum():>+8.1f}  avg={arr.mean():>+6.2f}  "
              f"win={100*(arr>0).mean():>5.1f}%  worst={arr.min():>+6.2f}")

    print()
    print_stats("Baseline (no stop)", val_base)
    print_stats(f"Global k={best_k_global}",   val_global)
    print_stats("Per-segment k",   val_var)

    # ── Fetch test paths and apply ──────────────────────────────────────
    print(f"\nFetching intraday paths for {len(test_df)} TEST trades...")
    test_lows = fetch_all_paths(test_df, api_key, cache, "TEST")
    test_df["low_pct"] = test_lows

    print("\n" + "="*70)
    print("TEST SET: per-segment k vs baseline vs best-global k")
    print("="*70)
    test_base, test_global, test_var = apply_variable_stop(test_df, segment_best_k, best_k_global)

    print()
    print_stats("Baseline (no stop)", test_base)
    print_stats(f"Global k={best_k_global}", test_global)
    print_stats("Per-segment k",  test_var)

    # ── Full detail CSV for test ─────────────────────────────────────────
    test_out = test_df[["ticker", "buy_datetime", "exit_datetime", "split",
                         "sector", "sec_group", "price_tier", "pred_tier",
                         "segment", "entry_price", "daily_vol_pct",
                         "pred_mean4", "ret_pct", "low_pct"]].copy()
    test_out["k_applied"] = test_out["segment"].map(segment_best_k).fillna(best_k_global)
    # "none" means no stop; convert to NaN for arithmetic (NaN stop_level_pct = no stop)
    k_numeric = pd.to_numeric(test_out["k_applied"], errors="coerce")
    test_out["stop_level_pct"] = -(k_numeric * test_out["daily_vol_pct"])
    test_out["ret_variable_stop"] = test_var
    test_out["ret_global_stop"]   = test_global
    test_out["ret_baseline"]      = test_base
    test_out.to_csv(OUT_DIR / "variable_stoploss_test_detail.csv", index=False)
    print(f"\nTest detail saved -> {OUT_DIR / 'variable_stoploss_test_detail.csv'}")

    # ── Segment-level breakdown on test ─────────────────────────────────
    print("\n" + "="*70)
    print("TEST SET: breakdown by segment")
    print("="*70)
    test_out["delta_var"] = test_out["ret_variable_stop"] - test_out["ret_baseline"]

    seg_summary = (
        test_out.groupby("segment")
        .agg(
            n=("ret_baseline", "count"),
            k=("k_applied", "first"),
            base_sum=("ret_baseline", "sum"),
            var_sum=("ret_variable_stop", "sum"),
            delta=("delta_var", "sum"),
        )
        .sort_values("delta", ascending=False)
    )
    print(f"\n{'Segment':<40}  {'n':>4}  {'k':>5}  {'BaseSum':>8}  {'VarSum':>8}  {'Delta':>7}")
    print("-" * 80)
    for seg, row in seg_summary.iterrows():
        print(f"{seg:<40}  {int(row['n']):>4}  {row['k']!s:>5}  "
              f"{row['base_sum']:>+8.1f}  {row['var_sum']:>+8.1f}  {row['delta']:>+7.1f}")

    # ── Segment k lookup table (for live use) ────────────────────────────
    k_table = pd.DataFrame([
        {"segment": seg, "k": k, "global_k_fallback": best_k_global}
        for seg, k in segment_best_k.items()
    ])
    k_table.to_csv(OUT_DIR / "variable_stoploss_k_table.csv", index=False)
    print(f"\nk-lookup table saved -> {OUT_DIR / 'variable_stoploss_k_table.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
