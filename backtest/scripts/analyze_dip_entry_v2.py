"""
Dip-Entry Analysis v2 — Full 659 test picks, variable limit order
==================================================================
Strategies tested:

  A  : Insider-price filter — skip if stock > T% above insider's purchase price.
       No change to mechanics; uses price_drift_filing_pct.

  B_fixed  : Intraday limit order at a fixed X% below open entry.
             Sweeps X on TRAINING, picks best X, validates on VAL, applies to TEST.

  B_var    : Intraday limit order at k × daily_vol_pct below open entry (ATR-scaled).
             Sweeps k on TRAINING per segment, validates on VAL, applies to TEST.
             Principle: wait for a "1-sigma intraday move" before entering;
             riskier stocks require a deeper dip in absolute %, but the same
             number of volatility units.

  C_combo  : A + B_var combined (insider filter AND volatility-scaled limit).

Metrics reported per strategy:
  n_trades    — trades taken
  freq_pct    — % of opportunities taken
  avg_ret     — avg return per taken trade
  sum_ret     — total P&L contribution (absolute)
  win_rate    — % of taken trades that were positive
  exp_daily   — avg_ret × (n_trades / calendar_days)  [frequency × quality]

Training split: first 60% chronologically (same as backtest / variable stoploss scripts).
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

MODEL_DIR = BASE / "models" / "prod4"
CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUT_DIR   = BASE / "backtest" / "out"
ET        = ZoneInfo("America/New_York")
HORIZON   = 1
MAX_WORKERS = 10

# ── Sweep ranges ────────────────────────────────────────────────────────
FIXED_X_VALUES  = [0, 1, 2, 3, 5, 7, 10]        # fixed % dip thresholds
VAR_K_VALUES    = np.round(np.arange(0.25, 3.25, 0.25), 2).tolist()  # k × vol
A_THRESHOLDS    = [-5, -2, 0, 2, 5, 8, 10, 15, 100]  # % above insider px
MIN_SEG_TRADES  = 20                              # min trades for per-segment k

# ── Segment definitions (same as variable stoploss) ──────────────────────
SECTOR_MAP = {"Health Care": "Biotech", "Financials": "Finance",
              "Information Technology": "Tech", "Communication Services": "Tech"}
def seg_sector(s): return SECTOR_MAP.get(s, "Other")
def seg_price(p):
    if p < 2:   return "micro"
    if p < 10:  return "small"
    return "large"
def seg_pred(pred, med): return "high" if pred >= med else "low"


# ── Price cache / Polygon (mirrors other scripts) ────────────────────────
class PriceCache:
    def __init__(self, d):
        self.dir = Path(d)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    def _p(self, t, k, f, to): return self.dir / f"{t}_{k}_{f}_{to}.json"
    def get(self, t, k, f, to):
        p = self._p(t, k, f, to)
        if p.exists():
            try:
                with open(p) as fh: return json.load(fh)
            except: return None
        return None
    def put(self, t, k, f, to, data):
        p = self._p(t, k, f, to)
        with self._lock:
            with open(p, "w") as fh: json.dump(data, fh)

_tls = threading.local()
def _client(key):
    if not hasattr(_tls, "c"): _tls.c = RESTClient(key)
    return _tls.c

def fetch_min(api_key, cache, ticker, d):
    ds = d.strftime("%Y-%m-%d")
    hit = cache.get(ticker, "min", ds, ds)
    if hit is not None: return hit
    try:
        bars = [{"t": a.timestamp, "o": a.open, "c": a.close, "h": a.high, "l": a.low}
                for a in _client(api_key).get_aggs(ticker, 1, "minute", d, d,
                adjusted=True, sort="asc", limit=50000) if a.timestamp and a.close]
    except Exception as e:
        print(f"  [WARN] {ticker} {ds}: {e}"); bars = []
    cache.put(ticker, "min", ds, ds, bars)
    return bars

def tdays(start, end):
    days, c = [], start
    while c <= end:
        if c.weekday() < 5: days.append(c)
        c += timedelta(days=1)
    return days

def intraday_low(api_key, cache, ticker, entry_dt, exit_dt):
    all_bars = []
    for d in tdays(entry_dt.date(), exit_dt.date()):
        all_bars.extend(fetch_min(api_key, cache, ticker, d))
    all_bars.sort(key=lambda b: b["t"])
    e_ms = int(entry_dt.timestamp() * 1000)
    x_ms = int(exit_dt.timestamp() * 1000)
    rel  = [b for b in all_bars if b["t"] >= e_ms and b["t"] <= x_ms]
    if not rel: return None
    ep = rel[0]["o"]
    if not ep or ep <= 0: return None
    return (min(b["l"] for b in rel) / ep - 1.0) * 100.0

def fetch_all(df, api_key, cache, label):
    results = {}
    def _work(idx, row):
        edt = row["buy_datetime"].to_pydatetime().replace(tzinfo=ET)
        xdt = row["exit_datetime"].to_pydatetime().replace(tzinfo=ET)
        return idx, intraday_low(api_key, cache, row["ticker"], edt, xdt)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_work, idx, row): idx for idx, row in df.iterrows()}
        done = 0
        for fut in as_completed(futs):
            idx, lp = fut.result(); results[idx] = lp; done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"  [{label}] {done}/{len(futs)}", end="\r", flush=True)
    print()
    return pd.Series(results, name="low_pct")


# ── Model scoring ────────────────────────────────────────────────────────
def align(model, X):
    cols = getattr(model, "feature_names_in_", None)
    return X[[str(c) for c in cols]].copy() if cols is not None else X
def pred(name, mdl, X):
    Xa = align(mdl, X)
    if name == "HGBR":             return mdl.predict(Xa)
    if name == "XGBoost":          return mdl.predict(to_xgb(Xa))
    if name == "ElasticNet":       return mdl.predict(to_linear_numeric(Xa))
    if name == "SplineElasticNet": return mdl.predict(Xa.copy())
    return predict_model(name, mdl, Xa)

def chrono_split(n):
    nt = max(1, min(int(round(n * 0.6)), n - 2))
    nv = max(1, min(int(round(n * 0.2)), n - nt - 1))
    return nt, nt + nv

def decile9_lb(signal):
    s = np.asarray(signal, dtype=float); s = s[np.isfinite(s)]
    o = np.argsort(s); bs = max(1, len(o) // 10)
    idx = o[8*bs: min(len(o), 9*bs)]
    return float(np.min(s[idx])) if len(idx) else float(np.quantile(s, 0.8))

def nbd_close(ts):
    return (ts + pd.offsets.BDay(1)).normalize() + pd.Timedelta(hours=16)

def load_picks():
    print("Scoring dataset...")
    df_raw = load_and_merge()
    df, features, _ = engineer_features(df_raw)
    assert features == FEATURES
    target = f"return_{HORIZON}d_pct"
    sub = df.dropna(subset=[target]).copy()
    lo, hi = sub[target].quantile([0.01, 0.99])
    sub[target] = sub[target].clip(lo, hi)
    sub["trade_date"]   = pd.to_datetime(sub["trade_date"],   errors="coerce")
    sub["buy_datetime"] = pd.to_datetime(sub["buy_datetime"], errors="coerce")
    sub = sub.dropna(subset=["trade_date","buy_datetime"]).sort_values("trade_date").reset_index(drop=True)

    X = sub[features].copy()
    n_train, s80 = chrono_split(len(sub))
    mdls = {nm: joblib.load(MODEL_DIR / f"model_{HORIZON}d_{nm}.pkl") for nm in MODEL_NAMES}
    pred_cols = []
    for nm, mdl in mdls.items():
        col = f"pred_{nm}"; sub[col] = pred(nm, mdl, X); pred_cols.append(col)
    sub["pred_mean4"] = sub[pred_cols].mean(axis=1)

    thr = decile9_lb(sub.loc[:n_train-1, "pred_mean4"].to_numpy())
    sub["split"] = "train"
    sub.loc[n_train:s80-1, "split"] = "val"
    sub.loc[s80:, "split"]          = "test"
    sub["exit_datetime"] = sub["buy_datetime"].apply(nbd_close)

    picks = sub[sub["pred_mean4"] > thr].copy()
    picks["ret_pct"] = pd.to_numeric(picks[target], errors="coerce")
    picks = picks.dropna(subset=["ret_pct"]).reset_index(drop=True)

    picks["daily_vol_pct"] = picks["prior_30d_vol"].clip(lower=0) / np.sqrt(252) * 100
    picks["entry_price"]   = picks["last_price_clean"].clip(lower=0.01)
    picks["sec_group"]     = picks["sector"].apply(seg_sector)
    picks["price_tier"]    = picks["entry_price"].apply(seg_price)
    tr_med = picks.loc[picks["split"] == "train", "pred_mean4"].median()
    picks["pred_tier"] = picks["pred_mean4"].apply(lambda p: seg_pred(p, tr_med))
    picks["segment"]   = picks["sec_group"] + "_" + picks["price_tier"] + "_" + picks["pred_tier"]

    counts = picks["split"].value_counts()
    print(f"  threshold={thr:.4f}%  train={counts.get('train',0)}  "
          f"val={counts.get('val',0)}  test={counts.get('test',0)}")
    return picks, thr


# ── Core limit simulation ────────────────────────────────────────────────

def limit_ret(orig_ret, low_pct, limit_pct):
    """Return (simulated_return, fired).
    limit_pct is negative (e.g. -5.0 means buy 5% below open).
    If low_pct <= limit_pct, the limit fires; return is from limit price.
    """
    if low_pct is None or np.isnan(low_pct):
        return None, False                   # no data → skip trade entirely
    if low_pct <= limit_pct:
        # Return from limit price: (1 + orig/100) / (1 - |limit_pct|/100) - 1
        denom = 1.0 - abs(limit_pct) / 100.0
        if denom <= 0: return None, False
        return ((1.0 + orig_ret / 100.0) / denom - 1.0) * 100.0, True
    return None, False                       # limit not reached → skip


def stats(df, mask, ret_series, total_n, cal_days, label):
    taken = ret_series[mask].dropna()
    n  = len(taken)
    return dict(
        label=label, n=n, freq=n / total_n * 100,
        avg  =float(taken.mean()) if n else 0.0,
        sumr =float(taken.sum())  if n else 0.0,
        win  =float((taken > 0).mean() * 100) if n else 0.0,
        expd =(n / cal_days) * float(taken.mean()) if (n and cal_days > 0) else 0.0,
    )


def sweep_fixed_x(df, x_values):
    """Sweep fixed dip thresholds on df. Returns best X and full stats dict."""
    best_x, best_expd = 0, -np.inf
    results = {}
    for X in x_values:
        if X == 0:
            mask = pd.Series(True, index=df.index)
            rets = df["ret_pct"].copy()
        else:
            lim  = -float(X)
            rows = df.apply(lambda r: limit_ret(r["low_pct"], lim, lim), axis=1)
            mask = pd.Series([r[1] for r in rows], index=df.index)
            rets = pd.Series([r[0] for r in rows], index=df.index)

        taken = rets[mask].dropna()
        n = len(taken)
        cal = max(1, (pd.to_datetime(df["buy_datetime"]).max()
                      - pd.to_datetime(df["buy_datetime"]).min()).days)
        expd = (n / cal) * float(taken.mean()) if n else 0.0
        results[X] = dict(n=n, avg=float(taken.mean()) if n else 0.0,
                          sumr=float(taken.sum()) if n else 0.0,
                          win=float((taken>0).mean()*100) if n else 0.0,
                          expd=expd, mask=mask, rets=rets)
        if expd > best_expd:
            best_expd, best_x = expd, X
    return best_x, results


def sweep_var_k(df, k_values):
    """Sweep k × daily_vol_pct limit on df. Returns best k and full stats dict."""
    best_k, best_expd = 0.0, -np.inf
    results = {}
    for k in [0.0] + k_values:
        if k == 0.0:
            mask = pd.Series(True, index=df.index)
            rets = df["ret_pct"].copy()
        else:
            masks, retlist = [], []
            for _, r in df.iterrows():
                lim_pct = -k * r["daily_vol_pct"]
                rt, fired = limit_ret(r["ret_pct"], r.get("low_pct"), lim_pct)
                masks.append(fired)
                retlist.append(rt)
            mask = pd.Series(masks, index=df.index)
            rets = pd.Series(retlist, index=df.index)

        taken = rets[mask].dropna()
        n = len(taken)
        cal = max(1, (pd.to_datetime(df["buy_datetime"]).max()
                      - pd.to_datetime(df["buy_datetime"]).min()).days)
        expd = (n / cal) * float(taken.mean()) if n else 0.0
        results[k] = dict(n=n, avg=float(taken.mean()) if n else 0.0,
                          sumr=float(taken.sum()) if n else 0.0,
                          win=float((taken>0).mean()*100) if n else 0.0,
                          expd=expd, mask=mask, rets=rets)
        if k > 0 and expd > best_expd:
            best_expd, best_k = expd, k
    return best_k, results


def apply_fixed_x(df, X):
    """Apply fixed X% limit to df; return ret Series (NaN = skip)."""
    if X == 0:
        return df["ret_pct"].copy()
    lim = -float(X)
    rets = df.apply(
        lambda r: limit_ret(r["ret_pct"], r.get("low_pct"), lim)[0], axis=1)
    return rets


def apply_var_k(df, seg_k, global_k):
    """Apply per-segment (or global fallback) k × vol limit; return ret Series."""
    rets = []
    for _, r in df.iterrows():
        k = seg_k.get(r["segment"], global_k)
        if k == 0.0:
            rets.append(r["ret_pct"])
        else:
            lim_pct = -k * r["daily_vol_pct"]
            rt, fired = limit_ret(r["ret_pct"], r.get("low_pct"), lim_pct)
            rets.append(rt if fired else None)
    return pd.Series(rets, index=df.index)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    load_dotenv(BASE / ".env")
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key: raise RuntimeError("POLYGON_API_KEY not set")

    picks, thr = load_picks()
    cache = PriceCache(CACHE_DIR)

    train_df = picks[picks["split"] == "train"].copy()
    val_df   = picks[picks["split"] == "val"].copy()
    test_df  = picks[picks["split"] == "test"].copy()

    # ── Intraday paths: test from saved detail CSV; train/val from cache ──
    test_detail = pd.read_csv(OUT_DIR / "variable_stoploss_test_detail.csv")
    test_detail["buy_datetime_key"] = pd.to_datetime(test_detail["buy_datetime"]).dt.tz_localize(None).dt.floor("s")
    test_df["buy_datetime_key"]     = pd.to_datetime(test_df["buy_datetime"]).dt.tz_localize(None).dt.floor("s")
    test_df = test_df.merge(
        test_detail[["ticker","buy_datetime_key","low_pct"]],
        on=["ticker","buy_datetime_key"], how="left"
    )
    print(f"Test low_pct matched: {test_df['low_pct'].notna().sum()}/{len(test_df)}")

    print(f"Fetching intraday for {len(train_df)} TRAIN trades (cached)...")
    train_df["low_pct"] = fetch_all(train_df, api_key, cache, "TRAIN")

    print(f"Fetching intraday for {len(val_df)} VAL trades (cached)...")
    val_df["low_pct"] = fetch_all(val_df, api_key, cache, "VAL")

    # Calendar days per split
    def cal_days(df):
        t = pd.to_datetime(df["buy_datetime"])
        return max(1.0, (t.max() - t.min()).days)
    tr_cal  = cal_days(train_df)
    val_cal = cal_days(val_df)
    tst_cal = cal_days(test_df)
    total_n = len(test_df)

    print(f"\nSplit sizes  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"Cal days     train={tr_cal:.0f}  val={val_cal:.0f}  test={tst_cal:.0f}")

    def pstats(label, taken):
        n = len(taken.dropna())
        a = float(taken.dropna().mean()) if n else 0.0
        s = float(taken.dropna().sum())  if n else 0.0
        w = float((taken.dropna()>0).mean()*100) if n else 0.0
        e = (n/tst_cal)*a if tst_cal else 0.0
        f = n/total_n*100
        print(f"  {label:<40}  n={n:4d}({f:4.0f}%)  avg={a:+6.2f}%  "
              f"sum={s:+8.1f}  win={w:4.1f}%  exp_d={e:+6.3f}%/d")
        return dict(label=label, n=n, freq=f, avg=a, sum=s, win=w, exp_daily=e)

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*72)
    print("BASELINE (no filter)")
    print("="*72)
    pstats("Baseline", test_df["ret_pct"])

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*72)
    print("STRATEGY A — Insider-price filter  (test only, no training needed)")
    print("="*72)
    print(f"\n  {'T%':>5}  {'n':>5}  {'freq':>5}  {'avg_ret':>8}  "
          f"{'sum_ret':>8}  {'win%':>6}  {'exp_daily':>9}")
    print("  " + "-"*60)
    pdc = test_df["price_drift_filing_pct"]
    rows_a = []
    for T in A_THRESHOLDS:
        mask = (pdc <= T) & pdc.notna() if T < 100 else pdc.notna()
        r = pstats(f"A  T<={T:+d}%", test_df.loc[mask, "ret_pct"])
        rows_a.append(r)

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*72)
    print("STRATEGY B_fixed — sweep fixed X% limit on TRAIN, apply to TEST")
    print("="*72)
    print("  Sweeping on training set...")
    best_x, train_bx = sweep_fixed_x(train_df, FIXED_X_VALUES)
    print(f"  Best fixed X on train (by exp_daily): X={best_x}%")
    print(f"\n  Training sweep:")
    print(f"  {'X%':>5}  {'n_tr':>6}  {'avg_tr':>8}  {'expd_tr':>8}")
    print("  " + "-"*40)
    for X, s in train_bx.items():
        mk = "  <-- best" if X == best_x else ""
        print(f"  {-X:>4}%  {s['n']:>6}  {s['avg']:>+8.2f}  {s['expd']:>+8.3f}{mk}")

    print(f"\n  Validation (X={best_x}%):")
    val_rets_bx = apply_fixed_x(val_df, best_x)
    pstats(f"B_fixed X={best_x}% on VAL", val_rets_bx)
    pstats(f"Baseline VAL",               val_df["ret_pct"])

    print(f"\n  Test results — B_fixed at best X={best_x}%:")
    rows_b_fixed = []
    for X in FIXED_X_VALUES:
        r = pstats(f"B_fixed X={X}%", apply_fixed_x(test_df, X))
        rows_b_fixed.append(r)

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*72)
    print("STRATEGY B_var — k × daily_vol_pct limit, trained per segment")
    print("="*72)

    # Global k sweep on training
    print("  Global k sweep on training...")
    best_k_global, train_bk = sweep_var_k(train_df, VAR_K_VALUES)
    print(f"  Best global k on train (by exp_daily): k={best_k_global}")
    print(f"\n  Training global sweep:")
    print(f"  {'k':>5}  {'n_tr':>6}  {'avg_tr':>8}  {'expd_tr':>8}")
    print("  " + "-"*40)
    for k, s in train_bk.items():
        mk = "  <-- best" if k == best_k_global else ""
        print(f"  {k:>5.2f}  {s['n']:>6}  {s['avg']:>+8.2f}  {s['expd']:>+8.3f}{mk}")

    # Per-segment k sweep on training
    print("\n  Per-segment k sweep on training...")
    seg_k: Dict[str, float] = {}
    for seg in sorted(train_df["segment"].unique()):
        grp = train_df[train_df["segment"] == seg]
        if len(grp) < MIN_SEG_TRADES:
            seg_k[seg] = best_k_global
            print(f"    {seg:<40} n={len(grp):3d}  -> global k={best_k_global}")
            continue
        bk, bk_stats = sweep_var_k(grp, VAR_K_VALUES)
        seg_k[seg] = bk
        base_expd = bk_stats[0.0]["expd"]
        best_expd = bk_stats[bk]["expd"]
        print(f"    {seg:<40} n={len(grp):3d}  best_k={bk:5.2f}  "
              f"base_expd={base_expd:+.3f}  best_expd={best_expd:+.3f}  "
              f"delta={best_expd-base_expd:+.3f}")

    # Validation
    print(f"\n  Validation — per-segment k vs global k vs baseline:")
    val_var_rets = apply_var_k(val_df, seg_k, best_k_global)
    val_glb_rets = apply_var_k(val_df, {}, best_k_global)
    pstats("B_var per-segment VAL",  val_var_rets)
    pstats(f"B_var global k={best_k_global} VAL", val_glb_rets)
    pstats("Baseline VAL",           val_df["ret_pct"])

    # Test
    print(f"\n  Test results — B_var per-segment k:")
    rows_b_var = []
    test_var_rets = apply_var_k(test_df, seg_k, best_k_global)
    test_glb_rets = apply_var_k(test_df, {}, best_k_global)
    rows_b_var.append(pstats("B_var per-segment TEST",   test_var_rets))
    rows_b_var.append(pstats(f"B_var global k={best_k_global} TEST", test_glb_rets))
    rows_b_var.append(pstats("Baseline TEST",            test_df["ret_pct"]))

    # k distribution actually applied
    print(f"\n  k values applied per segment on test:")
    for seg in sorted(test_df["segment"].unique()):
        k = seg_k.get(seg, best_k_global)
        n = int((test_df["segment"] == seg).sum())
        lim_med = -k * test_df.loc[test_df["segment"] == seg, "daily_vol_pct"].median()
        print(f"    {seg:<40} n={n:3d}  k={k:5.2f}  "
              f"median_limit={lim_med:+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*72)
    print("STRATEGY C_combo — A (insider filter) + B_var (vol-scaled limit)")
    print("  Using best trained k from B_var per segment.")
    print("="*72)

    COMBOS = [
        (100, False, "Baseline (no filter)"),
        (10,  False, "A only: T<=+10%"),
        (5,   False, "A only: T<=+5%"),
        (0,   False, "A only: T<=0%"),
        (100, True,  "B_var only (per-seg k)"),
        (10,  True,  "C: T<=+10% + B_var"),
        (5,   True,  "C: T<=+5%  + B_var"),
        (0,   True,  "C: T<=0%   + B_var"),
        (-2,  True,  "C: T<=-2%  + B_var"),
    ]

    print(f"\n  {'Label':<40}  {'n':>4}  {'freq':>5}  {'avg':>7}  "
          f"{'sum':>8}  {'win%':>5}  {'exp_d':>8}")
    print("  " + "-"*78)

    rows_c, all_rows = [], []
    for T, use_bvar, label in COMBOS:
        # A mask
        if T >= 100:
            mask_a = pd.Series(True, index=test_df.index)
        else:
            mask_a = (pdc <= T) & pdc.notna()

        if not use_bvar:
            taken = test_df.loc[mask_a, "ret_pct"]
            r = pstats(label, taken)
        else:
            bvar = apply_var_k(test_df, seg_k, best_k_global)
            taken = bvar[mask_a]
            r = pstats(label, taken)
        rows_c.append(r)

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*72)
    print("FINAL COMPARISON TABLE (test set)")
    print("="*72)
    print(f"\n  {'Strategy':<45}  {'n':>4}  {'freq%':>5}  {'avg%':>7}  "
          f"{'sum':>8}  {'win%':>5}  {'exp_d%/d':>9}")
    print("  " + "-"*82)

    summary = [
        ("Baseline",                          test_df["ret_pct"]),
        ("A: T<=+8% (prune run-aways)",       test_df.loc[(pdc<=8)&pdc.notna(), "ret_pct"]),
        ("A: T<=0%  (at/below insider price)",test_df.loc[(pdc<=0)&pdc.notna(), "ret_pct"]),
        ("B_fixed best X",                    apply_fixed_x(test_df, best_x)),
        ("B_var global k",                    test_glb_rets),
        ("B_var per-segment k",               test_var_rets),
        ("C: T<=+8% + B_var",                 apply_var_k(test_df[(pdc<=8)&pdc.notna()].copy(), seg_k, best_k_global)),
        ("C: T<=0%  + B_var",                 apply_var_k(test_df[(pdc<=0)&pdc.notna()].copy(), seg_k, best_k_global)),
    ]
    out_rows = []
    for lbl, taken in summary:
        n = len(taken.dropna())
        a = float(taken.dropna().mean()) if n else 0.0
        s = float(taken.dropna().sum())  if n else 0.0
        w = float((taken.dropna()>0).mean()*100) if n else 0.0
        e = (n/tst_cal)*a
        f = n/total_n*100
        print(f"  {lbl:<45}  {n:>4}  {f:>4.0f}%  {a:>+7.2f}  "
              f"{s:>+8.1f}  {w:>4.1f}%  {e:>+8.3f}")
        out_rows.append(dict(strategy=lbl, n=n, freq_pct=f, avg_ret=a,
                             sum_ret=s, win_pct=w, exp_daily=e))

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "dip_entry_v2_summary.csv", index=False)
    # Save k table
    pd.DataFrame([{"segment": s, "k": k} for s, k in seg_k.items()]).to_csv(
        OUT_DIR / "bvar_k_table.csv", index=False)
    print(f"\nSaved -> {OUT_DIR / 'dip_entry_v2_summary.csv'}")
    print(f"Saved -> {OUT_DIR / 'bvar_k_table.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
