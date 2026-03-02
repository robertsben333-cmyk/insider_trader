"""
Dip-Entry Analysis
==================
Tests whether entering on a pullback improves risk-adjusted returns.

Strategy A: Insider-price filter
    Only enter if the stock hasn't moved more than T% above (or below)
    the insider's own purchase price at the time we enter.
    Uses price_drift_filing_pct (already in the scored dataset).
    No change to mechanics — just a selection/skipping filter.

Strategy B: Intraday limit-order entry
    Instead of entering at market open, we place a limit order at
    (our_open_entry - X%), and only fill if the stock dips to that level.
    Return is computed from the limit price (not the session minimum).
    Assumption: limit fills at exactly entry*(1-X/100) if low_pct <= -X%.

Both are evaluated on the TEST set only (no training/tuning).

Key output metrics for each threshold:
  - n_trades        : number of trades taken
  - freq_pct        : % of opportunities taken
  - avg_ret         : average return per taken trade
  - sum_ret         : total return from all taken trades (absolute P&L contribution)
  - expected_daily  : avg_ret × (n_trades / test_calendar_days)
                      captures frequency × quality together
  - win_rate        : % of taken trades that were winners

Outputs:
  backtest/out/dip_entry_strategy_A.csv
  backtest/out/dip_entry_strategy_B.csv
  backtest/out/dip_entry_combined.csv   (A and B together)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from model_ensemble import MODEL_NAMES, predict_model, to_linear_numeric, to_xgb
from train_models import FEATURES, engineer_features, load_and_merge

OUT_DIR     = BASE / "backtest" / "out"
DETAIL_CSV  = OUT_DIR / "stoploss_trade_detail.csv"   # has low_pct for every test trade

# ─────────────────────── Sweep thresholds ──────────────────────────────

# Strategy A: max drift above insider price allowed (%)
# Negative = must be BELOW insider price; 0 = at or below; 5 = within 5% above
A_THRESHOLDS = [-5, -2, 0, 2, 5, 8, 10, 15, 100]   # 100 = no filter (baseline)

# Strategy B: required intraday dip from our open entry (%)
# 0 = any dip below open; positive means deeper dip required
B_THRESHOLDS = [0, 1, 2, 3, 5, 7, 10]               # 0 = no filter (baseline)

# ─────────────────────── Model scoring ─────────────────────────────────

MODEL_DIR = BASE / "models" / "prod4"
HORIZON   = 1


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


def align_features(model, X):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None: return X
    return X[[str(c) for c in cols]].copy()


def predict_aligned(name, model, X):
    Xa = align_features(model, X)
    if name == "HGBR":             return model.predict(Xa)
    if name == "XGBoost":          return model.predict(to_xgb(Xa))
    if name == "ElasticNet":       return model.predict(to_linear_numeric(Xa))
    if name == "SplineElasticNet": return model.predict(Xa.copy())
    return predict_model(name, model, Xa)


def load_test_picks() -> pd.DataFrame:
    """Load the test-set top-decile picks with all features."""
    print("Loading and scoring dataset...")
    df_raw = load_and_merge()
    df, features, _ = engineer_features(df_raw)
    assert features == FEATURES

    target = f"return_{HORIZON}d_pct"
    sub = df.dropna(subset=[target]).copy()
    lo, hi = sub[target].quantile([0.01, 0.99])
    sub[target] = sub[target].clip(lo, hi)
    sub["trade_date"]   = pd.to_datetime(sub["trade_date"],   errors="coerce")
    sub["buy_datetime"] = pd.to_datetime(sub["buy_datetime"], errors="coerce")
    sub = sub.dropna(subset=["trade_date", "buy_datetime"]).sort_values("trade_date").reset_index(drop=True)

    X = sub[features].copy()
    n_train, split_80 = chrono_split_60_20_20(len(sub))

    models = {nm: joblib.load(MODEL_DIR / f"model_{HORIZON}d_{nm}.pkl") for nm in MODEL_NAMES}
    pred_cols = []
    for nm, mdl in models.items():
        col = f"pred_{nm}"
        sub[col] = predict_aligned(nm, mdl, X)
        pred_cols.append(col)
    sub["pred_mean4"] = sub[pred_cols].mean(axis=1)

    train_signal = sub.loc[:n_train - 1, "pred_mean4"].to_numpy(dtype=float)
    threshold    = decile9_lower_bound(train_signal)
    print(f"  threshold={threshold:.4f}%  test start at row {split_80}")

    test = sub.iloc[split_80:].copy()
    test["ret_pct"] = pd.to_numeric(test[target], errors="coerce")
    test = test[test["pred_mean4"] > threshold].dropna(subset=["ret_pct"]).copy()
    print(f"  Test top-decile picks: {len(test)}")
    return test, threshold


# ─────────────────────── Join with intraday data ────────────────────────

def join_intraday(test_picks: pd.DataFrame) -> pd.DataFrame:
    """Join test picks with the pre-computed intraday low_pct."""
    detail = pd.read_csv(DETAIL_CSV, parse_dates=["entry_time", "exit_time"])
    # Normalise join keys
    test_picks = test_picks.copy()
    test_picks["buy_datetime_key"] = pd.to_datetime(
        test_picks["buy_datetime"]
    ).dt.tz_localize(None).dt.floor("s")
    detail["entry_time_key"] = pd.to_datetime(
        detail["entry_time"]
    ).dt.tz_localize(None).dt.floor("s")

    merged = test_picks.merge(
        detail[["ticker", "entry_time_key", "low_pct", "high_pct"]],
        left_on=["ticker", "buy_datetime_key"],
        right_on=["ticker", "entry_time_key"],
        how="left",
    )
    n_matched = merged["low_pct"].notna().sum()
    print(f"  Intraday data joined: {n_matched}/{len(merged)} trades matched")
    return merged


# ─────────────────────── Strategy metrics ───────────────────────────────

def strategy_stats(
    total_n: int,
    mask: pd.Series,
    returns: pd.Series,
    test_calendar_days: float,
    label: str,
) -> dict:
    taken = returns[mask].dropna()
    n = len(taken)
    freq = n / total_n * 100
    avg  = float(taken.mean()) if n else 0.0
    s    = float(taken.sum())  if n else 0.0
    wr   = float((taken > 0).mean() * 100) if n else 0.0
    # Expected daily = trades per day × avg return per trade
    exp_daily = (n / test_calendar_days) * avg if test_calendar_days > 0 else 0.0
    return dict(label=label, n_trades=n, freq_pct=freq, avg_ret=avg,
                sum_ret=s, win_rate=wr, expected_daily=exp_daily)


# ─────────────────────── Main ────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_picks, _ = load_test_picks()
    df = join_intraday(test_picks)

    # Calendar days in test period
    t_min = pd.to_datetime(df["buy_datetime"]).min()
    t_max = pd.to_datetime(df["buy_datetime"]).max()
    test_calendar_days = max(1.0, (t_max - t_min).days)
    total_n = len(df)
    print(f"\nTest period: {t_min.date()} to {t_max.date()}  ({test_calendar_days:.0f} calendar days)")
    print(f"Total test-set opportunities: {total_n}\n")

    # ── BASELINE ──────────────────────────────────────────────────────────
    print("="*72)
    print("BASELINE (no filter, enter at 9:45 AM / 15-min after filing)")
    print("="*72)
    base = strategy_stats(total_n, pd.Series([True]*total_n, index=df.index),
                          df["ret_pct"], test_calendar_days, "baseline")
    print(f"  n={base['n_trades']}  avg={base['avg_ret']:+.2f}%  "
          f"sum={base['sum_ret']:+.1f}  win={base['win_rate']:.1f}%  "
          f"exp_daily={base['expected_daily']:+.3f}%/day")

    # ── STRATEGY A: Insider-price filter ──────────────────────────────────
    print("\n" + "="*72)
    print("STRATEGY A: Insider-price filter (price_drift_filing_pct <= T%)")
    print("  Only enter if stock is within T% above (or below) insider's purchase price.")
    print("="*72)
    print(f"\n{'T(%)':>6}  {'n':>5}  {'freq%':>6}  {'avg_ret':>8}  "
          f"{'sum_ret':>8}  {'win%':>6}  {'exp_daily':>10}  {'note'}")
    print("-" * 72)

    rows_a = []
    pdc = df["price_drift_filing_pct"]
    for T in A_THRESHOLDS:
        mask = (pdc <= T) & pdc.notna()
        label = f"A_T{T:+d}pct"
        note  = "baseline" if T == 100 else (
            "at/below insider px" if T == 0 else
            f"within {T:+d}% of insider px"
        )
        r = strategy_stats(total_n, mask, df["ret_pct"], test_calendar_days, label)
        rows_a.append(r)
        marker = "  <- baseline" if T == 100 else ""
        print(f"  {T:>+4}%  {r['n_trades']:>5}  {r['freq_pct']:>5.1f}%  "
              f"{r['avg_ret']:>+8.2f}  {r['sum_ret']:>+8.1f}  "
              f"{r['win_rate']:>5.1f}%  {r['expected_daily']:>+9.3f}%/d  {note}{marker}")

    # Return distribution within A filters
    print(f"\n  price_drift_filing_pct distribution in test picks:")
    drift = pdc.dropna()
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    p{p:02d}: {np.percentile(drift, p):+.2f}%")
    print(f"    mean: {drift.mean():+.2f}%  "
          f"  [stock is this much above insider's purchase price at our entry]")

    # ── STRATEGY B: Intraday limit-order entry ────────────────────────────
    print("\n" + "="*72)
    print("STRATEGY B: Intraday limit-order entry (limit = open_entry * (1 - X%))")
    print("  Enter only if intraday low reaches the limit. Return from limit price.")
    print("="*72)
    print(f"\n{'X(%)':>6}  {'n':>5}  {'freq%':>6}  {'avg_ret':>8}  "
          f"{'sum_ret':>8}  {'win%':>6}  {'exp_daily':>10}  {'note'}")
    print("-" * 72)

    rows_b = []
    lp = df["low_pct"]
    rp = df["ret_pct"]

    for X in B_THRESHOLDS:
        if X == 0:
            # No dip requirement — enter at market, use original return
            mask = pd.Series([True]*len(df), index=df.index)
            adj_ret = rp.copy()
            label = "B_X0pct_baseline"
            note  = "no filter (enter at market open)"
        else:
            # Limit fires only if low_pct <= -X%
            mask = (lp <= -X) & lp.notna()
            # Return from limit price: (1 + orig_ret/100) / (1 - X/100) - 1
            adj_ret = ((1.0 + rp / 100.0) / (1.0 - X / 100.0) - 1.0) * 100.0
            label = f"B_X{X}pct"
            note  = f"limit at -{X}% from open"

        r = strategy_stats(total_n, mask, adj_ret[mask], test_calendar_days, label)
        r["dip_threshold_pct"] = X
        rows_b.append(r)
        marker = "  <- baseline" if X == 0 else ""
        print(f"  -{X:>3}%  {r['n_trades']:>5}  {r['freq_pct']:>5.1f}%  "
              f"{r['avg_ret']:>+8.2f}  {r['sum_ret']:>+8.1f}  "
              f"{r['win_rate']:>5.1f}%  {r['expected_daily']:>+9.3f}%/d  {note}{marker}")

    # ── STRATEGY C: Combined (A + B) ──────────────────────────────────────
    print("\n" + "="*72)
    print("STRATEGY C: Combined — insider-price filter T% AND intraday dip X%")
    print("="*72)

    COMBOS = [
        (100, 0),   # pure baseline
        (5,   0),   # A only, T=5%
        (100, 3),   # B only, X=3%
        (5,   3),   # both
        (2,   3),
        (0,   3),
        (5,   5),
        (2,   5),
        (0,   5),
    ]

    print(f"\n{'T(%)':>6}  {'X(%)':>5}  {'n':>5}  {'freq%':>6}  {'avg_ret':>8}  "
          f"{'sum_ret':>8}  {'win%':>6}  {'exp_daily':>10}")
    print("-" * 72)

    rows_c = []
    for T, X in COMBOS:
        if T == 100:
            mask_a = pd.Series([True]*len(df), index=df.index)
        else:
            mask_a = (pdc <= T) & pdc.notna()

        if X == 0:
            mask_b  = pd.Series([True]*len(df), index=df.index)
            adj_ret = rp.copy()
        else:
            mask_b  = (lp <= -X) & lp.notna()
            adj_ret = ((1.0 + rp / 100.0) / (1.0 - X / 100.0) - 1.0) * 100.0

        mask  = mask_a & mask_b
        label = f"C_T{T:+d}_X{X}"
        note  = "baseline" if (T == 100 and X == 0) else ""
        r = strategy_stats(total_n, mask, adj_ret[mask], test_calendar_days, label)
        r["T_threshold"] = T
        r["X_dip"]       = X
        rows_c.append(r)
        marker = "  <- baseline" if (T == 100 and X == 0) else ""
        print(f"  {T:>+4}%  -{X:>3}%  {r['n_trades']:>5}  {r['freq_pct']:>5.1f}%  "
              f"{r['avg_ret']:>+8.2f}  {r['sum_ret']:>+8.1f}  "
              f"{r['win_rate']:>5.1f}%  {r['expected_daily']:>+9.3f}%/d{marker}")

    # ── Frequency × quality analysis ─────────────────────────────────────
    print("\n" + "="*72)
    print("KEY INSIGHT: Return vs frequency trade-off for Strategy B")
    print("  For budget-curve: idle cash is a cost. Expected daily is the right metric.")
    print("="*72)
    print(f"\n  Intraday low distribution (reference):")
    lows = lp.dropna()
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    p{p:02d}: {np.percentile(lows, p):+.2f}%  "
              f"({100-p:.0f}% of trades dip at least this far intraday)")

    # ── Save outputs ──────────────────────────────────────────────────────
    pd.DataFrame(rows_a).to_csv(OUT_DIR / "dip_entry_strategy_A.csv", index=False)
    pd.DataFrame(rows_b).to_csv(OUT_DIR / "dip_entry_strategy_B.csv", index=False)
    pd.DataFrame(rows_c).to_csv(OUT_DIR / "dip_entry_combined.csv",   index=False)
    print(f"\nSaved -> {OUT_DIR / 'dip_entry_strategy_A.csv'}")
    print(f"Saved -> {OUT_DIR / 'dip_entry_strategy_B.csv'}")
    print(f"Saved -> {OUT_DIR / 'dip_entry_combined.csv'}")


if __name__ == "__main__":
    main()
