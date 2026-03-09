"""
Diagnose email trades for hypothesis C:
  Does the model have a systematic feature bias for recent/beaten-down stocks?

Fetches prices via yfinance, engineers features, runs models,
and prints a feature-level breakdown for each trade.
"""

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from pathlib import Path
from datetime import datetime, date, timedelta

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import train_models  # for FEATURES, engineer_features, SECTOR_ORDER, etc.

# ── Email trades ──────────────────────────────────────────────────────────────
EMAIL_TRADES = [
    {"ticker": "APCX",  "owner": "Derosa Thomas Joseph",      "trade_date": "2026-03-04", "buy_price": 0.3100, "title": "Dir", "email_score_1d": 2.397},
    {"ticker": "FBLG",  "owner": "O'Heeron Pete",             "trade_date": "2026-03-02", "buy_price": 0.4192, "title": "Dir", "email_score_1d": 1.907},
    {"ticker": "SABR",  "owner": "Constellation Software Inc","trade_date": "2026-02-27", "buy_price": 1.9900, "title": "10%", "email_score_1d": 1.846},
    {"ticker": "ABTC",  "owner": "Mateen Justin",             "trade_date": "2026-03-03", "buy_price": 1.2000, "title": "Dir", "email_score_1d": 1.636},
    {"ticker": "AIRS",  "owner": "Chernett Jorey",            "trade_date": "2026-03-03", "buy_price": 1.8500, "title": "Dir", "email_score_1d": 1.511},
    {"ticker": "EOSE",  "owner": "Dimitrief Alexander",       "trade_date": "2026-03-02", "buy_price": 6.2800, "title": "Dir", "email_score_1d": 1.361},
    {"ticker": "LAW",   "owner": "Goodman Robert P",          "trade_date": "2026-02-27", "buy_price": 4.1900, "title": "Dir", "email_score_1d": 1.355},
    {"ticker": "KDOZF", "owner": "Williams Tryon M",          "trade_date": "2026-03-03", "buy_price": 0.2441, "title": "Dir", "email_score_1d": 1.328},
    {"ticker": "SSP",   "owner": "Barmonde Charles L.",       "trade_date": "2026-03-03", "buy_price": 3.9900, "title": "Dir", "email_score_1d": 1.289},
    {"ticker": "THRY",  "owner": "Paulson & Co. Inc.",        "trade_date": "2026-02-27", "buy_price": 3.2000, "title": "10%", "email_score_1d": 1.289},
    {"ticker": "APCX",  "owner": "Lord Albert L",             "trade_date": "2026-02-27", "buy_price": 0.3250, "title": "Dir", "email_score_1d": 1.237},
    {"ticker": "HYMC",  "owner": "Sprott Eric",               "trade_date": "2026-03-04", "buy_price": 38.790, "title": "10%", "email_score_1d": 1.173},
    {"ticker": "NAUT",  "owner": "Patel Sujal M",             "trade_date": "2026-03-04", "buy_price": 2.4400, "title": "Dir", "email_score_1d": 1.171},
    {"ticker": "SSP",   "owner": "Klenzing Margaret Scripps", "trade_date": "2026-03-04", "buy_price": 4.5700, "title": "Dir", "email_score_1d": 1.115},
    {"ticker": "LAW",   "owner": "Hill Scott A",              "trade_date": "2026-03-02", "buy_price": 4.6800, "title": "Dir", "email_score_1d": 1.043},
    {"ticker": "KDOZF", "owner": "Williams Tryon M",          "trade_date": "2026-03-06", "buy_price": 0.2300, "title": "Dir", "email_score_1d": 0.998},
    {"ticker": "TTEC",  "owner": "Holtzman Marc",             "trade_date": "2026-03-02", "buy_price": 2.7000, "title": "Dir", "email_score_1d": 0.918},
    {"ticker": "NRGV",  "owner": "Ertel Thomas R",            "trade_date": "2026-03-04", "buy_price": 3.2550, "title": "Dir", "email_score_1d": 0.918},
    {"ticker": "EOSE",  "owner": "Mastrangelo Joe",           "trade_date": "2026-03-04", "buy_price": 6.6800, "title": "Dir", "email_score_1d": 0.910},
    {"ticker": "RUN",   "owner": "Ferber Alan",               "trade_date": "2026-03-03", "buy_price": 11.790, "title": "Dir", "email_score_1d": 0.880},
    {"ticker": "WD",    "owner": "Walker William M",          "trade_date": "2026-03-02", "buy_price": 50.350, "title": "Dir", "email_score_1d": 0.862},
    {"ticker": "XRAY",  "owner": "Lucier Gregory T",          "trade_date": "2026-03-02", "buy_price": 13.440, "title": "Dir", "email_score_1d": 0.856},
    {"ticker": "KDOZF", "owner": "Williams Tryon M",          "trade_date": "2026-03-04", "buy_price": 0.2300, "title": "Dir", "email_score_1d": 0.847},
    {"ticker": "HYMC",  "owner": "Sprott Eric",               "trade_date": "2026-03-05", "buy_price": 38.790, "title": "10%", "email_score_1d": 0.801},
    {"ticker": "CLRO",  "owner": "First Finance Ltd.",        "trade_date": "2026-03-02", "buy_price": 4.5700, "title": "10%", "email_score_1d": 0.797},
    {"ticker": "THRY",  "owner": "Rouse Paul D",              "trade_date": "2026-03-02", "buy_price": 3.2000, "title": "Dir", "email_score_1d": 0.786},
    {"ticker": "NAII",  "owner": "Ledoux Mark A",             "trade_date": "2026-02-27", "buy_price": 2.7700, "title": "Dir", "email_score_1d": 0.769},
    {"ticker": "NAII",  "owner": "Ledoux Mark A",             "trade_date": "2026-03-04", "buy_price": 2.7300, "title": "Dir", "email_score_1d": 0.762},
    {"ticker": "ABSI",  "owner": "Pangalos Menelas N",        "trade_date": "2026-02-27", "buy_price": 2.6600, "title": "Dir", "email_score_1d": 0.760},
    {"ticker": "VITL",  "owner": "Cyr William B.",            "trade_date": "2026-03-02", "buy_price": 19.890, "title": "Dir", "email_score_1d": 0.741},
    {"ticker": "THRY",  "owner": "Slater John",               "trade_date": "2026-03-04", "buy_price": 3.3050, "title": "Dir", "email_score_1d": 0.738},
    {"ticker": "GPK",   "owner": "Venturelli Larry M",        "trade_date": "2026-03-04", "buy_price": 11.220, "title": "Dir", "email_score_1d": 0.736},
    {"ticker": "DUOL",  "owner": "Shelton James H",           "trade_date": "2026-03-03", "buy_price": 101.54, "title": "Dir", "email_score_1d": 0.723},
    {"ticker": "FA",    "owner": "Nairne Douglas",            "trade_date": "2026-03-02", "buy_price": 12.220, "title": "Dir", "email_score_1d": 0.723},
    {"ticker": "CMTG",  "owner": "Mack Richard",              "trade_date": "2026-02-27", "buy_price": 2.7500, "title": "Dir", "email_score_1d": 0.722},
    {"ticker": "FRSH",  "owner": "Woodside Dennis",           "trade_date": "2026-03-02", "buy_price": 8.4700, "title": "Dir", "email_score_1d": 0.714},
]

df = pd.DataFrame(EMAIL_TRADES)
df["trade_date"] = pd.to_datetime(df["trade_date"])

# ── Load sector / market_type metadata ────────────────────────────────────────
meta_cache = pd.read_csv(ROOT / "data/ticker_meta_cache.csv")
meta_cache = meta_cache.set_index("ticker")

def get_sector(ticker):
    return meta_cache.loc[ticker, "sector"] if ticker in meta_cache.index else "Unknown"

def get_market_type(ticker):
    return meta_cache.loc[ticker, "market_type"] if ticker in meta_cache.index else "NON_TRADABLE"

def get_is_tradable(ticker):
    return int(meta_cache.loc[ticker, "is_tradable"]) if ticker in meta_cache.index else 0

df["sector"]      = df["ticker"].apply(get_sector)
df["market_type"] = df["ticker"].apply(get_market_type)
df["is_tradable"] = df["ticker"].apply(get_is_tradable)

# ── Price features via yfinance ────────────────────────────────────────────────
print("Fetching prices from yfinance...")
unique_tickers = df["ticker"].unique().tolist()

# Date range: oldest trade_date - 50 days through today
fetch_start = (df["trade_date"].min() - timedelta(days=55)).strftime("%Y-%m-%d")
fetch_end   = (df["trade_date"].max() + timedelta(days=2)).strftime("%Y-%m-%d")

price_data: dict[str, pd.DataFrame] = {}
for t in unique_tickers:
    try:
        hist = yf.download(t, start=fetch_start, end=fetch_end,
                           auto_adjust=True, progress=False)
        if not hist.empty:
            price_data[t] = hist
    except Exception as e:
        print(f"  WARNING: could not fetch {t}: {e}")

def compute_prior_return(ticker, trade_date, lookback_days, buy_price):
    if ticker not in price_data:
        return np.nan
    hist = price_data[ticker]["Close"].dropna()
    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
    cutoff = pd.Timestamp(trade_date) - timedelta(days=lookback_days)
    # find the most recent close on or before cutoff
    candidates = hist[hist.index.date <= cutoff.date()]
    if candidates.empty:
        return np.nan
    ref_price = float(candidates.iloc[-1])
    if ref_price <= 0:
        return np.nan
    return (buy_price - ref_price) / ref_price * 100.0

def compute_volatility(ticker, trade_date, lookback_days):
    if ticker not in price_data:
        return np.nan
    hist = price_data[ticker]["Close"].dropna()
    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
    cutoff = pd.Timestamp(trade_date) - timedelta(days=lookback_days)
    window = hist[hist.index.date >= cutoff.date()]
    window = window[window.index.date <= pd.Timestamp(trade_date).date()]
    if len(window) < 2:
        return np.nan
    log_rets = np.diff(np.log(window.values.astype(float)))
    return float(np.std(log_rets, ddof=1) * np.sqrt(252))

def compute_actual_return(ticker, trade_date, days_forward):
    """Compute actual return N trading days after trade_date."""
    if ticker not in price_data:
        return np.nan
    hist = price_data[ticker]["Close"].dropna()
    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
    td = pd.Timestamp(trade_date)
    # closes strictly after trade_date
    after = hist[hist.index > td]
    if len(after) < days_forward:
        return np.nan
    entry = float(after.iloc[0])
    exit_ = float(after.iloc[days_forward - 1])
    if entry <= 0:
        return np.nan
    return (exit_ / entry - 1) * 100.0

print("Computing price features...")
df["prior_30d_pct"] = df.apply(lambda r: compute_prior_return(r["ticker"], r["trade_date"], 30, r["buy_price"]), axis=1)
df["prior_10d_pct"] = df.apply(lambda r: compute_prior_return(r["ticker"], r["trade_date"], 10, r["buy_price"]), axis=1)
df["prior_5d_pct"]  = df.apply(lambda r: compute_prior_return(r["ticker"], r["trade_date"],  5, r["buy_price"]), axis=1)
df["prior_30d_vol"] = df.apply(lambda r: compute_volatility(r["ticker"], r["trade_date"], 30), axis=1)
df["prior_10d_vol"] = df.apply(lambda r: compute_volatility(r["ticker"], r["trade_date"], 10), axis=1)
df["prior_5d_vol"]  = df.apply(lambda r: compute_volatility(r["ticker"], r["trade_date"],  5), axis=1)
df["actual_2d_return_pct"] = df.apply(lambda r: compute_actual_return(r["ticker"], r["trade_date"], 2), axis=1)

# ── Historical insider features from insider_purchases.csv ────────────────────
print("Loading historical insider purchases for recency features...")
hist_insiders = pd.read_csv(ROOT / "backtest/data/insider_purchases.csv")
hist_insiders["trade_date"] = pd.to_datetime(hist_insiders["trade_date"], errors="coerce")
hist_insiders = hist_insiders.dropna(subset=["trade_date"])

def days_since_last_buy(owner_name, trade_date, same_ticker=None):
    mask = hist_insiders["owner_name"] == owner_name
    if same_ticker:
        mask = mask & (hist_insiders["ticker"] == same_ticker)
    prior = hist_insiders[mask & (hist_insiders["trade_date"] < trade_date)]
    if prior.empty:
        return 999.0
    last = prior["trade_date"].max()
    return float((trade_date - last).days)

def count_same_ticker_insiders_365d(ticker, trade_date):
    cutoff = trade_date - timedelta(days=365)
    mask = (hist_insiders["ticker"] == ticker) & \
           (hist_insiders["trade_date"] >= cutoff) & \
           (hist_insiders["trade_date"] < trade_date)
    return float(hist_insiders[mask]["owner_name"].nunique())

def count_prior_buys(owner_name, trade_date):
    mask = (hist_insiders["owner_name"] == owner_name) & \
           (hist_insiders["trade_date"] < trade_date)
    return float(hist_insiders[mask].shape[0])

df["days_since_last_buy_any"]         = df.apply(lambda r: days_since_last_buy(r["owner"], r["trade_date"]), axis=1)
df["days_since_last_buy_same_ticker"] = df.apply(lambda r: days_since_last_buy(r["owner"], r["trade_date"], r["ticker"]), axis=1)
df["same_ticker_insider_count_365d"]  = df.apply(lambda r: count_same_ticker_insiders_365d(r["ticker"], r["trade_date"]), axis=1)
df["insider_prior_buys"]              = df.apply(lambda r: count_prior_buys(r["owner"], r["trade_date"]), axis=1)
df["owner_name"] = df["owner"]

# ── Other required fields ─────────────────────────────────────────────────────
df["transaction_date"]  = df["trade_date"].dt.strftime("%Y-%m-%d") + " 16:00:00"
df["buy_datetime"]      = df["trade_date"].dt.strftime("%Y-%m-%d") + " 09:45:00"
df["trade_date_d"]      = df["trade_date"].dt.date
df["filing_gap_days"]   = (pd.Timestamp("2026-03-08") - df["trade_date"]).dt.days.clip(0, 5)
df["filing_hour_et"]    = 14.0 + 52.0/60.0  # 14:52 from Found At
df["value_usd"]         = df["buy_price"] * 10000  # rough estimate
df["price_drift_filing_pct"] = 0.0  # filing price ≈ trade price (conservative)
df["company_name"]      = df["ticker"]
df["event_key"]         = df["ticker"] + "|" + df["trade_date"].dt.strftime("%Y-%m-%d")
df["representative_transaction_date"] = df["transaction_date"]
df["n_insiders_in_cluster"] = df.groupby("ticker")["ticker"].transform("count").astype(float)

# ── Engineer features ─────────────────────────────────────────────────────────
feat_df, features, _ = train_models.engineer_features(df.copy())

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models...")
from live_scoring import load_models_and_policy, score_features, predict_model, HORIZONS, MODEL_NAMES

model_dir = ROOT / "models/prod4"
models_by_horizon, policy = load_models_and_policy(model_dir)

# ── Score ──────────────────────────────────────────────────────────────────────
print("Scoring...")
scored = score_features(feat_df, models_by_horizon, policy)
scored_1d = scored[scored["horizon_days"] == 1][["ticker", "trade_date", "pred_mean4"]].rename(
    columns={"pred_mean4": "model_score_1d"}
)

# ── Merge back for display ─────────────────────────────────────────────────────
scored_1d["trade_date"] = scored_1d["trade_date"].astype(str)
df["trade_date_str"] = df["trade_date"].dt.strftime("%Y-%m-%d")
out = df.merge(scored_1d, left_on=["ticker", "trade_date_str"], right_on=["ticker", "trade_date"], how="left")

# ── Display results ────────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "ticker", "trade_date_str", "buy_price",
    "prior_5d_pct", "prior_10d_pct", "prior_30d_pct",
    "prior_5d_vol", "prior_10d_vol", "prior_30d_vol",
    "email_score_1d", "model_score_1d", "actual_2d_return_pct",
]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:.3f}".format)

print("\n" + "="*120)
print("FEATURE SNAPSHOT — Email Trades (sorted by email score desc)")
print("="*120)
print(out[DISPLAY_COLS].sort_values("email_score_1d", ascending=False).to_string(index=False))

# ── Summary statistics ─────────────────────────────────────────────────────────
print("\n" + "="*120)
print("SUMMARY STATS — key features across all 36 trades")
print("="*120)
summary_cols = ["prior_5d_pct", "prior_10d_pct", "prior_30d_pct",
                "prior_5d_vol", "prior_10d_vol", "prior_30d_vol",
                "buy_price", "actual_2d_return_pct", "model_score_1d"]
print(out[summary_cols].describe().round(3).to_string())

# ── Compare with training distribution ────────────────────────────────────────
print("\n" + "="*120)
print("TRAINING DATA COMPARISON (from backtest_results_aggregated.csv)")
print("="*120)
try:
    train_df = pd.read_csv(ROOT / "backtest/data/backtest_results_aggregated.csv")
    compare_cols = [c for c in ["prior_5d_pct","prior_10d_pct","prior_30d_pct",
                                 "prior_5d_vol","prior_10d_vol","prior_30d_vol",
                                 "buy_price"] if c in train_df.columns]
    if compare_cols:
        print("Training data medians:")
        print(train_df[compare_cols].median().round(3))
        print("\nEmail trades medians:")
        print(out[compare_cols].median().round(3))
    else:
        print("  (Columns not found in aggregated CSV — skipping comparison)")
except Exception as e:
    print(f"  Could not load training data: {e}")

# ── Per-stock verdict ─────────────────────────────────────────────────────────
print("\n" + "="*120)
print("VERDICT — model_score_1d vs email_score_1d (delta shows re-run drift)")
print("="*120)
verdict = out[["ticker","trade_date_str","email_score_1d","model_score_1d","actual_2d_return_pct","prior_30d_pct","prior_30d_vol","buy_price"]].copy()
verdict["delta"] = verdict["model_score_1d"] - verdict["email_score_1d"]
print(verdict.sort_values("email_score_1d", ascending=False).to_string(index=False))
