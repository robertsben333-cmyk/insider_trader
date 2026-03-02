"""
Practice run: screen low-effort feature candidates against the current baseline.

Outputs:
  - research/outcomes/feature_screen_detail.csv
  - research/outcomes/feature_screen_summary.csv
  - research/outcomes/feature_screen_recommendations.csv
  - research/outcomes/feature_screen_meta.json
"""

from __future__ import annotations

import bisect
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from train_models import (  # noqa: E402
    FEATURES,
    HORIZONS,
    MODEL_PARAMS,
    clean_money,
    engineer_features,
    load_and_merge,
)

ET = ZoneInfo("America/New_York")
ORIGINAL_CSV = BASE / "backtest" / "data" / "insider_purchases.csv"
PRICE_CACHE = BASE / "backtest" / "data" / "price_cache"
OUT_DIR = BASE / "research" / "outcomes"


def clean_numeric(val) -> float:
    if not isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return np.nan
    s = val.replace("$", "").replace(",", "").replace("+", "").strip()
    if not s or s.lower() in ("n/a", "new"):
        return np.nan
    if "%" in s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _to_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _bisect_count_in_window(dates: List, current, days: int) -> int:
    if current is None:
        return 0
    left = bisect.bisect_left(dates, current - timedelta(days=days))
    right = bisect.bisect_left(dates, current)
    return max(0, right - left)


def _days_since_prev(dates: List, current) -> float:
    if current is None:
        return np.nan
    idx = bisect.bisect_left(dates, current)
    if idx <= 0:
        return np.nan
    prev = dates[idx - 1]
    return float((current - prev).days)


def build_history_maps() -> Tuple[dict, dict, dict]:
    hist = pd.read_csv(ORIGINAL_CSV, usecols=["owner_name", "ticker", "trade_date"])
    hist["trade_date_d"] = _to_date_series(hist["trade_date"])
    hist = hist.dropna(subset=["owner_name", "ticker", "trade_date_d"])

    owner_dates = defaultdict(list)
    owner_ticker_dates = defaultdict(list)
    ticker_dates = defaultdict(list)

    for rec in hist[["owner_name", "ticker", "trade_date_d"]].itertuples(index=False):
        owner_dates[rec.owner_name].append(rec.trade_date_d)
        owner_ticker_dates[(rec.owner_name, rec.ticker)].append(rec.trade_date_d)
        ticker_dates[rec.ticker].append(rec.trade_date_d)

    for d in (owner_dates, owner_ticker_dates, ticker_dates):
        for k in d:
            d[k] = sorted(set(d[k]))

    return owner_dates, owner_ticker_dates, ticker_dates


def add_history_features(
    df: pd.DataFrame,
    owner_dates: dict,
    owner_ticker_dates: dict,
    ticker_dates: dict,
) -> pd.DataFrame:
    out = df.copy()
    if "trade_date_d" in out.columns:
        td = _to_date_series(out["trade_date_d"])
    else:
        td = _to_date_series(out["trade_date"])

    owners = out["owner_name"].tolist()
    tickers = out["ticker"].tolist()
    td_list = td.tolist()

    days_any = []
    days_same_ticker = []
    c30 = []
    c90 = []
    c365_same_owner_ticker = []

    for owner, ticker, d in zip(owners, tickers, td_list):
        own_dates = owner_dates.get(owner, [])
        own_t_dates = owner_ticker_dates.get((owner, ticker), [])
        t_dates = ticker_dates.get(ticker, [])

        days_any.append(_days_since_prev(own_dates, d))
        days_same_ticker.append(_days_since_prev(own_t_dates, d))
        c30.append(_bisect_count_in_window(t_dates, d, 30))
        c90.append(_bisect_count_in_window(t_dates, d, 90))
        c365_same_owner_ticker.append(_bisect_count_in_window(own_t_dates, d, 365))

    out["days_since_last_buy_any"] = days_any
    out["days_since_last_buy_same_ticker"] = days_same_ticker
    out["ticker_buy_count_30d"] = np.log1p(np.asarray(c30, dtype=float))
    out["ticker_buy_count_90d"] = np.log1p(np.asarray(c90, dtype=float))
    out["same_ticker_insider_count_365d"] = np.log1p(np.asarray(c365_same_owner_ticker, dtype=float))
    return out


def load_day_cache_for_tickers(tickers: Iterable[str]) -> Dict[str, dict]:
    store: Dict[str, dict] = {}
    tickers = sorted(set(tickers))
    for t in tickers:
        day_files = glob.glob(str(PRICE_CACHE / f"{t}_day_*.json"))
        if not day_files:
            continue

        open_map: Dict = {}
        close_map: Dict = {}
        for path in day_files:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    bars = json.load(fh)
            except Exception:
                continue
            if not isinstance(bars, list):
                continue
            for b in bars:
                if not isinstance(b, dict):
                    continue
                ts = b.get("t")
                c = b.get("c")
                if ts is None or c is None:
                    continue
                try:
                    d = pd.Timestamp(ts, unit="ms", tz="UTC").tz_convert(ET).date()
                    close_map[d] = float(c)
                    if b.get("o") is not None:
                        open_map[d] = float(b["o"])
                except Exception:
                    continue

        if not close_map:
            continue
        dates = sorted(close_map.keys())
        store[t] = {"dates": dates, "open": open_map, "close": close_map}

    return store


def add_entry_day_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["buy_date_d"] = _to_date_series(out["buy_datetime"])
    day_store = load_day_cache_for_tickers(out["ticker"].dropna().unique().tolist())

    out["overnight_gap_to_entry"] = np.nan
    out["open_to_buy_move"] = np.nan

    for ticker, idx in out.groupby("ticker").groups.items():
        cache = day_store.get(ticker)
        if not cache:
            continue
        dates = cache["dates"]
        open_map = cache["open"]
        close_map = cache["close"]
        if not dates:
            continue

        for i in idx:
            buy_date = out.at[i, "buy_date_d"]
            buy_px = out.at[i, "buy_price"]
            if pd.isna(buy_date) or pd.isna(buy_px) or buy_px <= 0:
                continue

            pos = bisect.bisect_left(dates, buy_date)
            if pos > 0:
                prev_close = close_map.get(dates[pos - 1])
                if prev_close and prev_close > 0:
                    out.at[i, "overnight_gap_to_entry"] = (buy_px - prev_close) / prev_close * 100.0

            o = open_map.get(buy_date)
            if o and o > 0:
                out.at[i, "open_to_buy_move"] = (buy_px - o) / o * 100.0

    return out


def fetch_spy_5d_momentum(trade_dates: pd.Series) -> pd.Series:
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return pd.Series(np.nan, index=trade_dates.index)

    try:
        from polygon import RESTClient
    except Exception:
        return pd.Series(np.nan, index=trade_dates.index)

    td = _to_date_series(trade_dates)
    min_d = td.min()
    max_d = td.max()
    if pd.isna(min_d) or pd.isna(max_d):
        return pd.Series(np.nan, index=trade_dates.index)

    try:
        client = RESTClient(api_key=api_key, retries=3)
        aggs = client.get_aggs(
            ticker="SPY",
            multiplier=1,
            timespan="day",
            from_=min_d - timedelta(days=40),
            to=max_d + timedelta(days=5),
            adjusted=True,
            sort="asc",
            limit=50000,
        )
    except Exception:
        return pd.Series(np.nan, index=trade_dates.index)

    dates = []
    closes = []
    for a in aggs:
        try:
            d = pd.Timestamp(a.timestamp, unit="ms", tz="UTC").tz_convert(ET).date()
            c = float(a.close)
            dates.append(d)
            closes.append(c)
        except Exception:
            continue

    if not dates:
        return pd.Series(np.nan, index=trade_dates.index)

    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    closes = [closes[i] for i in order]

    spy_vals = []
    for d in td.tolist():
        if d is None:
            spy_vals.append(np.nan)
            continue
        i_now = bisect.bisect_right(dates, d) - 1
        i_lag = bisect.bisect_right(dates, d - timedelta(days=5)) - 1
        if i_now < 0 or i_lag < 0:
            spy_vals.append(np.nan)
            continue
        c_now = closes[i_now]
        c_lag = closes[i_lag]
        if c_lag <= 0:
            spy_vals.append(np.nan)
            continue
        spy_vals.append((c_now - c_lag) / c_lag * 100.0)

    return pd.Series(spy_vals, index=trade_dates.index, dtype=float)


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    owner_dates, owner_ticker_dates, ticker_dates = build_history_maps()
    out = add_history_features(out, owner_dates, owner_ticker_dates, ticker_dates)
    out = add_entry_day_features(out)

    txn = pd.to_datetime(out["transaction_date"], errors="coerce")
    out["filing_hour_et"] = txn.dt.hour + txn.dt.minute / 60.0
    out["is_after_close"] = np.where(txn.notna(), (txn.dt.hour >= 16).astype(float), np.nan)
    out["filing_weekday"] = txn.dt.dayofweek.astype(float)

    out["cluster_buy"] = out.get("cluster_buy", False).astype(float)
    out["insider_bought_ticker"] = out.get("insider_bought_ticker", 0).astype(float)

    out["momentum_slope"] = out["prior_5d_pct"] - out["prior_30d_pct"]
    out["vol_regime"] = _safe_ratio(out["prior_5d_vol"], out["prior_30d_vol"])

    qty_num = out["qty"].apply(clean_numeric)
    last_price = out["last_price_clean"] if "last_price_clean" in out.columns else out["last_price"].apply(clean_money)
    vps = _safe_ratio(out["value_usd"], qty_num)

    out["qty_log"] = np.log1p(qty_num.where(qty_num > 0))
    out["value_per_share_delta"] = _safe_ratio(vps - last_price, last_price) * 100.0

    spy_5d = fetch_spy_5d_momentum(out["trade_date"])
    out["market_relative_5d"] = out["prior_5d_pct"] - spy_5d

    num_clip_cols = [
        "days_since_last_buy_any",
        "days_since_last_buy_same_ticker",
        "momentum_slope",
        "vol_regime",
        "value_per_share_delta",
        "overnight_gap_to_entry",
        "open_to_buy_move",
        "market_relative_5d",
    ]
    for c in num_clip_cols:
        if c in out.columns:
            lo, hi = out[c].quantile([0.005, 0.995])
            out[c] = out[c].clip(lo, hi)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def top_decile_stats(pred: np.ndarray, y: np.ndarray) -> dict:
    order = np.argsort(pred)[::-1]
    n = max(1, len(order) // 10)
    top = y[order[:n]]
    bot = y[order[-n:]]
    return {
        "top_decile_mean": float(np.mean(top)),
        "bottom_decile_mean": float(np.mean(bot)),
        "decile_spread": float(np.mean(top) - np.mean(bot)),
        "top_decile_win_rate": float(np.mean(top > 0)),
        "n_decile": int(n),
    }


def chrono_split(sub: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(sub)
    n_train = int(round(n * 0.6))
    n_val = int(round(n * 0.2))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    s1 = n_train
    s2 = n_train + n_val
    return sub.iloc[:s1].copy(), sub.iloc[s1:s2].copy(), sub.iloc[s2:].copy()


def evaluate_one(
    sub: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> dict:
    keep = [c for c in feature_cols if c in sub.columns]
    X = sub[keep].copy()
    y = sub[target_col].to_numpy(dtype=float)

    for c in keep:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)

    train_df, val_df, test_df = chrono_split(pd.concat([X, sub[[target_col, "trade_date"]]], axis=1))
    X_tr = train_df[keep]
    y_tr = train_df[target_col].to_numpy(dtype=float)
    X_va = val_df[keep]
    y_va = val_df[target_col].to_numpy(dtype=float)
    X_te = test_df[keep]
    y_te = test_df[target_col].to_numpy(dtype=float)

    X_fit = pd.concat([X_tr, X_va], axis=0)
    y_fit = np.concatenate([y_tr, y_va], axis=0)

    model.fit(X_fit, y_fit)
    pred = model.predict(X_te)
    ds = top_decile_stats(pred, y_te)
    r2 = r2_score(y_te, pred)

    return {
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "r2_test": float(r2),
        **ds,
    }


def build_feature_sets(base_features: List[str], candidates: List[str]) -> Dict[str, List[str]]:
    sets = {"baseline": base_features.copy()}
    for c in candidates:
        sets[f"plus_{c}"] = base_features + [c]
    sets["plus_all_candidates"] = base_features + candidates
    return sets


def rank_recommendations(detail: pd.DataFrame, coverage: pd.Series) -> pd.DataFrame:
    s = detail[detail["scenario"].str.startswith("plus_") & (detail["scenario"] != "plus_all_candidates")].copy()
    s["feature"] = s["scenario"].str.replace("plus_", "", regex=False)

    agg = (
        s.groupby("feature", as_index=False)
        .agg(
            avg_delta_top_decile=("delta_top_decile_mean", "mean"),
            avg_delta_spread=("delta_decile_spread", "mean"),
            avg_delta_win_rate=("delta_top_decile_win_rate", "mean"),
            improved_horizons=("delta_top_decile_mean", lambda x: int((x > 0).sum())),
            degraded_horizons=("delta_top_decile_mean", lambda x: int((x < 0).sum())),
        )
        .sort_values(["avg_delta_top_decile", "avg_delta_spread"], ascending=False)
        .reset_index(drop=True)
    )

    agg["coverage"] = agg["feature"].map(coverage.to_dict()).astype(float)
    agg["recommendation"] = "skip_for_now"
    agg.loc[
        (agg["avg_delta_top_decile"] >= 0.15)
        & (agg["improved_horizons"] >= 3)
        & (agg["coverage"] >= 0.60),
        "recommendation",
    ] = "add_now"
    agg.loc[
        (agg["recommendation"] == "skip_for_now")
        & (agg["avg_delta_top_decile"] >= 0.02)
        & (agg["improved_horizons"] >= 2)
        & (agg["coverage"] >= 0.30),
        "recommendation",
    ] = "test_further"
    return agg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and merging source data...")
    raw = load_and_merge()
    print(f"  rows={len(raw):,}")

    raw = add_candidate_features(raw)
    df, base_features, _ = engineer_features(raw)

    candidates = [
        "cluster_buy",
        "insider_bought_ticker",
        "days_since_last_buy_any",
        "days_since_last_buy_same_ticker",
        "filing_hour_et",
        "is_after_close",
        "filing_weekday",
        "momentum_slope",
        "vol_regime",
        "qty_log",
        "value_per_share_delta",
        "ticker_buy_count_30d",
        "ticker_buy_count_90d",
        "same_ticker_insider_count_365d",
        "overnight_gap_to_entry",
        "open_to_buy_move",
        "market_relative_5d",
    ]

    feature_sets = build_feature_sets(base_features, candidates)
    detail_rows = []

    for h in HORIZONS:
        target = f"return_{h}d_pct"
        sub = df.dropna(subset=[target]).copy()
        sub[target] = sub[target].clip(sub[target].quantile(0.01), sub[target].quantile(0.99))
        sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
        sub = sub.dropna(subset=["trade_date"]).sort_values("trade_date").copy()

        print(f"\nHorizon {h}d | rows={len(sub):,}")
        baseline = evaluate_one(sub, target, feature_sets["baseline"])
        base_top = baseline["top_decile_mean"]
        base_spread = baseline["decile_spread"]
        base_wr = baseline["top_decile_win_rate"]

        detail_rows.append(
            {
                "horizon": h,
                "target": target,
                "scenario": "baseline",
                **baseline,
                "delta_top_decile_mean": 0.0,
                "delta_decile_spread": 0.0,
                "delta_top_decile_win_rate": 0.0,
            }
        )

        for name, feats in feature_sets.items():
            if name == "baseline":
                continue
            m = evaluate_one(sub, target, feats)
            detail_rows.append(
                {
                    "horizon": h,
                    "target": target,
                    "scenario": name,
                    **m,
                    "delta_top_decile_mean": float(m["top_decile_mean"] - base_top),
                    "delta_decile_spread": float(m["decile_spread"] - base_spread),
                    "delta_top_decile_win_rate": float(m["top_decile_win_rate"] - base_wr),
                }
            )
            print(
                f"  {name:<34} top_decile={m['top_decile_mean']:+.3f}% "
                f"(delta {m['top_decile_mean'] - base_top:+.3f})"
            )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["scenario"], as_index=False)
        .agg(
            avg_top_decile_mean=("top_decile_mean", "mean"),
            avg_decile_spread=("decile_spread", "mean"),
            avg_top_decile_win_rate=("top_decile_win_rate", "mean"),
            avg_r2_test=("r2_test", "mean"),
            avg_delta_top_decile_mean=("delta_top_decile_mean", "mean"),
            avg_delta_decile_spread=("delta_decile_spread", "mean"),
        )
        .sort_values(["avg_delta_top_decile_mean", "avg_delta_decile_spread"], ascending=False)
        .reset_index(drop=True)
    )

    coverage = pd.Series({c: float(df[c].notna().mean()) if c in df.columns else 0.0 for c in candidates})
    recs = rank_recommendations(detail, coverage)

    detail_path = OUT_DIR / "feature_screen_detail.csv"
    summary_path = OUT_DIR / "feature_screen_summary.csv"
    recs_path = OUT_DIR / "feature_screen_recommendations.csv"
    meta_path = OUT_DIR / "feature_screen_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    recs.to_csv(recs_path, index=False)

    meta = {
        "base_feature_count": len(base_features),
        "candidate_feature_count": len(candidates),
        "candidate_coverage": {k: float(v) for k, v in coverage.to_dict().items()},
        "outputs": {
            "detail": str(detail_path),
            "summary": str(summary_path),
            "recommendations": str(recs_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\nSaved outputs:")
    print(f"  {detail_path}")
    print(f"  {summary_path}")
    print(f"  {recs_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
