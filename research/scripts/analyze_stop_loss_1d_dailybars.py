"""
Analyze 1-day stop-loss levels using T+1 daily OHLC bars.

Model (long-only):
- Entry at existing backtest buy_price.
- Exit on first trading day after buy_date (T+1).
- If T+1 low <= stop_price, stop is considered hit.
- Gap-aware fill: if T+1 open <= stop_price, exit at open; else exit at stop_price.
- If not hit, exit at T+1 close.

Optional filter:
- Keep only rows where 1-day ensemble average prediction (p_mean) > 0.
  p_mean = mean(HGBR_pred, XGB_pred, HMM-XGB_pred) on the 1d test split.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from hmmlearn import hmm as hmmlib
from polygon import RESTClient
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

from train_models import FEATURES, engineer_features, load_and_merge

ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("data/price_cache")
DEFAULT_INPUT = "data/backtest_results_aggregated.csv"
MODEL_DIR = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_REGIMES = 4
HMM_COLS = ["prior_30d_pct", "log_buy_price"]

XGB_BASE = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    early_stopping_rounds=20,
    enable_categorical=True,
    tree_method="hist",
)


@dataclass
class T1Bar:
    open: float
    low: float
    close: float


class PriceCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, from_s: str, to_s: str) -> Path:
        return self.cache_dir / f"{ticker}_day_{from_s}_{to_s}.json"

    def get(self, ticker: str, from_s: str, to_s: str) -> Optional[list]:
        p = self._path(ticker, from_s, to_s)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, ticker: str, from_s: str, to_s: str, rows: list) -> None:
        p = self._path(ticker, from_s, to_s)
        p.write_text(json.dumps(rows), encoding="utf-8")


def to_xgb(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "officer_type_enc" in X.columns:
        X["officer_type_enc"] = X["officer_type_enc"].astype("category")
    return X


def train_xgb(X_tr: pd.DataFrame, y_tr: np.ndarray, params: dict = XGB_BASE) -> xgb.XGBRegressor:
    if len(X_tr) < 60:
        p2 = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        p2["n_estimators"] = 100
        m = xgb.XGBRegressor(**p2)
        m.fit(to_xgb(X_tr), y_tr, verbose=False)
        return m

    X_t2, X_v, y_t2, y_v = train_test_split(
        to_xgb(X_tr), y_tr, test_size=0.1, random_state=RANDOM_STATE
    )
    m = xgb.XGBRegressor(**params)
    m.fit(X_t2, y_t2, eval_set=[(X_v, y_v)], verbose=False)
    return m


def fit_hmm(X_seq: np.ndarray):
    valid = ~np.isnan(X_seq).any(axis=1)
    Xv = X_seq[valid]
    mu = Xv.mean(0)
    std = Xv.std(0) + 1e-8
    Xs = (Xv - mu) / std

    model = hmmlib.GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(Xs)
    return model, mu, std


def assign_regimes(model, mu: np.ndarray, std: np.ndarray, X_obs: np.ndarray, fallback: int) -> np.ndarray:
    states = np.full(len(X_obs), fallback, dtype=int)
    valid = ~np.isnan(X_obs).any(axis=1)
    if valid.sum() == 0:
        return states

    Xs = (X_obs[valid] - mu) / std
    log_em = np.stack(
        [
            multivariate_normal.logpdf(
                Xs,
                mean=model.means_[k],
                cov=model.covars_[k],
                allow_singular=True,
            )
            for k in range(N_REGIMES)
        ],
        axis=1,
    )
    states[valid] = log_em.argmax(axis=1)
    return states


def hmm_regime_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    sub: pd.DataFrame,
    idx_train,
    global_xgb: xgb.XGBRegressor,
) -> np.ndarray:
    tr_dates = pd.to_datetime(sub.loc[idx_train, "trade_date"]).values
    sort_order = np.argsort(tr_dates)
    unsort = np.argsort(sort_order)

    X_train_sorted = X_train[HMM_COLS].values[sort_order]
    hmm_model, hmm_mu, hmm_std = fit_hmm(X_train_sorted)

    valid_mask = ~np.isnan(X_train_sorted).any(axis=1)
    valid_preds = assign_regimes(hmm_model, hmm_mu, hmm_std, X_train_sorted[valid_mask], 0)
    modal_state = int(np.bincount(valid_preds, minlength=N_REGIMES).argmax())

    tr_states_sorted = assign_regimes(hmm_model, hmm_mu, hmm_std, X_train_sorted, modal_state)
    tr_states = tr_states_sorted[unsort]
    te_states = assign_regimes(hmm_model, hmm_mu, hmm_std, X_test[HMM_COLS].values, modal_state)

    regime_models: Dict[int, xgb.XGBRegressor] = {}
    for s in range(N_REGIMES):
        mask = tr_states == s
        n = int(mask.sum())
        if n < 50:
            regime_models[s] = global_xgb
        else:
            regime_models[s] = train_xgb(X_train.iloc[mask], y_train[mask])

    preds = np.zeros(len(X_test))
    for s in range(N_REGIMES):
        mask = te_states == s
        if mask.sum() == 0:
            continue
        preds[mask] = regime_models[s].predict(to_xgb(X_test.iloc[mask]))

    return preds


def compute_1d_pmean_testset() -> pd.DataFrame:
    df, features, _ = engineer_features(load_and_merge())

    tgt = "return_1d_pct"
    sub = df.dropna(subset=[tgt]).copy()
    lo, hi = sub[tgt].quantile([0.01, 0.99])
    sub[tgt] = sub[tgt].clip(lo, hi)

    X = sub[features].copy()
    y = sub[tgt].values

    X_tr, X_te, y_tr, _, idx_tr, idx_te = train_test_split(
        X,
        y,
        sub.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    hgbr = joblib.load(MODEL_DIR / "model_1d.pkl")
    p_hgbr = hgbr.predict(X_te)

    xgb_path = MODEL_DIR / "xgb_model_1d.pkl"
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
    else:
        xgb_model = train_xgb(X_tr, y_tr)
    p_xgb = xgb_model.predict(to_xgb(X_te))

    p_hmm = hmm_regime_predict(X_tr, y_tr, X_te, sub, idx_tr, xgb_model)
    p_mean = (p_hgbr + p_xgb + p_hmm) / 3.0

    return pd.DataFrame({"source_index": idx_te, "p_mean": p_mean})


def aggs_to_dicts(aggs) -> List[dict]:
    out = []
    for a in aggs:
        if not a.timestamp:
            continue
        out.append(
            {
                "t": a.timestamp,
                "o": float(a.open) if a.open is not None else None,
                "h": float(a.high) if a.high is not None else None,
                "l": float(a.low) if a.low is not None else None,
                "c": float(a.close) if a.close is not None else None,
            }
        )
    return out


def fetch_daily_bars(client: RESTClient, cache: PriceCache, ticker: str, from_d: date, to_d: date) -> List[dict]:
    from_s = from_d.strftime("%Y-%m-%d")
    to_s = to_d.strftime("%Y-%m-%d")

    cached = cache.get(ticker, from_s, to_s)
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
        rows = aggs_to_dicts(aggs)
    except Exception:
        rows = []

    cache.put(ticker, from_s, to_s, rows)
    return rows


def first_trading_day_after(daily_bars: List[dict], buy_date: date) -> Optional[T1Bar]:
    for bar in daily_bars:
        d = datetime.fromtimestamp(bar["t"] / 1000, tz=ET).date()
        if d > buy_date:
            if bar["o"] is None or bar["l"] is None or bar["c"] is None:
                return None
            return T1Bar(open=bar["o"], low=bar["l"], close=bar["c"])
    return None


def process_ticker(ticker: str, rows: List[dict], api_key: str, cache: PriceCache) -> List[dict]:
    client = RESTClient(api_key=api_key, retries=3)
    min_buy = min(r["buy_date"] for r in rows)
    max_buy = max(r["buy_date"] for r in rows)

    daily = fetch_daily_bars(client, cache, ticker, min_buy, max_buy + timedelta(days=14))

    out = []
    for r in rows:
        t1 = first_trading_day_after(daily, r["buy_date"])
        if t1 is None:
            continue

        buy = r["buy_price"]
        baseline_ret = (t1.close - buy) / buy * 100.0
        out.append(
            {
                "source_index": r["source_index"],
                "ticker": ticker,
                "buy_date": r["buy_date"],
                "buy_price": buy,
                "t1_open": t1.open,
                "t1_low": t1.low,
                "t1_close": t1.close,
                "baseline_return_1d_pct": baseline_ret,
            }
        )

    return out


def apply_stop_return(buy: float, t1_open: float, t1_low: float, t1_close: float, stop_pct: float) -> tuple[float, bool]:
    stop_price = buy * (1.0 - stop_pct / 100.0)

    if t1_low <= stop_price:
        exit_price = t1_open if t1_open <= stop_price else stop_price
        ret = (exit_price - buy) / buy * 100.0
        return ret, True

    ret = (t1_close - buy) / buy * 100.0
    return ret, False


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise SystemExit("POLYGON_API_KEY missing in .env")

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["ticker", "buy_datetime", "buy_price"]).copy()
    df["buy_datetime"] = pd.to_datetime(df["buy_datetime"], errors="coerce")
    df = df.dropna(subset=["buy_datetime"])
    df["buy_date"] = df["buy_datetime"].dt.date
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
    df = df.dropna(subset=["buy_price"])

    if args.cluster_only:
        if "n_insiders_in_cluster" not in df.columns:
            raise SystemExit("cluster-only requested but n_insiders_in_cluster column is missing")
        df["n_insiders_in_cluster"] = pd.to_numeric(df["n_insiders_in_cluster"], errors="coerce")
        df = df[df["n_insiders_in_cluster"] >= 2].copy()

    if args.pmean_gt0:
        if Path(args.input).name != "backtest_results_aggregated.csv":
            raise SystemExit("--pmean-gt0 currently requires input=data/backtest_results_aggregated.csv")
        preds = compute_1d_pmean_testset()
        keep_idx = set(preds.loc[preds["p_mean"] > 0, "source_index"].astype(int).tolist())
        df = df[df.index.isin(keep_idx)].copy()
        print(
            "Applied filter p_mean > 0 (1d test split): "
            f"kept {len(df):,} rows from aggregated input"
        )

    if df.empty:
        raise SystemExit("No rows left after filters")

    ticker_groups: Dict[str, List[dict]] = {}
    for idx, row in df.iterrows():
        ticker_groups.setdefault(row["ticker"], []).append(
            {
                "source_index": int(idx),
                "buy_date": row["buy_date"],
                "buy_price": float(row["buy_price"]),
            }
        )

    cache = PriceCache(CACHE_DIR)
    enriched: List[dict] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(process_ticker, tkr, rows, api_key, cache): tkr for tkr, rows in ticker_groups.items()}
        for fut in as_completed(futs):
            enriched.extend(fut.result())

    if not enriched:
        raise SystemExit("No trades had usable T+1 daily bars")

    e = pd.DataFrame(enriched)

    base = e["baseline_return_1d_pct"]
    print(f"Rows analyzed: {len(e):,}")
    print(
        "Baseline (no stop): "
        f"mean={base.mean():+.4f}% median={base.median():+.4f}% winrate={(base > 0).mean() * 100:.2f}%"
    )

    rows = []
    for stop in range(args.min_stop, args.max_stop + 1):
        rets = []
        hits = 0
        for _, r in e.iterrows():
            ret, hit = apply_stop_return(
                buy=float(r["buy_price"]),
                t1_open=float(r["t1_open"]),
                t1_low=float(r["t1_low"]),
                t1_close=float(r["t1_close"]),
                stop_pct=float(stop),
            )
            rets.append(ret)
            if hit:
                hits += 1

        s = pd.Series(rets)
        rows.append(
            {
                "stop_loss_pct": -stop,
                "mean_return_pct": s.mean(),
                "median_return_pct": s.median(),
                "win_rate_pct": (s > 0).mean() * 100.0,
                "stop_hit_pct": hits / len(s) * 100.0,
                "delta_vs_baseline": s.mean() - base.mean(),
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_return_pct", ascending=False)
    pd.set_option("display.max_rows", 200)
    print("\nStop sweep (sorted by mean return):")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\nSaved stop sweep to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--cluster-only", action="store_true", default=True)
    parser.add_argument("--all-events", action="store_true", help="Analyze all events, not only clusters")
    parser.add_argument("--pmean-gt0", action="store_true", help="Filter to rows with 1d ensemble p_mean > 0")
    parser.add_argument("--min-stop", type=int, default=1, help="Minimum stop percent, e.g. 1 means -1%")
    parser.add_argument("--max-stop", type=int, default=10, help="Maximum stop percent, e.g. 10 means -10%")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--output", default="data/stop_sweep_1d_dailybars.csv")

    args = parser.parse_args()
    if args.all_events:
        args.cluster_only = False
    run(args)
