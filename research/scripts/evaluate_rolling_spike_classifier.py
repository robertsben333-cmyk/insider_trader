from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit("xgboost is required. Install it with: py -m pip install xgboost") from exc

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
os.chdir(BASE)

import train_models
from research.scripts import build_unfiltered_aggregated_backtest as aggregate_builder


DEFAULT_RAW_CSV = BASE / "backtest" / "data" / "insider_purchases.csv"
DEFAULT_AGG_CSV = BASE / "backtest" / "data" / "backtest_results_aggregated_live_entry.csv"
DEFAULT_CACHE_DIR = BASE / "backtest" / "data" / "price_cache"
OUTPUT_DIR = BASE / "research" / "outcomes" / "models" / "rolling_spike_3d"

SUMMARY_JSON = OUTPUT_DIR / "summary.json"
SPLIT_METRICS_CSV = OUTPUT_DIR / "split_metrics.csv"
TEST_PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"
CALIBRATION_TABLE_CSV = OUTPUT_DIR / "calibration_table.csv"
FEATURE_IMPORTANCE_CSV = OUTPUT_DIR / "feature_importance_summary.csv"
DATASET_SUMMARY_CSV = OUTPUT_DIR / "dataset_summary.csv"
DATASET_SAMPLE_CSV = OUTPUT_DIR / "dataset_sample.csv"

MODEL_NAMES = ["HGB", "XGBoost", "ElasticNet", "SplineElasticNet"]
SEQUENTIAL_STATE_FEATURES = [
    "snapshot_day_idx",
    "days_remaining_in_10d_cycle",
    "return_since_first_snapshot_pct",
    "max_intraday_runup_so_far_pct",
    "snapshot_is_open",
]
METADATA_BASELINE_FEATURES = [
    "insider_type",
    "n_insiders_in_cluster",
    "filing_hour_et",
    "snapshot_is_open",
]

THRESHOLD_PCT = 8.0
LOOKAHEAD_DAYS = 3
MAX_CHECKS = 10
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
EMBARGO_TRADING_DAYS = 3
RANDOM_STATE = 42
PERMUTATION_REPEATS = 5
PERMUTATION_MAX_ROWS = 5000
TARGET_COL = "hit_8pct_intraday_next3d"

XGB_CLF_PARAMS = dict(
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
    objective="binary:logistic",
    eval_metric="logloss",
)

LINEAR_CV_CONFIG = [
    {
        "max_rows": 1000,
        "cv": 5,
        "cs": np.logspace(-4, 2, 12),
        "l1_ratios": [0.1, 0.3, 0.5, 0.7, 0.9],
        "spline_knots": 6,
    },
    {
        "max_rows": 5000,
        "cv": 4,
        "cs": np.logspace(-3, 1.5, 8),
        "l1_ratios": [0.1, 0.5, 0.9],
        "spline_knots": 5,
    },
    {
        "max_rows": None,
        "cv": 3,
        "cs": np.logspace(-2, 1, 6),
        "l1_ratios": [0.1, 0.5, 0.9],
        "spline_knots": 5,
    },
]


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    event_meta: pd.DataFrame
    boundaries: dict[str, str]


class ConstantProbabilityCalibrator:
    def __init__(self, p: float) -> None:
        self._p = float(min(max(p, 0.0), 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1, 1)
        p1 = np.full((len(arr), 1), self._p, dtype=float)
        return np.hstack([1.0 - p1, p1])


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("rolling_spike_3d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research pipeline for the rolling 3-day intraday spike classifier. "
            "Predicts whether a filing-driven name will hit +8%% within the next 3 trading days."
        )
    )
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--aggregate-csv", type=Path, default=DEFAULT_AGG_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--rebuild-aggregate", action="store_true")
    parser.add_argument("--cache-only-day", action="store_true")
    parser.add_argument("--limit-events", type=int, default=0, help="Optional cap for a smoke run.")
    return parser.parse_args()


def ensure_live_aggregate(args: argparse.Namespace, logger: logging.Logger) -> None:
    if args.aggregate_csv.exists() and not args.rebuild_aggregate:
        logger.info("Using existing live aggregate: %s", args.aggregate_csv)
        return
    logger.info("Building live aggregate: %s", args.aggregate_csv)
    aggregate_builder.build_dataset(
        input_csv=args.raw_csv,
        output_csv=args.aggregate_csv,
        cache_dir=args.cache_dir,
        max_workers=int(args.max_workers),
        entry_policy="live",
        supported_titles_only=True,
        allow_last_price_fallback=False,
        cache_only_day=bool(args.cache_only_day),
        logger=logger,
    )


def make_event_key_series(df: pd.DataFrame) -> pd.Series:
    trade_date = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df["ticker"].astype(str) + "|" + trade_date.fillna("")


def normalize_et_timestamp(ts: Any) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if pd.isna(out):
        return out
    if out.tzinfo is None:
        return out.tz_localize(train_models.ET)
    return out.tz_convert(train_models.ET)


def normalize_buy_datetime_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize(train_models.ET, ambiguous="NaT", nonexistent="shift_forward")
    return dt.dt.tz_convert(train_models.ET)


def is_intraday_regular_session(ts: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    ts_et = normalize_et_timestamp(ts)
    if ts_et.weekday() >= 5:
        return False
    open_dt = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_dt = ts_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_dt <= ts_et < close_dt


def first_snapshot_session(ts: pd.Timestamp, trading_dates: list[date]) -> tuple[date, str] | None:
    if pd.isna(ts) or not trading_dates:
        return None
    ts_et = normalize_et_timestamp(ts)
    trading_set = set(trading_dates)
    pos = bisect.bisect_left(trading_dates, ts_et.date())
    if is_intraday_regular_session(ts_et):
        if ts_et.date() in trading_set:
            return ts_et.date(), "close"
        return None
    if ts_et.weekday() >= 5:
        candidate_pos = pos
    else:
        market_open = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
        if ts_et < market_open:
            candidate_pos = pos
        else:
            candidate_pos = bisect.bisect_right(trading_dates, ts_et.date())
    if candidate_pos >= len(trading_dates):
        return None
    return trading_dates[candidate_pos], "open"


def build_snapshot_sessions(
    first_date: date,
    first_kind: str,
    trading_dates: list[date],
    max_checks: int = MAX_CHECKS,
) -> list[tuple[date, str]]:
    pos = bisect.bisect_left(trading_dates, first_date)
    if pos >= len(trading_dates) or trading_dates[pos] != first_date:
        return []
    sessions: list[tuple[date, str]] = [(first_date, first_kind)]
    next_pos = pos + 1
    if first_kind == "open":
        sessions.append((first_date, "close"))
    while next_pos < len(trading_dates) and len(sessions) < max_checks:
        sessions.append((trading_dates[next_pos], "close"))
        next_pos += 1
    return sessions[:max_checks]


def _safe_bar_price(bar: dict[str, Any], key: str) -> float | None:
    val = bar.get(key)
    if val is None:
        return None
    try:
        out = float(val)
    except Exception:
        return None
    if not np.isfinite(out) or out <= 0:
        return None
    return out


def close_on_or_before(
    trading_dates: list[date],
    close_by_date: dict[date, float],
    target_date: date,
) -> float | None:
    pos = bisect.bisect_right(trading_dates, target_date) - 1
    if pos < 0:
        return None
    return close_by_date.get(trading_dates[pos])


def compute_prior_momentum_pct(
    snapshot_price: float,
    trading_dates: list[date],
    close_by_date: dict[date, float],
    anchor_date: date,
    days: int,
) -> float:
    if not np.isfinite(snapshot_price) or snapshot_price <= 0:
        return np.nan
    prior_close = close_on_or_before(trading_dates, close_by_date, anchor_date - timedelta(days=days))
    if prior_close is None or not np.isfinite(prior_close) or prior_close <= 0:
        return np.nan
    return ((float(snapshot_price) / float(prior_close)) - 1.0) * 100.0


def compute_prior_vol(
    trading_dates: list[date],
    close_by_date: dict[date, float],
    anchor_date: date,
    days: int,
) -> float:
    cutoff = anchor_date - timedelta(days=days)
    closes = [close_by_date[d] for d in trading_dates if cutoff <= d <= anchor_date and d in close_by_date]
    if len(closes) < 2:
        return np.nan
    arr = np.asarray(closes, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)
    arr = arr[valid]
    if len(arr) < 3:
        return np.nan
    log_rets = np.diff(np.log(arr))
    if len(log_rets) < 2:
        return np.nan
    return float(np.std(log_rets, ddof=1) * np.sqrt(252.0))


def compute_max_runup_so_far_pct(
    first_snapshot_price: float,
    trading_dates: list[date],
    high_by_date: dict[date, float],
    current_date: date,
    current_kind: str,
) -> float:
    if not np.isfinite(first_snapshot_price) or first_snapshot_price <= 0:
        return np.nan
    highs = []
    for d in trading_dates:
        if d < current_date:
            high = high_by_date.get(d)
            if high is not None and np.isfinite(high) and high > 0:
                highs.append(float(high))
        elif d == current_date and current_kind == "close":
            high = high_by_date.get(d)
            if high is not None and np.isfinite(high) and high > 0:
                highs.append(float(high))
            break
        elif d >= current_date:
            break
    if not highs:
        return 0.0
    return ((max(highs) / float(first_snapshot_price)) - 1.0) * 100.0


def candidate_label_dates(
    trading_dates: list[date],
    current_date: date,
    current_kind: str,
    cycle_end_date: date,
    lookahead_days: int = LOOKAHEAD_DAYS,
) -> list[date]:
    pos = bisect.bisect_left(trading_dates, current_date)
    if pos >= len(trading_dates) or trading_dates[pos] != current_date:
        return []
    start_pos = pos if current_kind == "open" else pos + 1
    out: list[date] = []
    while start_pos < len(trading_dates) and len(out) < lookahead_days:
        d = trading_dates[start_pos]
        if d > cycle_end_date:
            break
        out.append(d)
        start_pos += 1
    return out


def next_interval_hit(
    trading_dates: list[date],
    high_by_date: dict[date, float],
    current_date: date,
    current_kind: str,
    snapshot_price: float,
) -> bool:
    if not np.isfinite(snapshot_price) or snapshot_price <= 0:
        return False
    threshold = float(snapshot_price) * (1.0 + THRESHOLD_PCT / 100.0)
    pos = bisect.bisect_left(trading_dates, current_date)
    if pos >= len(trading_dates) or trading_dates[pos] != current_date:
        return False
    if current_kind == "open":
        day_high = high_by_date.get(current_date)
        return bool(day_high is not None and np.isfinite(day_high) and float(day_high) >= threshold)
    next_pos = pos + 1
    if next_pos >= len(trading_dates):
        return False
    next_high = high_by_date.get(trading_dates[next_pos])
    return bool(next_high is not None and np.isfinite(next_high) and float(next_high) >= threshold)


def build_snapshot_rows_for_event(
    row: pd.Series,
    trading_dates: list[date],
    bar_by_date: dict[date, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not trading_dates:
        return []
    first = first_snapshot_session(pd.to_datetime(row["transaction_date"], errors="coerce"), trading_dates)
    if first is None:
        return []
    sessions = build_snapshot_sessions(first[0], first[1], trading_dates, max_checks=MAX_CHECKS)
    if not sessions:
        return []
    cycle_end_date = sessions[-1][0]
    close_by_date = {d: float(bar["c"]) for d, bar in bar_by_date.items() if _safe_bar_price(bar, "c") is not None}
    high_by_date = {d: float(bar["h"]) for d, bar in bar_by_date.items() if _safe_bar_price(bar, "h") is not None}

    rows: list[dict[str, Any]] = []
    first_snapshot_price: float | None = None
    for idx, (snap_date, snap_kind) in enumerate(sessions):
        bar = bar_by_date.get(snap_date)
        if bar is None:
            break
        price_key = "o" if snap_kind == "open" else "c"
        snapshot_price = _safe_bar_price(bar, price_key)
        if snapshot_price is None:
            break
        if first_snapshot_price is None:
            first_snapshot_price = snapshot_price

        label_dates = candidate_label_dates(trading_dates, snap_date, snap_kind, cycle_end_date, LOOKAHEAD_DAYS)
        if not label_dates:
            break
        threshold = float(snapshot_price) * (1.0 + THRESHOLD_PCT / 100.0)
        future_highs = [high_by_date.get(d, np.nan) for d in label_dates]
        hit_label = int(any(np.isfinite(h) and float(h) >= threshold for h in future_highs))

        snapshot_ts = datetime.combine(
            snap_date,
            time(hour=9, minute=30) if snap_kind == "open" else time(hour=16, minute=0),
            tzinfo=train_models.ET,
        )
        first_ref = float(first_snapshot_price)
        row_out = row.to_dict()
        row_out.update(
            {
                "snapshot_date": snap_date,
                "snapshot_timestamp": snapshot_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot_kind": snap_kind,
                "snapshot_is_open": int(snap_kind == "open"),
                "snapshot_price": float(snapshot_price),
                "buy_price": float(snapshot_price),
                "buy_datetime": snapshot_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot_day_idx": int(idx),
                "days_remaining_in_10d_cycle": int(MAX_CHECKS - idx - 1),
                "return_since_first_snapshot_pct": ((float(snapshot_price) / first_ref) - 1.0) * 100.0,
                "max_intraday_runup_so_far_pct": compute_max_runup_so_far_pct(
                    first_ref, trading_dates, high_by_date, snap_date, snap_kind
                ),
                "prior_30d_pct": compute_prior_momentum_pct(float(snapshot_price), trading_dates, close_by_date, snap_date, 30),
                "prior_10d_pct": compute_prior_momentum_pct(float(snapshot_price), trading_dates, close_by_date, snap_date, 10),
                "prior_5d_pct": compute_prior_momentum_pct(float(snapshot_price), trading_dates, close_by_date, snap_date, 5),
                "prior_30d_vol": compute_prior_vol(trading_dates, close_by_date, snap_date, 30),
                "prior_10d_vol": compute_prior_vol(trading_dates, close_by_date, snap_date, 10),
                "prior_5d_vol": compute_prior_vol(trading_dates, close_by_date, snap_date, 5),
                "price_drift_filing_pct": np.nan,
                "label_end_date": label_dates[-1],
                "lookahead_days_used": int(len(label_dates)),
                TARGET_COL: int(hit_label),
            }
        )
        rows.append(row_out)
        if next_interval_hit(trading_dates, high_by_date, snap_date, snap_kind, float(snapshot_price)):
            break
    return rows


def load_event_frame(aggregate_csv: Path, raw_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading live aggregate: %s", aggregate_csv)
    df = pd.read_csv(aggregate_csv)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["buy_datetime"] = normalize_buy_datetime_series(df["buy_datetime"])
    df["trade_date_d"] = df["trade_date"].dt.date
    df["event_key"] = make_event_key_series(df)

    logger.info("Merging owned_pct from raw CSV...")
    orig = pd.read_csv(raw_csv, usecols=["transaction_date", "ticker", "owner_name", "owned_pct"])
    orig["transaction_date"] = pd.to_datetime(orig["transaction_date"], errors="coerce").astype(str)
    orig["owned_pct_num"] = orig["owned_pct"].apply(train_models.clean_pct)
    orig = orig.drop_duplicates(["transaction_date", "ticker", "owner_name"])
    df["_merge_txn"] = df["transaction_date"].astype(str)
    df = df.merge(
        orig[["transaction_date", "ticker", "owner_name", "owned_pct_num"]].rename(
            columns={"transaction_date": "_merge_txn"}
        ),
        on=["_merge_txn", "ticker", "owner_name"],
        how="left",
    )
    df = df.drop(columns=["_merge_txn"], errors="ignore")

    df["value_usd"] = df["value"].apply(train_models.clean_money)
    df["last_price_clean"] = df["last_price"].apply(train_models.clean_money)

    logger.info("Computing static repeat-buyer features...")
    hist = pd.read_csv(raw_csv, usecols=["owner_name", "ticker", "trade_date"])
    hist["trade_date_d"] = pd.to_datetime(hist["trade_date"], errors="coerce").dt.date

    owner_dates: dict[str, list[date]] = {}
    owner_ticker_dates: dict[tuple[str, str], list[date]] = {}
    for rec in hist[["owner_name", "ticker", "trade_date_d"]].itertuples(index=False):
        if pd.isna(rec.trade_date_d):
            continue
        owner_dates.setdefault(rec.owner_name, []).append(rec.trade_date_d)
        owner_ticker_dates.setdefault((rec.owner_name, rec.ticker), []).append(rec.trade_date_d)
    for values in owner_dates.values():
        values.sort()
    for values in owner_ticker_dates.values():
        values.sort()

    def _days_since_prev(sorted_dates: list[date], cur_date: date) -> float:
        idx = bisect.bisect_left(sorted_dates, cur_date)
        if idx <= 0:
            return np.nan
        return float((cur_date - sorted_dates[idx - 1]).days)

    def _count_recent(sorted_dates: list[date], cur_date: date, window_days: int) -> float:
        left = bisect.bisect_left(sorted_dates, cur_date - timedelta(days=window_days))
        right = bisect.bisect_left(sorted_dates, cur_date)
        return float(max(0, right - left))

    prior_any = []
    prior_same_ticker = []
    days_last_any = []
    days_last_same_ticker = []
    count_365d = []
    for rec in df[["owner_name", "ticker", "trade_date_d"]].itertuples(index=False):
        owner_hist = owner_dates.get(rec.owner_name, [])
        owner_ticker_hist = owner_ticker_dates.get((rec.owner_name, rec.ticker), [])
        cur_date = rec.trade_date_d
        if pd.isna(cur_date):
            prior_any.append(np.nan)
            prior_same_ticker.append(np.nan)
            days_last_any.append(np.nan)
            days_last_same_ticker.append(np.nan)
            count_365d.append(np.nan)
            continue
        any_count = bisect.bisect_left(owner_hist, cur_date)
        same_count = bisect.bisect_left(owner_ticker_hist, cur_date)
        prior_any.append(float(any_count))
        prior_same_ticker.append(float(1 if same_count > 0 else 0))
        days_last_any.append(_days_since_prev(owner_hist, cur_date))
        days_last_same_ticker.append(_days_since_prev(owner_ticker_hist, cur_date))
        count_365d.append(_count_recent(owner_ticker_hist, cur_date, 365))
    df["insider_prior_buys"] = prior_any
    df["insider_bought_ticker"] = prior_same_ticker
    df["days_since_last_buy_any"] = days_last_any
    df["days_since_last_buy_same_ticker"] = days_last_same_ticker
    df["same_ticker_insider_count_365d"] = count_365d

    logger.info("Loading ticker metadata...")
    meta_map = train_models._load_ticker_metadata(df["ticker"].dropna().astype(str).unique().tolist())
    df["sector"] = df["ticker"].map(lambda t: meta_map.get(t, {}).get("sector", "Unknown")).fillna("Unknown")
    df["market_type"] = df["ticker"].map(
        lambda t: meta_map.get(t, {}).get("market_type", "NON_TRADABLE")
    ).fillna("NON_TRADABLE")
    df["is_tradable"] = df["ticker"].map(
        lambda t: int(meta_map.get(t, {}).get("is_tradable", 0))
    ).fillna(0).astype(int)
    return df


def build_snapshot_dataset(
    event_df: pd.DataFrame,
    cache_dir: Path,
    logger: logging.Logger,
    *,
    cache_only_day: bool = False,
    limit_events: int = 0,
) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    client = RESTClient(api_key=api_key, retries=3) if api_key else None
    cache = aggregate_builder.PriceCache(cache_dir)

    if limit_events > 0:
        ordered = event_df.sort_values(["transaction_date", "event_key"]).reset_index(drop=True)
        if len(ordered) > limit_events:
            sample_idx = np.linspace(0, len(ordered) - 1, num=limit_events, dtype=int)
            event_df = ordered.iloc[np.unique(sample_idx)].copy()
        else:
            event_df = ordered.copy()
        logger.info("Limiting run to %d chronologically-spread events for smoke/testing.", len(event_df))

    rows: list[dict[str, Any]] = []
    by_ticker = event_df.groupby("ticker", sort=True)
    total_tickers = int(by_ticker.ngroups)
    for idx, (ticker, grp) in enumerate(by_ticker, start=1):
        txn_dates = pd.to_datetime(grp["transaction_date"], errors="coerce")
        buy_dates = pd.to_datetime(grp["buy_datetime"], errors="coerce").dt.date
        valid_txn = [d.date() for d in txn_dates.dropna().tolist()]
        valid_buy = [d for d in buy_dates.tolist() if isinstance(d, date)]
        if not valid_txn and not valid_buy:
            continue
        min_d = min(valid_txn + valid_buy) - timedelta(days=60)
        max_d = max(valid_txn + valid_buy) + timedelta(days=40)
        if client is None:
            bars = cache.get_day_exact(str(ticker), min_d, max_d) or cache.get_day_covering(str(ticker), min_d, max_d) or []
        else:
            bars = aggregate_builder.get_daily_bars(client, cache, str(ticker), min_d, max_d, cache_only_day)
        dated = []
        for bar in bars:
            bar_date = train_models._bar_date_et(bar)
            if bar_date is None:
                continue
            dated.append((bar_date, bar))
        if not dated:
            continue
        dated = sorted(dated, key=lambda x: x[0])
        trading_dates = [d for d, _bar in dated]
        bar_by_date = {d: bar for d, bar in dated}
        for _, row in grp.iterrows():
            event_rows = build_snapshot_rows_for_event(row, trading_dates, bar_by_date)
            if not event_rows:
                continue
            for event_row in event_rows:
                lp = event_row.get("last_price_clean")
                if pd.notna(lp) and float(lp) > 0 and pd.notna(event_row.get("snapshot_price")):
                    event_row["price_drift_filing_pct"] = (
                        (float(event_row["snapshot_price"]) / float(lp)) - 1.0
                    ) * 100.0
                rows.append(event_row)
        if idx % 250 == 0 or idx == total_tickers:
            logger.info("Snapshot dataset progress: %d / %d tickers", idx, total_tickers)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.date
    out["label_end_date"] = pd.to_datetime(out["label_end_date"], errors="coerce").dt.date
    out["event_last_label_date"] = out.groupby("event_key")["label_end_date"].transform("max")
    out = out.sort_values(["snapshot_date", "event_key", "snapshot_day_idx"]).reset_index(drop=True)
    logger.info(
        "Built sequential snapshot dataset: %d rows across %d events",
        len(out),
        int(out["event_key"].nunique()),
    )
    return out


def engineer_model_frame(snapshot_df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, list[str], list[str]]:
    logger.info("Applying train_models.engineer_features to sequential frame...")
    pre = snapshot_df.copy()
    pre["transaction_date"] = pd.to_datetime(pre["transaction_date"], errors="coerce")
    engineered, _base_features, _caps = train_models.engineer_features(pre)
    full_features = list(train_models.FEATURES) + list(SEQUENTIAL_STATE_FEATURES)
    missing = [c for c in full_features if c not in engineered.columns]
    if missing:
        raise ValueError(f"Missing engineered columns for model frame: {missing}")
    engineered[TARGET_COL] = pd.to_numeric(engineered[TARGET_COL], errors="coerce").astype(int)
    engineered["snapshot_date"] = pd.to_datetime(engineered["snapshot_date"], errors="coerce").dt.date
    engineered["label_end_date"] = pd.to_datetime(engineered["label_end_date"], errors="coerce").dt.date
    engineered["event_last_label_date"] = pd.to_datetime(
        engineered["event_last_label_date"], errors="coerce"
    ).dt.date
    engineered = engineered.dropna(subset=full_features + [TARGET_COL, "event_key", "snapshot_date", "label_end_date"]).copy()
    engineered = engineered.sort_values(["snapshot_date", "event_key", "snapshot_day_idx"]).reset_index(drop=True)
    return engineered, full_features, list(METADATA_BASELINE_FEATURES)


def chronological_event_split(df: pd.DataFrame) -> SplitFrames:
    event_meta = (
        df.groupby("event_key", as_index=False)
        .agg(
            first_snapshot_date=("snapshot_date", "min"),
            last_snapshot_date=("snapshot_date", "max"),
            last_label_date=("label_end_date", "max"),
            n_rows=("event_key", "size"),
        )
        .sort_values(["first_snapshot_date", "event_key"])
        .reset_index(drop=True)
    )
    n_events = len(event_meta)
    if n_events < 6:
        raise ValueError("Need at least 6 events to form train/val/test splits.")

    def _partition_by_start_date(
        events: pd.DataFrame,
        target_left_count: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
        if len(events) < 2:
            raise ValueError("Need at least two events to form a split.")
        target_left_count = max(1, min(int(target_left_count), len(events) - 1))
        candidate_order = list(range(target_left_count, len(events)))
        candidate_order.extend(range(target_left_count - 1, 0, -1))
        best: tuple[int, int, pd.DataFrame, pd.DataFrame, str, str] | None = None
        for idx in candidate_order:
            right_start = pd.Timestamp(events.iloc[idx]["first_snapshot_date"])
            cutoff = (right_start - pd.offsets.BDay(EMBARGO_TRADING_DAYS)).date()
            left_mask = events["last_label_date"].to_numpy() < cutoff
            right_mask = events["first_snapshot_date"].to_numpy() >= right_start.date()
            left_count = int(left_mask.sum())
            right_count = int(right_mask.sum())
            if left_count == 0 or right_count == 0:
                continue
            left_events = events.loc[left_mask].copy().reset_index(drop=True)
            right_events = events.loc[right_mask].copy().reset_index(drop=True)
            distance = abs(left_count - target_left_count)
            purged = int(len(events) - left_count - right_count)
            candidate = (distance, purged, left_events, right_events, str(right_start.date()), str(cutoff))
            if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
                best = candidate
        if best is None:
            raise ValueError("Could not place an embargoed split boundary with the available event dates.")
        return best[2], best[3], best[4], best[5]

    train_target = int(round(n_events * TRAIN_FRAC))
    val_target = int(round(n_events * VAL_FRAC))
    train_events, remaining, val_start_str, train_cutoff = _partition_by_start_date(event_meta, train_target)
    if len(remaining) < 2:
        raise ValueError("Not enough events remain after choosing the train/validation boundary.")
    val_events, test_events, test_start_str, val_cutoff = _partition_by_start_date(remaining, val_target)
    if train_events.empty or val_events.empty or test_events.empty:
        raise ValueError("A split became empty after applying the embargo.")

    split_lookup: dict[str, str] = {}
    split_lookup.update({k: "train" for k in train_events["event_key"].tolist()})
    split_lookup.update({k: "val" for k in val_events["event_key"].tolist()})
    split_lookup.update({k: "test" for k in test_events["event_key"].tolist()})

    out = df.copy()
    out["split"] = out["event_key"].map(split_lookup)
    out = out[out["split"].notna()].copy()

    train = out[out["split"] == "train"].copy()
    val = out[out["split"] == "val"].copy()
    test = out[out["split"] == "test"].copy()

    train_keys = set(train["event_key"].tolist())
    val_keys = set(val["event_key"].tolist())
    test_keys = set(test["event_key"].tolist())
    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise AssertionError("Event overlap detected across train/val/test.")

    val_start_date = pd.to_datetime(val["snapshot_date"], errors="coerce").dt.date.min()
    test_start_date = pd.to_datetime(test["snapshot_date"], errors="coerce").dt.date.min()
    if pd.isna(val_start_date) or pd.isna(test_start_date):
        raise AssertionError("Could not determine validation/test start dates.")
    if bool((pd.to_datetime(train["snapshot_date"], errors="coerce").dt.date >= val_start_date).any()):
        raise AssertionError("Train snapshot_date leaks into validation window.")
    if bool((pd.to_datetime(train["label_end_date"], errors="coerce").dt.date >= val_start_date).any()):
        raise AssertionError("Train label_end_date leaks into validation window.")
    if bool((pd.to_datetime(val["snapshot_date"], errors="coerce").dt.date >= test_start_date).any()):
        raise AssertionError("Validation snapshot_date leaks into test window.")
    if bool((pd.to_datetime(val["label_end_date"], errors="coerce").dt.date >= test_start_date).any()):
        raise AssertionError("Validation label_end_date leaks into test window.")

    boundaries = {
        "validation_start_snapshot_date": val_start_str,
        "test_start_snapshot_date": test_start_str,
        "train_embargo_cutoff": train_cutoff,
        "validation_embargo_cutoff": val_cutoff,
    }
    return SplitFrames(train=train, val=val, test=test, event_meta=event_meta, boundaries=boundaries)


def linear_cv_config(n_rows: int) -> dict[str, Any]:
    n_rows = int(max(n_rows, 0))
    for cfg in LINEAR_CV_CONFIG:
        max_rows = cfg["max_rows"]
        if max_rows is None or n_rows <= max_rows:
            return cfg
    return LINEAR_CV_CONFIG[-1]


def to_xgb(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in ("officer_type_enc", "market_type_enc", "sector_enc"):
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def to_linear_numeric(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            codes = out[col].cat.codes.astype(float)
            codes[codes < 0] = np.nan
            out[col] = codes
    return out.astype(float)


def train_xgb_classifier(x_train: pd.DataFrame, y_train: np.ndarray) -> xgb.XGBClassifier:
    if len(x_train) < 60:
        params = {k: v for k, v in XGB_CLF_PARAMS.items() if k != "early_stopping_rounds"}
        params["n_estimators"] = 100
        model = xgb.XGBClassifier(**params)
        model.fit(to_xgb(x_train), y_train, verbose=False)
        return model
    cut = int(round(len(x_train) * 0.9))
    cut = max(1, min(cut, len(x_train) - 1))
    x_fit = to_xgb(x_train.iloc[:cut])
    y_fit = y_train[:cut]
    x_val = to_xgb(x_train.iloc[cut:])
    y_val = y_train[cut:]
    model = xgb.XGBClassifier(**XGB_CLF_PARAMS)
    model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
    return model


def train_elasticnet_classifier(x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    cfg = linear_cv_config(len(x_train))
    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=np.asarray(cfg["cs"], dtype=float),
                    cv=int(cfg["cv"]),
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratios=list(cfg["l1_ratios"]),
                    scoring="neg_log_loss",
                    max_iter=12000,
                    tol=1e-3,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(to_linear_numeric(x_train), y_train)
    return model


def train_spline_elasticnet_classifier(x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    cfg = linear_cv_config(len(x_train))
    x_in = x_train.copy()
    cat_cols = [c for c in ("officer_type_enc", "market_type_enc", "sector_enc") if c in x_in.columns]
    cont_cols = [c for c in x_in.columns if c not in cat_cols]
    for col in cont_cols:
        if isinstance(x_in[col].dtype, pd.CategoricalDtype):
            codes = x_in[col].cat.codes.astype(float)
            codes[codes < 0] = np.nan
            x_in[col] = codes
    prep = ColumnTransformer(
        transformers=[
            (
                "cont_spline",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=int(cfg["spline_knots"]),
                                degree=3,
                                include_bias=False,
                            ),
                        ),
                    ]
                ),
                cont_cols,
            ),
            (
                "cat_onehot",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("prep", prep),
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=np.asarray(cfg["cs"], dtype=float),
                    cv=int(cfg["cv"]),
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratios=list(cfg["l1_ratios"]),
                    scoring="neg_log_loss",
                    max_iter=15000,
                    tol=1e-3,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_in, y_train)
    return model


def train_main_models(x_train: pd.DataFrame, y_train: np.ndarray, logger: logging.Logger) -> dict[str, object]:
    models: dict[str, object] = {}
    hg_params = dict(train_models.MODEL_PARAMS)
    hg_params.update({"loss": "log_loss"})
    started = perf_counter()
    hg_model = HistGradientBoostingClassifier(
        learning_rate=hg_params["learning_rate"],
        max_iter=hg_params["max_iter"],
        max_leaf_nodes=hg_params["max_leaf_nodes"],
        min_samples_leaf=hg_params["min_samples_leaf"],
        l2_regularization=hg_params["l2_regularization"],
        max_bins=hg_params["max_bins"],
        early_stopping=hg_params["early_stopping"],
        validation_fraction=hg_params["validation_fraction"],
        n_iter_no_change=hg_params["n_iter_no_change"],
        random_state=hg_params["random_state"],
        categorical_features="from_dtype",
    )
    hg_model.fit(x_train, y_train)
    models["HGB"] = hg_model
    logger.info("  Trained HGB in %.1fs", perf_counter() - started)

    started = perf_counter()
    models["XGBoost"] = train_xgb_classifier(x_train, y_train)
    logger.info("  Trained XGBoost in %.1fs", perf_counter() - started)

    started = perf_counter()
    models["ElasticNet"] = train_elasticnet_classifier(x_train, y_train)
    logger.info("  Trained ElasticNet in %.1fs", perf_counter() - started)

    started = perf_counter()
    models["SplineElasticNet"] = train_spline_elasticnet_classifier(x_train, y_train)
    logger.info("  Trained SplineElasticNet in %.1fs", perf_counter() - started)
    return models


def train_metadata_baseline(x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    cat_cols = ["insider_type"]
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    model = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            num_cols,
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            cat_cols,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("clf", LogisticRegression(max_iter=10000, solver="lbfgs", random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train.copy(), y_train)
    return model


def predict_model_probability(model_name: str, model: object, x: pd.DataFrame) -> np.ndarray:
    if model_name == "HGB":
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if model_name == "XGBoost":
        return np.asarray(model.predict_proba(to_xgb(x))[:, 1], dtype=float)
    if model_name == "ElasticNet":
        return np.asarray(model.predict_proba(to_linear_numeric(x))[:, 1], dtype=float)
    if model_name == "SplineElasticNet":
        return np.asarray(model.predict_proba(x.copy())[:, 1], dtype=float)
    raise ValueError(f"Unknown model name: {model_name}")


def fit_platt_calibrator(raw_prob: np.ndarray, y_true: np.ndarray) -> object:
    x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return ConstantProbabilityCalibrator(float(y.mean()) if len(y) else 0.0)
    model = LogisticRegression(max_iter=10000, solver="lbfgs", random_state=RANDOM_STATE)
    model.fit(x, y)
    return model


def calibrated_probability(calibrator: object, raw_prob: np.ndarray) -> np.ndarray:
    x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
    return np.asarray(calibrator.predict_proba(x)[:, 1], dtype=float)


def metric_row(split: str, model_name: str, y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    base_rate = float(y.mean()) if len(y) else np.nan
    top_n = max(1, len(y) // 10)
    order = np.argsort(p)[::-1]
    top_hit_rate = float(y[order[:top_n]].mean()) if len(order) else np.nan
    row = {
        "split": split,
        "model": model_name,
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": base_rate,
        "pr_auc": np.nan,
        "roc_auc": np.nan,
        "brier": np.nan,
        "top_decile_hit_rate": top_hit_rate,
        "lift_vs_base_rate": float(top_hit_rate / base_rate) if base_rate and np.isfinite(base_rate) else np.nan,
    }
    if len(y) and len(np.unique(y)) >= 2:
        row["pr_auc"] = float(average_precision_score(y, p))
        row["roc_auc"] = float(roc_auc_score(y, p))
    if len(y):
        row["brier"] = float(brier_score_loss(y, p))
    return row


def calibration_table(split: str, model_name: str, y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    order = np.argsort(p)
    n = len(order)
    if n == 0:
        return pd.DataFrame(columns=["split", "model", "bucket", "rows", "mean_prob", "actual_rate"])
    bin_size = max(1, n // n_bins)
    rows: list[dict[str, Any]] = []
    for bucket in range(n_bins):
        start = bucket * bin_size
        end = n if bucket == n_bins - 1 else min(n, (bucket + 1) * bin_size)
        idx = order[start:end]
        if len(idx) == 0:
            continue
        rows.append(
            {
                "split": split,
                "model": model_name,
                "bucket": int(bucket + 1),
                "rows": int(len(idx)),
                "mean_prob": float(np.mean(p[idx])),
                "actual_rate": float(np.mean(y[idx])),
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_rows(
    models: dict[str, object],
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    x_eval_full = x_test
    y_eval_full = np.asarray(y_test, dtype=int)
    if len(x_eval_full) > PERMUTATION_MAX_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        keep_idx = np.sort(rng.choice(len(x_eval_full), size=PERMUTATION_MAX_ROWS, replace=False))
        x_eval_full = x_eval_full.iloc[keep_idx].copy()
        y_eval_full = y_eval_full[keep_idx]
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        model = models[model_name]
        x_eval = x_eval_full if model_name in {"HGB", "SplineElasticNet"} else (
            to_xgb(x_eval_full) if model_name == "XGBoost" else to_linear_numeric(x_eval_full)
        )
        perm = permutation_importance(
            model,
            x_eval,
            y_eval_full,
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="neg_brier_score",
        )
        imp = pd.DataFrame(
            {
                "model": model_name,
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values(["model", "importance_mean", "feature"], ascending=[True, False, True])
        imp["rank"] = imp.groupby("model").cumcount() + 1
        rows.extend(imp.to_dict("records"))
    return pd.DataFrame(rows)


def run_research(
    frame: SplitFrames,
    full_feature_cols: list[str],
    metadata_feature_cols: list[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = frame.train, frame.val, frame.test
    x_train = train_df[full_feature_cols].copy()
    y_train = train_df[TARGET_COL].to_numpy(dtype=int)
    x_val = val_df[full_feature_cols].copy()
    y_val = val_df[TARGET_COL].to_numpy(dtype=int)
    x_test = test_df[full_feature_cols].copy()
    y_test = test_df[TARGET_COL].to_numpy(dtype=int)

    logger.info("Training main classifier stack...")
    models = train_main_models(x_train, y_train, logger)
    val_probs = {name: predict_model_probability(name, mdl, x_val) for name, mdl in models.items()}
    test_probs = {name: predict_model_probability(name, mdl, x_test) for name, mdl in models.items()}
    val_mean = np.mean(np.column_stack([val_probs[name] for name in MODEL_NAMES]), axis=1)
    test_mean = np.mean(np.column_stack([test_probs[name] for name in MODEL_NAMES]), axis=1)

    logger.info("Fitting Platt calibrator on validation ensemble...")
    calibrator = fit_platt_calibrator(val_mean, y_val)
    val_cal = calibrated_probability(calibrator, val_mean)
    test_cal = calibrated_probability(calibrator, test_mean)

    logger.info("Training metadata baseline...")
    metadata_model = train_metadata_baseline(train_df[metadata_feature_cols].copy(), y_train)
    meta_val = np.asarray(
        metadata_model.predict_proba(val_df[metadata_feature_cols].copy())[:, 1],
        dtype=float,
    )
    meta_test = np.asarray(
        metadata_model.predict_proba(test_df[metadata_feature_cols].copy())[:, 1],
        dtype=float,
    )
    global_base_rate = float(y_train.mean()) if len(y_train) else 0.0
    global_val = np.full(len(y_val), global_base_rate, dtype=float)
    global_test = np.full(len(y_test), global_base_rate, dtype=float)

    metrics_rows: list[dict[str, Any]] = []
    for split_name, y_split, prob_map in (
        (
            "validation",
            y_val,
            {
                **val_probs,
                "ensemble_raw": val_mean,
                "ensemble_calibrated": val_cal,
                "metadata_baseline": meta_val,
                "global_base_rate": global_val,
            },
        ),
        (
            "test",
            y_test,
            {
                **test_probs,
                "ensemble_raw": test_mean,
                "ensemble_calibrated": test_cal,
                "metadata_baseline": meta_test,
                "global_base_rate": global_test,
            },
        ),
    ):
        for model_name, probs in prob_map.items():
            metrics_rows.append(metric_row(split_name, model_name, y_split, probs))
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["split", "model"]).reset_index(drop=True)

    calibration_df = pd.concat(
        [
            calibration_table("validation", "ensemble_calibrated", y_val, val_cal),
            calibration_table("test", "ensemble_calibrated", y_test, test_cal),
            calibration_table("validation", "metadata_baseline", y_val, meta_val),
            calibration_table("test", "metadata_baseline", y_test, meta_test),
        ],
        ignore_index=True,
    )

    test_predictions = test_df[
        [
            "event_key",
            "ticker",
            "company_name",
            "owner_name",
            "title",
            "trade_date",
            "transaction_date",
            "snapshot_timestamp",
            "snapshot_date",
            "snapshot_kind",
            "snapshot_day_idx",
            "snapshot_price",
            "days_remaining_in_10d_cycle",
            "return_since_first_snapshot_pct",
            "max_intraday_runup_so_far_pct",
            "lookahead_days_used",
            "label_end_date",
            TARGET_COL,
        ]
    ].copy()
    for model_name in MODEL_NAMES:
        test_predictions[f"prob_{model_name}"] = test_probs[model_name]
    test_predictions["prob_ensemble_raw"] = test_mean
    test_predictions["prob_ensemble_calibrated"] = test_cal
    test_predictions["prob_metadata_baseline"] = meta_test
    test_predictions["prob_global_base_rate"] = global_test

    logger.info("Computing permutation importance summary...")
    importance_df = permutation_importance_rows(models, x_test, y_test, full_feature_cols)
    return metrics_df, calibration_df, test_predictions, importance_df


def dataset_summary(frame: pd.DataFrame, split_frame: SplitFrames) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "section": "overall",
            "rows": int(len(frame)),
            "events": int(frame["event_key"].nunique()),
            "positive_rate": float(pd.to_numeric(frame[TARGET_COL], errors="coerce").mean()),
            "mean_rows_per_event": float(frame.groupby("event_key").size().mean()),
        }
    )
    for split_name, split_df in (("train", split_frame.train), ("val", split_frame.val), ("test", split_frame.test)):
        rows.append(
            {
                "section": split_name,
                "rows": int(len(split_df)),
                "events": int(split_df["event_key"].nunique()),
                "positive_rate": float(pd.to_numeric(split_df[TARGET_COL], errors="coerce").mean()),
                "mean_rows_per_event": float(split_df.groupby("event_key").size().mean()) if len(split_df) else np.nan,
            }
        )
    for source, sub in frame.groupby("snapshot_kind", dropna=False):
        rows.append(
            {
                "section": f"snapshot_kind={source}",
                "rows": int(len(sub)),
                "events": int(sub["event_key"].nunique()),
                "positive_rate": float(pd.to_numeric(sub[TARGET_COL], errors="coerce").mean()),
                "mean_rows_per_event": float(sub.groupby("event_key").size().mean()),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    metrics_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    importance_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    split_frame: SplitFrames,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(SPLIT_METRICS_CSV, index=False)
    calibration_df.to_csv(CALIBRATION_TABLE_CSV, index=False)
    test_predictions.to_csv(TEST_PREDICTIONS_CSV, index=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    dataset_summary_df.to_csv(DATASET_SUMMARY_CSV, index=False)
    test_predictions.head(250).to_csv(DATASET_SAMPLE_CSV, index=False)

    payload = {
        "target": TARGET_COL,
        "threshold_pct": THRESHOLD_PCT,
        "lookahead_days": LOOKAHEAD_DAYS,
        "max_checks": MAX_CHECKS,
        "embargo_trading_days": EMBARGO_TRADING_DAYS,
        "split_boundaries": split_frame.boundaries,
        "dataset_summary": dataset_summary_df.to_dict("records"),
        "top_metrics": metrics_df.to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    ensure_live_aggregate(args, logger)
    event_df = load_event_frame(args.aggregate_csv, args.raw_csv, logger)
    snapshot_df = build_snapshot_dataset(
        event_df,
        args.cache_dir,
        logger,
        cache_only_day=bool(args.cache_only_day),
        limit_events=int(args.limit_events),
    )
    if snapshot_df.empty:
        raise SystemExit("Sequential snapshot dataset is empty.")

    model_df, full_feature_cols, metadata_feature_cols = engineer_model_frame(snapshot_df, logger)
    split_frame = chronological_event_split(model_df)
    dataset_summary_df = dataset_summary(model_df, split_frame)

    metrics_df, calibration_df, test_predictions, importance_df = run_research(
        split_frame,
        full_feature_cols,
        metadata_feature_cols,
        logger,
    )
    write_outputs(metrics_df, calibration_df, test_predictions, importance_df, dataset_summary_df, split_frame)

    logger.info("Saved summary: %s", SUMMARY_JSON)
    logger.info("Saved split metrics: %s", SPLIT_METRICS_CSV)
    logger.info("Saved test predictions: %s", TEST_PREDICTIONS_CSV)
    logger.info("Saved calibration table: %s", CALIBRATION_TABLE_CSV)
    logger.info("Saved feature importance: %s", FEATURE_IMPORTANCE_CSV)
    logger.info("Saved dataset summary: %s", DATASET_SUMMARY_CSV)


if __name__ == "__main__":
    main()
