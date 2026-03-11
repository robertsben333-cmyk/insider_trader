from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
os.chdir(BASE)

import train_models
from openinsider_scraper import OpenInsiderScraper
from research.scripts import evaluate_live_gap_feature as gap_eval
from research.scripts import build_unfiltered_aggregated_backtest as aggregate_builder
from research.scripts.deploy_tplus2_open_day1_live import fit_models


DEFAULT_START_DATE = "2026-03-09"
DEFAULT_END_DATE = "2026-03-11"

BACKTEST_RAW_CSV = BASE / "backtest" / "data" / "insider_purchases.csv"
LIVE_AGG_CSV = BASE / "backtest" / "data" / "backtest_results_aggregated_live_entry.csv"
BACKTEST_CACHE_DIR = BASE / "backtest" / "data" / "price_cache"

VM_PREDICTIONS_CSV = BASE / "research" / "ibkr_vm_analysis" / "vm_live_predictions.csv"
VM_SIGNAL_ARCHIVE_CSV = BASE / "research" / "ibkr_vm_analysis" / "ibkr_paper_signal_archive.csv"
VM_JOURNAL_JSONL = BASE / "research" / "ibkr_vm_analysis" / "ibkr_paper_trader_journal.jsonl"
LIVE_CACHE_DIR = BASE / "live" / "data" / "price_cache"

OUTPUT_DIR = BASE / "research" / "ibkr_vm_analysis"
OUT_DETAIL_CSV = OUTPUT_DIR / "vm_gap_feature_replay_candidate_scores.csv"
OUT_DAILY_CSV = OUTPUT_DIR / "vm_gap_feature_replay_daily_summary.csv"
OUT_ACTUAL_CSV = OUTPUT_DIR / "vm_gap_feature_replay_actual_trades.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay March VM decisions with baseline vs live-gap-feature day-1 ensembles."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    return parser.parse_args()


def _normalize_date_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _candidate_id(df: pd.DataFrame) -> pd.Series:
    scored_at = pd.to_datetime(df["scored_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return df["event_key"].astype(str) + "|" + scored_at.fillna("")


def _clip_with_reference(series: pd.Series, reference: pd.Series, lo_q: float, hi_q: float) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce")
    valid = ref.dropna()
    if valid.empty:
        return pd.to_numeric(series, errors="coerce")
    lo, hi = valid.quantile([lo_q, hi_q])
    return pd.to_numeric(series, errors="coerce").clip(lo, hi)


def _make_n_insiders_label(n_series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(n_series, errors="coerce").fillna(1).astype(int).clip(lower=1, upper=5)
    mapping = {1: "1 (solo)", 2: "2", 3: "3", 4: "4", 5: "5+"}
    return vals.map(mapping)


def load_vm_candidates(start_date: str, end_date: str) -> pd.DataFrame:
    journal_rows: list[dict[str, Any]] = []
    with VM_JOURNAL_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                journal_rows.append(json.loads(line))
    if not journal_rows:
        return pd.DataFrame()

    journal = pd.json_normalize(journal_rows)
    journal = journal[journal["event_type"] == "candidate_ingested"].copy()
    journal["entry_trade_day"] = journal["payload.entry_trade_day"].astype(str)
    journal = journal[journal["entry_trade_day"].between(start_date, end_date)].copy()
    journal["candidate_id"] = journal["payload.candidate_id"].astype(str)
    journal["current_prod4_score"] = pd.to_numeric(journal["payload.signal_score"], errors="coerce")
    journal["recorded_at"] = pd.to_datetime(journal["recorded_at"], errors="coerce", utc=True)
    journal["journal_entry_bucket"] = journal["payload.entry_bucket"].astype(str)
    journal["intended_entry_at"] = pd.to_datetime(journal["payload.intended_entry_at"], errors="coerce", utc=True)
    journal["journal_buy_price_hint"] = pd.to_numeric(journal["payload.buy_price_hint"], errors="coerce")
    journal["journal_estimated_decile_score"] = pd.to_numeric(
        journal["payload.estimated_decile_score"], errors="coerce"
    )
    journal = journal.sort_values(["candidate_id", "recorded_at"]).drop_duplicates(
        subset=["candidate_id"], keep="last"
    )

    preds = pd.read_csv(VM_PREDICTIONS_CSV)
    preds = preds[pd.to_numeric(preds["horizon_days"], errors="coerce") == 1].copy()
    preds["candidate_id"] = _candidate_id(preds)
    preds["vm_pred_mean4"] = pd.to_numeric(preds["pred_mean4"], errors="coerce")
    preds["is_tradable"] = pd.to_numeric(preds["is_tradable"], errors="coerce").fillna(0).astype(int)
    preds["filing_gap_days"] = pd.to_numeric(preds["filing_gap_days"], errors="coerce")
    preds["n_insiders_in_cluster"] = pd.to_numeric(preds["n_insiders_in_cluster"], errors="coerce")
    preds["days_since_last_buy_any"] = pd.to_numeric(preds["days_since_last_buy_any"], errors="coerce")
    preds["days_since_last_buy_same_ticker"] = pd.to_numeric(preds["days_since_last_buy_same_ticker"], errors="coerce")
    preds["same_ticker_insider_count_365d"] = pd.to_numeric(preds["same_ticker_insider_count_365d"], errors="coerce")
    preds["filing_hour_et"] = pd.to_numeric(preds["filing_hour_et"], errors="coerce")
    preds["trade_date"] = pd.to_datetime(preds["trade_date"], errors="coerce")
    preds["transaction_date"] = pd.to_datetime(preds["transaction_date"], errors="coerce")
    preds = preds.dropna(subset=["candidate_id", "ticker", "trade_date", "transaction_date"]).copy()
    preds = preds.sort_values(["candidate_id"]).drop_duplicates(subset=["candidate_id"], keep="last")

    out = journal.merge(preds, on="candidate_id", how="left", suffixes=("", "_vm"))
    out["current_prod4_score"] = out["current_prod4_score"].fillna(out["vm_pred_mean4"])
    out["entry_trade_day"] = out["entry_trade_day"].fillna(_normalize_date_str(out["intended_entry_at"]))
    out["buy_datetime"] = out["intended_entry_at"].dt.tz_convert(train_models.ET).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["buy_datetime"] = pd.to_datetime(out["buy_datetime"], errors="coerce")
    out["journal_recorded_day"] = out["recorded_at"].dt.tz_convert(train_models.ET).dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def load_actual_trades(start_date: str, end_date: str) -> pd.DataFrame:
    archive = pd.read_csv(VM_SIGNAL_ARCHIVE_CSV)
    keep_status = {"filled", "partially_filled"}
    out = archive[
        archive["entry_trade_day"].between(start_date, end_date) & archive["status"].isin(list(keep_status))
    ].copy()
    return out.reset_index(drop=True)


def scrape_months(start_date: str, end_date: str) -> pd.DataFrame:
    scraper = OpenInsiderScraper()
    months = pd.period_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="M")
    rows: list[tuple] = []
    for month in months:
        rows.extend(list(scraper._get_data_for_month(int(month.year), int(month.month))))
    raw = pd.DataFrame(
        rows,
        columns=[
            "transaction_date",
            "trade_date",
            "ticker",
            "company_name",
            "owner_name",
            "title",
            "transaction_type",
            "last_price",
            "qty",
            "shares_held",
            "owned_pct",
            "value",
        ],
    )
    raw = raw.drop_duplicates().reset_index(drop=True)
    return raw


def build_combined_raw(local_raw: Path, scraped: pd.DataFrame) -> pd.DataFrame:
    hist = pd.read_csv(local_raw)
    cols = [
        "transaction_date",
        "trade_date",
        "ticker",
        "company_name",
        "owner_name",
        "title",
        "transaction_type",
        "last_price",
        "qty",
        "shares_held",
        "owned_pct",
        "value",
    ]
    hist = hist.reindex(columns=cols)
    scraped = scraped.reindex(columns=cols)
    out = pd.concat([hist, scraped], ignore_index=True)
    out = out.drop_duplicates(subset=cols, keep="last").reset_index(drop=True)
    out["trade_date_d"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    return out


def merge_vm_with_raw(vm_candidates: pd.DataFrame, combined_raw: pd.DataFrame) -> pd.DataFrame:
    raw = combined_raw.copy()
    raw["transaction_date"] = pd.to_datetime(raw["transaction_date"], errors="coerce")
    raw["trade_date_day"] = _normalize_date_str(raw["trade_date"])

    vm = vm_candidates.copy()
    vm["trade_date_day"] = _normalize_date_str(vm["trade_date"])
    merged = vm.merge(
        raw[
            [
                "transaction_date",
                "trade_date_day",
                "ticker",
                "owner_name",
                "title",
                "last_price",
                "qty",
                "owned_pct",
                "value",
            ]
        ],
        on=["transaction_date", "trade_date_day", "ticker", "owner_name", "title"],
        how="left",
    )
    return merged


def reconstruct_vm_entry_prices(frame: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    out["buy_datetime"] = pd.to_datetime(out["buy_datetime"], errors="coerce")
    buy_prices: list[float] = []
    price_sources: list[str] = []

    minute_cache: dict[tuple[str, date], list[dict]] = {}
    daily_cache: dict[tuple[str, date], float] = {}

    def _finite(val: object) -> bool:
        try:
            return bool(np.isfinite(float(val)))
        except (TypeError, ValueError):
            return False

    for row in out.itertuples(index=False):
        ticker = str(row.ticker)
        buy_dt = row.buy_datetime
        bucket = str(getattr(row, "journal_entry_bucket", "") or "")
        fallback_hint = float(row.journal_buy_price_hint) if pd.notna(getattr(row, "journal_buy_price_hint", np.nan)) else np.nan

        if pd.isna(buy_dt):
            buy_prices.append(fallback_hint if _finite(fallback_hint) else np.nan)
            price_sources.append("journal_hint_missing_buy_dt" if _finite(fallback_hint) else "missing_buy_dt")
            continue

        buy_date = buy_dt.date()
        mkey = (ticker, buy_date)
        if mkey not in minute_cache:
            minute_cache[mkey] = train_models.fetch_minute_bars(cache_dir, ticker, buy_date)

        px = train_models.find_price_at_or_after(minute_cache[mkey], int(buy_dt.timestamp() * 1000))
        if _finite(px):
            buy_prices.append(float(px))
            price_sources.append("minute_close")
            continue

        if bucket == "open":
            dkey = (ticker, buy_date)
            if dkey not in daily_cache:
                daily_cache[dkey] = aggregate_builder.find_open_on_date(
                    train_models.fetch_day_bars(cache_dir, ticker, buy_date - timedelta(days=1), buy_date + timedelta(days=1)),
                    buy_date,
                )
            daily_open = daily_cache[dkey]
            if _finite(daily_open):
                buy_prices.append(float(daily_open))
                price_sources.append("daily_open_fallback")
                continue

        if _finite(fallback_hint):
            buy_prices.append(float(fallback_hint))
            price_sources.append("journal_hint_fallback")
            continue

        buy_prices.append(np.nan)
        price_sources.append("missing")

    out["buy_price"] = buy_prices
    out["buy_price_source"] = price_sources
    return out


def add_prior_features(frame: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    out["trade_date_d"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    work = out[["ticker", "trade_date_d", "buy_price"]].drop_duplicates().copy()

    prior_30d_map: dict[tuple[str, date], float] = {}
    prior_10d_map: dict[tuple[str, date], float] = {}
    prior_5d_map: dict[tuple[str, date], float] = {}
    vol_30d_map: dict[tuple[str, date], float] = {}
    vol_10d_map: dict[tuple[str, date], float] = {}
    vol_5d_map: dict[tuple[str, date], float] = {}

    for row in work.itertuples(index=False):
        ticker = str(row.ticker)
        td = row.trade_date_d
        buy_px = float(row.buy_price) if pd.notna(row.buy_price) else np.nan
        key = (ticker, td)
        if td is None or not np.isfinite(buy_px) or buy_px <= 0:
            prior_30d_map[key] = np.nan
            prior_10d_map[key] = np.nan
            prior_5d_map[key] = np.nan
            vol_30d_map[key] = np.nan
            vol_10d_map[key] = np.nan
            vol_5d_map[key] = np.nan
            continue

        from_d = td - timedelta(days=train_models.FETCH_WINDOW)
        bars = train_models.fetch_day_bars(cache_dir, ticker, from_d, td)
        if not bars:
            prior_30d_map[key] = np.nan
            prior_10d_map[key] = np.nan
            prior_5d_map[key] = np.nan
            vol_30d_map[key] = np.nan
            vol_10d_map[key] = np.nan
            vol_5d_map[key] = np.nan
            continue

        bar_dates = [train_models._bar_date_et(bar) for bar in bars]

        def _mom(days: int) -> float:
            target = td - timedelta(days=days)
            close_px = None
            for bd, bar in zip(bar_dates, bars):
                if bd is None:
                    continue
                if bd <= target:
                    close_px = bar.get("c")
                elif bd > target:
                    break
            if close_px is None or float(close_px) <= 0:
                return np.nan
            return ((buy_px - float(close_px)) / float(close_px)) * 100.0

        def _vol(days: int) -> float:
            cutoff = td - timedelta(days=days)
            closes = [
                float(bar["c"])
                for bd, bar in zip(bar_dates, bars)
                if bd is not None and bd >= cutoff and bar.get("c") is not None
            ]
            if len(closes) < 2:
                return np.nan
            arr = np.array(closes, dtype=float)
            log_rets = np.diff(np.log(arr))
            return float(np.std(log_rets, ddof=1) * np.sqrt(252.0))

        prior_30d_map[key] = _mom(train_models.LOOKBACK_DAYS)
        prior_10d_map[key] = _mom(10)
        prior_5d_map[key] = _mom(5)
        vol_30d_map[key] = _vol(30)
        vol_10d_map[key] = _vol(10)
        vol_5d_map[key] = _vol(5)

    def _map_col(mapping: dict[tuple[str, date], float]) -> pd.Series:
        return out.apply(lambda r: mapping.get((str(r["ticker"]), r["trade_date_d"]), np.nan), axis=1)

    out["prior_30d_pct"] = _map_col(prior_30d_map)
    out["prior_10d_pct"] = _map_col(prior_10d_map)
    out["prior_5d_pct"] = _map_col(prior_5d_map)
    out["prior_30d_vol"] = _map_col(vol_30d_map)
    out["prior_10d_vol"] = _map_col(vol_10d_map)
    out["prior_5d_vol"] = _map_col(vol_5d_map)
    return out


def add_repeat_buyer_counts(frame: pd.DataFrame, combined_raw: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hist = combined_raw[["owner_name", "ticker", "trade_date_d"]].dropna().copy()

    owner_dates: dict[str, list[date]] = defaultdict(list)
    owner_ticker_dates: dict[tuple[str, str], list[date]] = defaultdict(list)
    for rec in hist.itertuples(index=False):
        owner_dates[str(rec.owner_name)].append(rec.trade_date_d)
        owner_ticker_dates[(str(rec.owner_name), str(rec.ticker))].append(rec.trade_date_d)

    for vals in owner_dates.values():
        vals.sort()
    for vals in owner_ticker_dates.values():
        vals.sort()

    out["insider_prior_buys"] = out.apply(
        lambda r: float(
            bisect.bisect_left(owner_dates[str(r["owner_name"])], r["trade_date_d"])
        )
        if pd.notna(r["trade_date_d"])
        else np.nan,
        axis=1,
    )
    return out


def add_sector_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    meta_map = train_models._load_ticker_metadata(sorted(set(out["ticker"].dropna().astype(str).tolist())))
    out["sector"] = out["ticker"].map(lambda t: meta_map.get(str(t), {}).get("sector", "Unknown")).fillna("Unknown")
    out["market_type"] = out["market_type"].fillna(
        out["ticker"].map(lambda t: meta_map.get(str(t), {}).get("market_type", "NON_TRADABLE"))
    )
    out["is_tradable"] = out["is_tradable"].fillna(
        out["ticker"].map(lambda t: meta_map.get(str(t), {}).get("is_tradable", 0))
    )
    out["is_tradable"] = pd.to_numeric(out["is_tradable"], errors="coerce").fillna(0).astype(int)
    return out


def build_vm_prefeature_frame(vm_candidates: pd.DataFrame, combined_raw: pd.DataFrame) -> pd.DataFrame:
    merged = merge_vm_with_raw(vm_candidates, combined_raw)
    merged = reconstruct_vm_entry_prices(merged, LIVE_CACHE_DIR)
    merged["value_usd"] = merged["value"].apply(train_models.clean_money)
    merged["last_price_clean"] = merged["last_price"].apply(train_models.clean_money)
    merged["price_drift_filing_pct"] = (
        (merged["buy_price"] - merged["last_price_clean"]) / merged["last_price_clean"] * 100.0
    )
    bad_drift = (merged["last_price_clean"] <= 0) | (merged["price_drift_filing_pct"] > 100) | (
        merged["price_drift_filing_pct"] < -50
    )
    merged.loc[bad_drift, "price_drift_filing_pct"] = np.nan
    merged["trade_date_d"] = pd.to_datetime(merged["trade_date"], errors="coerce").dt.date
    merged["n_insiders"] = pd.to_numeric(merged["n_insiders_in_cluster"], errors="coerce").fillna(1).astype(int)
    merged["cluster_buy"] = merged["n_insiders"] >= 2
    merged["n_insiders_label"] = _make_n_insiders_label(merged["n_insiders"])
    merged = add_prior_features(merged, LIVE_CACHE_DIR)
    merged = add_repeat_buyer_counts(merged, combined_raw)
    merged = add_sector_metadata(merged)
    merged["_source"] = "vm_candidate"
    return merged


def load_training_prefeature_frame() -> pd.DataFrame:
    original_agg = train_models.AGGREGATED_CSV
    original_raw = train_models.ORIGINAL_CSV
    original_cache = train_models.CACHE_DIR
    try:
        train_models.AGGREGATED_CSV = str(LIVE_AGG_CSV)
        train_models.ORIGINAL_CSV = str(BACKTEST_RAW_CSV)
        train_models.CACHE_DIR = str(BACKTEST_CACHE_DIR)
        merged = train_models.load_and_merge()
    finally:
        train_models.AGGREGATED_CSV = original_agg
        train_models.ORIGINAL_CSV = original_raw
        train_models.CACHE_DIR = original_cache
    merged["_source"] = "history"
    return merged


def add_reference_gap_clip(candidate_df: pd.DataFrame, history_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_gap, _ = gap_eval.add_live_gap_feature(history_df, BACKTEST_CACHE_DIR)
    cand_gap, _ = gap_eval.add_live_gap_feature(candidate_df, LIVE_CACHE_DIR)
    hist_gap["live_gap_to_entry_pct"] = _clip_with_reference(
        hist_gap["live_gap_to_entry_pct"], hist_gap["live_gap_to_entry_pct"], 0.005, 0.995
    )
    cand_gap["live_gap_to_entry_pct"] = _clip_with_reference(
        cand_gap["live_gap_to_entry_pct"], hist_gap["live_gap_to_entry_pct"], 0.005, 0.995
    )
    return hist_gap, cand_gap


def fit_replay_models(history_df: pd.DataFrame, base_features: list[str], start_date: str) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    hist_gap, _ = gap_eval.restrict_to_current_event_universe(history_df)
    hist_gap = gap_eval.prepare_model_frame(hist_gap)
    cutoff = pd.Timestamp(start_date)
    fit_df = hist_gap[pd.to_datetime(hist_gap["trade_date"], errors="coerce") < cutoff].copy()
    fit_df = fit_df.dropna(subset=["live_gap_to_entry_pct"]).copy()
    y = fit_df[gap_eval.CUSTOM_TARGET].to_numpy(dtype=float)
    baseline_models = fit_models(fit_df[base_features], y)
    candidate_models = fit_models(fit_df[base_features + ["live_gap_to_entry_pct"]], y)
    return baseline_models, candidate_models, fit_df


def rank_within_day(frame: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = frame.copy()
    out[rank_col] = (
        out.groupby("entry_trade_day")[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out


def build_daily_summary(
    scored: pd.DataFrame,
    actual_trades: pd.DataFrame,
    score_col_base: str,
    score_col_cand: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry_day, day_frame in scored.groupby("entry_trade_day", sort=True):
        actual_day = actual_trades[actual_trades["entry_trade_day"] == entry_day].copy()
        k = int(actual_day["candidate_id"].nunique())
        if k <= 0:
            continue

        actual_set = set(actual_day["candidate_id"].tolist())
        base_top = day_frame.sort_values(score_col_base, ascending=False).head(k)
        cand_top = day_frame.sort_values(score_col_cand, ascending=False).head(k)
        prod_top = day_frame.sort_values("current_prod4_score", ascending=False).head(k)

        base_set = set(base_top["candidate_id"].tolist())
        cand_set = set(cand_top["candidate_id"].tolist())
        prod_set = set(prod_top["candidate_id"].tolist())

        rows.append(
            {
                "entry_trade_day": entry_day,
                "available_candidates": int(day_frame["candidate_id"].nunique()),
                "actual_traded_candidates": k,
                "actual_vs_prod4_topk_overlap": int(len(actual_set & prod_set)),
                "actual_vs_replay_baseline_topk_overlap": int(len(actual_set & base_set)),
                "actual_vs_gap_topk_overlap": int(len(actual_set & cand_set)),
                "replay_baseline_vs_gap_topk_overlap": int(len(base_set & cand_set)),
                "gap_added_candidates": ", ".join(sorted(cand_set - base_set)),
                "gap_removed_candidates": ", ".join(sorted(base_set - cand_set)),
                "gap_added_tickers": ", ".join(sorted(cand_top[~cand_top["candidate_id"].isin(base_set)]["ticker"].tolist())),
                "gap_removed_tickers": ", ".join(sorted(base_top[~base_top["candidate_id"].isin(cand_set)]["ticker"].tolist())),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vm_candidates = load_vm_candidates(args.start_date, args.end_date)
    actual_trades = load_actual_trades(args.start_date, args.end_date)
    scraped = scrape_months(args.start_date, args.end_date)
    combined_raw = build_combined_raw(BACKTEST_RAW_CSV, scraped)

    history_pre = load_training_prefeature_frame()
    vm_pre = build_vm_prefeature_frame(vm_candidates, combined_raw)

    combined = pd.concat([history_pre, vm_pre], ignore_index=True, sort=False)
    engineered, base_features, _caps = train_models.engineer_features(combined)

    history_eng = engineered[engineered["_source"] == "history"].copy()
    vm_eng = engineered[engineered["_source"] == "vm_candidate"].copy()
    history_gap, vm_gap = add_reference_gap_clip(vm_eng, history_eng)

    baseline_models, candidate_models, fit_df = fit_replay_models(history_gap, base_features, args.start_date)

    vm_gap["baseline_replay_score"] = gap_eval.predict_all_models(
        baseline_models, vm_gap[base_features]
    )["pred_mean4"]
    vm_gap["gap_feature_replay_score"] = gap_eval.predict_all_models(
        candidate_models, vm_gap[base_features + ["live_gap_to_entry_pct"]]
    )["pred_mean4"]
    vm_gap["gap_score_delta"] = vm_gap["gap_feature_replay_score"] - vm_gap["baseline_replay_score"]
    vm_gap["entry_trade_day"] = _normalize_date_str(vm_gap["buy_datetime"])

    actual_trades = actual_trades.rename(columns={"status": "actual_status"})
    vm_gap = vm_gap.merge(
        actual_trades[["candidate_id", "actual_status", "sleeve_id"]],
        on="candidate_id",
        how="left",
    )
    vm_gap["actually_traded"] = vm_gap["actual_status"].notna()

    vm_gap = rank_within_day(vm_gap, "current_prod4_score", "current_prod4_rank")
    vm_gap = rank_within_day(vm_gap, "baseline_replay_score", "baseline_replay_rank")
    vm_gap = rank_within_day(vm_gap, "gap_feature_replay_score", "gap_feature_replay_rank")
    vm_gap["rank_change_gap_vs_baseline"] = vm_gap["baseline_replay_rank"] - vm_gap["gap_feature_replay_rank"]

    daily = build_daily_summary(vm_gap, actual_trades, "baseline_replay_score", "gap_feature_replay_score")

    actual_detail = vm_gap[vm_gap["actually_traded"]].copy()
    actual_detail = actual_detail.sort_values(["entry_trade_day", "gap_feature_replay_rank", "ticker"]).reset_index(drop=True)

    detail_cols = [
        "candidate_id",
        "ticker",
        "owner_name",
        "title",
        "entry_trade_day",
        "transaction_date",
        "trade_date",
        "buy_datetime",
        "buy_price",
        "last_price",
        "value",
        "live_entry_bucket",
        "prev_regular_close",
        "live_gap_to_entry_pct",
        "current_prod4_score",
        "baseline_replay_score",
        "gap_feature_replay_score",
        "gap_score_delta",
        "current_prod4_rank",
        "baseline_replay_rank",
        "gap_feature_replay_rank",
        "rank_change_gap_vs_baseline",
        "actually_traded",
        "actual_status",
        "sleeve_id",
    ]
    actual_cols = [
        "candidate_id",
        "ticker",
        "entry_trade_day",
        "actual_status",
        "current_prod4_score",
        "baseline_replay_score",
        "gap_feature_replay_score",
        "gap_score_delta",
        "live_gap_to_entry_pct",
        "current_prod4_rank",
        "baseline_replay_rank",
        "gap_feature_replay_rank",
        "rank_change_gap_vs_baseline",
    ]

    vm_gap[detail_cols].sort_values(["entry_trade_day", "gap_feature_replay_rank", "ticker"]).to_csv(OUT_DETAIL_CSV, index=False)
    daily.to_csv(OUT_DAILY_CSV, index=False)
    actual_detail[actual_cols].to_csv(OUT_ACTUAL_CSV, index=False)

    print(f"Replay training rows before {args.start_date}: {len(fit_df):,}")
    print(f"VM candidates scored for {args.start_date} to {args.end_date}: {len(vm_gap):,}")
    print(f"Actual traded candidates in window: {int(vm_gap['actually_traded'].sum()):,}")
    print("")
    if not daily.empty:
        print("Daily comparison:")
        print(daily.to_string(index=False))
        print("")
    print("Actual traded names, sorted by gap-feature replay rank:")
    print(actual_detail[actual_cols].to_string(index=False))
    print("")
    print(f"Saved candidate detail to: {OUT_DETAIL_CSV}")
    print(f"Saved daily summary to: {OUT_DAILY_CSV}")
    print(f"Saved actual-trade detail to: {OUT_ACTUAL_CSV}")


if __name__ == "__main__":
    main()
