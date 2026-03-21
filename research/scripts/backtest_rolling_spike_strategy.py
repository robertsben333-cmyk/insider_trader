from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
import os
import sys

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
os.chdir(BASE)

from research.scripts import build_unfiltered_aggregated_backtest as aggregate_builder
from research.scripts import evaluate_rolling_spike_classifier as rolling


OUTPUT_DIR = BASE / "research" / "outcomes" / "models" / "rolling_spike_3d_strategy"
SWEEP_CSV = OUTPUT_DIR / "threshold_sweep.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
VAL_CANDIDATES_CSV = OUTPUT_DIR / "validation_scored_candidates.csv"
TEST_CANDIDATES_CSV = OUTPUT_DIR / "test_scored_candidates.csv"
VAL_CURVE_CSV = OUTPUT_DIR / "validation_selected_curve.csv"
VAL_TRADES_CSV = OUTPUT_DIR / "validation_selected_trades.csv"
TEST_CURVE_CSV = OUTPUT_DIR / "test_selected_curve.csv"
TEST_TRADES_CSV = OUTPUT_DIR / "test_selected_trades.csv"

DEFAULT_SCORE_COL = "prob_ensemble_calibrated"
START_BUDGET_USD = 10_000.0
MAX_TICKER_WEIGHT = 0.25
MIN_VALIDATION_TRADES = 20
THRESHOLD_QUANTILES = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]


@dataclass
class Position:
    ticker: str
    event_key: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    invested_usd: float
    ret_pct: float
    score: float
    entry_price: float
    exit_price: float
    exit_reason: str
    snapshot_day_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Active backtest for the rolling spike classifier. "
            "Uses train for fitting, validation for score-threshold selection, and reports validation/test only."
        )
    )
    parser.add_argument("--raw-csv", type=Path, default=rolling.DEFAULT_RAW_CSV)
    parser.add_argument("--aggregate-csv", type=Path, default=rolling.DEFAULT_AGG_CSV)
    parser.add_argument("--cache-dir", type=Path, default=rolling.DEFAULT_CACHE_DIR)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--rebuild-aggregate", action="store_true")
    parser.add_argument("--cache-only-day", action="store_true")
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--start-budget-usd", type=float, default=START_BUDGET_USD)
    parser.add_argument("--max-ticker-weight", type=float, default=MAX_TICKER_WEIGHT)
    parser.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    parser.add_argument("--min-validation-trades", type=int, default=MIN_VALIDATION_TRADES)
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    return rolling.setup_logger()


def score_split_predictions(
    frame: rolling.SplitFrames,
    full_feature_cols: list[str],
    metadata_feature_cols: list[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = frame.train, frame.val, frame.test
    x_train = train_df[full_feature_cols].copy()
    y_train = train_df[rolling.TARGET_COL].to_numpy(dtype=int)
    x_val = val_df[full_feature_cols].copy()
    y_val = val_df[rolling.TARGET_COL].to_numpy(dtype=int)
    x_test = test_df[full_feature_cols].copy()

    logger.info("Training classifier stack for strategy backtest...")
    models = rolling.train_main_models(x_train, y_train, logger)
    val_probs = {name: rolling.predict_model_probability(name, mdl, x_val) for name, mdl in models.items()}
    test_probs = {name: rolling.predict_model_probability(name, mdl, x_test) for name, mdl in models.items()}
    val_mean = np.mean(np.column_stack([val_probs[name] for name in rolling.MODEL_NAMES]), axis=1)
    test_mean = np.mean(np.column_stack([test_probs[name] for name in rolling.MODEL_NAMES]), axis=1)

    logger.info("Fitting validation calibrator for strategy scores...")
    calibrator = rolling.fit_platt_calibrator(val_mean, y_val)
    val_cal = rolling.calibrated_probability(calibrator, val_mean)
    test_cal = rolling.calibrated_probability(calibrator, test_mean)

    logger.info("Training metadata baseline for reference columns...")
    metadata_model = rolling.train_metadata_baseline(train_df[metadata_feature_cols].copy(), y_train)
    meta_val = np.asarray(metadata_model.predict_proba(val_df[metadata_feature_cols].copy())[:, 1], dtype=float)
    meta_test = np.asarray(metadata_model.predict_proba(test_df[metadata_feature_cols].copy())[:, 1], dtype=float)

    keep_cols = [
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
        rolling.TARGET_COL,
    ]

    def _frame(split_name: str, split_df: pd.DataFrame, prob_map: dict[str, np.ndarray], mean_prob: np.ndarray, cal_prob: np.ndarray, meta_prob: np.ndarray) -> pd.DataFrame:
        out = split_df[keep_cols].copy()
        out["split"] = split_name
        for model_name in rolling.MODEL_NAMES:
            out[f"prob_{model_name}"] = prob_map[model_name]
        out["prob_ensemble_raw"] = mean_prob
        out["prob_ensemble_calibrated"] = cal_prob
        out["prob_metadata_baseline"] = meta_prob
        out["snapshot_timestamp"] = pd.to_datetime(out["snapshot_timestamp"], errors="coerce")
        out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.date
        out["label_end_date"] = pd.to_datetime(out["label_end_date"], errors="coerce").dt.date
        return out.sort_values(["snapshot_timestamp", "event_key", "snapshot_day_idx"]).reset_index(drop=True)

    val_predictions = _frame("validation", val_df, val_probs, val_mean, val_cal, meta_val)
    test_predictions = _frame("test", test_df, test_probs, test_mean, test_cal, meta_test)
    return val_predictions, test_predictions


def load_bar_lookup(
    candidates: pd.DataFrame,
    cache_dir: Path,
    logger: logging.Logger,
    *,
    cache_only_day: bool = False,
) -> dict[str, tuple[list[pd.Timestamp], dict[Any, dict[str, Any]]]]:
    cache = aggregate_builder.PriceCache(cache_dir)
    bar_lookup: dict[str, tuple[list[pd.Timestamp], dict[Any, dict[str, Any]]]] = {}
    tickers = candidates["ticker"].dropna().astype(str).unique().tolist()
    total = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        sub = candidates[candidates["ticker"].astype(str) == ticker]
        min_date = pd.to_datetime(sub["snapshot_date"], errors="coerce").dt.date.min()
        max_date = pd.to_datetime(sub["label_end_date"], errors="coerce").dt.date.max()
        if pd.isna(min_date) or pd.isna(max_date):
            continue
        bars = cache.get_day_exact(ticker, min_date, max_date) or cache.get_day_covering(ticker, min_date, max_date) or []
        dated: list[tuple[Any, dict[str, Any]]] = []
        for bar in bars:
            bar_date = rolling.train_models._bar_date_et(bar)
            if bar_date is None:
                continue
            dated.append((bar_date, bar))
        if not dated:
            continue
        dated.sort(key=lambda x: x[0])
        trading_dates = [d for d, _bar in dated]
        bar_by_date = {d: bar for d, bar in dated}
        bar_lookup[ticker] = (trading_dates, bar_by_date)
        if idx % 500 == 0 or idx == total:
            logger.info("Loaded daily bars for %d / %d strategy tickers", idx, total)
    return bar_lookup


def resolve_trade_exit(
    row: pd.Series,
    trading_dates: list[Any],
    bar_by_date: dict[Any, dict[str, Any]],
) -> dict[str, Any] | None:
    snapshot_date = pd.to_datetime(row["snapshot_date"], errors="coerce").date()
    label_end_date = pd.to_datetime(row["label_end_date"], errors="coerce").date()
    snapshot_kind = str(row["snapshot_kind"])
    snapshot_price = float(row["snapshot_price"])
    if not np.isfinite(snapshot_price) or snapshot_price <= 0:
        return None

    label_dates = rolling.candidate_label_dates(
        trading_dates=trading_dates,
        current_date=snapshot_date,
        current_kind=snapshot_kind,
        cycle_end_date=label_end_date,
        lookahead_days=rolling.LOOKAHEAD_DAYS,
    )
    if not label_dates:
        return None

    threshold = snapshot_price * (1.0 + rolling.THRESHOLD_PCT / 100.0)
    exit_date = label_end_date
    exit_reason = "time_exit_close"
    for d in label_dates:
        high = rolling._safe_bar_price(bar_by_date.get(d, {}), "h")
        if high is not None and float(high) >= threshold:
            exit_date = d
            exit_reason = "target_hit_close"
            break

    exit_price = rolling._safe_bar_price(bar_by_date.get(exit_date, {}), "c")
    if exit_price is None:
        return None

    entry_ts = pd.to_datetime(row["snapshot_timestamp"], errors="coerce")
    exit_ts = pd.Timestamp(exit_date) + pd.Timedelta(hours=16)
    ret_pct = (float(exit_price) / snapshot_price - 1.0) * 100.0
    return {
        "entry_time": entry_ts,
        "exit_time": exit_ts,
        "exit_date": exit_date,
        "exit_price": float(exit_price),
        "ret_pct": float(ret_pct),
        "exit_reason": exit_reason,
        "threshold_price": float(threshold),
        "holding_calendar_days": float((exit_ts - entry_ts).total_seconds() / 86400.0) if pd.notna(entry_ts) else np.nan,
    }


def attach_trade_outcomes(
    predictions: pd.DataFrame,
    bar_lookup: dict[str, tuple[list[Any], dict[Any, dict[str, Any]]]],
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(predictions)
    for idx, row in enumerate(predictions.itertuples(index=False), start=1):
        row_series = pd.Series(row._asdict())
        ticker = str(row_series["ticker"])
        lookup = bar_lookup.get(ticker)
        if lookup is None:
            continue
        resolved = resolve_trade_exit(row_series, lookup[0], lookup[1])
        if resolved is None:
            continue
        out = row_series.to_dict()
        out.update(resolved)
        rows.append(out)
        if idx % 2000 == 0 or idx == total:
            logger.info("Resolved strategy exits for %d / %d scored rows", idx, total)
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    out_df["entry_time"] = pd.to_datetime(out_df["entry_time"], errors="coerce")
    out_df["exit_time"] = pd.to_datetime(out_df["exit_time"], errors="coerce")
    out_df["trade_day"] = out_df["entry_time"].dt.date
    return out_df.sort_values(["entry_time", "event_key", "snapshot_day_idx"]).reset_index(drop=True)


def candidate_thresholds(scores: pd.Series) -> list[float]:
    vals = pd.to_numeric(scores, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return []
    thresholds = {float(np.quantile(vals, q)) for q in THRESHOLD_QUANTILES}
    thresholds.add(float(np.quantile(vals, 0.995)))
    thresholds.add(float(np.quantile(vals, 0.999)))
    return sorted(t for t in thresholds if np.isfinite(t))


def simulate_budget_curve(
    picks: pd.DataFrame,
    *,
    score_col: str,
    start_budget_usd: float,
    max_ticker_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks = picks.dropna(subset=["entry_time", "exit_time", "ret_pct", "ticker", "event_key", score_col]).copy()
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame()

    picks = picks.sort_values(["entry_time", score_col], ascending=[True, False]).reset_index(drop=True)
    cash = float(start_budget_usd)
    open_positions: list[Position] = []
    trade_log: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []

    start_ts = pd.to_datetime(picks["entry_time"].min()) - pd.Timedelta(minutes=1)
    curve.append({"timestamp": start_ts, "cash_usd": cash, "invested_usd": 0.0, "budget_usd": cash})

    def mark(ts: pd.Timestamp) -> None:
        invested = float(sum(p.invested_usd for p in open_positions))
        curve.append({"timestamp": ts, "cash_usd": cash, "invested_usd": invested, "budget_usd": cash + invested})

    def close_until(ts: pd.Timestamp) -> None:
        nonlocal cash, open_positions
        while True:
            due = [p for p in open_positions if p.exit_time <= ts]
            if not due:
                break
            next_exit = min(p.exit_time for p in due)
            closing = [p for p in open_positions if p.exit_time == next_exit]
            open_positions = [p for p in open_positions if p.exit_time != next_exit]
            for pos in closing:
                proceeds = pos.invested_usd * (1.0 + pos.ret_pct / 100.0)
                cash += proceeds
                trade_log.append(
                    {
                        "ticker": pos.ticker,
                        "event_key": pos.event_key,
                        "entry_time": pos.entry_time,
                        "exit_time": pos.exit_time,
                        "invested_usd": pos.invested_usd,
                        "ret_pct": pos.ret_pct,
                        "proceeds_usd": proceeds,
                        "pnl_usd": proceeds - pos.invested_usd,
                        "score": pos.score,
                        "entry_price": pos.entry_price,
                        "exit_price": pos.exit_price,
                        "exit_reason": pos.exit_reason,
                        "snapshot_day_idx": pos.snapshot_day_idx,
                    }
                )
            mark(next_exit)

    for entry_time, batch in picks.groupby("entry_time", sort=True):
        entry_time = pd.to_datetime(entry_time)
        close_until(entry_time)
        batch = batch.sort_values(score_col, ascending=False)
        active_event_keys = {p.event_key for p in open_positions}
        for _, row in batch.iterrows():
            if cash <= 0:
                break
            event_key = str(row["event_key"])
            if event_key in active_event_keys:
                continue
            invested_total = float(sum(p.invested_usd for p in open_positions))
            budget_now = cash + invested_total
            if budget_now <= 0:
                break
            ticker = str(row["ticker"])
            ticker_exposure = float(sum(p.invested_usd for p in open_positions if p.ticker == ticker))
            ticker_cap = float(max_ticker_weight) * budget_now
            alloc = min(cash, max(0.0, ticker_cap - ticker_exposure))
            if alloc <= 0:
                continue
            cash -= alloc
            pos = Position(
                ticker=ticker,
                event_key=event_key,
                entry_time=entry_time,
                exit_time=pd.to_datetime(row["exit_time"]),
                invested_usd=float(alloc),
                ret_pct=float(row["ret_pct"]),
                score=float(row[score_col]),
                entry_price=float(row["snapshot_price"]),
                exit_price=float(row["exit_price"]),
                exit_reason=str(row["exit_reason"]),
                snapshot_day_idx=int(row["snapshot_day_idx"]),
            )
            open_positions.append(pos)
            active_event_keys.add(event_key)
        mark(entry_time)

    if open_positions:
        last_exit = max(p.exit_time for p in open_positions)
        close_until(last_exit + pd.Timedelta(seconds=1))

    curve_df = pd.DataFrame(curve).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    trades_df = pd.DataFrame(trade_log).sort_values(["entry_time", "event_key"]).reset_index(drop=True)
    return curve_df, trades_df


def summarize_backtest(
    picks: pd.DataFrame,
    curve_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    split_name: str,
    threshold: float,
) -> dict[str, Any]:
    if curve_df.empty or trades_df.empty:
        return {
            "split": split_name,
            "threshold": float(threshold),
            "candidate_rows": int(len(picks)),
            "candidate_days": int(pd.to_datetime(picks["entry_time"], errors="coerce").dt.date.nunique()) if not picks.empty else 0,
            "trades_executed": 0,
            "start_budget_usd": np.nan,
            "end_budget_usd": np.nan,
            "total_return_pct": np.nan,
            "annualized_return_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "win_rate_pct": np.nan,
            "mean_trade_ret_pct": np.nan,
            "median_trade_ret_pct": np.nan,
            "avg_holding_days": np.nan,
            "target_hit_rate_pct": np.nan,
        }

    start_budget = float(curve_df["budget_usd"].iloc[0])
    end_budget = float(curve_df["budget_usd"].iloc[-1])
    start_time = pd.to_datetime(curve_df["timestamp"].iloc[0])
    end_time = pd.to_datetime(curve_df["timestamp"].iloc[-1])
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1.0)
    annualized_return = ((end_budget / start_budget) ** (365.25 / elapsed_days) - 1.0) * 100.0
    running_peak = curve_df["budget_usd"].cummax()
    max_drawdown = (((curve_df["budget_usd"] / running_peak) - 1.0) * 100.0).min()
    rets = pd.to_numeric(trades_df["ret_pct"], errors="coerce")
    holding = (pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])).dt.total_seconds() / 86400.0
    target_hit_rate = (trades_df["exit_reason"] == "target_hit_close").mean() * 100.0
    return {
        "split": split_name,
        "threshold": float(threshold),
        "candidate_rows": int(len(picks)),
        "candidate_days": int(pd.to_datetime(picks["entry_time"], errors="coerce").dt.date.nunique()),
        "trades_executed": int(len(trades_df)),
        "start_budget_usd": start_budget,
        "end_budget_usd": end_budget,
        "total_return_pct": (end_budget / start_budget - 1.0) * 100.0,
        "annualized_return_pct": float(annualized_return),
        "max_drawdown_pct": float(max_drawdown),
        "win_rate_pct": float((rets > 0).mean() * 100.0),
        "mean_trade_ret_pct": float(rets.mean()),
        "median_trade_ret_pct": float(rets.median()),
        "avg_holding_days": float(holding.mean()),
        "target_hit_rate_pct": float(target_hit_rate),
    }


def run_threshold_sweep(
    val_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
    *,
    score_col: str,
    start_budget_usd: float,
    max_ticker_weight: float,
    min_validation_trades: int,
) -> tuple[pd.DataFrame, dict[str, Any], tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    sweep_rows: list[dict[str, Any]] = []
    best_val: dict[str, Any] | None = None
    best_val_curve = (pd.DataFrame(), pd.DataFrame())
    best_test_curve = (pd.DataFrame(), pd.DataFrame())

    for threshold in candidate_thresholds(val_candidates[score_col]):
        val_picks = val_candidates[pd.to_numeric(val_candidates[score_col], errors="coerce") >= float(threshold)].copy()
        test_picks = test_candidates[pd.to_numeric(test_candidates[score_col], errors="coerce") >= float(threshold)].copy()

        val_curve, val_trades = simulate_budget_curve(
            val_picks,
            score_col=score_col,
            start_budget_usd=start_budget_usd,
            max_ticker_weight=max_ticker_weight,
        )
        test_curve, test_trades = simulate_budget_curve(
            test_picks,
            score_col=score_col,
            start_budget_usd=start_budget_usd,
            max_ticker_weight=max_ticker_weight,
        )
        val_summary = summarize_backtest(val_picks, val_curve, val_trades, split_name="validation", threshold=threshold)
        test_summary = summarize_backtest(test_picks, test_curve, test_trades, split_name="test", threshold=threshold)
        sweep_rows.extend([val_summary, test_summary])

        if val_summary["trades_executed"] < int(min_validation_trades):
            continue
        candidate = {
            "threshold": float(threshold),
            "validation": val_summary,
            "test": test_summary,
        }
        if best_val is None:
            best_val = candidate
            best_val_curve = (val_curve, val_trades)
            best_test_curve = (test_curve, test_trades)
            continue
        current_key = (
            float(candidate["validation"]["total_return_pct"]),
            float(candidate["validation"]["mean_trade_ret_pct"]),
            float(candidate["validation"]["win_rate_pct"]),
            -float(candidate["validation"]["max_drawdown_pct"]),
        )
        best_key = (
            float(best_val["validation"]["total_return_pct"]),
            float(best_val["validation"]["mean_trade_ret_pct"]),
            float(best_val["validation"]["win_rate_pct"]),
            -float(best_val["validation"]["max_drawdown_pct"]),
        )
        if current_key > best_key:
            best_val = candidate
            best_val_curve = (val_curve, val_trades)
            best_test_curve = (test_curve, test_trades)

    if best_val is None:
        raise ValueError("No validation threshold met the minimum trade requirement.")
    sweep_df = pd.DataFrame(sweep_rows).sort_values(["split", "threshold"]).reset_index(drop=True)
    return sweep_df, best_val, best_val_curve, best_test_curve


def write_outputs(
    sweep_df: pd.DataFrame,
    best_result: dict[str, Any],
    val_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
    val_curve: pd.DataFrame,
    val_trades: pd.DataFrame,
    test_curve: pd.DataFrame,
    test_trades: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(SWEEP_CSV, index=False)
    val_candidates.to_csv(VAL_CANDIDATES_CSV, index=False)
    test_candidates.to_csv(TEST_CANDIDATES_CSV, index=False)
    val_curve.to_csv(VAL_CURVE_CSV, index=False)
    val_trades.to_csv(VAL_TRADES_CSV, index=False)
    test_curve.to_csv(TEST_CURVE_CSV, index=False)
    test_trades.to_csv(TEST_TRADES_CSV, index=False)

    payload = {
        "score_col": str(args.score_col),
        "start_budget_usd": float(args.start_budget_usd),
        "max_ticker_weight": float(args.max_ticker_weight),
        "min_validation_trades": int(args.min_validation_trades),
        "best_threshold": float(best_result["threshold"]),
        "validation": best_result["validation"],
        "test": best_result["test"],
        "notes": [
            "Model fitting uses train only.",
            "Threshold selection uses validation only.",
            "Reported strategy performance should be read from the test section.",
            "No slippage or commissions are applied.",
            "Exit rule: +8% hit in the forward window exits at that day's close; otherwise exit at the last allowed close.",
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    rolling.ensure_live_aggregate(args, logger)
    event_df = rolling.load_event_frame(args.aggregate_csv, args.raw_csv, logger)
    snapshot_df = rolling.build_snapshot_dataset(
        event_df,
        args.cache_dir,
        logger,
        cache_only_day=bool(args.cache_only_day),
        limit_events=int(args.limit_events),
    )
    if snapshot_df.empty:
        raise SystemExit("Sequential snapshot dataset is empty.")

    model_df, full_feature_cols, metadata_feature_cols = rolling.engineer_model_frame(snapshot_df, logger)
    split_frame = rolling.chronological_event_split(model_df)

    val_predictions, test_predictions = score_split_predictions(split_frame, full_feature_cols, metadata_feature_cols, logger)
    combined = pd.concat([val_predictions, test_predictions], ignore_index=True)
    bar_lookup = load_bar_lookup(combined, args.cache_dir, logger, cache_only_day=bool(args.cache_only_day))

    val_candidates = attach_trade_outcomes(val_predictions, bar_lookup, logger)
    test_candidates = attach_trade_outcomes(test_predictions, bar_lookup, logger)

    if val_candidates.empty or test_candidates.empty:
        raise SystemExit("Validation/test candidates are empty after resolving exits.")

    sweep_df, best_result, val_curve_pack, test_curve_pack = run_threshold_sweep(
        val_candidates,
        test_candidates,
        score_col=str(args.score_col),
        start_budget_usd=float(args.start_budget_usd),
        max_ticker_weight=float(args.max_ticker_weight),
        min_validation_trades=int(args.min_validation_trades),
    )
    write_outputs(
        sweep_df=sweep_df,
        best_result=best_result,
        val_candidates=val_candidates,
        test_candidates=test_candidates,
        val_curve=val_curve_pack[0],
        val_trades=val_curve_pack[1],
        test_curve=test_curve_pack[0],
        test_trades=test_curve_pack[1],
        args=args,
    )
    logger.info("Saved strategy summary: %s", SUMMARY_JSON)
    logger.info("Saved threshold sweep: %s", SWEEP_CSV)
    logger.info("Saved validation trades: %s", VAL_TRADES_CSV)
    logger.info("Saved test trades: %s", TEST_TRADES_CSV)


if __name__ == "__main__":
    main()
