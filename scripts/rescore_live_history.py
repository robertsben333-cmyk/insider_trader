from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import live_scoring
from live_trading.strategy_settings import ACTIVE_STRATEGY, LIVE_PATHS


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rescore full local live event history with the deployed live models."
    )
    p.add_argument("--raw-file", default=LIVE_PATHS.raw_file)
    p.add_argument("--aggregated-file", default=LIVE_PATHS.aggregated_file)
    p.add_argument("--predictions-file", default=LIVE_PATHS.predictions_file)
    p.add_argument("--cache-dir", default=LIVE_PATHS.cache_dir)
    p.add_argument("--sector-cache-file", default=LIVE_PATHS.sector_cache_file)
    p.add_argument("--model-dir", default=ACTIVE_STRATEGY.model_dir)
    p.add_argument("--alert-snapshot-file", default=LIVE_PATHS.alert_snapshot_file)
    p.add_argument("--alert-history-file", default=LIVE_PATHS.alert_history_file)
    p.add_argument(
        "--day1-decile-score-threshold",
        type=float,
        default=live_scoring.DEFAULT_DAY1_DECILE_SCORE_THRESHOLD,
    )
    p.add_argument(
        "--day1-decile-cutoff-file",
        default=ACTIVE_STRATEGY.day1_decile_cutoff_file,
    )
    p.add_argument(
        "--day1-benchmark-file",
        default=ACTIVE_STRATEGY.day1_benchmark_file,
    )
    p.add_argument(
        "--advice-base-alloc-fraction",
        type=float,
        default=live_scoring.DEFAULT_ADVICE_BASE_ALLOC_FRACTION,
    )
    p.add_argument(
        "--advice-bonus-fraction",
        type=float,
        default=live_scoring.DEFAULT_ADVICE_BONUS_FRACTION,
    )
    return p


def main() -> None:
    load_dotenv()
    args = build_arg_parser().parse_args()
    logger = live_scoring.setup_logger()

    raw_path = Path(args.raw_file)
    aggregated_path = Path(args.aggregated_file)
    predictions_path = Path(args.predictions_file)
    alert_snapshot_path = Path(args.alert_snapshot_file)
    alert_history_path = Path(args.alert_history_file)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw live file not found: {raw_path}")
    if not aggregated_path.exists():
        raise FileNotFoundError(f"Aggregated live file not found: {aggregated_path}")

    models_by_horizon, policy = live_scoring.load_models_and_policy(Path(args.model_dir))
    threshold, threshold_source = live_scoring.load_day1_pred_mean_threshold(
        decile_score_threshold=args.day1_decile_score_threshold,
        cutoff_csv=Path(args.day1_decile_cutoff_file),
        benchmark_csv=Path(args.day1_benchmark_file),
        logger=logger,
    )
    advice_deciles, advice_raw_cutoffs, advice_curve_source = live_scoring.load_day1_decile_curve(
        cutoff_csv=Path(args.day1_decile_cutoff_file),
        benchmark_csv=Path(args.day1_benchmark_file),
        logger=logger,
    )

    aggregated_df = pd.read_csv(aggregated_path, dtype=str, keep_default_na=False)
    event_keys = (
        live_scoring.make_event_key_series(aggregated_df["ticker"], aggregated_df["trade_date"])
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    logger.info("Scoring full live history for %d event keys", len(event_keys))

    feat_df = live_scoring.compute_features_for_pending(
        temp_agg_path=aggregated_path,
        pending_event_keys=event_keys,
        raw_file=raw_path,
        cache_dir=Path(args.cache_dir),
        sector_cache_file=Path(args.sector_cache_file),
    )
    logger.info("Feature rows ready for full-history scoring: %d", len(feat_df))

    scored = live_scoring.score_features(feat_df, models_by_horizon, policy)
    if "representative_transaction_date" in scored.columns:
        scored["scored_at"] = (
            scored["representative_transaction_date"]
            .fillna(scored.get("transaction_date", ""))
            .astype(str)
        )
    scored = scored.sort_values(
        ["representative_transaction_date", "event_key", "horizon_days"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(predictions_path, index=False)
    logger.info("Rebuilt predictions -> %s (rows=%d)", predictions_path, len(scored))

    picks_df, pred_col = live_scoring.select_day1_pred_mean_candidates(scored, threshold)
    if not picks_df.empty:
        picks_df = live_scoring.apply_linear_allocation_advice(
            picks_df=picks_df,
            pred_col=pred_col,
            decile_score_threshold=args.day1_decile_score_threshold,
            decile_scores=advice_deciles,
            raw_cutoffs=advice_raw_cutoffs,
            base_alloc_fraction=args.advice_base_alloc_fraction,
            bonus_fraction=args.advice_bonus_fraction,
        )

    history_export = live_scoring.build_alert_export_rows(
        scored_df=scored,
        picks_df=picks_df,
        pred_col=pred_col,
        threshold=threshold,
        decile_score_threshold=args.day1_decile_score_threshold,
        threshold_source=threshold_source,
    )
    alert_history_path.parent.mkdir(parents=True, exist_ok=True)
    history_export.to_csv(alert_history_path, index=False)
    logger.info("Rebuilt alert history -> %s (rows=%d)", alert_history_path, len(history_export))

    if history_export.empty:
        latest_export = pd.DataFrame(columns=pd.Index(live_scoring.ALERT_EXPORT_COLUMNS))
    else:
        latest_scored_at = history_export["scored_at"].astype(str).max()
        latest_export = history_export[history_export["scored_at"].astype(str) == latest_scored_at].copy()
        latest_export = latest_export.sort_values(
            ["scored_at", "score_1d", "ticker"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    alert_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    latest_export.to_csv(alert_snapshot_path, index=False)
    logger.info(
        "Rebuilt latest alert snapshot -> %s (rows=%d, threshold_source=%s, advice_curve_source=%s)",
        alert_snapshot_path,
        len(latest_export),
        threshold_source,
        advice_curve_source,
    )


if __name__ == "__main__":
    main()
