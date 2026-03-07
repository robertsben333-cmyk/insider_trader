from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


@dataclass(frozen=True)
class ActiveStrategy:
    strategy_id: str
    description: str
    model_dir: str
    backup_model_dir: str
    day1_decile_cutoff_file: str
    day1_benchmark_file: str
    benchmark_curve_json: str
    sell_after_trading_days: int
    sell_session: str
    sell_rule_label: str
    day1_decile_score_threshold: float
    day1_raw_threshold_fallback: float
    advice_base_alloc_fraction: float
    advice_bonus_fraction: float
    exit_policy_review_date: str


@dataclass(frozen=True)
class LivePaths:
    scraper_config: str
    raw_file: str
    aggregated_file: str
    predictions_file: str
    temp_aggregate_file: str
    cache_dir: str
    sector_cache_file: str
    alert_snapshot_file: str
    alert_history_file: str
    vm_sync_dir: str
    vm_predictions_file: str
    vm_backtest_summary_file: str
    vm_backtest_detail_file: str
    vm_early_exit_summary_file: str
    vm_early_exit_detail_file: str
    vm_early_exit_walkforward_summary_file: str
    vm_early_exit_walkforward_detail_file: str
    vm_dynamic_path_summary_file: str
    vm_dynamic_path_grid_file: str
    vm_dynamic_path_detail_file: str
    vm_stoploss_summary_file: str
    vm_stoploss_detail_file: str


@dataclass(frozen=True)
class RuntimeDefaults:
    months_back: int
    alert_recipient: str
    near_open_window_hours: int
    near_open_interval_minutes: int
    far_interval_minutes: int
    historical_backtest_horizons: tuple[int, ...]
    early_exit_thresholds: tuple[float, ...]
    early_exit_initial_train_size: int
    early_exit_min_train_flagged: int


ACTIVE_STRATEGY = ActiveStrategy(
    strategy_id="day1_tplus2_open",
    description="Day-1 ensemble signal with exit at the next market open two trading days after entry.",
    model_dir="models/prod4",
    backup_model_dir="models_r_day1/prod4",
    day1_decile_cutoff_file="backtest/out/investable_decile_score_sweep_0005_tplus2_open.csv",
    day1_benchmark_file="research/outcomes/models/equal4_deciles_time_split_tplus2_open_live.csv",
    benchmark_curve_json="research/outcomes/models/equal4_deciles_time_split_tplus2_open_live.json",
    sell_after_trading_days=2,
    sell_session="open",
    sell_rule_label="sell_at_open_2_trading_days_after_buy",
    day1_decile_score_threshold=0.87,
    day1_raw_threshold_fallback=0.6091125803233034,
    advice_base_alloc_fraction=0.25,
    advice_bonus_fraction=0.25,
    exit_policy_review_date="2026-03-01",
)

LIVE_PATHS = LivePaths(
    scraper_config="config.yaml",
    raw_file="live/data/insider_purchases.csv",
    aggregated_file="live/data/event_history_aggregated.csv",
    predictions_file="live/data/live_predictions.csv",
    temp_aggregate_file="live/data/live_scoring_temp_aggregate.csv",
    cache_dir="live/data/price_cache",
    sector_cache_file="live/data/sector_cache.csv",
    alert_snapshot_file="live/data/latest_alert_candidates.csv",
    alert_history_file="live/data/alert_candidate_history.csv",
    vm_sync_dir="live/data/vm_sync",
    vm_predictions_file="live/data/vm_sync/historical_recommended_predictions.csv",
    vm_backtest_summary_file="live/data/vm_sync/historical_recommended_backtest_summary.csv",
    vm_backtest_detail_file="live/data/vm_sync/historical_recommended_backtest_detail.csv",
    vm_early_exit_summary_file="live/data/vm_sync/historical_recommended_early_exit_summary.csv",
    vm_early_exit_detail_file="live/data/vm_sync/historical_recommended_early_exit_detail.csv",
    vm_early_exit_walkforward_summary_file="live/data/vm_sync/historical_recommended_early_exit_walkforward_summary.csv",
    vm_early_exit_walkforward_detail_file="live/data/vm_sync/historical_recommended_early_exit_walkforward_detail.csv",
    vm_dynamic_path_summary_file="live/data/vm_sync/historical_recommended_dynamic_path_summary.csv",
    vm_dynamic_path_grid_file="live/data/vm_sync/historical_recommended_dynamic_path_grid.csv",
    vm_dynamic_path_detail_file="live/data/vm_sync/historical_recommended_dynamic_path_detail.csv",
    vm_stoploss_summary_file="live/data/vm_sync/historical_recommended_stoploss_summary.csv",
    vm_stoploss_detail_file="live/data/vm_sync/historical_recommended_stoploss_detail.csv",
)

RUNTIME_DEFAULTS = RuntimeDefaults(
    months_back=1,
    alert_recipient="xavierjjc@outlook.com",
    near_open_window_hours=2,
    near_open_interval_minutes=30,
    far_interval_minutes=120,
    historical_backtest_horizons=(0, 1, 3, 5, 10),
    early_exit_thresholds=(-0.5, -1.0, -2.0, -3.0, -5.0),
    early_exit_initial_train_size=10,
    early_exit_min_train_flagged=2,
)
