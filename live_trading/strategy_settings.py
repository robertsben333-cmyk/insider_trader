from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
    trader_state_dir: str
    trader_state_file: str
    trader_journal_file: str
    trader_signal_archive_file: str
    dashboard_dir: str
    dashboard_baseline_file: str
    dashboard_executions_file: str
    dashboard_equity_history_file: str
    dashboard_latest_snapshot_file: str
    gateway_env_file: str


@dataclass(frozen=True)
class RuntimeDefaults:
    months_back: int
    alert_recipient: str
    market_hours_interval_minutes: int
    near_open_window_hours: int
    near_open_interval_minutes: int
    far_interval_minutes: int
    historical_backtest_horizons: tuple[int, ...]
    early_exit_thresholds: tuple[float, ...]
    early_exit_initial_train_size: int
    early_exit_min_train_flagged: int


@dataclass(frozen=True)
class IbkrConfig:
    host: str
    port: int
    client_id: int
    account_id: str
    paper_trading: bool
    connect_timeout_seconds: float
    readonly: bool


@dataclass(frozen=True)
class GatewayRuntimeConfig:
    display: str
    xvfb_geometry: str
    gateway_host: str
    gateway_port: int
    ready_timeout_seconds: int
    launch_script: str
    wait_script: str


@dataclass(frozen=True)
class TradingBudgetConfig:
    initial_strategy_budget: float
    sleeve_count: int
    sleeve_fraction: float
    rotation_anchor_date: str
    compound_pnl: bool
    max_fraction_single_name: float
    max_fraction_two_names: float
    max_fraction_three_plus_names: float
    long_only: bool
    whole_shares_only: bool

    def anchor_date(self) -> date:
        return date.fromisoformat(self.rotation_anchor_date)


@dataclass(frozen=True)
class ExecutionPolicy:
    cycle_seconds: int
    buy_cutoff_time: str
    cancel_unfilled_time: str
    replace_interval_seconds: int
    buy_limit_buffer_bps: float
    sell_limit_buffer_bps: float
    min_order_notional: float
    quote_wait_seconds: float
    routing_exchange: str
    currency: str
    open_order_poll_seconds: int


@dataclass(frozen=True)
class DashboardRuntimeConfig:
    sync_interval_seconds: int
    recent_trades_limit: int
    auto_refresh_seconds: int
    streamlit_host: str
    streamlit_port: int


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
    trader_state_dir="live/data/trader_state",
    trader_state_file="live/data/trader_state/ibkr_paper_trader_state.json",
    trader_journal_file="live/data/trader_state/ibkr_paper_trader_journal.jsonl",
    trader_signal_archive_file="live/data/trader_state/ibkr_paper_signal_archive.csv",
    dashboard_dir="live/data/dashboard",
    dashboard_baseline_file="live/data/dashboard/baseline.json",
    dashboard_executions_file="live/data/dashboard/executions.jsonl",
    dashboard_equity_history_file="live/data/dashboard/equity_history.csv",
    dashboard_latest_snapshot_file="live/data/dashboard/latest_snapshot.json",
    gateway_env_file="/etc/insider_trades.env",
)

RUNTIME_DEFAULTS = RuntimeDefaults(
    months_back=1,
    alert_recipient="xavierjjc@outlook.com",
    market_hours_interval_minutes=1,
    near_open_window_hours=2,
    near_open_interval_minutes=30,
    far_interval_minutes=120,
    historical_backtest_horizons=(0, 1, 3, 5, 10),
    early_exit_thresholds=(-0.5, -1.0, -2.0, -3.0, -5.0),
    early_exit_initial_train_size=10,
    early_exit_min_train_flagged=2,
)

IBKR_CONFIG = IbkrConfig(
    host="127.0.0.1",
    port=4002,
    client_id=17,
    account_id="DUN175042",
    paper_trading=True,
    connect_timeout_seconds=10.0,
    readonly=False,
)

GATEWAY_RUNTIME = GatewayRuntimeConfig(
    display=":1",
    xvfb_geometry="1440x900x24",
    gateway_host="127.0.0.1",
    gateway_port=4002,
    ready_timeout_seconds=120,
    launch_script="scripts/vm/start_ibgateway.sh",
    wait_script="scripts/vm/wait_for_ibgateway.py",
)

TRADING_BUDGET = TradingBudgetConfig(
    initial_strategy_budget=10_000.0,
    sleeve_count=2,
    sleeve_fraction=0.5,
    rotation_anchor_date="2026-01-05",
    compound_pnl=True,
    max_fraction_single_name=0.80,
    max_fraction_two_names=0.60,
    max_fraction_three_plus_names=0.40,
    long_only=True,
    whole_shares_only=True,
)

EXECUTION_POLICY = ExecutionPolicy(
    cycle_seconds=30,
    buy_cutoff_time="15:30",
    cancel_unfilled_time="15:55",
    replace_interval_seconds=45,
    buy_limit_buffer_bps=50.0,
    sell_limit_buffer_bps=50.0,
    min_order_notional=100.0,
    quote_wait_seconds=2.0,
    routing_exchange="SMART",
    currency="USD",
    open_order_poll_seconds=5,
)

DASHBOARD_CONFIG = DashboardRuntimeConfig(
    sync_interval_seconds=30,
    recent_trades_limit=25,
    auto_refresh_seconds=30,
    streamlit_host="127.0.0.1",
    streamlit_port=8501,
)
