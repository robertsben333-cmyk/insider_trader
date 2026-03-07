# Execution Research

## Scope

This category covers entry timing, exit timing, capital recycling, stop-loss studies, and intraday path rules.

## Main findings

- `backtest/out/entry_timing_summary.csv` shows earlier execution was better in the sampled timing sweep; the `1m` delay outperformed the older `15m` baseline on average return.
- `backtest/out/exit_recycling_strategy_comparison.json` supports the current live direction: the fully retrained `portfolio_tplus2_open_signal_tplus2_open_exit` variant was the strongest of the compared portfolio policies.
- Stop-loss and dynamic-path work remain research tools rather than current production rules; they are useful for future app controls but are not the active default strategy.

## Key scripts

- `research/scripts/analyze_entry_timing_sensitivity.py`
- `research/scripts/compare_exit_recycling_strategies.py`
- `research/scripts/analyze_stop_loss_1d_dailybars.py`
- `backtest/scripts/analyze_entry_timing.py`
- `backtest/scripts/analyze_exit_timing.py`
- `backtest/scripts/analyze_investable_early_exit.py`
- `backtest/scripts/analyze_trailing_stop_market_conditions.py`
- `scripts/analyze_vm_early_exit_signals.py`
- `scripts/analyze_vm_dynamic_path_exit.py`
- `scripts/analyze_vm_stoploss.py`

## Key outputs

- `backtest/out/entry_timing_summary.csv`
- `backtest/out/exit_recycling_strategy_comparison.json`
- `backtest/out/investable_early_exit_summary.csv`
- `live/data/vm_sync/historical_recommended_backtest_summary.csv`
- `live/data/vm_sync/historical_recommended_early_exit_walkforward_summary.csv`
