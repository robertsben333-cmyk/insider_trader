# Live Transition

## Scope

This category tracks the work that bridges research into the live runtime and eventually into an app.

## Main findings

- `research/scripts/deploy_tplus2_open_day1_live.py` formalized the current operating strategy: day-1 ensemble gating with a `T+2 open` exit and archived backups of the previous production bundle.
- The VM-sync scripts provide a paper-trading style audit trail for live alerts, historical recommended trades, and post-entry management experiments.
- The new `live_trading/` folder should be treated as the source area for future app orchestration, while `research/` stays focused on evidence and historical decisions.

## Key scripts

- `research/scripts/deploy_tplus2_open_day1_live.py`
- `live_scoring.py`
- `scripts/rescore_live_history.py`
- `scripts/backtest_vm_recommendations.py`
- `scripts/evaluate_vm_early_exit_walkforward.py`
- `scripts/pull_vm_live_csvs.ps1`

## Key artifacts

- `live_trading/models/day1_tplus2_open/manifest.json`
- `models/prod4/ensemble_policy.json`
- `models/prod4/eval_time_split_day1_tplus2_open.json`
- `backtest/out/investable_decile_score_sweep_0005_tplus2_open.csv`
- `research/outcomes/models/equal4_deciles_time_split_tplus2_open_live.json`
