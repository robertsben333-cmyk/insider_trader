# Model Research

## Scope

This category covers model-family selection, ensemble design, and the research artifacts behind the current live signal.

## Main findings

- `research/outcomes/models/model_comparison.json` shows the tree-based models were strongest single-model candidates, especially on `1d` where `HGBR` had the best top-decile spread in the sampled comparison.
- The live stack converged on a four-model ensemble: `HGBR`, `XGBoost`, `ElasticNet`, and `SplineElasticNet`.
- `research/outcomes/models/equal4_deciles_time_split_tplus2_open_live.json` is the clearest active-strategy artifact: in the time split, top decile mean actual excess return was about `1.72%` versus roughly `-0.17%` in the bottom decile for the `return_2d_open_pct` target.

## Key scripts

- `model_ensemble.py`
- `research/scripts/compare_models.py`
- `research/scripts/compare_equal_ensembles.py`
- `research/scripts/compare_weighted_vs_equal_trainval.py`
- `research/scripts/evaluate_prod4_stacking.py`
- `research/scripts/report_equal4_deciles.py`
- `research/scripts/report_equal4_deciles_tplus2_open.py`
- `research/scripts/reestimate_nextday_close_profit_trail_models.py`

## Key outputs

- `research/outcomes/models/model_comparison.json`
- `research/outcomes/models/equal4_deciles_time_split.json`
- `research/outcomes/models/equal4_deciles_time_split_tplus2_open.json`
- `research/outcomes/models/equal4_deciles_time_split_tplus2_open_live.json`
- `research/outcomes/models/prod4_stacking_eval.json`
- `research/outcomes/models/weighted_vs_equal_trainval_time_split.json`
