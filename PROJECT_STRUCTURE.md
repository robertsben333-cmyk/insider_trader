# Project Structure

## Production entrypoints
- `openinsider_scraper.py`
  - Scrapes insider transactions to `backtest/data/insider_purchases.csv`.
- `backtest/scripts/run_backtest.py`
  - Builds backtest outcomes in `backtest/data/`.
- `model_ensemble.py`
  - Unified 4-model training + ensemble policy script.
  - Uses equal weights for `1d/3d`.
  - Uses validation-optimized weights for `5d/10d`.
  - Saves deployable artifacts to `models/prod4/`.

## Folders
- `backtest/`
  - `scripts/`: backtest scripts
  - `data/`: backtest datasets/results/cache
- `models/prod4/`
  - Production model files and ensemble policy/evaluation
- `research/`
  - `scripts/`: ad-hoc analysis/experiments
  - `outcomes/`: research JSON/CSV/charts and legacy model artifacts

## Typical run order
1. `py openinsider_scraper.py`
2. `py backtest/scripts/run_backtest.py`
3. `py model_ensemble.py`
