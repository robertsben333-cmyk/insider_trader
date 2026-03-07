# Research Map

The raw experiments still live in `research/scripts/`, `research/outcomes/`, and `backtest/out/`, but the summaries below group them by decision area so the future app work can separate validated ideas from ad hoc exploration.

## Categories

- `feature_research/README.md`: feature screening, missing-data behavior, and descriptive return studies.
- `model_research/README.md`: model stack comparisons, ensemble work, and the active `T+2 open` signal research.
- `execution_research/README.md`: entry timing, exits, recycling, stop-loss, and path-based trade management.
- `live_transition/README.md`: deployment-oriented work that connects research artifacts to the live runtime.

## Current live takeaway

- The active live bundle is the `day1_tplus2_open` strategy documented in `live_trading/models/day1_tplus2_open/manifest.json`.
- Research supports a four-model ensemble with decile gating and a `sell_at_open_2_trading_days_after_buy` exit rule.
- The new `live_trading/` folder is the runtime boundary for app development; `research/` remains the evidence trail.
