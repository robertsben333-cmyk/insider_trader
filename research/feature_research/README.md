# Feature Research

## Scope

This category covers feature usefulness, descriptive return studies, and data-quality effects.

## Main findings

- `research/outcomes/feature_screen_summary.csv` shows the strongest uplift came from broad candidate additions (`plus_all_candidates`), while filing-time and repeat-insider recency features were modestly helpful.
- `research/outcomes/models/ownership_missingness_summary.csv` indicates ownership fields are not essential; the `no_ownership_feature` variants slightly improved several test metrics versus the baseline.
- The return-distribution and cluster-buy work suggest trade value, prior momentum, and multi-insider context matter more than raw ownership coverage.

## Key scripts

- `research/scripts/screen_low_effort_features.py`
- `research/scripts/analyze_ownership_missingness.py`
- `research/scripts/analyze_cluster_buys.py`
- `research/scripts/analyze_returns_by_price.py`
- `research/scripts/analyze_returns_by_price_gap.py`
- `research/scripts/analyze_returns_by_prior_momentum.py`
- `research/scripts/analyze_returns_by_value_and_pct.py`

## Key outputs

- `research/outcomes/feature_screen_summary.csv`
- `research/outcomes/feature_screen_recommendations.csv`
- `research/outcomes/models/ownership_missingness_summary.csv`
- `research/outcomes/charts/cluster_buy_vs_solo.png`
- `research/outcomes/charts/returns_by_n_insiders.png`
