"""
Cluster-buy analysis + ML-safe aggregation
==========================================
1. Analysis: relationship between cluster buys and returns
   - Binary (cluster vs solo) bar charts at all horizons
   - Returns by n_insiders count (1, 2, 3, 4+)

2. Aggregation: collapse cluster buys to a single row per (ticker, trade_date)
   using the 2nd-filed insider's record, so each event is independent.
   Saves to data/backtest_results_aggregated.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
INPUT_CSV  = "data/backtest_results.csv"
OUTPUT_CSV = "data/backtest_results_aggregated.csv"
OUT_DIR    = "data/charts"
HORIZONS   = [1, 3, 5, 10]
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Load & label ─────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["trade_date_d"]     = pd.to_datetime(df["trade_date"]).dt.date

# Count insiders per (ticker, trade_date)
counts = (
    df.groupby(["ticker", "trade_date_d"])
    .size()
    .reset_index(name="n_insiders")
)
df = df.merge(counts, on=["ticker", "trade_date_d"])
df["cluster_buy"] = df["n_insiders"] >= 2
df["n_insiders_label"] = df["n_insiders"].clip(upper=5).map(
    {1: "1 (solo)", 2: "2", 3: "3", 4: "4", 5: "5+"}
)

print(f"Total rows    : {len(df):,}")
print(f"Solo events   : {(~df.cluster_buy).sum():,}  ({(~df.cluster_buy).mean()*100:.1f}%)")
print(f"Cluster rows  : {df.cluster_buy.sum():,}  ({df.cluster_buy.mean()*100:.1f}%)")
print(f"Cluster events: {counts[counts.n_insiders >= 2]['n_insiders'].count():,}")

# ── Dark style helper ────────────────────────────────────────────────
DARK_BG   = "#1E1E1E"
DARK_AX   = "#2B2B2B"
ORANGE    = "#FF6B35"
BLUE      = "#5584AC"
GREEN     = "#00C851"
RED       = "#FF4444"

def style_ax(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors="white", labelsize=11)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, axis="y", alpha=0.2, color="white")


# ══════════════════════════════════════════════════════════════════════
# CHART 1: Grouped bar – Cluster vs Solo, all horizons
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
fig.suptitle(
    "Insider Officer Purchases — Cluster Buy vs Solo Buy\n"
    f"(solo n={( ~df.cluster_buy).sum():,}  |  cluster n={df.cluster_buy.sum():,} rows)",
    fontsize=15, fontweight="bold", color="white"
)
fig.patch.set_facecolor(DARK_BG)

for ax, metric, label in zip(axes, ["mean", "median"], ["Mean Return (%)", "Median Return (%)"]):
    solo_vals    = []
    cluster_vals = []
    for h in HORIZONS:
        col = f"return_{h}d_pct"
        solo_v    = getattr(df.loc[~df.cluster_buy, col].dropna(), metric)()
        cluster_v = getattr(df.loc[ df.cluster_buy, col].dropna(), metric)()
        solo_vals.append(solo_v)
        cluster_vals.append(cluster_v)
        print(f"  {h}d  solo {metric}: {solo_v:+.2f}%   cluster {metric}: {cluster_v:+.2f}%")

    x = np.arange(len(HORIZONS))
    w = 0.35
    bars1 = ax.bar(x - w/2, solo_vals,    width=w, label="Solo (1 insider)",   color=BLUE,  alpha=0.85)
    bars2 = ax.bar(x + w/2, cluster_vals, width=w, label="Cluster (2+ insiders)", color=ORANGE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}-Day" for h in HORIZONS])
    ax.axhline(0, color="white", linewidth=0.7, alpha=0.4, linestyle="--")
    ax.set_ylabel(label, fontsize=12)
    ax.set_xlabel("Holding Period", fontsize=12)
    ax.legend(facecolor=DARK_AX, edgecolor="white", labelcolor="white", fontsize=11)

    # Annotate bars
    for bar in bars1:
        h_v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h_v + (0.05 if h_v >= 0 else -0.15),
                f"{h_v:+.2f}%", ha="center", va="bottom", fontsize=9, color="white")
    for bar in bars2:
        h_v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h_v + (0.05 if h_v >= 0 else -0.15),
                f"{h_v:+.2f}%", ha="center", va="bottom", fontsize=9, color="white")

    style_ax(fig, ax)

plt.tight_layout()
out = f"{OUT_DIR}/cluster_buy_vs_solo.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"\nSaved {out}")


# ══════════════════════════════════════════════════════════════════════
# CHART 2: Mean & Median returns by n_insiders (1 / 2 / 3 / 4 / 5+)
# ══════════════════════════════════════════════════════════════════════
label_order = ["1 (solo)", "2", "3", "4", "5+"]
colors_n    = [BLUE, "#7FC8A9", GREEN, ORANGE, RED]

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig.suptitle(
    "Returns by Number of Insiders Buying Same Ticker on Same Day",
    fontsize=15, fontweight="bold", color="white"
)
fig.patch.set_facecolor(DARK_BG)

for ax, h in zip(axes.flat, HORIZONS):
    col = f"return_{h}d_pct"
    grp = df.groupby("n_insiders_label")[col]

    means   = grp.mean().reindex(label_order)
    medians = grp.median().reindex(label_order)
    ns      = grp.count().reindex(label_order)

    x = np.arange(len(label_order))
    w = 0.35
    ax.bar(x - w/2, means.values,   width=w, color=ORANGE, alpha=0.85, label="Mean")
    ax.bar(x + w/2, medians.values, width=w, color=BLUE,   alpha=0.85, label="Median")

    ax.set_xticks(x)
    ax.set_xticklabels(label_order)
    ax.axhline(0, color="white", linewidth=0.7, alpha=0.4, linestyle="--")
    ax.set_title(f"{h}-Day Hold", fontsize=13, color="white")
    ax.set_ylabel("Return (%)", fontsize=11)
    ax.set_xlabel("# Insiders Same Day", fontsize=11)
    ax.legend(facecolor=DARK_AX, edgecolor="white", labelcolor="white", fontsize=10)

    # Annotate with n
    for i, (lbl, n_val) in enumerate(zip(label_order, ns.values)):
        ax.text(i, ax.get_ylim()[0] * 0.85,
                f"n={int(n_val) if not np.isnan(n_val) else 0}",
                ha="center", fontsize=8, color="#AAAAAA")

    style_ax(fig, ax)

plt.tight_layout()
out = f"{OUT_DIR}/returns_by_n_insiders.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# CHART 3: Win-rate by n_insiders
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig.suptitle(
    "Win Rate (% Trades > 0%) by Number of Insiders Buying Same Day",
    fontsize=15, fontweight="bold", color="white"
)
fig.patch.set_facecolor(DARK_BG)

for ax, h in zip(axes.flat, HORIZONS):
    col = f"return_{h}d_pct"
    winrate = (
        df.groupby("n_insiders_label")[col]
        .apply(lambda s: (s.dropna() > 0).mean() * 100)
        .reindex(label_order)
    )
    n_counts = df.groupby("n_insiders_label")[col].count().reindex(label_order)

    bar_colors = [GREEN if v >= 50 else RED for v in winrate.fillna(0)]
    bars = ax.bar(label_order, winrate.values, color=bar_colors, alpha=0.85)
    ax.axhline(50, color="white", linewidth=1.2, alpha=0.6, linestyle="--", label="50% baseline")

    ax.set_title(f"{h}-Day Hold", fontsize=13, color="white")
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_xlabel("# Insiders Same Day", fontsize=11)
    ax.set_ylim(0, 80)
    ax.legend(facecolor=DARK_AX, edgecolor="white", labelcolor="white", fontsize=10)

    for bar, n_val in zip(bars, n_counts.values):
        h_v = bar.get_height()
        if not np.isnan(h_v):
            ax.text(bar.get_x() + bar.get_width()/2, h_v + 0.5,
                    f"{h_v:.1f}%", ha="center", va="bottom", fontsize=10, color="white")
        ax.text(bar.get_x() + bar.get_width()/2, 2,
                f"n={int(n_val) if not np.isnan(n_val) else 0}",
                ha="center", va="bottom", fontsize=8, color="#AAAAAA")

    style_ax(fig, ax)

plt.tight_layout()
out = f"{OUT_DIR}/winrate_by_n_insiders.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# AGGREGATION: one row per (ticker, trade_date)
# Strategy: for n==1 keep the only row; for n>=2 take the 2nd-filed row
# ══════════════════════════════════════════════════════════════════════
print("\nAggregating cluster buys...")

# Sort by filing time within each group
df_sorted = df.sort_values(["ticker", "trade_date_d", "transaction_date"])

def pick_representative(group):
    """
    n=1  → return the single row
    n>=2 → return the 2nd row (index 1, 0-based) — the moment we see a *second*
            insider confirm the buy, which is our actual entry signal.
    """
    if len(group) == 1:
        return group.iloc[[0]]
    return group.iloc[[1]]

agg = (
    df_sorted
    .groupby(["ticker", "trade_date_d"], group_keys=False)
    .apply(pick_representative)
    .reset_index(drop=True)
)

# Add cluster metadata columns
agg = agg.merge(
    counts.rename(columns={"n_insiders": "n_insiders_in_cluster"}),
    on=["ticker", "trade_date_d"],
    suffixes=("", "_dup"),
)
# Drop any _dup column from double-merge
agg = agg[[c for c in agg.columns if not c.endswith("_dup")]]

agg.to_csv(OUTPUT_CSV, index=False)
print(f"\nOriginal rows : {len(df):,}")
print(f"Aggregated rows: {len(agg):,}  (one per ticker/trade_date event)")
print(f"Rows removed  : {len(df) - len(agg):,}  (duplicate cluster entries)")
print(f"\nAggregated file: {OUTPUT_CSV}")

# Quick sanity: compare returns of aggregated vs raw
print("\n--- Return comparison: raw vs aggregated ---")
print(f"  {'Horizon':>8}  {'Raw mean':>10}  {'Agg mean':>10}  {'Raw winrate':>12}  {'Agg winrate':>12}")
for h in HORIZONS:
    col = f"return_{h}d_pct"
    raw_m  = df[col].dropna().mean()
    agg_m  = agg[col].dropna().mean()
    raw_wr = (df[col].dropna() > 0).mean() * 100
    agg_wr = (agg[col].dropna() > 0).mean() * 100
    print(f"  {h:>6}d  {raw_m:>+9.2f}%  {agg_m:>+9.2f}%  {raw_wr:>11.1f}%  {agg_wr:>11.1f}%")

print("\nDone.")
