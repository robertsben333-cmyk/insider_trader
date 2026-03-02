"""
Scatter-plot analysis: insider purchase returns vs stock price
=============================================================
Generates one scatter plot per holding-period horizon (1d, 3d, 5d, 10d).
X-axis = buy price (log scale), Y-axis = return %.
Includes a binned-average trend line for clarity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ───────────────────────────────────────────────────────────
INPUT_CSV = "data/backtest_results.csv"
OUT_DIR = "data/charts"
HORIZONS = [1, 3, 5, 10]

# Price outlier cap — ignore buy_price > $5 000 (very few, distort x-axis)
PRICE_CAP = 5_000

# ── Load & clean ─────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df = df[df["buy_price"].between(0.01, PRICE_CAP)].copy()
print(f"Loaded {len(df):,} trades (after capping price < ${PRICE_CAP:,})")

# ── Binned-average helper ────────────────────────────────────────────

def binned_avg(x, y, n_bins=30):
    """Return (bin_centers, bin_means) using log-spaced bins."""
    log_edges = np.linspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    centres, means = [], []
    for lo, hi in zip(log_edges[:-1], log_edges[1:]):
        mask = (np.log10(x) >= lo) & (np.log10(x) < hi)
        if mask.sum() >= 10:                       # need enough points
            centres.append(10 ** ((lo + hi) / 2))
            means.append(y[mask].mean())
    return np.array(centres), np.array(means)


# ── Plot each horizon ────────────────────────────────────────────────

for h in HORIZONS:
    col = f"return_{h}d_pct"
    sub = df[["buy_price", col]].dropna()

    # Clip extreme returns for better visualisation (keep data, just limit axes)
    ret_lo, ret_hi = sub[col].quantile(0.005), sub[col].quantile(0.995)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter
    ax.scatter(
        sub["buy_price"],
        sub[col],
        s=4, alpha=0.15, color="#5584AC", edgecolors="none", rasterized=True,
    )

    # Binned average trend line
    bx, by = binned_avg(sub["buy_price"].values, sub[col].values, n_bins=35)
    ax.plot(bx, by, color="#FF6B35", linewidth=2.5, label="Binned average", zorder=5)

    # Zero line
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")

    # Axes
    ax.set_xscale("log")
    ax.set_xlim(sub["buy_price"].min() * 0.8, sub["buy_price"].max() * 1.2)
    ax.set_ylim(ret_lo, ret_hi)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:.2f}"))

    ax.set_xlabel("Buy Price (log scale)", fontsize=13)
    ax.set_ylabel(f"{h}-Day Return (%)", fontsize=13)
    ax.set_title(
        f"Insider Officer Purchases — {h}-Day Return vs Stock Price\n"
        f"({len(sub):,} trades, buy 15 min after SEC filing)",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="upper right")

    # Style
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_facecolor("#2B2B2B")
    ax.tick_params(colors="white", labelsize=11)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(fontsize=12, loc="upper right", facecolor="#2B2B2B", edgecolor="white", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, alpha=0.2, color="white")

    out_path = f"{OUT_DIR}/scatter_return_{h}d_vs_price.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path}")

print("\nDone.")
