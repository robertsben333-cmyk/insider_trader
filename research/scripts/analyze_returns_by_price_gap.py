"""
Scatter-plot analysis: returns vs (buy_price - insider_price) / insider_price
==============================================================================
The "price gap" is how much the stock had already moved between the insider's
reported trade price and the price we actually buy at (15 min after SEC filing).
A positive gap means the market already ran up before we could buy.
A negative gap means we got in cheaper than the insider (stock dipped after trade).

One scatter plot per horizon (1d, 3d, 5d, 10d).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ──────────────────────────────────────────────────────────
INPUT_CSV = "data/backtest_results.csv"
OUT_DIR   = "data/charts"
HORIZONS  = [1, 3, 5, 10]

# Cap the gap axis: ignore trades where gap > ±50 % (data artefacts / penny stocks)
GAP_CAP = 50.0

# ── Load & compute gap ───────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

def clean_money(s):
    if not isinstance(s, str):
        try:
            return float(s)
        except Exception:
            return np.nan
    s = s.replace("+", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

df["last_price_clean"] = df["last_price"].apply(clean_money)

# price gap = how much stock moved from insider buy → our buy (%)
df["price_gap_pct"] = (
    (df["buy_price"] - df["last_price_clean"]) / df["last_price_clean"] * 100
)

# Drop inf / NaN gaps and apply cap
df = df[np.isfinite(df["price_gap_pct"])].copy()
df_capped = df[df["price_gap_pct"].between(-GAP_CAP, GAP_CAP)].copy()

print(f"Total rows     : {len(df):,}")
print(f"After ±{GAP_CAP}% cap : {len(df_capped):,}")
print(f"\nPrice gap stats (capped):")
print(df_capped["price_gap_pct"].describe().to_string())
print(f"\nNegative gap (we buy cheaper than insider): {(df_capped['price_gap_pct'] < 0).sum():,}")
print(f"Positive gap  (stock ran before we bought): {(df_capped['price_gap_pct'] > 0).sum():,}")

# ── Binned-average helper (linear bins) ────────────────────────────
def binned_avg(x, y, n_bins=40):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centres, means, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= 15:
            centres.append((lo + hi) / 2)
            means.append(y[mask].mean())
            counts.append(mask.sum())
    return np.array(centres), np.array(means), np.array(counts)


# ── Plot each horizon ────────────────────────────────────────────────
for h in HORIZONS:
    ret_col = f"return_{h}d_pct"
    sub = df_capped[["price_gap_pct", ret_col]].dropna()

    ret_lo = sub[ret_col].quantile(0.005)
    ret_hi = sub[ret_col].quantile(0.995)

    fig, ax = plt.subplots(figsize=(13, 7))

    # Scatter
    ax.scatter(
        sub["price_gap_pct"], sub[ret_col],
        s=4, alpha=0.12, color="#5584AC", edgecolors="none", rasterized=True,
    )

    # Binned average
    bx, by, bn = binned_avg(sub["price_gap_pct"].values, sub[ret_col].values, n_bins=40)
    ax.plot(bx, by, color="#FF6B35", linewidth=2.5, label="Binned average", zorder=5)

    # Reference lines
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")
    ax.axvline(0, color="#AAAAAA", linewidth=0.8, alpha=0.5, linestyle=":")

    # Shade regions
    ax.axvspan(-GAP_CAP, 0, alpha=0.04, color="#00C851", label="We buy cheaper than insider")
    ax.axvspan(0, GAP_CAP,  alpha=0.04, color="#FF4444", label="Stock ran before we bought")

    ax.set_xlim(-GAP_CAP, GAP_CAP)
    ax.set_ylim(ret_lo, ret_hi)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlabel("Price Gap: (Our Buy Price − Insider Price) / Insider Price", fontsize=13)
    ax.set_ylabel(f"{h}-Day Return (%)", fontsize=13)
    ax.set_title(
        f"Insider Officer Purchases — {h}-Day Return vs Price Gap\n"
        f"({len(sub):,} trades  |  median gap: {sub['price_gap_pct'].median():+.1f}%)",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right")

    # Dark theme
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_facecolor("#2B2B2B")
    ax.tick_params(colors="white", labelsize=11)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(fontsize=11, loc="upper right",
              facecolor="#2B2B2B", edgecolor="white", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, alpha=0.2, color="white")

    out_path = f"{OUT_DIR}/scatter_return_{h}d_vs_price_gap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path}")

print("\nDone.")
