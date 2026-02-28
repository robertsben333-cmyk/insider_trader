"""
Scatter-plot analysis: insider purchase returns vs trade value & % ownership
============================================================================
For each of the 4 holding-period horizons (1d, 3d, 5d, 10d):
  • scatter_return_Xd_vs_value.png  — return vs dollar value of the trade
  • scatter_return_Xd_vs_pct.png    — return vs % of company owned after trade
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ──────────────────────────────────────────────────────────
BACKTEST_CSV = "data/backtest_results.csv"
ORIGINAL_CSV = "data/insider_purchases.csv"
OUT_DIR      = "data/charts"
HORIZONS     = [1, 3, 5, 10]

# ── Load backtest results ────────────────────────────────────────────
df = pd.read_csv(BACKTEST_CSV)
print(f"Backtest rows: {len(df):,}")

# ── Clean 'value' column  (+$31,283 → 31283) ────────────────────────
def clean_money(s):
    if not isinstance(s, str):
        return np.nan
    s = s.replace("+", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

df["value_usd"] = df["value"].apply(clean_money)

# ── Merge owned_pct from original CSV ───────────────────────────────
orig = pd.read_csv(ORIGINAL_CSV, usecols=["transaction_date", "ticker", "owner_name", "owned_pct"])

def clean_pct(s):
    """
    Clean ownership-% strings from OpenInsider.
    Valid range: 0 – 100 %. Negative values and >100% are data artefacts → NaN.
    '>999%' → NaN (clamp out),  '+21%' → 21,  'New'/'n/a' → NaN, '0%' → NaN (too small to be useful as log x).
    """
    if not isinstance(s, str):
        return np.nan
    s = s.strip()
    if s.lower() in ("new", "n/a", ""):
        return np.nan
    # Strip leading > + signs
    s = s.lstrip(">+").replace("%", "").strip()
    try:
        v = float(s)
    except ValueError:
        return np.nan
    # Keep only sensible 0–100 % range; 0 % → NaN (can't log-scale a zero)
    if v <= 0 or v > 100:
        return np.nan
    return v

orig["owned_pct_num"] = orig["owned_pct"].apply(clean_pct)

# Merge on transaction_date + ticker + owner_name
df = df.merge(
    orig[["transaction_date", "ticker", "owner_name", "owned_pct_num"]],
    on=["transaction_date", "ticker", "owner_name"],
    how="left",
)
print(f"After merge – value valid: {df['value_usd'].notna().sum():,}  |  "
      f"owned_pct valid: {df['owned_pct_num'].notna().sum():,}")

# ── Binned-average helper (log-spaced) ──────────────────────────────
def binned_avg(x, y, n_bins=30, log=True):
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if log:
        edges = np.linspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
        get_bin = lambda v: np.log10(v)
    else:
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
        get_bin = lambda v: v
    centres, means = [], []
    bv = get_bin(x)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (bv >= lo) & (bv < hi)
        if mask.sum() >= 10:
            mid = (lo + hi) / 2
            centres.append(10**mid if log else mid)
            means.append(y[mask].mean())
    return np.array(centres), np.array(means)


# ── Generic scatter-plot function ───────────────────────────────────
def make_scatter(x_series, y_series, h, xlabel, title_var, filename,
                 log_x=True, x_formatter=None, x_cap=None):
    sub = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
    if x_cap:
        sub = sub[sub["x"] <= x_cap]
    if len(sub) < 50:
        print(f"  Skipping {filename} — too few rows ({len(sub)})")
        return

    ret_lo = sub["y"].quantile(0.005)
    ret_hi = sub["y"].quantile(0.995)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(sub["x"], sub["y"],
               s=4, alpha=0.15, color="#5584AC", edgecolors="none", rasterized=True)

    bx, by = binned_avg(sub["x"].values, sub["y"].values, n_bins=35, log=log_x)
    ax.plot(bx, by, color="#FF6B35", linewidth=2.5, label="Binned average", zorder=5)

    ax.axhline(0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")

    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(ret_lo, ret_hi)

    if x_formatter:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_formatter))

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(f"{h}-Day Return (%)", fontsize=13)
    ax.set_title(
        f"Insider Officer Purchases — {h}-Day Return vs {title_var}\n"
        f"({len(sub):,} trades, buy 15 min after SEC filing)",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="upper right")

    # Dark theme
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_facecolor("#2B2B2B")
    ax.tick_params(colors="white", labelsize=11)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(fontsize=12, loc="upper right",
              facecolor="#2B2B2B", edgecolor="white", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, alpha=0.2, color="white")

    out_path = f"{OUT_DIR}/{filename}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Generate plots ───────────────────────────────────────────────────
fmt_usd = lambda v, _: (f"${v/1e6:.0f}M" if v >= 1e6
                         else f"${v/1e3:.0f}K" if v >= 1e3
                         else f"${v:.0f}")
fmt_pct = lambda v, _: f"{v:.0f}%"

for h in HORIZONS:
    ret_col = f"return_{h}d_pct"

    # --- Value of trade ---
    make_scatter(
        x_series   = df["value_usd"],
        y_series   = df[ret_col],
        h          = h,
        xlabel     = "Trade Value USD (log scale)",
        title_var  = "Trade Value",
        filename   = f"scatter_return_{h}d_vs_value.png",
        log_x      = True,
        x_formatter= fmt_usd,
        x_cap      = 100_000_000,   # cap at $100 M to avoid a few mega outliers
    )

    # --- % ownership held ---
    make_scatter(
        x_series   = df["owned_pct_num"],
        y_series   = df[ret_col],
        h          = h,
        xlabel     = "% of Company Owned After Trade (log scale)",
        title_var  = "% Ownership Held",
        filename   = f"scatter_return_{h}d_vs_pct.png",
        log_x      = True,          # log scale — most values cluster 0–10 %
        x_formatter= fmt_pct,
        x_cap      = 100,           # already capped in clean_pct
    )

print("\nDone.")
