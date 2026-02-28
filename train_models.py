"""
ML Pipeline: Insider Purchase Return Prediction
================================================
Trains one HistGradientBoostingRegressor per return window (1d, 3d, 5d, 10d).
Features: prior 30d momentum, stock price, % ownership, trade value,
          n_insiders, officer type, filing gap days.

Handles non-linear effects and interactions natively via gradient-boosted trees.
Saves models to models/ and prints summary statistics with focus on
positive-return precision (especially for larger >2% predicted returns).
"""

import os
import json
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train")

# ── Config ──────────────────────────────────────────────────────────
AGGREGATED_CSV = "data/backtest_results_aggregated.csv"
ORIGINAL_CSV   = "data/insider_purchases.csv"
CACHE_DIR      = "data/price_cache"
MODEL_DIR      = "models"
HORIZONS       = [1, 3, 5, 10]
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
LOOKBACK_DAYS  = 30
FETCH_WINDOW   = 50
ET             = ZoneInfo("America/New_York")

FEATURES = [
    "prior_30d_pct",
    "log_buy_price",
    "owned_pct_num",
    "log_value_usd",
    "n_insiders_in_cluster",
    "officer_type_enc",
    "filing_gap_days",
    "is_10pct",
]

MODEL_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.1,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
    categorical_features="from_dtype",
)

OFFICER_KEYWORDS = [
    ("CEO", r"\bCEO\b|Co-CEO"),
    ("CFO", r"\bCFO\b"),
    ("COO", r"\bCOO\b"),
    ("COB", r"\bCOB\b|Chairman"),
    ("Pres", r"\bPres\b|\bPresident\b"),
    ("EVP", r"\bEVP\b"),
    ("SVP", r"\bSVP\b"),
    ("VP", r"\bVP\b"),
    ("GC", r"\bGC\b|General Counsel"),
]

# ── Helpers ──────────────────────────────────────────────────────────

def clean_money(s) -> float:
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


def clean_pct(s) -> float:
    if not isinstance(s, str):
        return np.nan
    s = s.strip()
    if s.lower() in ("new", "n/a", ""):
        return np.nan
    s = s.lstrip(">+").replace("%", "").strip()
    try:
        v = float(s)
    except ValueError:
        return np.nan
    if v <= 0 or v > 100:
        return np.nan
    return v


def extract_officer_type(title: str) -> str:
    import re
    if not isinstance(title, str):
        return "Other"
    for name, pattern in OFFICER_KEYWORDS:
        if re.search(pattern, title, re.IGNORECASE):
            return name
    return "Other"


def find_close_on_or_before(bars: list, target_date: date):
    result = None
    for bar in bars:
        bar_date = pd.Timestamp(bar["t"], unit="ms", tz="UTC").tz_convert(ET).date()
        if bar_date <= target_date:
            result = bar["c"]
        else:
            break
    return result


# ── Step 1: Load & Merge ────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    logger.info("Loading aggregated dataset...")
    df = pd.read_csv(AGGREGATED_CSV)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["trade_date_d"] = pd.to_datetime(df["trade_date"]).dt.date
    logger.info(f"  Rows: {len(df):,}")

    # ── Merge owned_pct ──
    logger.info("Merging owned_pct from original CSV...")
    orig = pd.read_csv(ORIGINAL_CSV, usecols=["transaction_date", "ticker", "owner_name", "owned_pct"])
    orig["transaction_date"] = pd.to_datetime(orig["transaction_date"]).astype(str)
    df["_merge_txn"] = df["transaction_date"].astype(str)
    orig["owned_pct_num"] = orig["owned_pct"].apply(clean_pct)
    orig = orig.drop_duplicates(["transaction_date", "ticker", "owner_name"])
    df = df.merge(
        orig[["transaction_date", "ticker", "owner_name", "owned_pct_num"]].rename(
            columns={"transaction_date": "_merge_txn"}
        ),
        on=["_merge_txn", "ticker", "owner_name"],
        how="left",
    )
    df.drop(columns=["_merge_txn"], inplace=True)
    logger.info(f"  owned_pct valid: {df['owned_pct_num'].notna().sum():,} / {len(df):,}")

    # ── Clean value_usd ──
    df["value_usd"] = df["value"].apply(clean_money)
    df["last_price_clean"] = df["last_price"].apply(clean_money)

    # ── Compute prior_30d_pct from price cache ──
    logger.info("Computing prior_30d_pct from price cache...")
    cache_dir = Path(CACHE_DIR)
    unique_pairs = df[["ticker", "trade_date_d", "buy_price"]].drop_duplicates(["ticker", "trade_date_d"])

    prior_map = {}
    hits = 0
    misses = 0
    for _, row in unique_pairs.iterrows():
        ticker = row["ticker"]
        td = row["trade_date_d"]
        buy_px = row["buy_price"]

        from_d = td - timedelta(days=FETCH_WINDOW)
        fs = from_d.strftime("%Y-%m-%d")
        ts = td.strftime("%Y-%m-%d")
        cache_path = cache_dir / f"{ticker}_lkbk_{fs}_{ts}.json"

        if not cache_path.exists():
            misses += 1
            prior_map[(ticker, td)] = np.nan
            continue

        try:
            with open(cache_path) as fh:
                bars = json.load(fh)
        except Exception:
            misses += 1
            prior_map[(ticker, td)] = np.nan
            continue

        target_date = td - timedelta(days=LOOKBACK_DAYS)
        close_30d = find_close_on_or_before(bars, target_date)

        if close_30d is None or close_30d <= 0 or buy_px is None or buy_px <= 0:
            misses += 1
            prior_map[(ticker, td)] = np.nan
            continue

        prior_map[(ticker, td)] = (buy_px - close_30d) / close_30d * 100
        hits += 1

    df["prior_30d_pct"] = df.apply(lambda r: prior_map.get((r["ticker"], r["trade_date_d"]), np.nan), axis=1)
    logger.info(f"  prior_30d_pct: {hits:,} hits, {misses:,} misses")

    return df


# ── Step 2: Feature Engineering ──────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple:
    logger.info("Engineering features...")
    caps = {}

    # log_buy_price: winsorize at 99.5th, then log1p
    cap_bp = df["buy_price"].quantile(0.995)
    caps["buy_price_cap"] = float(cap_bp)
    df["log_buy_price"] = np.log1p(df["buy_price"].clip(upper=cap_bp))

    # log_value_usd: zeros → NaN, winsorize, log1p
    df.loc[df["value_usd"] <= 0, "value_usd"] = np.nan
    cap_val = df["value_usd"].quantile(0.995)
    caps["value_usd_cap"] = float(cap_val)
    df["log_value_usd"] = np.log1p(df["value_usd"].clip(upper=cap_val))

    # prior_30d_pct: clip
    df["prior_30d_pct"] = df["prior_30d_pct"].clip(-150, 300)

    # officer_type
    df["officer_type"] = df["title"].apply(extract_officer_type)
    logger.info(f"  Officer types: {df['officer_type'].value_counts().to_dict()}")

    # Ordinal encode officer_type → set category dtype for HGBR
    cat_map = {name: i for i, (name, _) in enumerate(OFFICER_KEYWORDS)}
    cat_map["Other"] = len(OFFICER_KEYWORDS)
    df["officer_type_enc"] = df["officer_type"].map(cat_map).astype("category")

    # is_10pct
    df["is_10pct"] = df["title"].str.contains("10%", na=False).astype(int)

    # n_insiders_in_cluster: ensure int
    if "n_insiders_in_cluster" not in df.columns:
        if "n_insiders" in df.columns:
            df["n_insiders_in_cluster"] = df["n_insiders"]
        else:
            df["n_insiders_in_cluster"] = 1

    logger.info(f"  Feature matrix ready: {len(df):,} rows x {len(FEATURES)} features")
    return df, FEATURES, caps


# ── Step 3: Train One Window ─────────────────────────────────────────

def train_one_window(df: pd.DataFrame, features: list, horizon: int) -> dict:
    target_col = f"return_{horizon}d_pct"

    # Drop rows without target
    sub = df[features + [target_col]].dropna(subset=[target_col]).copy()

    # Winsorize target at 1%-99%
    lo = sub[target_col].quantile(0.01)
    hi = sub[target_col].quantile(0.99)
    sub[target_col] = sub[target_col].clip(lo, hi)

    X = sub[features]
    y = sub[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    logger.info(f"  {horizon}d: train={len(X_train):,}, test={len(X_test):,}, target clip=[{lo:.2f}, {hi:.2f}]")

    # Fit
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    n_iter = model.n_iter_

    # Predict
    y_pred = model.predict(X_test)

    # ── Standard metrics ──
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # ── Directional accuracy ──
    dir_acc = (np.sign(y_pred) == np.sign(y_test.values)).mean()

    # ── Decile spread ──
    order = np.argsort(y_pred)[::-1]
    n10 = max(1, len(order) // 10)
    top_dec_actual = y_test.values[order[:n10]].mean()
    bot_dec_actual = y_test.values[order[-n10:]].mean()
    dec_spread = top_dec_actual - bot_dec_actual

    # ── Feature importance (permutation-based) ──
    perm_imp = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    fi = dict(zip(features, perm_imp.importances_mean))
    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])

    # ── Positive-return precision metrics ──
    # When model predicts positive, how often is actual positive?
    pred_pos_mask = y_pred > 0
    if pred_pos_mask.sum() > 0:
        prec_pos = (y_test.values[pred_pos_mask] > 0).mean()
        n_pred_pos = int(pred_pos_mask.sum())
    else:
        prec_pos = 0.0
        n_pred_pos = 0

    # When model predicts >2%, how often is actual positive?
    pred_gt2_mask = y_pred > 2.0
    if pred_gt2_mask.sum() > 0:
        prec_gt2_pos = (y_test.values[pred_gt2_mask] > 0).mean()
        prec_gt2_gt2 = (y_test.values[pred_gt2_mask] > 2.0).mean()
        mean_actual_gt2 = y_test.values[pred_gt2_mask].mean()
        n_pred_gt2 = int(pred_gt2_mask.sum())
    else:
        prec_gt2_pos = 0.0
        prec_gt2_gt2 = 0.0
        mean_actual_gt2 = 0.0
        n_pred_gt2 = 0

    # When model predicts >5%, how often is actual positive?
    pred_gt5_mask = y_pred > 5.0
    if pred_gt5_mask.sum() > 0:
        prec_gt5_pos = (y_test.values[pred_gt5_mask] > 0).mean()
        mean_actual_gt5 = y_test.values[pred_gt5_mask].mean()
        n_pred_gt5 = int(pred_gt5_mask.sum())
    else:
        prec_gt5_pos = 0.0
        mean_actual_gt5 = 0.0
        n_pred_gt5 = 0

    # ── Quintile analysis ──
    quintile_stats = []
    quintile_idx = np.argsort(y_pred)
    q_size = len(quintile_idx) // 5
    for q in range(5):
        start = q * q_size
        end = (q + 1) * q_size if q < 4 else len(quintile_idx)
        q_idx = quintile_idx[start:end]
        q_actual = y_test.values[q_idx]
        quintile_stats.append({
            "quintile": q + 1,
            "mean_pred": float(y_pred[q_idx].mean()),
            "mean_actual": float(q_actual.mean()),
            "win_rate": float((q_actual > 0).mean()),
            "n": int(len(q_idx)),
        })

    # ── Save model ──
    model_path = Path(MODEL_DIR) / f"model_{horizon}d.pkl"
    joblib.dump(model, model_path)

    return {
        "horizon": horizon,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_iter": n_iter,
        "target_clip_lo": float(lo),
        "target_clip_hi": float(hi),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "dir_acc": float(dir_acc),
        "top_dec_mean": float(top_dec_actual),
        "bot_dec_mean": float(bot_dec_actual),
        "dec_spread": float(dec_spread),
        "feature_importance": fi_sorted,
        "prec_pos": float(prec_pos),
        "n_pred_pos": n_pred_pos,
        "prec_gt2_pos": float(prec_gt2_pos),
        "prec_gt2_gt2": float(prec_gt2_gt2),
        "mean_actual_gt2": float(mean_actual_gt2),
        "n_pred_gt2": n_pred_gt2,
        "prec_gt5_pos": float(prec_gt5_pos),
        "mean_actual_gt5": float(mean_actual_gt5),
        "n_pred_gt5": n_pred_gt5,
        "quintile_stats": quintile_stats,
        "model_path": str(model_path),
    }


# ── Step 4: Summary Report ──────────────────────────────────────────

def print_summary(all_metrics: dict):
    w = 74
    print("\n" + "=" * w)
    print("  INSIDER PURCHASE ML PIPELINE - RESULTS SUMMARY")
    print("=" * w)
    print(f"  Random split : {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test (seed={RANDOM_STATE})")
    print(f"  Model        : HistGradientBoostingRegressor")
    print(f"  Features ({len(FEATURES)}): {', '.join(FEATURES)}")

    for h in HORIZONS:
        m = all_metrics[h]
        print(f"\n  {'='*68}")
        print(f"  {h}-DAY RETURN MODEL")
        print(f"  {'='*68}")
        print(f"    Train/test       : {m['n_train']:,} / {m['n_test']:,}")
        print(f"    Iterations       : {m['n_iter']}  (early stopped)")
        print(f"    Target clip      : [{m['target_clip_lo']:+.2f}%, {m['target_clip_hi']:+.2f}%]")
        print()
        print(f"    -- Regression Metrics --")
        print(f"    R2               : {m['r2']:.4f}")
        print(f"    MAE              : {m['mae']:.2f} pp")
        print(f"    RMSE             : {m['rmse']:.2f} pp")
        print()
        print(f"    -- Directional Accuracy --")
        print(f"    Overall          : {m['dir_acc']*100:.1f}%")
        print()
        print(f"    -- Positive-Return Precision (KEY METRICS) --")
        print(f"    Pred > 0%  : {m['n_pred_pos']:,} trades -> {m['prec_pos']*100:.1f}% actual > 0%")
        print(f"    Pred > 2%  : {m['n_pred_gt2']:,} trades -> {m['prec_gt2_pos']*100:.1f}% actual > 0%, "
              f"{m['prec_gt2_gt2']*100:.1f}% actual > 2%, mean actual: {m['mean_actual_gt2']:+.2f}%")
        print(f"    Pred > 5%  : {m['n_pred_gt5']:,} trades -> {m['prec_gt5_pos']*100:.1f}% actual > 0%, "
              f"mean actual: {m['mean_actual_gt5']:+.2f}%")
        print()
        print(f"    -- Decile Analysis --")
        print(f"    Top 10% pred  -> mean actual: {m['top_dec_mean']:+.2f}%")
        print(f"    Bot 10% pred  -> mean actual: {m['bot_dec_mean']:+.2f}%")
        print(f"    Spread         : {m['dec_spread']:+.2f} pp")
        print()
        print(f"    -- Quintile Breakdown (sorted by prediction) --")
        print(f"    {'Q':>4}  {'Mean Pred':>10}  {'Mean Actual':>12}  {'Win Rate':>9}  {'n':>6}")
        for qs in m["quintile_stats"]:
            print(f"    {qs['quintile']:>4}  {qs['mean_pred']:>+9.2f}%  {qs['mean_actual']:>+11.2f}%  "
                  f"{qs['win_rate']*100:>8.1f}%  {qs['n']:>6,}")
        print()
        print(f"    -- Feature Importance --")
        for fname, fimp in m["feature_importance"]:
            bar = "#" * int(fimp * 50)
            print(f"    {fname:<25} {fimp:.3f}  {bar}")

    print("\n" + "=" * w + "\n")


# ── Step 5: Save Metadata ───────────────────────────────────────────

def save_metadata(all_metrics: dict, features: list, caps: dict):
    from datetime import datetime
    import sklearn
    meta = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "features": features,
        "model_params": MODEL_PARAMS,
        "winsorization_caps": caps,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "windows": {},
    }
    for h in HORIZONS:
        m = all_metrics[h]
        meta["windows"][f"{h}d"] = {
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "n_iter": m["n_iter"],
            "target_clip": [m["target_clip_lo"], m["target_clip_hi"]],
            "r2": m["r2"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "dir_acc": m["dir_acc"],
            "dec_spread": m["dec_spread"],
            "prec_pos": m["prec_pos"],
            "prec_gt2_pos": m["prec_gt2_pos"],
            "prec_gt2_gt2": m["prec_gt2_gt2"],
            "prec_gt5_pos": m["prec_gt5_pos"],
            "feature_importance": dict(m["feature_importance"]),
            "quintile_stats": m["quintile_stats"],
            "model_path": m["model_path"],
        }
    out_path = Path(MODEL_DIR) / "model_metadata.json"
    # Need custom serializer for category dtype
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Metadata saved to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    df = load_and_merge()
    df, features, caps = engineer_features(df)

    all_metrics = {}
    for h in HORIZONS:
        logger.info(f"\nTraining {h}-day model...")
        all_metrics[h] = train_one_window(df, features, h)

    print_summary(all_metrics)
    save_metadata(all_metrics, features, caps)
    logger.info("All done.")


if __name__ == "__main__":
    main()
