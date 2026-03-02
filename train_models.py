"""
ML Pipeline: Insider Purchase Return Prediction
================================================
Trains one HistGradientBoostingRegressor per return window (1d, 3d, 5d, 10d).
For 3d/5d/10d targets, returns are converted to per-day compounded equivalents.
Features: prior 30d momentum, stock price, % ownership, trade value,
          n_insiders, officer type, filing gap days.

Handles non-linear effects and interactions natively via gradient-boosted trees.
Saves models to models/ and prints summary statistics with focus on
positive-return precision (especially for larger >2% predicted returns).
"""

import os
import json
import logging
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance

load_dotenv()

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train")

# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Config ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
AGGREGATED_CSV  = "backtest/data/backtest_results_aggregated.csv"
ORIGINAL_CSV    = "backtest/data/insider_purchases.csv"
CACHE_DIR       = "backtest/data/price_cache"
SECTOR_CACHE    = "data/sector_cache.csv"
TICKER_META_CACHE = "data/ticker_meta_cache.csv"
MODEL_DIR       = "models"
HORIZONS        = [1, 3, 5, 10]
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
LOOKBACK_DAYS   = 30
FETCH_WINDOW    = 50
SECTOR_WORKERS  = 10
ET              = ZoneInfo("America/New_York")

FEATURES = [
    "prior_30d_pct",
    "prior_5d_pct",
    "prior_10d_pct",
    "prior_30d_vol",
    "prior_10d_vol",
    "prior_5d_vol",
    "log_buy_price",
    "price_drift_filing_pct",
    "owned_pct_num",
    "log_value_usd",
    "n_insiders_in_cluster",
    "is_officer",
    "is_director",
    "is_10pct",
    "n_insiders_x_officer",
    "n_insiders_x_director",
    "n_insiders_x_10pct",
    "officer_type_enc",
    "filing_gap_days",
    "insider_prior_buys_log",
    "days_since_last_buy_any",
    "days_since_last_buy_same_ticker",
    "same_ticker_insider_count_365d",
    "filing_hour_et",
    "market_type_enc",
    "sector_enc",
]

# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ SIC ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ GICS sector mapping ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
# Ranges are non-overlapping and ordered; first match wins.
_SIC_RANGES = [
    (100,  999,  "Consumer Staples"),       # agriculture
    (1000, 1299, "Materials"),              # mining (non-oil)
    (1300, 1399, "Energy"),                 # oil & gas extraction
    (1400, 1499, "Materials"),              # non-metallic minerals
    (1500, 1799, "Industrials"),            # construction
    (2000, 2199, "Consumer Staples"),       # food & tobacco
    (2200, 2399, "Consumer Discretionary"), # textiles & apparel
    (2400, 2829, "Materials"),              # lumber, paper, chemicals
    (2830, 2836, "Health Care"),            # pharmaceuticals
    (2837, 2899, "Materials"),              # other chemicals
    (2900, 2999, "Energy"),                 # petroleum refining
    (3000, 3399, "Materials"),              # rubber, glass, metals
    (3400, 3599, "Industrials"),            # fabricated metals, machinery
    (3600, 3699, "Information Technology"), # electronic equipment
    (3700, 3716, "Consumer Discretionary"), # motor vehicles
    (3717, 3799, "Industrials"),            # other transport equipment
    (3800, 3851, "Health Care"),            # lab & medical instruments
    (3852, 3999, "Industrials"),            # misc manufacturing
    (4000, 4799, "Industrials"),            # transportation
    (4800, 4899, "Communication Services"), # telecom
    (4900, 4999, "Utilities"),
    (5000, 5199, "Industrials"),            # wholesale
    (5200, 5999, "Consumer Discretionary"), # retail
    (6000, 6499, "Financials"),             # banking, insurance, securities
    (6500, 6799, "Real Estate"),
    (6800, 6999, "Financials"),             # holding companies
    (7000, 7369, "Consumer Discretionary"), # hotels, personal services
    (7370, 7379, "Information Technology"), # computer programming & data processing
    (7380, 7999, "Consumer Discretionary"), # misc services, entertainment
    (8000, 8099, "Health Care"),            # health services
    (8100, 8999, "Industrials"),            # legal, engineering, misc services
    (9100, 9999, "Industrials"),            # public administration
]

SECTOR_ORDER = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials", "Information Technology",
    "Communication Services", "Utilities", "Real Estate", "Unknown",
]
MARKET_TYPE_ORDER = ["NYSE", "NASDAQ", "AMEX", "OTC", "NON_TRADABLE"]


def sic_to_sector(sic_code) -> str:
    if not sic_code:
        return "Unknown"
    try:
        s = int(sic_code)
    except (ValueError, TypeError):
        return "Unknown"
    for lo, hi, sector in _SIC_RANGES:
        if lo <= s <= hi:
            return sector
    return "Unknown"


def normalize_market_type(primary_exchange, active_flag) -> str:
    if active_flag is False:
        return "NON_TRADABLE"
    txt = "" if primary_exchange is None else str(primary_exchange).upper()
    if ("NASDAQ" in txt) or ("XNAS" in txt):
        return "NASDAQ"
    if ("AMEX" in txt) or ("XASE" in txt) or ("NYSE AMERICAN" in txt):
        return "AMEX"
    if ("NYSE" in txt) or ("XNYS" in txt):
        return "NYSE"
    if ("OTC" in txt) or ("OTCM" in txt) or ("PINK" in txt):
        return "OTC"
    return "NON_TRADABLE"

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
OFFICER_TITLE_PATTERN = re.compile(
    r"\b(COB|Chairman|CEO|Co-CEO|Pres|President|COO|CFO|GC|VP|SVP|EVP)\b",
    re.IGNORECASE,
)
DIRECTOR_TITLE_PATTERN = re.compile(r"\bdirector\b|\bdir\b", re.IGNORECASE)
TEN_PCT_TITLE_PATTERN = re.compile(r"10%|10 percent|10pct|ten percent", re.IGNORECASE)

# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Helpers ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

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
    if not isinstance(title, str):
        return "Other"
    for name, pattern in OFFICER_KEYWORDS:
        if re.search(pattern, title, re.IGNORECASE):
            return name
    return "Other"


def classify_insider_type(title: str) -> str:
    """
    Mutually-exclusive type bucket used as fixed effects:
      TenPct > Director > Officer > Other.
    """
    txt = "" if not isinstance(title, str) else title
    if TEN_PCT_TITLE_PATTERN.search(txt):
        return "TenPct"
    if DIRECTOR_TITLE_PATTERN.search(txt):
        return "Director"
    if OFFICER_TITLE_PATTERN.search(txt):
        return "Officer"
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

def apply_per_day_adjusted_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-day total return targets into per-day compounded returns."""
    out = df.copy()

    marker_col = "returns_are_per_day_adjusted"
    needs_adjust = pd.Series(True, index=out.index)
    if marker_col in out.columns:
        marker_raw = out[marker_col].astype(str).str.strip().str.lower()
        marker_true = marker_raw.isin({"1", "true", "t", "yes", "y"})
        needs_adjust = ~marker_true
        if not bool(needs_adjust.any()):
            logger.info("Detected pre-adjusted multi-day return targets; skipping conversion.")
            return out

    changed = []
    for h in [3, 5, 10]:
        col = f"return_{h}d_pct"
        if col not in out.columns:
            continue
        raw = pd.to_numeric(out[col], errors="coerce") / 100.0
        base = 1.0 + raw
        valid = base >= 0.0
        row_mask = valid & needs_adjust
        adj = pd.Series(np.nan, index=out.index, dtype=float)
        adj.loc[row_mask] = (base.loc[row_mask] ** (1.0 / float(h))) - 1.0
        out.loc[needs_adjust, col] = adj.loc[needs_adjust] * 100.0
        changed.append(col)

    if changed:
        out[marker_col] = True
        logger.info("Applied per-day compounded target adjustment to: %s", ", ".join(changed))
    return out

# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Step 1: Load & Merge ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

def load_and_merge() -> pd.DataFrame:
    logger.info("Loading aggregated dataset...")
    df = pd.read_csv(AGGREGATED_CSV)
    df = apply_per_day_adjusted_targets(df)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["trade_date_d"] = pd.to_datetime(df["trade_date"]).dt.date
    logger.info(f"  Rows: {len(df):,}")

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Merge owned_pct ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
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

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Clean value_usd ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    df["value_usd"] = df["value"].apply(clean_money)
    df["last_price_clean"] = df["last_price"].apply(clean_money)

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Compute prior momentum + volatility features from price cache ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    logger.info("Computing prior momentum and volatility features from price cache...")
    cache_dir = Path(CACHE_DIR)
    unique_pairs = df[["ticker", "trade_date_d", "buy_price"]].drop_duplicates(["ticker", "trade_date_d"])

    prior_30d_map  = {}
    prior_10d_map  = {}
    prior_5d_map   = {}
    vol_30d_map    = {}
    vol_10d_map    = {}
    vol_5d_map     = {}
    hits = 0
    misses = 0
    for _, row in unique_pairs.iterrows():
        ticker = row["ticker"]
        td     = row["trade_date_d"]
        buy_px = row["buy_price"]

        from_d = td - timedelta(days=FETCH_WINDOW)
        cache_path = cache_dir / f"{ticker}_lkbk_{from_d.strftime('%Y-%m-%d')}_{td.strftime('%Y-%m-%d')}.json"

        nan6 = lambda: (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        if not cache_path.exists():
            misses += 1
            prior_30d_map[(ticker, td)] = prior_10d_map[(ticker, td)] = prior_5d_map[(ticker, td)] = np.nan
            vol_30d_map[(ticker, td)]   = vol_10d_map[(ticker, td)]   = vol_5d_map[(ticker, td)]   = np.nan
            continue

        try:
            with open(cache_path) as fh:
                bars = json.load(fh)
        except Exception:
            misses += 1
            prior_30d_map[(ticker, td)] = prior_10d_map[(ticker, td)] = prior_5d_map[(ticker, td)] = np.nan
            vol_30d_map[(ticker, td)]   = vol_10d_map[(ticker, td)]   = vol_5d_map[(ticker, td)]   = np.nan
            continue

        if not buy_px or buy_px <= 0:
            misses += 1
            prior_30d_map[(ticker, td)] = prior_10d_map[(ticker, td)] = prior_5d_map[(ticker, td)] = np.nan
            vol_30d_map[(ticker, td)]   = vol_10d_map[(ticker, td)]   = vol_5d_map[(ticker, td)]   = np.nan
            continue

        # Precompute bar dates once per file
        bar_dates = [
            pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert(ET).date()
            for b in bars
        ]

        def _mom(days):
            target = td - timedelta(days=days)
            c = None
            for bd, b in zip(bar_dates, bars):
                if bd <= target:
                    c = b["c"]
                elif bd > target:
                    break
            return (buy_px - c) / c * 100 if (c and c > 0) else np.nan

        def _vol(days):
            cutoff = td - timedelta(days=days)
            closes = [b["c"] for bd, b in zip(bar_dates, bars) if bd >= cutoff]
            if len(closes) < 2:
                return np.nan
            a = np.array(closes, dtype=float)
            log_rets = np.diff(np.log(a))
            return float(np.std(log_rets, ddof=1) * np.sqrt(252))

        prior_30d_map[(ticker, td)] = _mom(LOOKBACK_DAYS)
        prior_10d_map[(ticker, td)] = _mom(10)
        prior_5d_map[(ticker, td)]  = _mom(5)
        vol_30d_map[(ticker, td)]   = _vol(30)
        vol_10d_map[(ticker, td)]   = _vol(10)
        vol_5d_map[(ticker, td)]    = _vol(5)
        hits += 1

    def _map_col(m):
        return df.apply(lambda r: m.get((r["ticker"], r["trade_date_d"]), np.nan), axis=1)

    df["prior_30d_pct"] = _map_col(prior_30d_map)
    df["prior_10d_pct"] = _map_col(prior_10d_map)
    df["prior_5d_pct"]  = _map_col(prior_5d_map)
    df["prior_30d_vol"] = _map_col(vol_30d_map)
    df["prior_10d_vol"] = _map_col(vol_10d_map)
    df["prior_5d_vol"]  = _map_col(vol_5d_map)
    logger.info(f"  momentum+vol features: {hits:,} hits, {misses:,} misses")

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Price drift from insider's purchase price to our entry ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    # last_price_clean ÃƒÂ¢Ã¢â‚¬Â°Ã‹â€  price the insider paid (from Form 4).
    # buy_price = market price ~15 min after open on the day after filing.
    # Captures how much of the "pop" on the announcement has already happened.
    # Flag large discrepancies (>100% or <-50%) as bad data ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ NaN.
    logger.info("Computing price_drift_filing_pct...")
    raw_drift = (df["buy_price"] - df["last_price_clean"]) / df["last_price_clean"] * 100
    bad_data  = (df["last_price_clean"] <= 0) | (raw_drift > 100) | (raw_drift < -50)
    df["price_drift_filing_pct"] = raw_drift.where(~bad_data, other=np.nan)
    valid_drift = df["price_drift_filing_pct"].notna().sum()
    logger.info(f"  price_drift_filing_pct: {valid_drift:,} valid ({100*valid_drift/len(df):.1f}%)")

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Insider repeat-buyer features ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    # Use the full insider_purchases.csv (2001-present) to count prior purchases
    # for each (owner_name) and (owner_name, ticker) STRICTLY before this trade_date.
    # Only purchases before the current trade date are counted ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ no look-ahead leakage.
    logger.info("Computing insider repeat-buyer features...")
    import bisect
    from collections import defaultdict

    hist = pd.read_csv(ORIGINAL_CSV, usecols=["owner_name", "ticker", "trade_date"])
    hist["trade_date_d"] = pd.to_datetime(hist["trade_date"]).dt.date

    # Build sorted date lists per owner (any ticker) and per (owner, ticker)
    owner_dates = defaultdict(list)
    owner_ticker_dates = defaultdict(list)
    for rec in hist[["owner_name", "ticker", "trade_date_d"]].itertuples(index=False):
        owner_dates[rec.owner_name].append(rec.trade_date_d)
        owner_ticker_dates[(rec.owner_name, rec.ticker)].append(rec.trade_date_d)

    for k in owner_dates:
        owner_dates[k].sort()
    for k in owner_ticker_dates:
        owner_ticker_dates[k].sort()

    def _days_since_prev(sorted_dates, cur_date):
        if pd.isna(cur_date):
            return np.nan
        idx = bisect.bisect_left(sorted_dates, cur_date)
        if idx <= 0:
            return np.nan
        prev = sorted_dates[idx - 1]
        return float((cur_date - prev).days)

    def _count_recent(sorted_dates, cur_date, window_days):
        if pd.isna(cur_date):
            return np.nan
        left = bisect.bisect_left(sorted_dates, cur_date - timedelta(days=window_days))
        right = bisect.bisect_left(sorted_dates, cur_date)
        return float(max(0, right - left))

    df["insider_prior_buys"] = df.apply(
        lambda r: bisect.bisect_left(owner_dates[r["owner_name"]], r["trade_date_d"]),
        axis=1,
    )
    df["insider_bought_ticker"] = df.apply(
        lambda r: int(bisect.bisect_left(owner_ticker_dates[(r["owner_name"], r["ticker"])], r["trade_date_d"]) > 0),
        axis=1,
    )
    df["days_since_last_buy_any"] = df.apply(
        lambda r: _days_since_prev(owner_dates[r["owner_name"]], r["trade_date_d"]),
        axis=1,
    )
    df["days_since_last_buy_same_ticker"] = df.apply(
        lambda r: _days_since_prev(owner_ticker_dates[(r["owner_name"], r["ticker"])], r["trade_date_d"]),
        axis=1,
    )
    df["same_ticker_insider_count_365d"] = df.apply(
        lambda r: _count_recent(
            owner_ticker_dates[(r["owner_name"], r["ticker"])],
            r["trade_date_d"],
            365,
        ),
        axis=1,
    )
    logger.info(f"  repeat buyers (any ticker): {(df['insider_prior_buys'] > 0).sum():,}")
    logger.info(f"  repeat buyers (same ticker): {df['insider_bought_ticker'].sum():,}")
    logger.info(f"  days_since_last_buy_any non-null: {df['days_since_last_buy_any'].notna().sum():,}")
    logger.info(f"  days_since_last_buy_same_ticker non-null: {df['days_since_last_buy_same_ticker'].notna().sum():,}")

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Ticker metadata (sector + market type) from Polygon reference API ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    logger.info("Loading ticker metadata from Polygon...")
    tickers = df["ticker"].unique().tolist()
    meta_map = _load_ticker_metadata(tickers)
    df["sector"] = df["ticker"].map(lambda t: meta_map.get(t, {}).get("sector", "Unknown")).fillna("Unknown")
    df["market_type"] = df["ticker"].map(
        lambda t: meta_map.get(t, {}).get("market_type", "NON_TRADABLE")
    ).fillna("NON_TRADABLE")
    df["is_tradable"] = (
        df["ticker"].map(lambda t: int(meta_map.get(t, {}).get("is_tradable", 0))).fillna(0).astype(int)
    )
    dist = df["sector"].value_counts().to_dict()
    mdist = df["market_type"].value_counts().to_dict()
    logger.info(f"  Sector distribution: {dist}")
    logger.info(f"  Market type distribution: {mdist}")
    logger.info(f"  Tradable rows: {int(df['is_tradable'].sum()):,} / {len(df):,}")

    return df


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Polygon sector fetch (with local CSV cache) ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

_thread_local = threading.local()

def _polygon_client():
    if not hasattr(_thread_local, "client"):
        api_key = os.getenv("POLYGON_API_KEY")
        _thread_local.client = RESTClient(api_key=api_key, retries=3)
    return _thread_local.client

def _fetch_one_ticker_meta(ticker):
    try:
        details = _polygon_client().get_ticker_details(ticker)
        sic = getattr(details, "sic_code", None)
        primary_exchange = getattr(details, "primary_exchange", None)
        active_flag = getattr(details, "active", None)
        market_type = normalize_market_type(primary_exchange, active_flag)
        is_tradable = int(market_type != "NON_TRADABLE")
        return ticker, {
            "sector": sic_to_sector(sic),
            "market_type": market_type,
            "is_tradable": is_tradable,
        }
    except Exception:
        return ticker, {"sector": "Unknown", "market_type": "NON_TRADABLE", "is_tradable": 0}


def _load_ticker_metadata(tickers: list) -> dict:
    """
    Return:
      {ticker: {"sector": str, "market_type": str, "is_tradable": 0/1}}
    Fetches missing entries from Polygon and persists cache.
    """
    meta_path = Path(TICKER_META_CACHE)
    sec_path = Path(SECTOR_CACHE)

    cached = {}
    if meta_path.exists():
        try:
            meta_df = pd.read_csv(meta_path, dtype=str).fillna("")
            for r in meta_df.itertuples(index=False):
                tr = str(getattr(r, "is_tradable", "0")).strip().lower()
                is_tradable = 1 if tr in {"1", "true", "t", "yes", "y"} else 0
                cached[getattr(r, "ticker")] = {
                    "sector": getattr(r, "sector", "Unknown") or "Unknown",
                    "market_type": getattr(r, "market_type", "NON_TRADABLE") or "NON_TRADABLE",
                    "is_tradable": is_tradable,
                }
        except Exception:
            cached = {}

    # Backward compatibility: seed sector from old sector cache if present.
    if sec_path.exists():
        try:
            sec_df = pd.read_csv(sec_path, dtype=str).fillna("")
            for r in sec_df.itertuples(index=False):
                t = getattr(r, "ticker")
                s = getattr(r, "sector", "Unknown") or "Unknown"
                if t not in cached:
                    cached[t] = {"sector": s, "market_type": "", "is_tradable": 0}
                elif not cached[t].get("sector") or cached[t]["sector"] == "Unknown":
                    cached[t]["sector"] = s
        except Exception:
            pass

    missing = [t for t in tickers if t not in cached or cached[t].get("market_type", "") == ""]
    if not missing:
        mvals = [cached[t].get("market_type", "") for t in tickers if t in cached]
        if mvals and all(v in {"", "NON_TRADABLE"} for v in mvals):
            logger.info("  Refreshing ticker metadata: cache market_type values are all blank/NON_TRADABLE.")
            missing = list(tickers)
    if missing:
        logger.info(f"  Fetching {len(missing):,} new tickers from Polygon reference API...")
        with ThreadPoolExecutor(max_workers=SECTOR_WORKERS) as pool:
            futures = {pool.submit(_fetch_one_ticker_meta, t): t for t in missing}
            for fut in as_completed(futures):
                ticker, meta = fut.result()
                cached[ticker] = meta

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for t, v in sorted(cached.items()):
            rows.append(
                {
                    "ticker": t,
                    "sector": v.get("sector", "Unknown"),
                    "market_type": v.get("market_type", "NON_TRADABLE"),
                    "is_tradable": int(v.get("is_tradable", 0)),
                }
            )
        pd.DataFrame(rows).to_csv(meta_path, index=False)
        pd.DataFrame([{"ticker": r["ticker"], "sector": r["sector"]} for r in rows]).to_csv(sec_path, index=False)
        logger.info(f"  Ticker meta cache updated -> {meta_path}")
        logger.info(f"  Sector cache updated -> {sec_path}")
    else:
        logger.info(f"  All {len(tickers):,} tickers found in ticker meta cache.")

    return cached


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Step 2: Feature Engineering ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

def engineer_features(df: pd.DataFrame) -> tuple:
    logger.info("Engineering features...")
    caps = {}

    # log_buy_price: winsorize at 99.5th, then log1p
    cap_bp = df["buy_price"].quantile(0.995)
    caps["buy_price_cap"] = float(cap_bp)
    df["log_buy_price"] = np.log1p(df["buy_price"].clip(upper=cap_bp))

    # log_value_usd: zeros ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ NaN, winsorize, log1p
    df.loc[df["value_usd"] <= 0, "value_usd"] = np.nan
    cap_val = df["value_usd"].quantile(0.995)
    caps["value_usd_cap"] = float(cap_val)
    df["log_value_usd"] = np.log1p(df["value_usd"].clip(upper=cap_val))

    # prior momentum: clip to same range
    df["prior_30d_pct"] = df["prior_30d_pct"].clip(-150, 300)
    df["prior_10d_pct"] = df["prior_10d_pct"].clip(-150, 300)
    df["prior_5d_pct"]  = df["prior_5d_pct"].clip(-150, 300)

    # prior volatility: annualized daily vol; clip at 99th pct to remove
    # extreme outliers from thin-market days with tiny price gaps
    for col in ["prior_30d_vol", "prior_10d_vol", "prior_5d_vol"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    # price_drift_filing_pct: clip to [-25, 25]; NaN already set for bad rows
    df["price_drift_filing_pct"] = df["price_drift_filing_pct"].clip(-25, 25)

    # insider_prior_buys_log: log1p of count (handles skew; 0 stays 0)
    df["insider_prior_buys_log"] = np.log1p(df["insider_prior_buys"])
    df["same_ticker_insider_count_365d"] = np.log1p(df["same_ticker_insider_count_365d"].clip(lower=0))

    # filing time (Eastern, from OpenInsider transaction timestamp).
    txn = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["filing_hour_et"] = txn.dt.hour + (txn.dt.minute / 60.0)

    # Long-tailed recency features: cap only extreme right tail.
    for col in ["days_since_last_buy_any", "days_since_last_buy_same_ticker"]:
        cap = df[col].quantile(0.995)
        if pd.notna(cap):
            caps[f"{col}_cap"] = float(cap)
            df[col] = df[col].clip(lower=0, upper=cap)

    # Exclude non-tradable venues from model training/scoring.
    if "is_tradable" in df.columns:
        before = len(df)
        df = df[df["is_tradable"] == 1].copy()
        logger.info("  Tradability filter: %d / %d rows kept", len(df), before)

    # market type fixed effect
    mkt_map = {m: i for i, m in enumerate(MARKET_TYPE_ORDER)}
    df["market_type"] = df.get("market_type", "NON_TRADABLE")
    df["market_type_enc"] = df["market_type"].map(mkt_map).fillna(mkt_map["NON_TRADABLE"]).astype("category")

    # sector: ordinal encode ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ category dtype for HGBR / XGBoost
    sec_map = {s: i for i, s in enumerate(SECTOR_ORDER)}
    df["sector_enc"] = df["sector"].map(sec_map).fillna(sec_map["Unknown"]).astype("category")

    # officer_type
    df["officer_type"] = df["title"].apply(extract_officer_type)
    logger.info(f"  Officer types: {df['officer_type'].value_counts().to_dict()}")

    # Ordinal encode officer_type ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ set category dtype for HGBR
    cat_map = {name: i for i, (name, _) in enumerate(OFFICER_KEYWORDS)}
    cat_map["Other"] = len(OFFICER_KEYWORDS)
    df["officer_type_enc"] = df["officer_type"].map(cat_map).astype("category")

    # n_insiders_in_cluster: ensure int
    if "n_insiders_in_cluster" not in df.columns:
        if "n_insiders" in df.columns:
            df["n_insiders_in_cluster"] = df["n_insiders"]
        else:
            df["n_insiders_in_cluster"] = 1
    df["n_insiders_in_cluster"] = pd.to_numeric(df["n_insiders_in_cluster"], errors="coerce").fillna(1).clip(lower=1)

    # Fixed effects for insider type (Officer / Director / TenPct) plus interactions.
    # This allows n_insiders impact to differ by insider cohort.
    df["insider_type"] = df["title"].apply(classify_insider_type)
    df["is_officer"] = (df["insider_type"] == "Officer").astype(int)
    df["is_director"] = (df["insider_type"] == "Director").astype(int)
    df["is_10pct"] = (df["insider_type"] == "TenPct").astype(int)
    df["is_supported_type"] = (df["insider_type"] != "Other").astype(int)

    n_in = df["n_insiders_in_cluster"].astype(float)
    df["n_insiders_x_officer"] = n_in * df["is_officer"]
    df["n_insiders_x_director"] = n_in * df["is_director"]
    df["n_insiders_x_10pct"] = n_in * df["is_10pct"]
    logger.info("  Insider type mix: %s", df["insider_type"].value_counts().to_dict())
    logger.info("  Supported-type rows (Officer/Director/TenPct): %d / %d", int(df["is_supported_type"].sum()), len(df))

    logger.info(f"  Feature matrix ready: {len(df):,} rows x {len(FEATURES)} features")
    return df, FEATURES, caps


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Step 3: Train One Window ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

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

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Standard metrics ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Directional accuracy ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    dir_acc = (np.sign(y_pred) == np.sign(y_test.values)).mean()

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Decile spread ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    order = np.argsort(y_pred)[::-1]
    n10 = max(1, len(order) // 10)
    top_dec_actual = y_test.values[order[:n10]].mean()
    bot_dec_actual = y_test.values[order[-n10:]].mean()
    dec_spread = top_dec_actual - bot_dec_actual

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Feature importance (permutation-based) ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
    perm_imp = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    fi = dict(zip(features, perm_imp.importances_mean))
    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Positive-return precision metrics ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
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

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Quintile analysis ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
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

    # ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Save model ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬
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


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Step 4: Summary Report ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

def print_summary(all_metrics: dict):
    w = 74
    print("\n" + "=" * w)
    print("  INSIDER PURCHASE ML PIPELINE - RESULTS SUMMARY")
    print("=" * w)
    print(f"  Random split : {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test (seed={RANDOM_STATE})")
    print(f"  Model        : HistGradientBoostingRegressor")
    print(f"  Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print("  Target definition: 1d is total return; 3d/5d/10d are per-day compounded returns.")

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


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Step 5: Save Metadata ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

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


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Main ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

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


