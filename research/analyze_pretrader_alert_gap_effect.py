from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


BASE = Path(__file__).resolve().parents[1]
ALERT_HISTORY = BASE / "live" / "data" / "alert_candidate_history.csv"
OUT_CSV = BASE / "research" / "outcomes" / "models" / "pretrader_alert_gap_effect.csv"
POLYGON_BASE_URL = "https://api.polygon.io"


def fetch_daily_bars(session: requests.Session, api_key: str, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    r = session.get(
        url,
        params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["date", "open", "close"])
    out = pd.DataFrame(
        {
            "date": [
                pd.to_datetime(row["t"], unit="ms", utc=True).tz_convert("America/New_York").date().isoformat()
                for row in rows
            ],
            "open": [float(row["o"]) for row in rows],
            "close": [float(row["c"]) for row in rows],
        }
    )
    return out


def build_frame(cutoff_date: str) -> pd.DataFrame:
    df = pd.read_csv(ALERT_HISTORY)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
    df["pred_mean4"] = pd.to_numeric(df["pred_mean4"], errors="coerce")
    df["estimated_decile_score"] = pd.to_numeric(df["estimated_decile_score"], errors="coerce")
    df = df.dropna(subset=["ticker", "trade_date", "buy_price"]).copy()
    df = df[df["trade_date"] < pd.Timestamp(cutoff_date)].copy()
    df = df.sort_values(["scored_at", "event_key"]).drop_duplicates(subset=["event_key"], keep="last").reset_index(drop=True)

    load_dotenv(BASE / ".env")
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not found.")

    session = requests.Session()
    rows: list[dict] = []
    today = datetime.now().date().isoformat()

    for ticker, grp in df.groupby("ticker", sort=True):
        start = (grp["trade_date"].min() - pd.Timedelta(days=10)).date().isoformat()
        bars = fetch_daily_bars(session, api_key, str(ticker), start, today)
        if len(bars) < 2:
            continue
        by_day = bars.set_index("date")
        trade_days = bars["date"].tolist()

        for _, row in grp.iterrows():
            trade_day = row["trade_date"].date().isoformat()
            if trade_day not in by_day.index:
                continue
            idx = trade_days.index(trade_day)
            if idx == 0:
                continue
            prev_day = trade_days[idx - 1]
            if idx + 2 >= len(trade_days):
                continue
            t2_day = trade_days[idx + 2]

            prev_close = float(by_day.loc[prev_day, "close"])
            trade_open = float(by_day.loc[trade_day, "open"])
            buy_price = float(row["buy_price"])
            t2_close = float(by_day.loc[t2_day, "close"])
            if min(prev_close, trade_open, buy_price, t2_close) <= 0:
                continue

            rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": row["ticker"],
                    "trade_date": trade_day,
                    "buy_price": buy_price,
                    "pred_mean4": row["pred_mean4"],
                    "estimated_decile_score": row["estimated_decile_score"],
                    "prev_close_day": prev_day,
                    "prev_close": prev_close,
                    "trade_open": trade_open,
                    "overnight_gap_pct": ((trade_open / prev_close) - 1.0) * 100.0,
                    "buy_open_slippage_pct": ((buy_price / trade_open) - 1.0) * 100.0,
                    "tplus2_close_day": t2_day,
                    "tplus2_close": t2_close,
                    "return_from_buyprice_to_tplus2_close_pct": ((t2_close / buy_price) - 1.0) * 100.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def print_summary(frame: pd.DataFrame) -> None:
    print(f"Rows analyzed: {len(frame)}")
    print(f"Period: {frame['trade_date'].min()} -> {frame['trade_date'].max()}")
    print("")
    pearson = frame["overnight_gap_pct"].corr(frame["return_from_buyprice_to_tplus2_close_pct"])
    spearman = frame["overnight_gap_pct"].corr(frame["return_from_buyprice_to_tplus2_close_pct"], method="spearman")
    print(f"Pearson:  {pearson:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(
        frame.assign(bucket=frame["overnight_gap_pct"].apply(lambda x: "gap_down_or_flat" if x <= 0 else "gap_up"))
        .groupby("bucket")["return_from_buyprice_to_tplus2_close_pct"]
        .agg(["count", "mean", "median"])
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pre-trader live alerts for overnight gap effect.")
    parser.add_argument("--cutoff-date", default="2026-03-09")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    frame = build_frame(args.cutoff_date)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    print_summary(frame)
    print("")
    print(f"Saved detail to: {args.out_csv}")


if __name__ == "__main__":
    main()
