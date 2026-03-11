from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ANALYSIS_CSV = BASE / "research" / "ibkr_vm_analysis" / "ibkr_open_gap_predictor_analysis.csv"
STATE_JSON = BASE / "research" / "ibkr_vm_analysis" / "ibkr_paper_trader_state.json"
OUT_CSV = BASE / "research" / "ibkr_vm_analysis" / "ibkr_gap_threshold_impact.csv"


def load_frame() -> pd.DataFrame:
    analysis = pd.read_csv(ANALYSIS_CSV)
    state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    lots = pd.DataFrame(state["lots"])[["lot_id", "entry_value", "realized_pnl", "status"]].copy()
    lots["entry_value"] = pd.to_numeric(lots["entry_value"], errors="coerce")
    lots["realized_pnl"] = pd.to_numeric(lots["realized_pnl"], errors="coerce")

    frame = analysis.merge(lots, on=["lot_id", "status"], how="left", suffixes=("", "_state"))
    frame["actual_return_pct"] = pd.to_numeric(frame["actual_return_pct"], errors="coerce")
    frame["current_mark_return_pct"] = pd.to_numeric(frame["current_mark_return_pct"], errors="coerce")
    frame["overnight_gap_pct"] = pd.to_numeric(frame["overnight_gap_pct"], errors="coerce")

    # Closed lots use actual realized P&L from trader state; open lots use current mark estimate.
    frame["pnl_usd_est"] = frame["realized_pnl"]
    open_mask = frame["status"].eq("open") & frame["entry_value"].notna() & frame["current_mark_return_pct"].notna()
    frame.loc[open_mask, "pnl_usd_est"] = (
        frame.loc[open_mask, "entry_value"] * frame.loc[open_mask, "current_mark_return_pct"] / 100.0
    )

    frame["return_pct_est"] = frame["pnl_usd_est"] / frame["entry_value"] * 100.0
    return frame


def summarize(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    kept = frame[frame["overnight_gap_pct"] <= threshold].copy()
    dropped = frame[frame["overnight_gap_pct"] > threshold].copy()

    deployed = float(kept["entry_value"].sum())
    pnl = float(kept["pnl_usd_est"].sum())
    dropped_deployed = float(dropped["entry_value"].sum())
    dropped_pnl = float(dropped["pnl_usd_est"].sum())

    return {
        "max_gap_pct": float(threshold),
        "lots_kept": int(len(kept)),
        "lots_dropped": int(len(dropped)),
        "capital_kept_usd": deployed,
        "capital_dropped_usd": dropped_deployed,
        "pnl_kept_usd_est": pnl,
        "pnl_dropped_usd_est": dropped_pnl,
        "return_kept_pct_est": (pnl / deployed * 100.0) if deployed else float("nan"),
        "avg_trade_return_pct": float(kept["actual_return_pct"].mean()) if not kept.empty else float("nan"),
        "median_trade_return_pct": float(kept["actual_return_pct"].median()) if not kept.empty else float("nan"),
        "winrate_pct": float((kept["actual_return_pct"] > 0).mean() * 100.0) if not kept.empty else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep for IBKR overnight-gap benchmark impact.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    frame = load_frame()
    thresholds = sorted(set([-999.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] + frame["overnight_gap_pct"].dropna().round(6).tolist()))
    table = pd.DataFrame([summarize(frame, threshold) for threshold in thresholds]).sort_values("max_gap_pct").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)

    baseline = summarize(frame, float("inf"))
    best_by_pnl = table.sort_values(["pnl_kept_usd_est", "return_kept_pct_est"], ascending=False).iloc[0]
    best_by_return = table.sort_values(["return_kept_pct_est", "pnl_kept_usd_est"], ascending=False).iloc[0]

    print("Baseline (no filter):")
    print(pd.Series(baseline).to_string())
    print("")
    print("Best by estimated P&L kept:")
    print(best_by_pnl.to_string())
    print("")
    print("Best by estimated return on kept capital:")
    print(best_by_return.to_string())
    print("")
    print("Coarse thresholds:")
    coarse = table[table["max_gap_pct"].isin([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])].copy()
    print(coarse.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("")
    print(f"Saved sweep to: {args.out_csv}")


if __name__ == "__main__":
    main()
