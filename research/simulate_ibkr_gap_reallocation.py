from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ANALYSIS_CSV = BASE / "research" / "ibkr_vm_analysis" / "ibkr_open_gap_predictor_analysis.csv"
STATE_JSON = BASE / "research" / "ibkr_vm_analysis" / "ibkr_paper_trader_state.json"
OUT_CSV = BASE / "research" / "ibkr_vm_analysis" / "ibkr_gap_reallocation_sweep.csv"

MAX_FRACTION_SINGLE = 0.80
MAX_FRACTION_TWO = 0.60
MAX_FRACTION_THREE_PLUS = 0.40


def max_name_fraction(n_names: int) -> float:
    if n_names <= 1:
        return MAX_FRACTION_SINGLE
    if n_names == 2:
        return MAX_FRACTION_TWO
    return MAX_FRACTION_THREE_PLUS


def load_name_frame() -> pd.DataFrame:
    analysis = pd.read_csv(ANALYSIS_CSV)
    state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    lots = pd.DataFrame(state["lots"])[
        ["lot_id", "entry_value", "realized_pnl", "status", "sleeve_id", "entry_trade_day", "ticker", "due_exit_at"]
    ].copy()
    sleeves = pd.DataFrame(state["sleeves"])[["sleeve_id", "starting_cash"]].copy()
    lots["entry_value"] = pd.to_numeric(lots["entry_value"], errors="coerce")
    lots["realized_pnl"] = pd.to_numeric(lots["realized_pnl"], errors="coerce")

    frame = analysis.merge(lots, on=["lot_id", "status"], how="left", suffixes=("", "_state"))
    frame["current_mark_return_pct"] = pd.to_numeric(frame["current_mark_return_pct"], errors="coerce")
    frame["overnight_gap_pct"] = pd.to_numeric(frame["overnight_gap_pct"], errors="coerce")

    frame["pnl_usd_est"] = frame["realized_pnl"]
    open_mask = frame["status"].eq("open") & frame["entry_value"].notna() & frame["current_mark_return_pct"].notna()
    frame.loc[open_mask, "pnl_usd_est"] = (
        frame.loc[open_mask, "entry_value"] * frame.loc[open_mask, "current_mark_return_pct"] / 100.0
    )
    frame["due_exit_day"] = pd.to_datetime(frame["due_exit_at"], errors="coerce").dt.strftime("%Y-%m-%d")

    name_frame = (
        frame.groupby(["sleeve_id", "entry_trade_day", "ticker"], as_index=False)
        .agg(
            overnight_gap_pct=("overnight_gap_pct", "first"),
            entry_value=("entry_value", "sum"),
            pnl_usd_est=("pnl_usd_est", "sum"),
            due_exit_day=("due_exit_day", "first"),
        )
    )
    name_frame["return_pct_est"] = name_frame["pnl_usd_est"] / name_frame["entry_value"] * 100.0
    return name_frame.merge(sleeves, on="sleeve_id", how="left")


def allocate_day(day_frame: pd.DataFrame, budget: float) -> pd.DataFrame:
    work = day_frame.copy().reset_index(drop=True)
    n_names = len(work)
    if n_names == 0 or budget <= 0:
        work["alloc_usd"] = 0.0
        work["alloc_pnl_usd_est"] = 0.0
        return work

    cap_fraction = max_name_fraction(n_names)
    cap_usd = budget * cap_fraction
    remaining_budget = float(budget)
    remaining = set(work.index.tolist())
    alloc = pd.Series(0.0, index=work.index, dtype=float)
    weights = work["entry_value"].clip(lower=0.0).astype(float)

    while remaining and remaining_budget > 1e-9:
        remaining_idx = sorted(remaining)
        weight_sum = float(weights.loc[remaining_idx].sum())
        if weight_sum <= 0:
            equal = remaining_budget / float(len(remaining_idx))
            for idx in remaining_idx:
                take = min(equal, cap_usd - alloc.loc[idx])
                alloc.loc[idx] += max(take, 0.0)
            break

        saturated: list[int] = []
        for idx in remaining_idx:
            target = remaining_budget * float(weights.loc[idx]) / weight_sum
            room = cap_usd - float(alloc.loc[idx])
            take = min(target, room)
            alloc.loc[idx] += max(take, 0.0)
            if room - take <= 1e-9:
                saturated.append(idx)

        used = float(alloc.sum())
        remaining_budget = max(0.0, budget - used)
        if not saturated:
            if remaining_budget > 1e-9:
                # Final proportional top-up into remaining room.
                rooms = pd.Series({idx: cap_usd - alloc.loc[idx] for idx in remaining_idx})
                rooms = rooms[rooms > 1e-9]
                if rooms.empty:
                    break
                room_sum = float(rooms.sum())
                for idx, room in rooms.items():
                    alloc.loc[idx] += remaining_budget * float(room) / room_sum
                remaining_budget = 0.0
            break
        for idx in saturated:
            remaining.discard(idx)

    work["alloc_usd"] = alloc
    work["alloc_pnl_usd_est"] = work["alloc_usd"] * work["return_pct_est"] / 100.0
    return work


def simulate_threshold(name_frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    total_start_cash = 0.0
    total_alloc = 0.0
    total_pnl = 0.0
    kept_names = 0
    dropped_names = 0
    used_days = 0
    ending_cash = 0.0

    for sleeve_id, sleeve_frame in name_frame.groupby("sleeve_id", sort=True):
        sleeve_frame = sleeve_frame.sort_values(["entry_trade_day", "ticker"]).copy()
        cash = float(sleeve_frame["starting_cash"].iloc[0])
        total_start_cash += cash
        pending_exits: list[tuple[str, float]] = []

        for entry_day, day_all in sleeve_frame.groupby("entry_trade_day", sort=True):
            realized_today = sum(pnl_plus_principal for due_day, pnl_plus_principal in pending_exits if due_day <= entry_day)
            cash += realized_today
            pending_exits = [(due_day, amt) for due_day, amt in pending_exits if due_day > entry_day]

            kept = day_all[day_all["overnight_gap_pct"] <= threshold].copy()
            dropped_names += int(len(day_all) - len(kept))
            if kept.empty or cash <= 0:
                continue

            used_days += 1
            kept_names += int(len(kept))
            allocated = allocate_day(kept, cash)
            used_cash = float(allocated["alloc_usd"].sum())
            total_alloc += used_cash
            total_pnl += float(allocated["alloc_pnl_usd_est"].sum())
            cash -= used_cash

            for _, row in allocated.iterrows():
                proceeds = float(row["alloc_usd"]) + float(row["alloc_pnl_usd_est"])
                pending_exits.append((str(row["due_exit_day"]), proceeds))

        cash += sum(amt for _, amt in pending_exits)
        ending_cash += cash

    return {
        "max_gap_pct": float(threshold),
        "kept_names": int(kept_names),
        "dropped_names": int(dropped_names),
        "days_with_positions": int(used_days),
        "starting_cash_usd": float(total_start_cash),
        "allocated_capital_usd": float(total_alloc),
        "estimated_pnl_usd": float(total_pnl),
        "estimated_return_pct": (total_pnl / total_alloc * 100.0) if total_alloc else float("nan"),
        "ending_cash_usd": float(ending_cash),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate sleeve/day reallocation after applying an overnight-gap filter.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    name_frame = load_name_frame()
    thresholds = sorted(set([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] + name_frame["overnight_gap_pct"].dropna().round(6).tolist()))
    rows = [simulate_threshold(name_frame, threshold) for threshold in thresholds]
    rows.append(simulate_threshold(name_frame, float("inf")))
    table = pd.DataFrame(rows).sort_values("max_gap_pct").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)

    baseline = table[table["max_gap_pct"] == float("inf")].iloc[0]
    best_pnl = table.sort_values(["estimated_pnl_usd", "estimated_return_pct"], ascending=False).iloc[0]

    print("Baseline (reallocate, no filter):")
    print(baseline.to_string())
    print("")
    print("Best threshold by estimated P&L:")
    print(best_pnl.to_string())
    print("")
    coarse = table[table["max_gap_pct"].isin([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, float("inf")])].copy()
    print(coarse.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("")
    print(f"Saved sweep to: {args.out_csv}")


if __name__ == "__main__":
    main()
