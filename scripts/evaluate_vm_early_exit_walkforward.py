from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import median

from live_trading.strategy_settings import LIVE_PATHS, RUNTIME_DEFAULTS


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def signal_columns(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return sorted(
        col
        for col in rows[0].keys()
        if col.startswith("ret_") and col.endswith("_pct") and col != "final_ret_1d_pct"
    )


def safe_float(raw: str) -> float | None:
    try:
        return float(raw)
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pct_positive(values: list[float]) -> float | None:
    if not values:
        return None
    return 100.0 * sum(v > 0 for v in values) / len(values)


def evaluate_candidate(rows: list[dict], signal: str, threshold: float) -> dict | None:
    eligible = []
    flagged_signal_returns = []
    flagged_final_returns = []
    strategy_returns = []
    baseline_returns = []

    for row in rows:
        final_ret = safe_float(str(row.get("final_ret_1d_pct", "")))
        signal_ret = safe_float(str(row.get(signal, "")))
        if final_ret is None or signal_ret is None:
            continue
        eligible.append(row)
        baseline_returns.append(final_ret)
        if signal_ret <= threshold:
            flagged_signal_returns.append(signal_ret)
            flagged_final_returns.append(final_ret)
            strategy_returns.append(signal_ret)
        else:
            strategy_returns.append(final_ret)

    if not eligible:
        return None

    total_losers = sum(v < 0 for v in baseline_returns)
    flagged_losers = sum(v < 0 for v in flagged_final_returns)
    return {
        "signal": signal,
        "threshold_pct": threshold,
        "n_trades": len(eligible),
        "n_flagged": len(flagged_final_returns),
        "flag_rate_pct": 100.0 * len(flagged_final_returns) / len(eligible),
        "loser_precision_pct": (100.0 * flagged_losers / len(flagged_final_returns)) if flagged_final_returns else None,
        "loser_recall_pct": (100.0 * flagged_losers / total_losers) if total_losers else None,
        "baseline_mean_ret_pct": mean(baseline_returns),
        "strategy_mean_ret_pct": mean(strategy_returns),
        "delta_mean_ret_pct": mean(strategy_returns) - mean(baseline_returns),
        "baseline_median_ret_pct": median(baseline_returns),
        "strategy_median_ret_pct": median(strategy_returns),
        "baseline_win_rate_pct": pct_positive(baseline_returns),
        "strategy_win_rate_pct": pct_positive(strategy_returns),
    }


def apply_candidate(row: dict, signal: str, threshold: float) -> dict | None:
    final_ret = safe_float(str(row.get("final_ret_1d_pct", "")))
    signal_ret = safe_float(str(row.get(signal, "")))
    if final_ret is None:
        return None
    if signal_ret is None:
        return {
            "signal_available": 0,
            "flagged": 0,
            "baseline_ret_pct": final_ret,
            "strategy_ret_pct": final_ret,
            "signal_ret_pct": "",
        }
    flagged = int(signal_ret <= threshold)
    return {
        "signal_available": 1,
        "flagged": flagged,
        "baseline_ret_pct": final_ret,
        "strategy_ret_pct": signal_ret if flagged else final_ret,
        "signal_ret_pct": signal_ret,
    }


def choose_best_candidate(
    train_rows: list[dict],
    candidates: list[tuple[str, float]],
    min_train_flagged: int,
) -> dict | None:
    scored: list[dict] = []
    for signal, threshold in candidates:
        metrics = evaluate_candidate(train_rows, signal, threshold)
        if metrics is None:
            continue
        if int(metrics["n_flagged"]) < min_train_flagged:
            continue
        scored.append(metrics)

    if not scored:
        return None

    scored.sort(
        key=lambda r: (
            float(r["strategy_mean_ret_pct"]),
            float(r["loser_precision_pct"]) if r["loser_precision_pct"] is not None else -1.0,
            -float(r["flag_rate_pct"]),
        ),
        reverse=True,
    )
    return scored[0]


def fmt(value: float | None) -> float | str:
    if value is None:
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward train/test evaluation for VM early-exit rules."
    )
    parser.add_argument(
        "--input",
        default=LIVE_PATHS.vm_early_exit_detail_file,
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(RUNTIME_DEFAULTS.early_exit_thresholds),
    )
    parser.add_argument("--initial-train-size", type=int, default=RUNTIME_DEFAULTS.early_exit_initial_train_size)
    parser.add_argument("--min-train-flagged", type=int, default=RUNTIME_DEFAULTS.early_exit_min_train_flagged)
    parser.add_argument(
        "--summary-out",
        default=LIVE_PATHS.vm_early_exit_walkforward_summary_file,
    )
    parser.add_argument(
        "--detail-out",
        default=LIVE_PATHS.vm_early_exit_walkforward_detail_file,
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    rows = sorted(rows, key=lambda r: (str(r.get("scored_at", "")), str(r.get("event_key", ""))))
    signals = signal_columns(rows)
    candidates = [(signal, threshold) for signal in signals for threshold in args.thresholds]

    fold_rows: list[dict] = []
    selected_rules: list[str] = []

    for idx in range(args.initial_train_size, len(rows)):
        train_rows = rows[:idx]
        test_row = rows[idx]
        best = choose_best_candidate(train_rows, candidates, args.min_train_flagged)
        if best is None:
            continue

        applied = apply_candidate(test_row, str(best["signal"]), float(best["threshold_pct"]))
        if applied is None:
            continue

        selected_rule = f"{best['signal']} <= {best['threshold_pct']}"
        selected_rules.append(selected_rule)
        fold_rows.append(
            {
                "test_index": idx,
                "train_n": len(train_rows),
                "selected_signal": best["signal"],
                "selected_threshold_pct": best["threshold_pct"],
                "selected_rule": selected_rule,
                "train_candidate_n": best["n_trades"],
                "train_candidate_flagged": best["n_flagged"],
                "train_candidate_flag_rate_pct": best["flag_rate_pct"],
                "train_candidate_loser_precision_pct": fmt(best["loser_precision_pct"]),
                "train_candidate_loser_recall_pct": fmt(best["loser_recall_pct"]),
                "train_baseline_mean_ret_pct": best["baseline_mean_ret_pct"],
                "train_strategy_mean_ret_pct": best["strategy_mean_ret_pct"],
                "train_delta_mean_ret_pct": best["delta_mean_ret_pct"],
                "scored_at": test_row["scored_at"],
                "ticker": test_row["ticker"],
                "event_key": test_row["event_key"],
                "baseline_ret_pct": applied["baseline_ret_pct"],
                "signal_available": applied["signal_available"],
                "signal_ret_pct": applied["signal_ret_pct"],
                "flagged": applied["flagged"],
                "strategy_ret_pct": applied["strategy_ret_pct"],
                "improvement_pct": applied["strategy_ret_pct"] - applied["baseline_ret_pct"],
            }
        )

    baseline_returns = [float(r["baseline_ret_pct"]) for r in fold_rows]
    strategy_returns = [float(r["strategy_ret_pct"]) for r in fold_rows]
    improvements = [float(r["improvement_pct"]) for r in fold_rows]
    flagged_returns = [float(r["strategy_ret_pct"]) for r in fold_rows if int(r["flagged"]) == 1]

    selected_counts = Counter(selected_rules)
    summary_rows = [
        {
            "metric": "n_test_trades",
            "value": len(fold_rows),
        },
        {
            "metric": "baseline_mean_ret_pct",
            "value": fmt(mean(baseline_returns)),
        },
        {
            "metric": "strategy_mean_ret_pct",
            "value": fmt(mean(strategy_returns)),
        },
        {
            "metric": "delta_mean_ret_pct",
            "value": fmt(mean(strategy_returns) - mean(baseline_returns) if baseline_returns else None),
        },
        {
            "metric": "baseline_median_ret_pct",
            "value": fmt(median(baseline_returns) if baseline_returns else None),
        },
        {
            "metric": "strategy_median_ret_pct",
            "value": fmt(median(strategy_returns) if strategy_returns else None),
        },
        {
            "metric": "baseline_win_rate_pct",
            "value": fmt(pct_positive(baseline_returns)),
        },
        {
            "metric": "strategy_win_rate_pct",
            "value": fmt(pct_positive(strategy_returns)),
        },
        {
            "metric": "n_flagged_test_trades",
            "value": sum(int(r["flagged"]) for r in fold_rows),
        },
        {
            "metric": "flag_rate_test_pct",
            "value": fmt(100.0 * sum(int(r["flagged"]) for r in fold_rows) / len(fold_rows) if fold_rows else None),
        },
        {
            "metric": "mean_improvement_pct",
            "value": fmt(mean(improvements)),
        },
        {
            "metric": "median_improvement_pct",
            "value": fmt(median(improvements) if improvements else None),
        },
        {
            "metric": "mean_flagged_strategy_ret_pct",
            "value": fmt(mean(flagged_returns)),
        },
        {
            "metric": "most_selected_rule",
            "value": selected_counts.most_common(1)[0][0] if selected_counts else "",
        },
        {
            "metric": "most_selected_rule_count",
            "value": selected_counts.most_common(1)[0][1] if selected_counts else "",
        },
    ]

    detail_out = Path(args.detail_out)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    with detail_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()) if fold_rows else [])
        if fold_rows:
            writer.writeheader()
            writer.writerows(fold_rows)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"test_trades={len(fold_rows)}")
    for row in summary_rows:
        print(f"{row['metric']}={row['value']}")
    if selected_counts:
        print("rule_counts:")
        for rule, count in selected_counts.most_common():
            print(f"{rule}: {count}")


if __name__ == "__main__":
    main()
