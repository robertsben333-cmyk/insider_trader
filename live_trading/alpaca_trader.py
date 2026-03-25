from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-untyped]

from live_trading.alpaca_broker import AlpacaBrokerAdapter
from live_trading.broker import DryRunBrokerAdapter
from live_trading.ibkr_paper_trader import IbkrPaperTrader, target_cycle_seconds
from live_trading.market_calendar import ET
from live_trading.strategy_settings import (
    ALPACA_CONFIG,
    ALPACA_LIVE_PATHS,
    EXECUTION_POLICY,
    TRADING_BUDGET,
)
from live_trading.trader_state import StateStore


def setup_logger() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    logger = logging.getLogger("alpaca_trader")
    logger.setLevel(logging.INFO)
    return logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alpaca live-trading service for the insider-trading strategy."
    )
    parser.add_argument("--state-file", default=ALPACA_LIVE_PATHS.trader_state_file)
    parser.add_argument("--journal-file", default=ALPACA_LIVE_PATHS.trader_journal_file)
    parser.add_argument(
        "--signal-archive-file", default=ALPACA_LIVE_PATHS.trader_signal_archive_file
    )
    parser.add_argument(
        "--alert-snapshot-file", default=ALPACA_LIVE_PATHS.alert_snapshot_file
    )
    parser.add_argument(
        "--cycle-seconds", type=int, default=EXECUTION_POLICY.cycle_seconds
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the in-memory DryRunBrokerAdapter.",
    )
    return parser


def main() -> None:
    load_dotenv()
    logger = setup_logger()
    args = build_arg_parser().parse_args()

    if args.dry_run:
        broker: DryRunBrokerAdapter | AlpacaBrokerAdapter = DryRunBrokerAdapter()
        broker.connect()
    else:
        api_key = os.environ.get(ALPACA_CONFIG.api_key_env_var, "")
        api_secret = os.environ.get(ALPACA_CONFIG.api_secret_env_var, "")
        if not api_key or not api_secret:
            raise RuntimeError(
                f"Missing Alpaca credentials: set {ALPACA_CONFIG.api_key_env_var} and "
                f"{ALPACA_CONFIG.api_secret_env_var} environment variables."
            )
        broker = AlpacaBrokerAdapter(
            api_key=api_key,
            api_secret=api_secret,
            paper_trading=ALPACA_CONFIG.paper_trading,
            connect_timeout_seconds=ALPACA_CONFIG.connect_timeout_seconds,
            data_feed=ALPACA_CONFIG.data_feed,
        )
        broker.connect()

    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = Path(args.journal_file)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    store = StateStore(state_path, journal_path)
    trader = IbkrPaperTrader(
        broker=broker,
        store=store,
        alert_snapshot_path=Path(args.alert_snapshot_file),
        signal_archive_path=Path(args.signal_archive_file),
        logger=logger,
        budget_config=TRADING_BUDGET,
        execution_policy=EXECUTION_POLICY,
    )

    try:
        while True:
            start = time.time()
            now_et = datetime.now(ET)
            trader.run_once(now_et)
            if args.once:
                break
            elapsed = time.time() - start
            target_cycle = target_cycle_seconds(now_et, float(args.cycle_seconds), EXECUTION_POLICY)
            sleep_seconds = max(0.0, float(target_cycle) - elapsed)
            logger.info("Cycle complete. Sleeping %.1f sec", sleep_seconds)
            time.sleep(sleep_seconds)
    finally:
        if not args.dry_run:
            broker.disconnect()


if __name__ == "__main__":
    main()
