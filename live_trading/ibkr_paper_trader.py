from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import logging
import math
import time
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
import pandas as pd

from live_trading.broker import (
    BrokerAdapter,
    BrokerFillView,
    BrokerOrderRequest,
    DryRunBrokerAdapter,
    IbkrBrokerAdapter,
)
from live_trading.market_calendar import (
    ET,
    UTC,
    exit_at_tplus_open,
    is_regular_trading_hours,
    is_weekend_shutdown_window,
    parse_iso_datetime,
    parse_time_hhmm,
    previous_trading_day,
    seconds_until_weekend_shutdown_end,
)
from live_trading.signal_intake import load_signal_candidates
from live_trading.strategy_settings import (
    ACTIVE_STRATEGY,
    EXECUTION_POLICY,
    IBKR_CONFIG,
    LIVE_PATHS,
    TRADING_BUDGET,
    ExecutionPolicy,
    TradingBudgetConfig,
)
from live_trading.trader_state import (
    FillEvent,
    PendingOrder,
    PlannedExit,
    PositionLot,
    SignalCandidate,
    SleeveState,
    StateStore,
    TraderStateSnapshot,
    new_id,
    utc_now_iso,
)


TERMINAL_CANDIDATE_STATUSES = {"expired", "rejected", "filled"}
CANCELLED_ORDER_STATUSES = {"Cancelled", "ApiCancelled", "Inactive"}


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("ibkr_paper_trader")


def _to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def _parse_fill_timestamp(raw: str) -> datetime:
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return datetime.now(UTC)
    return ts.to_pydatetime()


class IbkrPaperTrader:
    def __init__(
        self,
        *,
        broker: BrokerAdapter,
        store: StateStore,
        alert_snapshot_path: Path,
        signal_archive_path: Path,
        logger: logging.Logger,
        budget_config: TradingBudgetConfig = TRADING_BUDGET,
        execution_policy: ExecutionPolicy = EXECUTION_POLICY,
    ) -> None:
        self.broker = broker
        self.store = store
        self.alert_snapshot_path = alert_snapshot_path
        self.signal_archive_path = signal_archive_path
        self.logger = logger
        self.budget_config = budget_config
        self.execution_policy = execution_policy

    def run_once(self, now: datetime | None = None) -> None:
        now_et = _to_et(now or datetime.now(ET))
        if is_weekend_shutdown_window(now_et):
            self.logger.info("Weekend shutdown active; skipping cycle until Monday 00:00 ET.")
            return
        state = self.store.load()
        self._initialize_state(state)
        self._connect_broker()
        state.metadata["last_cycle_at"] = now_et.isoformat()
        state.metadata["last_account_snapshot"] = asdict(self.broker.get_account_snapshot())

        self._reconcile_orders_and_fills(state)
        self._refresh_sleeve_equity(state)
        self._ingest_signals(state)
        self._ensure_planned_exits(state)
        self._expire_stale_candidates(state, now_et)
        self._manage_intraday_replacements(state, now_et)
        self._manage_exit_orders(state, now_et)
        self._manage_entry_orders(state, now_et)
        self._expire_stale_candidates(state, now_et)
        self._archive_candidates(state)
        self.store.save(state)

    def _connect_broker(self) -> None:
        if not self.broker.is_connected():
            self.broker.connect()

    def _initialize_state(self, state: TraderStateSnapshot) -> None:
        if state.sleeves:
            return
        sleeve_cash = self.budget_config.initial_strategy_budget / float(self.budget_config.sleeve_count)
        for idx in range(self.budget_config.sleeve_count):
            state.sleeves.append(
                SleeveState(
                    sleeve_id=f"sleeve_{idx}",
                    starting_cash=sleeve_cash,
                    cash_balance=sleeve_cash,
                    last_equity=sleeve_cash,
                    last_marked_at=utc_now_iso(),
                )
            )
        state.metadata["initialized_at"] = utc_now_iso()
        self.store.append_journal(
            "state_initialized",
            {
                "initial_strategy_budget": self.budget_config.initial_strategy_budget,
                "sleeve_count": self.budget_config.sleeve_count,
                "sleeve_cash": sleeve_cash,
            },
        )

    def _refresh_sleeve_equity(self, state: TraderStateSnapshot) -> None:
        mark_prices: dict[str, float] = {}
        for lot in state.lots:
            if lot.status != "open" or lot.quantity <= 0:
                continue
            lot.last_mark_price = self._mark_price_for_symbol(mark_prices, lot.ticker, lot.last_mark_price or lot.avg_entry_price)
        for sleeve in state.sleeves:
            invested = 0.0
            for lot in state.lots:
                if lot.status != "open" or lot.quantity <= 0 or lot.sleeve_id != sleeve.sleeve_id:
                    continue
                mark = lot.last_mark_price if lot.last_mark_price is not None else lot.avg_entry_price
                invested += float(mark) * float(lot.quantity)
            sleeve.last_equity = sleeve.cash_balance + invested
            sleeve.last_marked_at = utc_now_iso()

    def _mark_price_for_symbol(self, cache: dict[str, float], symbol: str, fallback: float | None) -> float:
        key = symbol.upper()
        if key not in cache:
            quote = self.broker.get_quote(key)
            ref = quote.reference_price("SELL", fallback)
            cache[key] = float(ref if ref is not None else fallback if fallback is not None else 0.0)
        return cache[key]

    def _ingest_signals(self, state: TraderStateSnapshot) -> None:
        known = {candidate.candidate_id for candidate in state.candidates}
        for candidate in load_signal_candidates(
            self.alert_snapshot_path,
            budget_config=self.budget_config,
            execution_policy=self.execution_policy,
        ):
            if candidate.candidate_id in known:
                continue
            state.candidates.append(candidate)
            self.store.append_journal("candidate_ingested", asdict(candidate))

    def _ensure_planned_exits(self, state: TraderStateSnapshot) -> None:
        known = {row.lot_id for row in state.planned_exits}
        for lot in state.lots:
            if lot.status != "open" or lot.lot_id in known:
                continue
            planned = PlannedExit(
                exit_id=new_id("exit"),
                lot_id=lot.lot_id,
                ticker=lot.ticker,
                sleeve_id=lot.sleeve_id,
                due_at=lot.due_exit_at,
            )
            state.planned_exits.append(planned)
            self.store.append_journal("exit_planned", asdict(planned))

    def _expire_stale_candidates(self, state: TraderStateSnapshot, now_et: datetime) -> None:
        cancel_cutoff = parse_time_hhmm(self.execution_policy.cancel_unfilled_time)
        cancel_cutoff_dt = now_et.replace(hour=cancel_cutoff.hour, minute=cancel_cutoff.minute, second=0, microsecond=0)
        for candidate in state.candidates:
            if candidate.status in TERMINAL_CANDIDATE_STATUSES:
                continue
            expiry = parse_iso_datetime(candidate.expires_at)
            if now_et <= expiry:
                continue
            order = self._find_order(state, candidate.active_order_id)
            if order is not None and order.broker_order_id is not None and now_et >= cancel_cutoff_dt:
                self.broker.cancel_order(order.broker_order_id)
            if candidate.linked_lot_id:
                candidate.status = "filled"
                candidate.last_reason = "entry_window_closed_after_partial_fill"
            else:
                candidate.status = "expired"
                candidate.last_reason = "entry_window_closed_without_fill"
            candidate.active_order_id = None
            candidate.last_updated_at = utc_now_iso()

    def _manage_exit_orders(self, state: TraderStateSnapshot, now_et: datetime) -> None:
        for planned in state.planned_exits:
            if planned.status == "filled":
                continue
            due_at = parse_iso_datetime(planned.due_at)
            if now_et < due_at:
                continue
            lot = self._find_lot(state, planned.lot_id)
            if lot is None or lot.status != "open" or lot.quantity <= 0:
                planned.status = "filled"
                continue
            current_order = self._find_order(state, lot.active_exit_order_id)
            if current_order is not None:
                if self._order_is_stale(current_order, now_et):
                    self._cancel_order(state, current_order, "stale_exit_reprice")
                continue
            self._submit_exit_order(state, lot, planned, reason="scheduled_exit")

    def _manage_intraday_replacements(self, state: TraderStateSnapshot, now_et: datetime) -> None:
        if not is_regular_trading_hours(now_et):
            return

        previous_day = previous_trading_day(now_et.date())
        reserved_lot_ids = {
            str(candidate.replacement_for_lot_id)
            for candidate in state.candidates
            if candidate.replacement_for_lot_id and candidate.status not in TERMINAL_CANDIDATE_STATUSES
        }
        eligible = self._eligible_candidates(state, now_et)
        for candidate in eligible:
            if candidate.entry_bucket != "intraday":
                continue
            if candidate.status in TERMINAL_CANDIDATE_STATUSES or candidate.linked_lot_id or candidate.active_order_id:
                continue
            if self._candidate_has_funding(state, candidate):
                continue

            replacement_lot = self._find_lot(state, candidate.replacement_for_lot_id)
            if replacement_lot is None:
                replacement_lot = self._select_intraday_replacement_lot(
                    state=state,
                    previous_day=previous_day,
                    reserved_lot_ids=reserved_lot_ids,
                    candidate_decile=float(candidate.estimated_decile_score),
                )
                if replacement_lot is None:
                    continue
                candidate.replacement_for_lot_id = replacement_lot.lot_id
                candidate.sleeve_id = replacement_lot.sleeve_id
                candidate.last_reason = "intraday_replacement_selected"
                candidate.last_updated_at = utc_now_iso()
                reserved_lot_ids.add(replacement_lot.lot_id)
                self.store.append_journal(
                    "intraday_replacement_selected",
                    {
                        "candidate_id": candidate.candidate_id,
                        "candidate_ticker": candidate.ticker,
                        "candidate_decile": float(candidate.estimated_decile_score),
                        "replacement_lot_id": replacement_lot.lot_id,
                        "replacement_ticker": replacement_lot.ticker,
                        "replacement_decile": float(replacement_lot.entry_estimated_decile_score),
                        "replacement_trade_day": replacement_lot.entry_trade_day,
                    },
                )

            if replacement_lot.status != "open" or replacement_lot.quantity <= 0:
                continue

            planned = self._planned_exit_for_lot(state, replacement_lot)
            if planned is None:
                planned = PlannedExit(
                    exit_id=new_id("exit"),
                    lot_id=replacement_lot.lot_id,
                    ticker=replacement_lot.ticker,
                    sleeve_id=replacement_lot.sleeve_id,
                    due_at=now_et.isoformat(),
                )
                state.planned_exits.append(planned)
            else:
                planned.due_at = now_et.isoformat()
            self._submit_exit_order(state, replacement_lot, planned, reason="intraday_replacement_exit")

    def _manage_entry_orders(self, state: TraderStateSnapshot, now_et: datetime) -> None:
        eligible = self._eligible_candidates(state, now_et)
        open_batches: dict[tuple[str, str], list[SignalCandidate]] = {}
        for candidate in eligible:
            if candidate.entry_bucket == "open" and candidate.linked_lot_id is None and candidate.active_order_id is None:
                open_batches.setdefault((candidate.sleeve_id, candidate.intended_entry_at), []).append(candidate)
                continue
            self._submit_entry_for_candidate(state, candidate, now_et)

        for _, batch in sorted(open_batches.items(), key=lambda row: row[0]):
            self._manage_open_batch_entry_orders(state, batch, now_et)

    def _submit_entry_for_candidate(
        self,
        state: TraderStateSnapshot,
        candidate: SignalCandidate,
        now_et: datetime,
    ) -> bool:
        current_order = self._find_order(state, candidate.active_order_id)
        if current_order is not None:
            if self._order_is_stale(current_order, now_et):
                self._cancel_order(state, current_order, "stale_entry_reprice")
            return False

        sleeve = self._find_sleeve(state, candidate.sleeve_id)
        if sleeve is None:
            return False

        target_notional = self._target_notional_for_candidate(state, candidate, sleeve)
        committed_notional = self._committed_notional_for_candidate(state, candidate.candidate_id)
        remaining_notional = max(0.0, target_notional - committed_notional)
        if remaining_notional < self.execution_policy.min_order_notional:
            if candidate.linked_lot_id:
                candidate.status = "filled"
            return False

        limit_price = self._build_limit_price("BUY", candidate.ticker, candidate.buy_price_hint)
        if limit_price is None:
            candidate.last_reason = "missing_reference_price"
            candidate.last_updated_at = utc_now_iso()
            return False

        max_cash = min(sleeve.cash_balance, remaining_notional)
        quantity = int(math.floor(max_cash / float(limit_price)))
        reserve = float(quantity) * float(limit_price)
        if quantity < 1 or reserve < self.execution_policy.min_order_notional:
            candidate.last_reason = "insufficient_sleeve_cash"
            candidate.last_updated_at = utc_now_iso()
            return False
        return self._submit_entry_order(state, sleeve, candidate, quantity, limit_price, reserve)

    def _manage_open_batch_entry_orders(
        self,
        state: TraderStateSnapshot,
        batch: list[SignalCandidate],
        now_et: datetime,
    ) -> None:
        if not batch:
            return
        sleeve = self._find_sleeve(state, batch[0].sleeve_id)
        if sleeve is None or sleeve.cash_balance < self.execution_policy.min_order_notional:
            return

        workable: list[dict[str, object]] = []
        for candidate in batch:
            current_order = self._find_order(state, candidate.active_order_id)
            if current_order is not None:
                if self._order_is_stale(current_order, now_et):
                    self._cancel_order(state, current_order, "stale_entry_reprice")
                continue

            limit_price = self._build_limit_price("BUY", candidate.ticker, candidate.buy_price_hint)
            if limit_price is None:
                candidate.last_reason = "missing_reference_price"
                candidate.last_updated_at = utc_now_iso()
                continue
            workable.append(
                {
                    "candidate": candidate,
                    "limit_price": float(limit_price),
                    "weight": max(float(candidate.advised_allocation_fraction), 1e-9),
                }
            )
        if not workable:
            return

        available_cash = float(sleeve.cash_balance)
        total_weight = float(sum(float(row["weight"]) for row in workable))
        allocations: list[dict[str, object]] = []
        reserved_total = 0.0

        for row in workable:
            limit_price = float(row["limit_price"])
            desired_cash = available_cash * float(row["weight"]) / total_weight
            quantity = int(math.floor(desired_cash / limit_price))
            reserve = float(quantity) * limit_price
            allocations.append(
                {
                    "candidate": row["candidate"],
                    "limit_price": limit_price,
                    "quantity": quantity,
                    "reserve": reserve,
                }
            )
            reserved_total += reserve

        leftover_cash = max(0.0, available_cash - reserved_total)
        while leftover_cash >= min(float(row["limit_price"]) for row in allocations):
            best = next(
                (
                    row
                    for row in sorted(
                        allocations,
                        key=lambda item: (
                            -float(cast(SignalCandidate, item["candidate"]).signal_score),
                            cast(SignalCandidate, item["candidate"]).ticker,
                        ),
                    )
                    if float(row["limit_price"]) <= leftover_cash
                ),
                None,
            )
            if best is None:
                break
            best["quantity"] = int(best["quantity"]) + 1
            best["reserve"] = float(best["reserve"]) + float(best["limit_price"])
            leftover_cash -= float(best["limit_price"])

        for row in allocations:
            candidate = cast(SignalCandidate, row["candidate"])
            quantity = int(row["quantity"])
            reserve = float(row["reserve"])
            limit_price = float(row["limit_price"])
            if quantity < 1 or reserve < self.execution_policy.min_order_notional:
                candidate.last_reason = "insufficient_sleeve_cash"
                candidate.last_updated_at = utc_now_iso()
                continue
            self._submit_entry_order(state, sleeve, candidate, quantity, limit_price, reserve)

    def _submit_entry_order(
        self,
        state: TraderStateSnapshot,
        sleeve: SleeveState,
        candidate: SignalCandidate,
        quantity: int,
        limit_price: float,
        reserve: float,
    ) -> bool:
        if quantity < 1 or reserve < self.execution_policy.min_order_notional:
            return False
        if sleeve.cash_balance + 1e-9 < reserve:
            candidate.last_reason = "insufficient_sleeve_cash"
            candidate.last_updated_at = utc_now_iso()
            return False

        sleeve.cash_balance -= reserve
        order = PendingOrder(
            local_order_id=new_id("ord"),
            kind="entry",
            side="BUY",
            ticker=candidate.ticker,
            sleeve_id=candidate.sleeve_id,
            quantity=quantity,
            limit_price=limit_price,
            placed_at=utc_now_iso(),
            status="submitted",
            reserved_cash=reserve,
            candidate_id=candidate.candidate_id,
        )
        try:
            broker_order = self.broker.place_order(
                BrokerOrderRequest(
                    order_ref=order.local_order_id,
                    symbol=order.ticker,
                    side=order.side,
                    quantity=order.quantity,
                    limit_price=order.limit_price,
                )
            )
        except Exception:
            sleeve.cash_balance += reserve
            raise
        order.broker_order_id = broker_order.broker_order_id
        order.broker_status = broker_order.status
        state.pending_orders.append(order)
        candidate.active_order_id = order.local_order_id
        candidate.status = "ordered"
        candidate.last_reason = ""
        candidate.last_updated_at = utc_now_iso()
        self.store.append_journal("entry_order_submitted", asdict(order))
        return True

    def _eligible_candidates(self, state: TraderStateSnapshot, now_et: datetime) -> list[SignalCandidate]:
        out: list[SignalCandidate] = []
        for candidate in state.candidates:
            if candidate.status in TERMINAL_CANDIDATE_STATUSES:
                continue
            intended = parse_iso_datetime(candidate.intended_entry_at)
            expiry = parse_iso_datetime(candidate.expires_at)
            if intended > now_et or expiry < now_et:
                continue
            out.append(candidate)
        out.sort(key=lambda row: (row.intended_entry_at, -row.signal_score, row.ticker))
        return out

    def _candidate_has_funding(self, state: TraderStateSnapshot, candidate: SignalCandidate) -> bool:
        sleeve = self._find_sleeve(state, candidate.sleeve_id)
        if sleeve is None:
            return False
        target_notional = self._target_notional_for_candidate(state, candidate, sleeve)
        committed_notional = self._committed_notional_for_candidate(state, candidate.candidate_id)
        remaining_notional = max(0.0, target_notional - committed_notional)
        return min(sleeve.cash_balance, remaining_notional) >= self.execution_policy.min_order_notional

    def _select_intraday_replacement_lot(
        self,
        *,
        state: TraderStateSnapshot,
        previous_day,
        reserved_lot_ids: set[str],
        candidate_decile: float,
    ) -> PositionLot | None:
        cohort: list[PositionLot] = []
        prev_day_iso = previous_day.isoformat()
        for lot in state.lots:
            if lot.status != "open" or lot.quantity <= 0:
                continue
            if lot.active_exit_order_id or lot.lot_id in reserved_lot_ids:
                continue
            if lot.entry_trade_day != prev_day_iso:
                continue
            cohort.append(lot)
        if not cohort:
            return None
        cohort.sort(
            key=lambda lot: (
                float(lot.entry_estimated_decile_score),
                lot.opened_at,
                lot.ticker,
            )
        )
        target = cohort[0]
        if float(candidate_decile) <= float(target.entry_estimated_decile_score):
            return None
        return target

    def _target_notional_for_candidate(
        self,
        state: TraderStateSnapshot,
        candidate: SignalCandidate,
        sleeve: SleeveState,
    ) -> float:
        batch = [
            row
            for row in state.candidates
            if row.sleeve_id == candidate.sleeve_id
            and row.intended_entry_at[:10] == candidate.intended_entry_at[:10]
            and row.status not in {"expired", "rejected"}
        ]
        batch_size = max(1, len(batch))
        if batch_size == 1:
            cap_fraction = self.budget_config.max_fraction_single_name
        elif batch_size == 2:
            cap_fraction = self.budget_config.max_fraction_two_names
        else:
            cap_fraction = self.budget_config.max_fraction_three_plus_names
        target_fraction = min(cap_fraction, candidate.advised_allocation_fraction if candidate.advised_allocation_fraction > 0 else cap_fraction)
        sleeve_equity = sleeve.last_equity if sleeve.last_equity > 0 else sleeve.cash_balance
        return max(0.0, sleeve_equity * target_fraction)

    def _committed_notional_for_candidate(self, state: TraderStateSnapshot, candidate_id: str) -> float:
        reserved = 0.0
        for order in state.pending_orders:
            if order.kind != "entry" or order.candidate_id != candidate_id:
                continue
            if order.status in {"cancelled"}:
                continue
            reserved += max(0.0, order.reserved_cash - order.released_cash)
        for lot in state.lots:
            if lot.candidate_id == candidate_id and lot.status == "open":
                reserved += float(lot.entry_value)
        return reserved

    def _build_limit_price(self, side: str, symbol: str, fallback: float | None) -> float | None:
        quote = self.broker.get_quote(symbol)
        reference = quote.reference_price(side, fallback)
        if reference is None or reference <= 0:
            return None
        if side.upper() == "BUY":
            adjusted = reference * (1.0 + self.execution_policy.buy_limit_buffer_bps / 10_000.0)
        else:
            adjusted = reference * max(0.0, 1.0 - self.execution_policy.sell_limit_buffer_bps / 10_000.0)
        return round(float(adjusted), 2)

    def _reconcile_orders_and_fills(self, state: TraderStateSnapshot) -> None:
        known_exec_ids = {row.execution_id for row in state.fills}
        for fill in self.broker.list_fills():
            if not fill.execution_id or fill.execution_id in known_exec_ids:
                continue
            self._apply_fill(state, fill)
            known_exec_ids.add(fill.execution_id)

        broker_orders = {row.broker_order_id: row for row in self.broker.list_orders(include_closed=True)}
        for order in state.pending_orders:
            if order.broker_order_id is None:
                continue
            broker_order = broker_orders.get(order.broker_order_id)
            if broker_order is None:
                continue
            order.broker_status = broker_order.status
            order.filled_quantity = broker_order.filled_quantity
            if broker_order.status in CANCELLED_ORDER_STATUSES:
                self._finalize_cancelled_order(state, order, broker_order.status)
            elif broker_order.status == "Filled" or order.filled_quantity >= order.quantity:
                self._finalize_filled_order(state, order)

    def _apply_fill(self, state: TraderStateSnapshot, fill: BrokerFillView) -> None:
        order = self._find_order_by_broker_ref(state, fill.broker_order_id, fill.order_ref)
        if order is None:
            return
        event = FillEvent(
            fill_id=new_id("fill"),
            execution_id=fill.execution_id,
            local_order_id=order.local_order_id,
            broker_order_id=fill.broker_order_id,
            ticker=fill.symbol,
            side=fill.side,
            quantity=int(fill.quantity),
            price=float(fill.price),
            filled_at=fill.filled_at,
            commission=float(fill.commission),
        )
        state.fills.append(event)
        order.filled_quantity += int(fill.quantity)
        order.actual_notional += float(fill.quantity) * float(fill.price)
        order.broker_status = "PartiallyFilled" if order.filled_quantity < order.quantity else "Filled"

        if order.kind == "entry":
            self._apply_entry_fill(state, order, event)
        else:
            self._apply_exit_fill(state, order, event)
        self.store.append_journal("fill_applied", asdict(event))

    def _apply_entry_fill(self, state: TraderStateSnapshot, order: PendingOrder, event: FillEvent) -> None:
        candidate = next((row for row in state.candidates if row.candidate_id == order.candidate_id), None)
        if candidate is None:
            return
        lot = next((row for row in state.lots if row.entry_order_id == order.local_order_id), None)
        if lot is None:
            filled_at = _parse_fill_timestamp(event.filled_at)
            due_exit_at = exit_at_tplus_open(filled_at, ACTIVE_STRATEGY.sell_after_trading_days)
            lot = PositionLot(
                lot_id=new_id("lot"),
                candidate_id=candidate.candidate_id,
                ticker=candidate.ticker,
                sleeve_id=candidate.sleeve_id,
                entry_order_id=order.local_order_id,
                opened_at=filled_at.isoformat(),
                due_exit_at=due_exit_at.isoformat(),
                entry_signal_score=float(candidate.signal_score),
                entry_estimated_decile_score=float(candidate.estimated_decile_score),
                entry_trade_day=candidate.entry_trade_day or filled_at.date().isoformat(),
            )
            state.lots.append(lot)
            candidate.linked_lot_id = lot.lot_id
        lot.entry_quantity += int(event.quantity)
        lot.quantity += int(event.quantity)
        lot.entry_value += float(event.quantity) * float(event.price)
        candidate.status = "partially_filled"
        candidate.last_updated_at = utc_now_iso()

    def _apply_exit_fill(self, state: TraderStateSnapshot, order: PendingOrder, event: FillEvent) -> None:
        lot = self._find_lot(state, order.lot_id)
        sleeve = self._find_sleeve(state, order.sleeve_id)
        if lot is None or sleeve is None:
            return
        quantity = min(int(event.quantity), int(lot.quantity))
        proceeds = float(quantity) * float(event.price)
        cost_basis = float(quantity) * float(lot.avg_entry_price)
        pnl = proceeds - cost_basis - float(event.commission)
        sleeve.cash_balance += proceeds
        sleeve.realized_pnl += pnl
        lot.quantity -= quantity
        lot.exit_value += proceeds
        lot.realized_pnl += pnl
        if lot.quantity <= 0:
            lot.quantity = 0
            lot.status = "closed"
            planned = next((row for row in state.planned_exits if row.lot_id == lot.lot_id), None)
            if planned is not None:
                planned.status = "filled"

    def _finalize_cancelled_order(self, state: TraderStateSnapshot, order: PendingOrder, broker_status: str) -> None:
        if order.status == "cancelled":
            return
        order.status = "cancelled"
        order.broker_status = broker_status
        if order.kind == "entry":
            sleeve = self._find_sleeve(state, order.sleeve_id)
            if sleeve is not None:
                refund = max(0.0, order.reserved_cash - order.actual_notional - order.released_cash)
                sleeve.cash_balance += refund
                order.released_cash += refund
            candidate = next((row for row in state.candidates if row.candidate_id == order.candidate_id), None)
            if candidate is not None:
                candidate.active_order_id = None
                candidate.status = "partially_filled" if candidate.linked_lot_id else "pending"
                candidate.last_updated_at = utc_now_iso()
        else:
            lot = self._find_lot(state, order.lot_id)
            if lot is not None:
                lot.active_exit_order_id = None
        self.store.append_journal("order_cancelled", asdict(order))

    def _finalize_filled_order(self, state: TraderStateSnapshot, order: PendingOrder) -> None:
        if order.status == "filled":
            return
        order.status = "filled"
        order.broker_status = "Filled"
        if order.kind == "entry":
            sleeve = self._find_sleeve(state, order.sleeve_id)
            if sleeve is not None:
                refund = max(0.0, order.reserved_cash - order.actual_notional - order.released_cash)
                sleeve.cash_balance += refund
                order.released_cash += refund
            candidate = next((row for row in state.candidates if row.candidate_id == order.candidate_id), None)
            if candidate is not None:
                candidate.active_order_id = None
                candidate.status = "partially_filled" if candidate.linked_lot_id else "pending"
                candidate.last_updated_at = utc_now_iso()
        else:
            lot = self._find_lot(state, order.lot_id)
            if lot is not None:
                lot.active_exit_order_id = None
        self.store.append_journal("order_filled", asdict(order))

    def _cancel_order(self, state: TraderStateSnapshot, order: PendingOrder, reason: str) -> None:
        if order.broker_order_id is not None:
            self.broker.cancel_order(order.broker_order_id)
        order.status = "cancel_requested"
        order.broker_status = reason
        self.store.append_journal(
            "order_cancel_requested",
            {"local_order_id": order.local_order_id, "broker_order_id": order.broker_order_id, "reason": reason},
        )

    def _archive_candidates(self, state: TraderStateSnapshot) -> None:
        rows = [
            {
                "candidate_id": row.candidate_id,
                "ticker": row.ticker,
                "status": row.status,
                "signal_score": row.signal_score,
                "estimated_decile_score": row.estimated_decile_score,
                "sleeve_id": row.sleeve_id,
                "entry_bucket": row.entry_bucket,
                "entry_trade_day": row.entry_trade_day,
                "intended_entry_at": row.intended_entry_at,
                "expires_at": row.expires_at,
                "replacement_for_lot_id": row.replacement_for_lot_id,
                "last_reason": row.last_reason,
                "last_updated_at": row.last_updated_at,
            }
            for row in state.candidates
        ]
        self.signal_archive_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(self.signal_archive_path, index=False)

    def _find_sleeve(self, state: TraderStateSnapshot, sleeve_id: str) -> SleeveState | None:
        return next((row for row in state.sleeves if row.sleeve_id == sleeve_id), None)

    def _find_order(self, state: TraderStateSnapshot, local_order_id: str | None) -> PendingOrder | None:
        if not local_order_id:
            return None
        return next((row for row in state.pending_orders if row.local_order_id == local_order_id), None)

    def _find_order_by_broker_ref(
        self,
        state: TraderStateSnapshot,
        broker_order_id: int | None,
        order_ref: str,
    ) -> PendingOrder | None:
        for order in state.pending_orders:
            if broker_order_id is not None and order.broker_order_id == broker_order_id:
                return order
            if order_ref and order.local_order_id == order_ref:
                return order
        return None

    def _find_lot(self, state: TraderStateSnapshot, lot_id: str | None) -> PositionLot | None:
        if not lot_id:
            return None
        return next((row for row in state.lots if row.lot_id == lot_id), None)

    def _planned_exit_for_lot(self, state: TraderStateSnapshot, lot: PositionLot) -> PlannedExit | None:
        return next((row for row in state.planned_exits if row.lot_id == lot.lot_id), None)

    def _submit_exit_order(
        self,
        state: TraderStateSnapshot,
        lot: PositionLot,
        planned: PlannedExit,
        *,
        reason: str,
    ) -> bool:
        current_order = self._find_order(state, lot.active_exit_order_id)
        if current_order is not None:
            return False
        limit_price = self._build_limit_price("SELL", lot.ticker, lot.last_mark_price or lot.avg_entry_price)
        if limit_price is None:
            self.logger.warning("No exit quote available for %s", lot.ticker)
            return False
        order = PendingOrder(
            local_order_id=new_id("ord"),
            kind="exit",
            side="SELL",
            ticker=lot.ticker,
            sleeve_id=lot.sleeve_id,
            quantity=int(lot.quantity),
            limit_price=limit_price,
            placed_at=utc_now_iso(),
            status="submitted",
            lot_id=lot.lot_id,
        )
        broker_order = self.broker.place_order(
            BrokerOrderRequest(
                order_ref=order.local_order_id,
                symbol=order.ticker,
                side=order.side,
                quantity=order.quantity,
                limit_price=order.limit_price,
            )
        )
        order.broker_order_id = broker_order.broker_order_id
        order.broker_status = broker_order.status
        state.pending_orders.append(order)
        lot.active_exit_order_id = order.local_order_id
        planned.last_order_id = order.local_order_id
        planned.status = "submitted"
        self.store.append_journal(reason, asdict(order))
        return True

    def _order_is_stale(self, order: PendingOrder, now_et: datetime) -> bool:
        placed_at = parse_iso_datetime(order.placed_at)
        return (now_et - placed_at.astimezone(ET)).total_seconds() >= self.execution_policy.replace_interval_seconds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IBKR paper-trading service for the live insider-trading strategy.")
    parser.add_argument("--host", default=IBKR_CONFIG.host)
    parser.add_argument("--port", type=int, default=IBKR_CONFIG.port)
    parser.add_argument("--client-id", type=int, default=IBKR_CONFIG.client_id)
    parser.add_argument("--account-id", default=IBKR_CONFIG.account_id)
    parser.add_argument("--state-file", default=LIVE_PATHS.trader_state_file)
    parser.add_argument("--journal-file", default=LIVE_PATHS.trader_journal_file)
    parser.add_argument("--signal-archive-file", default=LIVE_PATHS.trader_signal_archive_file)
    parser.add_argument("--alert-snapshot-file", default=LIVE_PATHS.alert_snapshot_file)
    parser.add_argument("--cycle-seconds", type=int, default=EXECUTION_POLICY.cycle_seconds)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Use the in-memory broker adapter instead of IBKR.")
    return parser


def make_broker(args) -> BrokerAdapter:
    if args.dry_run:
        broker = DryRunBrokerAdapter()
        broker.connect()
        return broker
    return IbkrBrokerAdapter(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        account_id=args.account_id,
        connect_timeout_seconds=IBKR_CONFIG.connect_timeout_seconds,
        readonly=IBKR_CONFIG.readonly,
        routing_exchange=EXECUTION_POLICY.routing_exchange,
        currency=EXECUTION_POLICY.currency,
        quote_wait_seconds=EXECUTION_POLICY.quote_wait_seconds,
    )


def main() -> None:
    load_dotenv()
    logger = setup_logger()
    args = build_arg_parser().parse_args()
    broker = make_broker(args)
    store = StateStore(Path(args.state_file), Path(args.journal_file))
    trader = IbkrPaperTrader(
        broker=broker,
        store=store,
        alert_snapshot_path=Path(args.alert_snapshot_file),
        signal_archive_path=Path(args.signal_archive_file),
        logger=logger,
    )

    try:
        while True:
            start = time.time()
            now_et = datetime.now(ET)
            if is_weekend_shutdown_window(now_et):
                sleep_seconds = seconds_until_weekend_shutdown_end(now_et)
                logger.info("Weekend shutdown active. Sleeping %.1f hours until Monday 00:00 ET.", sleep_seconds / 3600.0)
                if args.once:
                    break
                time.sleep(sleep_seconds)
                continue

            trader.run_once(now_et)
            if args.once:
                break
            elapsed = time.time() - start
            target_cycle = (
                EXECUTION_POLICY.open_order_poll_seconds
                if is_regular_trading_hours(now_et)
                else float(args.cycle_seconds)
            )
            sleep_seconds = max(0.0, float(target_cycle) - elapsed)
            logger.info("Cycle complete. Sleeping %.1f sec", sleep_seconds)
            time.sleep(sleep_seconds)
    finally:
        if not args.dry_run:
            broker.disconnect()


if __name__ == "__main__":
    main()
