from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


@dataclass
class SignalCandidate:
    candidate_id: str
    event_key: str
    ticker: str
    scored_at: str
    intended_entry_at: str
    expires_at: str
    sleeve_id: str
    signal_score: float
    estimated_decile_score: float
    advised_allocation_fraction: float
    score_column: str
    buy_price_hint: float | None = None
    status: str = "pending"
    linked_lot_id: str | None = None
    active_order_id: str | None = None
    last_reason: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    last_updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class PositionLot:
    lot_id: str
    candidate_id: str
    ticker: str
    sleeve_id: str
    entry_order_id: str
    opened_at: str
    due_exit_at: str
    entry_quantity: int = 0
    quantity: int = 0
    entry_value: float = 0.0
    exit_value: float = 0.0
    realized_pnl: float = 0.0
    last_mark_price: float | None = None
    status: str = "open"
    active_exit_order_id: str | None = None

    @property
    def avg_entry_price(self) -> float:
        if self.entry_quantity <= 0 or self.entry_value <= 0:
            return 0.0
        return float(self.entry_value) / float(self.entry_quantity)


@dataclass
class PlannedExit:
    exit_id: str
    lot_id: str
    ticker: str
    sleeve_id: str
    due_at: str
    status: str = "scheduled"
    last_order_id: str | None = None


@dataclass
class PendingOrder:
    local_order_id: str
    kind: str
    side: str
    ticker: str
    sleeve_id: str
    quantity: int
    limit_price: float
    placed_at: str
    status: str
    broker_order_id: int | None = None
    broker_status: str = ""
    filled_quantity: int = 0
    reserved_cash: float = 0.0
    actual_notional: float = 0.0
    released_cash: float = 0.0
    replace_count: int = 0
    candidate_id: str | None = None
    lot_id: str | None = None


@dataclass
class FillEvent:
    fill_id: str
    execution_id: str
    local_order_id: str
    broker_order_id: int | None
    ticker: str
    side: str
    quantity: int
    price: float
    filled_at: str
    commission: float = 0.0


@dataclass
class SleeveState:
    sleeve_id: str
    starting_cash: float
    cash_balance: float
    realized_pnl: float = 0.0
    last_equity: float = 0.0
    last_marked_at: str | None = None


@dataclass
class TraderStateSnapshot:
    version: int = 1
    sleeves: list[SleeveState] = field(default_factory=list)
    candidates: list[SignalCandidate] = field(default_factory=list)
    lots: list[PositionLot] = field(default_factory=list)
    planned_exits: list[PlannedExit] = field(default_factory=list)
    pending_orders: list[PendingOrder] = field(default_factory=list)
    fills: list[FillEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraderStateSnapshot":
        return cls(
            version=int(data.get("version", 1)),
            sleeves=[SleeveState(**row) for row in data.get("sleeves", [])],
            candidates=[SignalCandidate(**row) for row in data.get("candidates", [])],
            lots=[PositionLot(**row) for row in data.get("lots", [])],
            planned_exits=[PlannedExit(**row) for row in data.get("planned_exits", [])],
            pending_orders=[PendingOrder(**row) for row in data.get("pending_orders", [])],
            fills=[FillEvent(**row) for row in data.get("fills", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateStore:
    def __init__(self, state_path: Path, journal_path: Path) -> None:
        self.state_path = state_path
        self.journal_path = journal_path

    def load(self) -> TraderStateSnapshot:
        if not self.state_path.exists():
            return TraderStateSnapshot()
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return TraderStateSnapshot.from_dict(raw)

    def save(self, state: TraderStateSnapshot) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def append_journal(self, event_type: str, payload: dict[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "recorded_at": utc_now_iso(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
