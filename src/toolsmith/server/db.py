"""In-memory commerce database backing the tool server.

Each episode gets its own ``Database`` loaded from the JSON fixtures in
``data/domain``. Tasks are graded against the final state, so the database has
to be restored between episodes; ``snapshot``/``restore`` handle that without
re-reading the fixtures 800 times per evaluation.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TABLES = (
    "products",
    "inventory",
    "customers",
    "orders",
    "shipments",
    "tickets",
    "refunds",
)

ORDER_STATES = ("pending", "processing", "shipped", "delivered", "cancelled", "returned")
TICKET_STATES = ("open", "pending_customer", "resolved", "closed")
REFUNDABLE_STATES = ("cancelled", "returned")


class DomainError(Exception):
    """Raised by tool implementations for conditions the agent can act on."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Database:
    products: dict[str, dict[str, Any]]
    inventory: dict[str, dict[str, Any]]
    customers: dict[str, dict[str, Any]]
    orders: dict[str, dict[str, Any]]
    shipments: dict[str, dict[str, Any]]
    tickets: dict[str, dict[str, Any]]
    refunds: dict[str, dict[str, Any]]
    clock: str = "2026-01-17T09:00:00Z"
    _counters: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._counters is None:
            self._counters = {"order": 0, "ticket": 0, "refund": 0, "shipment": 0}

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, domain_dir: str | Path) -> Database:
        root = Path(domain_dir)
        payload = {name: json.loads((root / f"{name}.json").read_text()) for name in TABLES}
        return cls(**payload)

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                **{name: getattr(self, name) for name in TABLES},
                "clock": self.clock,
                "_counters": self._counters,
            }
        )

    def restore(self, snapshot: dict[str, Any]) -> None:
        restored = copy.deepcopy(snapshot)
        for name in TABLES:
            setattr(self, name, restored[name])
        self.clock = restored["clock"]
        self._counters = restored["_counters"]

    # -- lookups ---------------------------------------------------------

    def product(self, product_id: str) -> dict[str, Any]:
        record = self.products.get(product_id)
        if record is None:
            raise DomainError("not_found", f"No product with id {product_id}.")
        return record

    def customer(self, customer_id: str) -> dict[str, Any]:
        record = self.customers.get(customer_id)
        if record is None:
            raise DomainError("not_found", f"No customer with id {customer_id}.")
        return record

    def order(self, order_id: str) -> dict[str, Any]:
        record = self.orders.get(order_id)
        if record is None:
            raise DomainError("not_found", f"No order with id {order_id}.")
        return record

    def ticket(self, ticket_id: str) -> dict[str, Any]:
        record = self.tickets.get(ticket_id)
        if record is None:
            raise DomainError("not_found", f"No ticket with id {ticket_id}.")
        return record

    def stock(self, sku: str) -> dict[str, Any]:
        record = self.inventory.get(sku)
        if record is None:
            raise DomainError("not_found", f"No inventory record for sku {sku}.")
        return record

    # -- mutation helpers ------------------------------------------------

    def next_id(self, kind: str) -> str:
        self._counters[kind] = self._counters.get(kind, 0) + 1
        prefix = {"order": "W", "ticket": "T", "refund": "R", "shipment": "S"}[kind]
        return f"{prefix}{self._counters[kind]:05d}"

    def now(self) -> str:
        return self.clock

    def advance(self, seconds: int) -> None:
        moment = datetime.strptime(self.clock, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        moment = moment.fromtimestamp(moment.timestamp() + seconds, tz=UTC)
        self.clock = moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    def order_total(self, order: dict[str, Any]) -> float:
        return round(sum(item["unit_price"] * item["quantity"] for item in order["items"]), 2)

    def reserve(self, sku: str, quantity: int) -> None:
        record = self.stock(sku)
        if record["available"] < quantity:
            raise DomainError(
                "insufficient_stock",
                f"Only {record['available']} units of {sku} are available.",
            )
        record["available"] -= quantity
        record["reserved"] += quantity

    def release(self, sku: str, quantity: int) -> None:
        record = self.stock(sku)
        record["available"] += quantity
        record["reserved"] = max(0, record["reserved"] - quantity)
