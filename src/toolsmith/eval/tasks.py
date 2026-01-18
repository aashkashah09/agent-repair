"""Task suite: loading, validation and grading.

A task pairs an instruction given to the simulated user with a set of checks
run against the database once the episode ends. Checks see both the final
database and the snapshot taken before the episode started, so "did not create
anything" is as expressible as "created exactly this".

Grading is on final state plus, where the task is a question rather than an
action, on what the agent told the user. Nothing grades the trajectory: two
different tool sequences that leave the same state are both correct.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..server.db import Database

PERSONAS = ("underspecified", "goal_change", "misinformed", "abandonment")
DOMAINS = ("commerce", "support")


class TaskError(Exception):
    pass


@dataclass
class Task:
    task_id: str
    domain: str
    persona: str
    instruction: str
    known: dict[str, Any]
    checks: list[dict[str, Any]]
    tools_expected: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Task:
        return cls(
            task_id=payload["task_id"],
            domain=payload["domain"],
            persona=payload["persona"],
            instruction=payload["instruction"],
            known=payload.get("known", {}),
            checks=payload["checks"],
            tools_expected=payload.get("tools_expected", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "persona": self.persona,
            "instruction": self.instruction,
            "known": self.known,
            "checks": self.checks,
            "tools_expected": self.tools_expected,
        }


def load_tasks(path: str | Path) -> list[Task]:
    tasks = [
        Task.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: list[Task]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise TaskError(f"duplicate task_id {task.task_id}")
        seen.add(task.task_id)
        if task.domain not in DOMAINS:
            raise TaskError(f"{task.task_id}: unknown domain {task.domain!r}")
        if task.persona not in PERSONAS:
            raise TaskError(f"{task.task_id}: unknown persona {task.persona!r}")
        if not task.instruction.strip():
            raise TaskError(f"{task.task_id}: empty instruction")
        if not task.checks:
            raise TaskError(f"{task.task_id}: no checks")
        for check in task.checks:
            if check.get("type") not in CHECKS:
                raise TaskError(f"{task.task_id}: unknown check {check.get('type')!r}")


# -- grading -------------------------------------------------------------
#
# Every check takes the final database, the pre-episode snapshot, the check
# body and the agent's closing message, and answers a single yes/no question.

Baseline = dict[str, Any]


def _check_order_status(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    order = db.orders.get(check["order_id"])
    return order is not None and order["status"] == check["equals"]


def _check_order_items(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    order = db.orders.get(check["order_id"])
    if order is None:
        return False
    for item in order["items"]:
        if item["sku"] == check["sku"]:
            return item["quantity"] == check["quantity"]
    return check.get("quantity", 0) == 0


def _check_order_lacks_sku(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    order = db.orders.get(check["order_id"])
    return order is not None and all(i["sku"] != check["sku"] for i in order["items"])


def _check_order_created(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    wanted = {entry["sku"]: entry["quantity"] for entry in check["items"]}
    for order_id, order in db.orders.items():
        if order_id in base["orders"]:
            continue
        if order["customer_id"] != check["customer_id"]:
            continue
        if order["status"] not in ("pending", "processing"):
            continue
        if {item["sku"]: item["quantity"] for item in order["items"]} == wanted:
            return True
    return False


def _check_no_new_orders(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    return set(db.orders) <= set(base["orders"])


def _check_refund(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    for refund_id, refund in db.refunds.items():
        if refund_id in base["refunds"]:
            continue
        if refund["order_id"] != check["order_id"]:
            continue
        if "amount" in check and abs(refund["amount"] - check["amount"]) > 0.01:
            continue
        if "method" in check and refund["method"] != check["method"]:
            continue
        return True
    return False


def _check_no_refund(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    return all(
        refund["order_id"] != check["order_id"]
        for refund_id, refund in db.refunds.items()
        if refund_id not in base["refunds"]
    )


def _check_refund_total(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    total = sum(r["amount"] for r in db.refunds.values() if r["order_id"] == check["order_id"])
    return abs(total - check["amount"]) <= 0.01


def _check_ticket_created(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    for ticket_id, ticket in db.tickets.items():
        if ticket_id in base["tickets"]:
            continue
        if ticket["customer_id"] != check["customer_id"]:
            continue
        if "category" in check and ticket["category"] != check["category"]:
            continue
        if "order_id" in check and ticket["order_id"] != check["order_id"]:
            continue
        return True
    return False


def _check_no_new_tickets(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    return set(db.tickets) <= set(base["tickets"])


def _check_ticket_field(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    ticket = db.tickets.get(check["ticket_id"])
    return ticket is not None and ticket[check["field"]] == check["equals"]


def _check_ticket_note(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    ticket = db.tickets.get(check["ticket_id"])
    if ticket is None:
        return False
    before = len(base["tickets"].get(check["ticket_id"], {}).get("notes", []))
    return len(ticket["notes"]) >= before + check.get("added", 1)


def _check_customer_field(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    customer = db.customers.get(check["customer_id"])
    if customer is None:
        return False
    return str(customer[check["field"]]).strip().lower() == str(check["equals"]).strip().lower()


def _check_address_present(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    customer = db.customers.get(check["customer_id"])
    if customer is None:
        return False
    for address in customer["addresses"]:
        if all(
            str(address.get(key, "")).strip().lower() == str(value).strip().lower()
            for key, value in check["fields"].items()
        ):
            return True
    return False


def _check_inventory_available(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    record = db.inventory.get(check["sku"])
    return record is not None and record["available"] == check["available"]


def _check_state_unchanged(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    return (
        set(db.orders) <= set(base["orders"])
        and set(db.tickets) <= set(base["tickets"])
        and set(db.refunds) <= set(base["refunds"])
        and all(
            db.orders[order_id]["status"] == base["orders"][order_id]["status"]
            for order_id in base["orders"]
        )
    )


def _check_communicated(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    haystack = re.sub(r"[\s,]+", " ", reply.lower())
    return any(str(value).lower() in haystack for value in check["any_of"])


def _check_communicated_all(db: Database, base: Baseline, check: dict[str, Any], reply: str) -> bool:
    haystack = re.sub(r"[\s,]+", " ", reply.lower())
    return all(str(value).lower() in haystack for value in check["all_of"])


CHECKS = {
    "order_status": _check_order_status,
    "order_items": _check_order_items,
    "order_lacks_sku": _check_order_lacks_sku,
    "order_created": _check_order_created,
    "no_new_orders": _check_no_new_orders,
    "refund": _check_refund,
    "no_refund": _check_no_refund,
    "refund_total": _check_refund_total,
    "ticket_created": _check_ticket_created,
    "no_new_tickets": _check_no_new_tickets,
    "ticket_field": _check_ticket_field,
    "ticket_note": _check_ticket_note,
    "customer_field": _check_customer_field,
    "address_present": _check_address_present,
    "inventory_available": _check_inventory_available,
    "state_unchanged": _check_state_unchanged,
    "communicated": _check_communicated,
    "communicated_all": _check_communicated_all,
}


@dataclass
class Grade:
    success: bool
    failed_checks: list[str]


def grade(task: Task, db: Database, baseline: Baseline, final_reply: str) -> Grade:
    failed = [
        f"{index}:{check['type']}"
        for index, check in enumerate(task.checks)
        if not CHECKS[check["type"]](db, baseline, check, final_reply)
    ]
    return Grade(success=not failed, failed_checks=failed)


def iter_tasks(path: str | Path) -> Iterator[Task]:
    yield from load_tasks(path)
