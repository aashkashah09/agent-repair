"""Tool implementations for the commerce and support domain.

Fourteen tools covering catalogue lookup, order lifecycle, refunds, shipment
tracking, and support tickets. Implementations are deliberately independent of
the published schemas: ``seeding`` perturbs schemas and, for a few defect
patterns, wraps the return values here, so the behaviour a suite exercises is
always the behaviour these functions define.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .db import REFUNDABLE_STATES, Database, DomainError

REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}

# Closed value sets the implementations enforce. Kept here rather than inline
# so the published schemas and the permission analysis have one thing to agree
# with.
CANCELLATION_REASONS = (
    "no_longer_needed",
    "ordered_by_mistake",
    "found_better_price",
    "defective_or_damaged",
)
REFUND_METHODS = ("original_payment", "store_credit")
TICKET_CATEGORIES = ("shipping", "billing", "product_defect", "return_request", "other")
TICKET_STATUSES = ("open", "pending_customer", "resolved", "closed")
TICKET_PRIORITIES = ("low", "normal", "high", "urgent")
MODIFY_MODES = ("add", "remove", "replace")
ORDER_STATUSES = ("pending", "processing", "shipped", "delivered", "cancelled", "returned")
PAGE_SIZE = 20

ACCEPTED_VALUES: dict[tuple[str, str], tuple[str, ...]] = {
    ("cancel_order", "reason"): CANCELLATION_REASONS,
    ("process_refund", "method"): REFUND_METHODS,
    ("create_support_ticket", "category"): TICKET_CATEGORIES,
    ("update_support_ticket", "status"): TICKET_STATUSES,
    ("update_support_ticket", "priority"): TICKET_PRIORITIES,
    ("modify_order_items", "mode"): MODIFY_MODES,
    ("list_orders", "status"): ORDER_STATUSES,
}


def tool(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[name] = fn
        fn.tool_name = name  # type: ignore[attr-defined]
        return fn

    return register


def _parse_timestamp(value: str) -> datetime:
    """Accept an ISO-8601 instant. Anything else is a caller error."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DomainError(
            "invalid_argument",
            f"{value!r} is not an ISO-8601 timestamp (expected e.g. 2026-01-01T00:00:00Z).",
        ) from exc
    if moment.tzinfo is None:
        raise DomainError(
            "invalid_argument",
            f"{value!r} is missing a UTC offset (expected e.g. 2026-01-01T00:00:00Z).",
        )
    return moment.astimezone(UTC)


def _require(value: Any, field: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise DomainError("invalid_argument", f"{field} is required.")
    return value


def _enum(value: str, allowed: tuple[str, ...], field: str) -> str:
    if value not in allowed:
        raise DomainError(
            "invalid_argument",
            f"{field} must be one of {', '.join(allowed)}; got {value!r}.",
        )
    return value


# -- catalogue -----------------------------------------------------------


@tool("search_products")
def search_products(
    db: Database,
    query: str,
    category: str | None = None,
    max_price: float | None = None,
    in_stock_only: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    _require(query, "query")
    known = {product["category"] for product in db.products.values()}
    if category is not None and category not in known:
        raise DomainError(
            "invalid_argument",
            f"Unknown category {category!r}. Known categories: {', '.join(sorted(known))}.",
        )
    terms = [word for word in re.split(r"\W+", query.lower()) if word]
    hits = []
    for product in db.products.values():
        haystack = f"{product['name']} {product['description']} {product['category']}".lower()
        if terms and not any(term in haystack for term in terms):
            continue
        if category is not None and product["category"] != category:
            continue
        if max_price is not None and product["price"] > max_price:
            continue
        available = sum(db.inventory[sku]["available"] for sku in product["skus"])
        if in_stock_only and available <= 0:
            continue
        hits.append(
            {
                "product_id": product["product_id"],
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "units_available": available,
            }
        )
    hits.sort(key=lambda row: (-row["units_available"], row["price"]))
    return {"results": hits[:limit], "total_matches": len(hits)}


@tool("get_product_detail")
def get_product_detail(db: Database, product_id: str) -> dict[str, Any]:
    product = db.product(_require(product_id, "product_id"))
    variants = [
        {
            "sku": sku,
            "option": db.inventory[sku]["option"],
            "available": db.inventory[sku]["available"],
        }
        for sku in product["skus"]
    ]
    return {
        "product_id": product["product_id"],
        "name": product["name"],
        "category": product["category"],
        "description": product["description"],
        "price": product["price"],
        "variants": variants,
    }


@tool("check_inventory")
def check_inventory(db: Database, sku: str) -> dict[str, Any]:
    record = db.stock(_require(sku, "sku"))
    return {
        "sku": sku,
        "product_id": record["product_id"],
        "option": record["option"],
        "available": record["available"],
        "reserved": record["reserved"],
        "restock_eta": record["restock_eta"],
    }


# -- customers -----------------------------------------------------------


@tool("get_customer")
def get_customer(
    db: Database,
    customer_id: str | None = None,
    email: str | None = None,
) -> dict[str, Any]:
    if customer_id is None and email is None:
        raise DomainError("invalid_argument", "Provide either customer_id or email.")
    if customer_id is not None:
        record = db.customer(customer_id)
    else:
        matches = [c for c in db.customers.values() if c["email"].lower() == email.lower()]
        if not matches:
            raise DomainError("not_found", f"No customer with email {email}.")
        record = matches[0]
    return {
        "customer_id": record["customer_id"],
        "name": record["name"],
        "email": record["email"],
        "phone": record["phone"],
        "membership": record["membership"],
        "addresses": record["addresses"],
        "payment_methods": [
            {"payment_method_id": pm["payment_method_id"], "label": pm["label"]}
            for pm in record["payment_methods"]
        ],
    }


@tool("update_customer_profile")
def update_customer_profile(
    db: Database,
    customer_id: str,
    email: str | None = None,
    phone: str | None = None,
    address: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = db.customer(_require(customer_id, "customer_id"))
    updated: list[str] = []
    if email is not None:
        if "@" not in email:
            raise DomainError("invalid_argument", f"{email!r} is not a valid email address.")
        record["email"] = email
        updated.append("email")
    if phone is not None:
        record["phone"] = phone
        updated.append("phone")
    if address is not None:
        if not isinstance(address, dict):
            raise DomainError(
                "invalid_argument",
                "address must be an object with line1, city, state and postal_code.",
            )
        missing = [f for f in ("line1", "city", "state", "postal_code") if not address.get(f)]
        if missing:
            raise DomainError(
                "invalid_argument",
                f"address is missing required field(s): {', '.join(missing)}.",
            )
        entry = {
            "address_id": address.get("address_id") or f"A{len(record['addresses']) + 1:03d}",
            "line1": address["line1"],
            "line2": address.get("line2", ""),
            "city": address["city"],
            "state": address["state"],
            "postal_code": address["postal_code"],
            "country": address.get("country", "US"),
        }
        existing = {a["address_id"]: i for i, a in enumerate(record["addresses"])}
        if entry["address_id"] in existing:
            record["addresses"][existing[entry["address_id"]]] = entry
        else:
            record["addresses"].append(entry)
        updated.append("address")
    if not updated:
        raise DomainError("invalid_argument", "No fields to update were supplied.")
    return {"customer_id": customer_id, "updated_fields": updated, "profile": record}


# -- orders --------------------------------------------------------------


@tool("list_orders")
def list_orders(
    db: Database,
    customer_id: str,
    status: str | None = None,
    since: str | None = None,
    limit: int = PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    db.customer(_require(customer_id, "customer_id"))
    rows = [o for o in db.orders.values() if o["customer_id"] == customer_id]
    if status is not None:
        rows = [o for o in rows if o["status"] == status]
    if since is not None:
        threshold = _parse_timestamp(since)
        rows = [o for o in rows if _parse_timestamp(o["placed_at"]) >= threshold]
    rows.sort(key=lambda o: o["placed_at"], reverse=True)
    total = len(rows)
    start = int(cursor) if cursor else 0
    page = rows[start : start + min(limit, PAGE_SIZE)]
    next_cursor = str(start + len(page)) if start + len(page) < total else None
    return {
        "orders": [
            {
                "order_id": o["order_id"],
                "status": o["status"],
                "placed_at": o["placed_at"],
                "total": db.order_total(o),
                "item_count": sum(item["quantity"] for item in o["items"]),
            }
            for o in page
        ],
        "total_matches": total,
        "next_cursor": next_cursor,
    }


@tool("get_order")
def get_order(db: Database, order_id: str) -> dict[str, Any]:
    order = db.order(_require(order_id, "order_id"))
    return {
        "order_id": order["order_id"],
        "customer_id": order["customer_id"],
        "status": order["status"],
        "placed_at": order["placed_at"],
        "items": order["items"],
        "total": db.order_total(order),
        "shipping_address": order["shipping_address"],
        "payment_method_id": order["payment_method_id"],
        "shipment_id": order.get("shipment_id"),
    }


@tool("create_order")
def create_order(
    db: Database,
    customer_id: str,
    items: list[dict[str, Any]],
    payment_method_id: str,
    address_id: str | None = None,
) -> dict[str, Any]:
    customer = db.customer(_require(customer_id, "customer_id"))
    if not items:
        raise DomainError("invalid_argument", "items must contain at least one entry.")
    known_payment = {pm["payment_method_id"] for pm in customer["payment_methods"]}
    if payment_method_id not in known_payment:
        raise DomainError(
            "invalid_argument",
            f"{payment_method_id} is not a payment method on this account.",
        )
    addresses = {a["address_id"]: a for a in customer["addresses"]}
    if address_id is None:
        if len(addresses) != 1:
            raise DomainError(
                "invalid_argument",
                "address_id is required when the account has more than one address.",
            )
        address = next(iter(addresses.values()))
    else:
        if address_id not in addresses:
            raise DomainError(
                "invalid_argument", f"{address_id} is not an address on this account."
            )
        address = addresses[address_id]

    resolved = []
    for entry in items:
        sku = _require(entry.get("sku"), "items[].sku")
        quantity = int(entry.get("quantity", 1))
        if quantity < 1:
            raise DomainError("invalid_argument", "items[].quantity must be at least 1.")
        stock = db.stock(sku)
        resolved.append(
            {
                "sku": sku,
                "product_id": stock["product_id"],
                "option": stock["option"],
                "quantity": quantity,
                "unit_price": db.product(stock["product_id"])["price"],
            }
        )
    for entry in resolved:
        db.reserve(entry["sku"], entry["quantity"])

    order_id = db.next_id("order")
    order = {
        "order_id": order_id,
        "customer_id": customer_id,
        "status": "pending",
        "placed_at": db.now(),
        "items": resolved,
        "shipping_address": address,
        "payment_method_id": payment_method_id,
        "shipment_id": None,
    }
    db.orders[order_id] = order
    return {"order_id": order_id, "status": "pending", "total": db.order_total(order)}


@tool("cancel_order")
def cancel_order(db: Database, order_id: str, reason: str) -> dict[str, Any]:
    order = db.order(_require(order_id, "order_id"))
    _enum(_require(reason, "reason"), CANCELLATION_REASONS, "reason")
    if order["status"] in ("shipped", "delivered"):
        raise DomainError(
            "invalid_state",
            f"Order {order_id} is {order['status']} and can no longer be cancelled; "
            "open a return request instead.",
        )
    if order["status"] == "cancelled":
        raise DomainError("invalid_state", f"Order {order_id} is already cancelled.")
    for item in order["items"]:
        db.release(item["sku"], item["quantity"])
    order["status"] = "cancelled"
    order["cancelled_at"] = db.now()
    order["cancellation_reason"] = reason
    return {"order_id": order_id, "status": "cancelled", "refundable_total": db.order_total(order)}


@tool("modify_order_items")
def modify_order_items(
    db: Database,
    order_id: str,
    mode: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    order = db.order(_require(order_id, "order_id"))
    _enum(_require(mode, "mode"), MODIFY_MODES, "mode")
    if order["status"] != "pending":
        raise DomainError(
            "invalid_state",
            f"Order {order_id} is {order['status']}; items can only be modified while pending.",
        )
    if not items:
        raise DomainError("invalid_argument", "items must contain at least one entry.")

    by_sku = {item["sku"]: item for item in order["items"]}
    if mode == "add":
        for entry in items:
            sku = _require(entry.get("sku"), "items[].sku")
            quantity = int(entry.get("quantity", 1))
            db.reserve(sku, quantity)
            if sku in by_sku:
                by_sku[sku]["quantity"] += quantity
            else:
                stock = db.stock(sku)
                order["items"].append(
                    {
                        "sku": sku,
                        "product_id": stock["product_id"],
                        "option": stock["option"],
                        "quantity": quantity,
                        "unit_price": db.product(stock["product_id"])["price"],
                    }
                )
    elif mode == "remove":
        for entry in items:
            sku = _require(entry.get("sku"), "items[].sku")
            if sku not in by_sku:
                raise DomainError("not_found", f"Order {order_id} does not contain {sku}.")
            quantity = int(entry.get("quantity", by_sku[sku]["quantity"]))
            if quantity > by_sku[sku]["quantity"]:
                raise DomainError(
                    "invalid_argument",
                    f"Order {order_id} only has {by_sku[sku]['quantity']} units of {sku}.",
                )
            db.release(sku, quantity)
            by_sku[sku]["quantity"] -= quantity
        order["items"] = [item for item in order["items"] if item["quantity"] > 0]
        if not order["items"]:
            raise DomainError(
                "invalid_argument",
                "Removing these items would empty the order; cancel it instead.",
            )
    else:  # replace
        for entry in items:
            old_sku = _require(entry.get("sku"), "items[].sku")
            new_sku = _require(entry.get("replacement_sku"), "items[].replacement_sku")
            if old_sku not in by_sku:
                raise DomainError("not_found", f"Order {order_id} does not contain {old_sku}.")
            quantity = by_sku[old_sku]["quantity"]
            db.reserve(new_sku, quantity)
            db.release(old_sku, quantity)
            stock = db.stock(new_sku)
            by_sku[old_sku].update(
                {
                    "sku": new_sku,
                    "product_id": stock["product_id"],
                    "option": stock["option"],
                    "unit_price": db.product(stock["product_id"])["price"],
                }
            )
    return {"order_id": order_id, "items": order["items"], "total": db.order_total(order)}


@tool("process_refund")
def process_refund(
    db: Database,
    order_id: str,
    method: str,
    amount: float | None = None,
) -> dict[str, Any]:
    order = db.order(_require(order_id, "order_id"))
    _enum(_require(method, "method"), REFUND_METHODS, "method")
    if order["status"] not in REFUNDABLE_STATES:
        raise DomainError(
            "invalid_state",
            f"Order {order_id} is {order['status']}; it must be cancelled or returned "
            "before a refund can be issued.",
        )
    already = sum(r["amount"] for r in db.refunds.values() if r["order_id"] == order_id)
    ceiling = round(db.order_total(order) - already, 2)
    if ceiling <= 0:
        raise DomainError("invalid_state", f"Order {order_id} has already been fully refunded.")
    value = ceiling if amount is None else round(float(amount), 2)
    if value <= 0:
        raise DomainError("invalid_argument", "amount must be greater than zero.")
    if value > ceiling:
        raise DomainError(
            "invalid_argument",
            f"amount {value} exceeds the {ceiling} still refundable on order {order_id}.",
        )
    refund_id = db.next_id("refund")
    db.refunds[refund_id] = {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": value,
        "method": method,
        "issued_at": db.now(),
    }
    return {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": value,
        "method": method,
        "remaining_refundable": round(ceiling - value, 2),
    }


# -- fulfilment and support ---------------------------------------------


@tool("track_shipment")
def track_shipment(
    db: Database,
    shipment_id: str | None = None,
    order_id: str | None = None,
) -> dict[str, Any]:
    if shipment_id is None and order_id is None:
        raise DomainError("invalid_argument", "Provide either shipment_id or order_id.")
    if shipment_id is None:
        order = db.order(order_id)
        shipment_id = order.get("shipment_id")
        if shipment_id is None:
            raise DomainError(
                "not_found",
                f"Order {order_id} has not shipped yet; there is no tracking information.",
            )
    record = db.shipments.get(shipment_id)
    if record is None:
        raise DomainError("not_found", f"No shipment with id {shipment_id}.")
    return {
        "shipment_id": shipment_id,
        "order_id": record["order_id"],
        "carrier": record["carrier"],
        "tracking_number": record["tracking_number"],
        "status": record["status"],
        "estimated_delivery": record["estimated_delivery"],
        "events": record["events"],
    }


@tool("create_support_ticket")
def create_support_ticket(
    db: Database,
    customer_id: str,
    subject: str,
    body: str,
    category: str,
    order_id: str | None = None,
) -> dict[str, Any]:
    db.customer(_require(customer_id, "customer_id"))
    _require(subject, "subject")
    _require(body, "body")
    _enum(_require(category, "category"), TICKET_CATEGORIES, "category")
    if order_id is not None:
        order = db.order(order_id)
        if order["customer_id"] != customer_id:
            raise DomainError(
                "invalid_argument",
                f"Order {order_id} does not belong to customer {customer_id}.",
            )
    ticket_id = db.next_id("ticket")
    db.tickets[ticket_id] = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "subject": subject,
        "body": body,
        "category": category,
        "status": "open",
        "priority": "normal",
        "opened_at": db.now(),
        "notes": [],
    }
    return {"ticket_id": ticket_id, "status": "open", "category": category}


@tool("update_support_ticket")
def update_support_ticket(
    db: Database,
    ticket_id: str,
    status: str | None = None,
    note: str | None = None,
    priority: str | None = None,
) -> dict[str, Any]:
    ticket = db.ticket(_require(ticket_id, "ticket_id"))
    if status is None and note is None and priority is None:
        raise DomainError("invalid_argument", "Provide at least one of status, note or priority.")
    if ticket["status"] == "closed":
        raise DomainError("invalid_state", f"Ticket {ticket_id} is closed and cannot be updated.")
    if status is not None:
        _enum(status, TICKET_STATUSES, "status")
        ticket["status"] = status
    if priority is not None:
        _enum(priority, TICKET_PRIORITIES, "priority")
        ticket["priority"] = priority
    if note is not None:
        ticket["notes"].append({"at": db.now(), "text": note})
    return {
        "ticket_id": ticket_id,
        "status": ticket["status"],
        "priority": ticket["priority"],
        "note_count": len(ticket["notes"]),
    }
