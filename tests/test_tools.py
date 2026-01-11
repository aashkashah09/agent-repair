import pytest

from toolsmith.server.db import DomainError
from toolsmith.server.tools import REGISTRY, TICKET_CATEGORIES


def call(db, name, **arguments):
    return REGISTRY[name](db, **arguments)


# -- catalogue -----------------------------------------------------------


def test_search_rejects_an_unknown_category(db):
    with pytest.raises(DomainError) as excinfo:
        call(db, "search_products", query="jacket", category="furniture")
    assert excinfo.value.code == "invalid_argument"


def test_search_reports_the_full_match_count(db):
    result = call(db, "search_products", query="merino", limit=1)
    assert len(result["results"]) == 1
    assert result["total_matches"] >= 1


def test_product_detail_lists_every_variant(db):
    product_id = next(iter(db.products))
    detail = call(db, "get_product_detail", product_id=product_id)
    assert len(detail["variants"]) == len(db.products[product_id]["skus"])


def test_check_inventory_takes_a_sku_not_a_product_id(db):
    with pytest.raises(DomainError) as excinfo:
        call(db, "check_inventory", sku="P0001")
    assert excinfo.value.code == "not_found"


# -- customers -----------------------------------------------------------


def test_customer_lookup_needs_one_of_the_two_keys(db):
    with pytest.raises(DomainError):
        call(db, "get_customer")


def test_customer_lookup_by_email_is_case_insensitive(db):
    record = db.customers["C0004"]
    found = call(db, "get_customer", email=record["email"].upper())
    assert found["customer_id"] == "C0004"


def test_profile_update_requires_a_structured_address(db):
    with pytest.raises(DomainError) as excinfo:
        call(db, "update_customer_profile", customer_id="C0004", address="12 Anywhere St")
    assert excinfo.value.code == "invalid_argument"


def test_profile_update_rejects_an_incomplete_address(db):
    with pytest.raises(DomainError):
        call(
            db, "update_customer_profile", customer_id="C0004",
            address={"line1": "12 Anywhere St", "city": "Portland"},
        )


def test_profile_update_needs_something_to_change(db):
    with pytest.raises(DomainError):
        call(db, "update_customer_profile", customer_id="C0004")


# -- orders --------------------------------------------------------------


def test_list_orders_paginates_and_reports_the_total(db):
    first = call(db, "list_orders", customer_id="C0001")
    assert len(first["orders"]) <= 20
    assert first["total_matches"] > len(first["orders"])
    assert first["next_cursor"] is not None

    second = call(db, "list_orders", customer_id="C0001", cursor=first["next_cursor"])
    seen = {o["order_id"] for o in first["orders"]} & {o["order_id"] for o in second["orders"]}
    assert not seen


def test_list_orders_requires_a_timestamp_with_an_offset(db):
    with pytest.raises(DomainError) as excinfo:
        call(db, "list_orders", customer_id="C0001", since="2026-01-01")
    assert excinfo.value.code == "invalid_argument"

    ok = call(db, "list_orders", customer_id="C0001", since="2026-01-01T00:00:00Z")
    assert "orders" in ok


def test_cancel_requires_an_accepted_reason(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "pending")
    with pytest.raises(DomainError):
        call(db, "cancel_order", order_id=order_id, reason="customer_request")


def test_cancel_releases_reserved_stock(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "pending")
    item = db.orders[order_id]["items"][0]
    before = db.inventory[item["sku"]]["available"]
    call(db, "cancel_order", order_id=order_id, reason="no_longer_needed")
    assert db.inventory[item["sku"]]["available"] == before + item["quantity"]
    assert db.orders[order_id]["status"] == "cancelled"


def test_a_shipped_order_cannot_be_cancelled(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "shipped")
    with pytest.raises(DomainError) as excinfo:
        call(db, "cancel_order", order_id=order_id, reason="no_longer_needed")
    assert excinfo.value.code == "invalid_state"


def test_create_order_reserves_stock(db):
    sku = next(s for s, r in db.inventory.items() if r["available"] >= 3)
    before = db.inventory[sku]["available"]
    result = call(
        db, "create_order", customer_id="C0004",
        items=[{"sku": sku, "quantity": 2}], payment_method_id="PM001",
        address_id=db.customers["C0004"]["addresses"][0]["address_id"],
    )
    assert result["status"] == "pending"
    assert db.inventory[sku]["available"] == before - 2


def test_create_order_refuses_more_units_than_exist(db):
    sku = next(s for s, r in db.inventory.items() if 0 < r["available"] < 50)
    with pytest.raises(DomainError) as excinfo:
        call(
            db, "create_order", customer_id="C0004",
            items=[{"sku": sku, "quantity": 999}], payment_method_id="PM001",
            address_id=db.customers["C0004"]["addresses"][0]["address_id"],
        )
    assert excinfo.value.code == "insufficient_stock"


def test_address_is_required_on_a_multi_address_account(db):
    customer_id = next(c for c, r in db.customers.items() if len(r["addresses"]) > 1)
    sku = next(s for s, r in db.inventory.items() if r["available"] >= 1)
    with pytest.raises(DomainError):
        call(
            db, "create_order", customer_id=customer_id,
            items=[{"sku": sku}],
            payment_method_id=db.customers[customer_id]["payment_methods"][0]["payment_method_id"],
        )


def test_modify_is_limited_to_pending_orders(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "processing")
    sku = db.orders[order_id]["items"][0]["sku"]
    with pytest.raises(DomainError) as excinfo:
        call(db, "modify_order_items", order_id=order_id, mode="remove",
             items=[{"sku": sku}])
    assert excinfo.value.code == "invalid_state"


def test_replace_mode_needs_a_replacement_sku(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "pending")
    sku = db.orders[order_id]["items"][0]["sku"]
    with pytest.raises(DomainError):
        call(db, "modify_order_items", order_id=order_id, mode="replace",
             items=[{"sku": sku}])


def test_an_order_cannot_be_emptied_by_removal(db):
    order_id = next(
        o for o, r in db.orders.items() if r["status"] == "pending" and len(r["items"]) == 1
    )
    sku = db.orders[order_id]["items"][0]["sku"]
    with pytest.raises(DomainError):
        call(db, "modify_order_items", order_id=order_id, mode="remove",
             items=[{"sku": sku}])


# -- refunds -------------------------------------------------------------


def test_refund_requires_a_terminal_order(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "pending")
    with pytest.raises(DomainError) as excinfo:
        call(db, "process_refund", order_id=order_id, method="original_payment")
    assert excinfo.value.code == "invalid_state"


def test_refund_after_cancelling_succeeds(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "pending")
    call(db, "cancel_order", order_id=order_id, reason="ordered_by_mistake")
    result = call(db, "process_refund", order_id=order_id, method="original_payment")
    assert result["amount"] > 0
    assert result["remaining_refundable"] == 0


def test_partial_refunds_accumulate_to_the_total(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "pending")
    call(db, "cancel_order", order_id=order_id, reason="ordered_by_mistake")
    total = db.order_total(db.orders[order_id])
    first = call(db, "process_refund", order_id=order_id, method="store_credit",
                 amount=round(total / 4, 2))
    assert first["remaining_refundable"] == pytest.approx(total - first["amount"], abs=0.01)
    call(db, "process_refund", order_id=order_id, method="store_credit")
    with pytest.raises(DomainError):
        call(db, "process_refund", order_id=order_id, method="store_credit")


def test_refund_cannot_exceed_the_outstanding_balance(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "pending")
    call(db, "cancel_order", order_id=order_id, reason="ordered_by_mistake")
    with pytest.raises(DomainError):
        call(db, "process_refund", order_id=order_id, method="original_payment",
             amount=999999.0)


# -- fulfilment and support ---------------------------------------------


def test_tracking_an_unshipped_order_is_not_found(db):
    order_id = next(o for o, r in db.orders.items() if r["shipment_id"] is None)
    with pytest.raises(DomainError) as excinfo:
        call(db, "track_shipment", order_id=order_id)
    assert excinfo.value.code == "not_found"


def test_tracking_returns_the_full_scan_history(db):
    order_id = next(o for o, r in db.orders.items() if r["status"] == "delivered")
    result = call(db, "track_shipment", order_id=order_id)
    assert len(result["events"]) == len(db.shipments[result["shipment_id"]]["events"])


@pytest.mark.parametrize("category", TICKET_CATEGORIES)
def test_every_documented_ticket_category_is_accepted(db, category):
    result = call(
        db, "create_support_ticket", customer_id="C0004",
        subject="s", body="b", category=category,
    )
    assert result["category"] == category


def test_ticket_must_belong_to_the_customer(db):
    order_id, order = next(iter(db.orders.items()))
    other = next(c for c in db.customers if c != order["customer_id"])
    with pytest.raises(DomainError):
        call(db, "create_support_ticket", customer_id=other, subject="s", body="b",
             category="shipping", order_id=order_id)


def test_ticket_update_needs_something_to_change(db):
    ticket_id = next(iter(db.tickets))
    with pytest.raises(DomainError):
        call(db, "update_support_ticket", ticket_id=ticket_id)


def test_a_closed_ticket_is_immutable(db):
    ticket_id = next(iter(db.tickets))
    call(db, "update_support_ticket", ticket_id=ticket_id, status="closed")
    with pytest.raises(DomainError) as excinfo:
        call(db, "update_support_ticket", ticket_id=ticket_id, note="one more thing")
    assert excinfo.value.code == "invalid_state"


def test_urgent_priority_is_accepted(db):
    ticket_id = next(iter(db.tickets))
    result = call(db, "update_support_ticket", ticket_id=ticket_id, priority="urgent")
    assert result["priority"] == "urgent"
