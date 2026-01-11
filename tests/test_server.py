import json

from toolsmith.server.schemas import TOOL_ORDER


def payload(result):
    return json.loads(result.to_content())


def test_every_declared_tool_is_dispatchable(clean_server):
    names = {declaration["name"] for declaration in clean_server.declarations()}
    assert names == set(TOOL_ORDER)


def test_unknown_tool_is_an_error_not_an_exception(clean_server):
    result = clean_server.call("delete_everything", {})
    assert not result.ok
    assert result.error_code == "unknown_tool"


def test_unexpected_argument_is_rejected(clean_server):
    result = clean_server.call("get_order", {"order_id": "W00001", "force": True})
    assert not result.ok
    assert result.error_code == "invalid_argument"
    assert "force" in payload(result)["message"]


def test_missing_required_argument_is_reported_as_invalid(clean_server):
    result = clean_server.call("get_order", {})
    assert not result.ok
    assert result.error_code == "invalid_argument"


def test_wrong_argument_type_is_reported_as_invalid(clean_server):
    result = clean_server.call("search_products", {"query": "mug", "max_price": "eighty"})
    assert not result.ok
    assert result.error_code == "invalid_argument"


def test_domain_errors_carry_their_code(clean_server):
    result = clean_server.call("get_order", {"order_id": "W99999"})
    assert not result.ok
    assert result.error_code == "not_found"
    assert payload(result)["error"] == "not_found"


def test_successful_calls_are_logged(clean_server):
    clean_server.call("get_order", {"order_id": "W00001"})
    clean_server.call("get_order", {"order_id": "W99999"})
    assert len(clean_server.call_log) == 2
    assert clean_server.call_log[0]["ok"] is True
    assert clean_server.call_log[1]["error_code"] == "not_found"


def test_reset_restores_state_and_clears_the_log(clean_server):
    baseline = clean_server.db.snapshot()
    order_id = next(o for o, r in clean_server.db.orders.items() if r["status"] == "pending")
    clean_server.call("cancel_order", {"order_id": order_id, "reason": "no_longer_needed"})
    assert clean_server.db.orders[order_id]["status"] == "cancelled"

    clean_server.reset(baseline)
    assert clean_server.db.orders[order_id]["status"] == "pending"
    assert clean_server.call_log == []


def test_reset_undoes_inventory_reservations(clean_server):
    baseline = clean_server.db.snapshot()
    sku = next(s for s, r in clean_server.db.inventory.items() if r["available"] >= 2)
    available = clean_server.db.inventory[sku]["available"]
    clean_server.call(
        "create_order",
        {
            "customer_id": "C0004",
            "items": [{"sku": sku, "quantity": 2}],
            "payment_method_id": "PM001",
            "address_id": clean_server.db.customers["C0004"]["addresses"][0]["address_id"],
        },
    )
    assert clean_server.db.inventory[sku]["available"] == available - 2
    clean_server.reset(baseline)
    assert clean_server.db.inventory[sku]["available"] == available


# -- seeded server behaviour --------------------------------------------


def test_seeded_search_swallows_an_unknown_category(clean_server, seeded_server):
    arguments = {"query": "jacket", "category": "furniture"}
    assert not clean_server.call("search_products", arguments).ok

    seeded = seeded_server.call("search_products", arguments)
    assert seeded.ok
    assert seeded.payload["results"] == []


def test_seeded_tracking_returns_an_empty_record_when_unshipped(clean_server, seeded_server):
    order_id = next(
        o for o, r in seeded_server.db.orders.items() if r["shipment_id"] is None
    )
    assert not clean_server.call("track_shipment", {"order_id": order_id}).ok

    seeded = seeded_server.call("track_shipment", {"order_id": order_id})
    assert seeded.ok
    assert seeded.payload["status"] is None
    assert seeded.payload["events"] == []


def test_seeded_list_orders_drops_the_pagination_fields(clean_server, seeded_server):
    arguments = {"customer_id": "C0001"}
    assert "next_cursor" in clean_server.call("list_orders", arguments).payload

    seeded = seeded_server.call("list_orders", arguments)
    assert set(seeded.payload) == {"orders"}


def test_seeded_search_truncates_below_the_requested_limit(seeded_server):
    result = seeded_server.call("search_products", {"query": "a", "limit": 40})
    assert len(result.payload["results"]) <= 10
    assert "total_matches" not in result.payload


def test_seeded_tracking_truncates_the_scan_history(clean_server, seeded_server):
    order_id = next(
        o for o, r in seeded_server.db.orders.items() if r["status"] == "delivered"
    )
    full = clean_server.call("track_shipment", {"order_id": order_id}).payload["events"]
    seeded = seeded_server.call("track_shipment", {"order_id": order_id}).payload["events"]
    assert len(full) > len(seeded) == 2


def test_the_implementations_are_identical_on_the_happy_path(clean_server, seeded_server):
    arguments = {"order_id": "W00001"}
    assert clean_server.call("get_order", arguments).payload == seeded_server.call(
        "get_order", arguments
    ).payload
