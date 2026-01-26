from toolsmith.optimizer.permissions import (
    accepted_parameters,
    diff,
    expands_permissions,
)
from toolsmith.server.schemas import ToolSchema


def schema(**overrides) -> ToolSchema:
    body = {
        "name": "cancel_order",
        "description": "Cancel an order. The order must not have shipped.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order identifier."},
                "reason": {
                    "type": "string",
                    "description": "Why.",
                    "enum": ["no_longer_needed", "ordered_by_mistake"],
                },
            },
            "required": ["order_id", "reason"],
        },
        "error_returns": [
            {"code": "not_found", "when": "No such order."},
            {"code": "invalid_state", "when": "Already cancelled."},
        ],
    }
    body.update(overrides)
    return ToolSchema.from_dict(body)


def kinds(before, after):
    return {expansion.kind for expansion in diff(before, after)}


def test_rewording_is_not_an_expansion():
    after = schema()
    after.description = "Cancel an order that must not have shipped yet."
    assert not expands_permissions(schema(), after)


def test_dropping_a_stated_restriction_is_an_expansion():
    after = schema()
    after.description = "Cancel an order."
    assert "restriction_removed" in kinds(schema(), after)


def test_a_restriction_moved_into_a_parameter_still_counts_as_stated():
    after = schema()
    after.description = "Cancel an order."
    after.input_schema["properties"]["order_id"]["description"] = (
        "Order identifier. The order must not have shipped."
    )
    assert not expands_permissions(schema(), after)


def test_dropping_an_enum_is_an_expansion():
    after = schema()
    after.input_schema["properties"]["reason"].pop("enum")
    assert "enum_removed" in kinds(schema(), after)


def test_adding_a_value_the_tool_accepts_is_not_an_expansion():
    after = schema()
    after.input_schema["properties"]["reason"]["enum"] = [
        "no_longer_needed",
        "ordered_by_mistake",
        "found_better_price",
    ]
    assert not expands_permissions(schema(), after)


def test_adding_a_value_the_tool_rejects_is_an_expansion():
    after = schema()
    after.input_schema["properties"]["reason"]["enum"] = [
        "no_longer_needed",
        "ordered_by_mistake",
        "any_reason_at_all",
    ]
    assert "enum_widened" in kinds(schema(), after)


def test_making_a_required_parameter_optional_is_an_expansion():
    after = schema()
    after.input_schema["required"] = ["order_id"]
    assert "required_dropped" in kinds(schema(), after)


def test_publishing_a_parameter_the_tool_already_accepts_is_not_an_expansion():
    before = schema()
    before.input_schema["properties"].pop("reason")
    before.input_schema["required"] = ["order_id"]
    after = schema()
    after.input_schema["required"] = ["order_id"]
    assert "new_parameter" not in kinds(before, after)


def test_a_parameter_the_tool_does_not_accept_is_an_expansion():
    after = schema()
    after.input_schema["properties"]["force"] = {
        "type": "boolean",
        "description": "Cancel regardless of state.",
    }
    assert "new_parameter" in kinds(schema(), after)


def test_dropping_a_documented_error_return_is_an_expansion():
    after = schema()
    after.error_returns = [{"code": "not_found", "when": "No such order."}]
    assert "error_return_removed" in kinds(schema(), after)


def test_relaxing_a_numeric_bound_is_an_expansion():
    before = ToolSchema.from_dict(
        {
            "name": "search_products",
            "description": "Search.",
            "input_schema": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "maximum": 50}},
                "required": [],
            },
            "error_returns": [],
        }
    )
    after = ToolSchema.from_dict(before.to_dict())
    after.input_schema["properties"]["limit"]["maximum"] = 500
    assert "bound_relaxed" in kinds(before, after)

    dropped = ToolSchema.from_dict(before.to_dict())
    dropped.input_schema["properties"]["limit"].pop("maximum")
    assert "bound_removed" in kinds(before, dropped)


def test_a_schema_is_not_an_expansion_of_itself():
    assert diff(schema(), schema()) == []


def test_accepted_parameters_comes_from_the_implementation():
    assert accepted_parameters("cancel_order") == {"order_id", "reason"}
    assert "db" not in accepted_parameters("get_order")
    assert accepted_parameters("no_such_tool") == set()
