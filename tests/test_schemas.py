import json

import pytest

from toolsmith.config import REPO_ROOT
from toolsmith.server.schemas import TOOL_ORDER, SchemaError, SchemaSet, ToolSchema
from toolsmith.server.tools import ACCEPTED_VALUES, REGISTRY

SCHEMA_SETS = sorted(
    path for path in (REPO_ROOT / "data" / "schemas").iterdir() if path.is_dir()
)
# accept_all is the ablation: it carries the revisions the gate refused, so it
# is deliberately not held to the invariants the deployed sets are.
DEPLOYED_SETS = [path for path in SCHEMA_SETS if path.name != "accept_all"]


def implementation_parameters(name: str) -> set[str]:
    import inspect

    return {p for p in inspect.signature(REGISTRY[name]).parameters if p != "db"}


@pytest.mark.parametrize("directory", SCHEMA_SETS, ids=lambda p: p.name)
def test_every_schema_set_validates(directory):
    SchemaSet.load(directory).validate()


@pytest.mark.parametrize("directory", DEPLOYED_SETS, ids=lambda p: p.name)
def test_declared_parameters_exist_on_the_implementation(directory):
    schemas = SchemaSet.load(directory)
    for name in TOOL_ORDER:
        declared = set(schemas[name].parameters)
        accepted = implementation_parameters(name)
        assert declared <= accepted, f"{directory.name}/{name}: {declared - accepted}"


def test_the_ablation_advertises_parameters_the_tools_reject():
    """What the permission check was there to stop.

    Two of the blocked revisions add parameters no implementation takes. An
    agent reading the accept-all declarations will construct calls the server
    rejects outright, which is the concrete cost of promoting on measured
    improvement alone.
    """
    schemas = SchemaSet.load(REPO_ROOT / "data" / "schemas" / "accept_all")
    undeliverable = {
        name: sorted(set(schemas[name].parameters) - implementation_parameters(name))
        for name in TOOL_ORDER
        if set(schemas[name].parameters) - implementation_parameters(name)
    }
    assert undeliverable == {"get_customer": ["include_orders", "shipping_speed"]}


def test_the_ablation_offers_an_enum_value_the_tool_rejects():
    schemas = SchemaSet.load(REPO_ROOT / "data" / "schemas" / "accept_all")
    declared = set(schemas["update_support_ticket"].parameters["priority"]["enum"])
    assert declared - set(ACCEPTED_VALUES[("update_support_ticket", "priority")]) == {"critical"}


def test_the_ablation_drops_a_required_parameter():
    gated = SchemaSet.load(REPO_ROOT / "data" / "schemas" / "round4")
    everything = SchemaSet.load(REPO_ROOT / "data" / "schemas" / "accept_all")
    assert "payment_method_id" in gated["create_order"].required
    assert "payment_method_id" not in everything["create_order"].required


def test_the_reference_set_declares_enums_exactly(clean_schemas):
    for (tool, parameter), allowed in ACCEPTED_VALUES.items():
        node = clean_schemas[tool].parameters.get(parameter)
        if node is None or "enum" not in node:
            continue
        assert set(node["enum"]) == set(allowed), f"{tool}.{parameter}"


def test_round_trip_through_disk_is_lossless(clean_schemas, tmp_path):
    clean_schemas.save(tmp_path)
    reloaded = SchemaSet.load(tmp_path)
    for name in TOOL_ORDER:
        assert reloaded[name].to_dict() == clean_schemas[name].to_dict()


def test_copy_is_independent(clean_schemas):
    duplicate = clean_schemas.copy()
    duplicate["cancel_order"].description = "changed"
    duplicate["cancel_order"].input_schema["properties"]["reason"]["description"] = "changed"
    assert clean_schemas["cancel_order"].description != "changed"
    assert (
        clean_schemas["cancel_order"].input_schema["properties"]["reason"]["description"]
        != "changed"
    )


def test_from_dict_does_not_alias_the_source(clean_schemas):
    source = clean_schemas["create_order"]
    clone = ToolSchema.from_dict(source.to_dict())
    clone.input_schema["properties"]["items"]["description"] = "changed"
    assert source.input_schema["properties"]["items"]["description"] != "changed"


def test_mcp_declaration_carries_the_error_returns(clean_schemas):
    declaration = clean_schemas["process_refund"].to_mcp()
    assert "invalid_state" in declaration["description"]
    assert set(declaration) == {"name", "description", "input_schema"}


def test_missing_key_is_rejected():
    with pytest.raises(SchemaError):
        ToolSchema.from_dict({"name": "x", "description": "y"})


def test_required_parameter_must_be_declared(clean_schemas, tmp_path):
    clean_schemas["get_order"].input_schema["required"] = ["nonexistent"]
    with pytest.raises(SchemaError):
        clean_schemas.validate()


def test_replace_rejects_an_unknown_tool(clean_schemas):
    stray = ToolSchema.from_dict(
        {
            "name": "not_a_tool",
            "description": "x",
            "input_schema": {"type": "object", "properties": {}},
            "error_returns": [],
        }
    )
    with pytest.raises(SchemaError):
        clean_schemas.replace(stray)


def test_committed_schema_files_are_formatted_consistently():
    for directory in SCHEMA_SETS:
        for name in TOOL_ORDER:
            path = directory / f"{name}.json"
            text = path.read_text()
            assert text.endswith("\n")
            assert text == json.dumps(json.loads(text), indent=2) + "\n"
