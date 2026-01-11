import json

import pytest

from toolsmith.config import REPO_ROOT
from toolsmith.server.schemas import TOOL_ORDER, SchemaError, SchemaSet, ToolSchema
from toolsmith.server.tools import REGISTRY

SCHEMA_SETS = sorted(
    path for path in (REPO_ROOT / "data" / "schemas").iterdir() if path.is_dir()
)
DEPLOYED_SETS = SCHEMA_SETS


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


def test_round_trip_through_disk_is_lossless(clean_schemas, tmp_path):
    clean_schemas.save(tmp_path)
    reloaded = SchemaSet.load(tmp_path)
    for name in TOOL_ORDER:
        assert reloaded[name].to_dict() == clean_schemas[name].to_dict()


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
