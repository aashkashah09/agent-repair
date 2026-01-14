"""Schema sets: the artefact the repair loop actually edits.

A schema set is a directory of one JSON document per tool, each holding the
MCP tool declaration (``name``, ``description``, ``input_schema``) plus the
``error_returns`` block that documents what the tool emits when a call fails.
The optimizer rewrites these documents; the tool implementations never change.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import REGISTRY

TOOL_ORDER = (
    "search_products",
    "get_product_detail",
    "check_inventory",
    "get_customer",
    "update_customer_profile",
    "list_orders",
    "get_order",
    "create_order",
    "cancel_order",
    "modify_order_items",
    "process_refund",
    "track_shipment",
    "create_support_ticket",
    "update_support_ticket",
)

REQUIRED_KEYS = ("name", "description", "input_schema", "error_returns")


class SchemaError(Exception):
    pass


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]
    error_returns: list[dict[str, str]]
    revision: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolSchema:
        missing = [key for key in REQUIRED_KEYS if key not in payload]
        if missing:
            raise SchemaError(f"tool schema missing key(s): {', '.join(missing)}")
        # Deep-copied so that building a schema from another one's dict gives an
        # independent object: the optimizer edits candidates in place while the
        # gate still needs the unmodified original to diff against.
        return cls(
            name=payload["name"],
            description=payload["description"],
            input_schema=deepcopy(payload["input_schema"]),
            error_returns=deepcopy(payload["error_returns"]),
            revision=payload.get("revision", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "error_returns": self.error_returns,
            "revision": self.revision,
        }

    def to_mcp(self) -> dict[str, Any]:
        """Declaration as handed to the agent."""
        description = self.description
        if self.error_returns:
            lines = [f"- {e['code']}: {e['when']}" for e in self.error_returns]
            description = f"{description}\n\nErrors:\n" + "\n".join(lines)
        return {
            "name": self.name,
            "description": description,
            "input_schema": self.input_schema,
        }

    @property
    def parameters(self) -> dict[str, Any]:
        return self.input_schema.get("properties", {})

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required", []))


class SchemaSet:
    """One directory of tool schemas, addressable by tool name."""

    def __init__(self, tools: dict[str, ToolSchema], source: Path | None = None):
        self.tools = tools
        self.source = source

    # -- io ---------------------------------------------------------------

    @classmethod
    def load(cls, directory: str | Path) -> SchemaSet:
        root = Path(directory)
        if not root.is_dir():
            raise SchemaError(f"schema set directory not found: {root}")
        tools: dict[str, ToolSchema] = {}
        for name in TOOL_ORDER:
            path = root / f"{name}.json"
            if not path.exists():
                raise SchemaError(f"schema set {root.name} is missing {name}.json")
            tools[name] = ToolSchema.from_dict(json.loads(path.read_text()))
        return cls(tools, source=root)

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name in TOOL_ORDER:
            path = root / f"{name}.json"
            path.write_text(json.dumps(self.tools[name].to_dict(), indent=2) + "\n")

    def copy(self) -> SchemaSet:
        payload = {name: ToolSchema.from_dict(t.to_dict()) for name, t in self.tools.items()}
        return SchemaSet(payload, source=self.source)

    # -- access -----------------------------------------------------------

    def __getitem__(self, name: str) -> ToolSchema:
        return self.tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __iter__(self):
        return iter(self.tools[name] for name in TOOL_ORDER)

    def replace(self, schema: ToolSchema) -> None:
        if schema.name not in self.tools:
            raise SchemaError(f"unknown tool {schema.name}")
        self.tools[schema.name] = schema

    def declarations(self) -> list[dict[str, Any]]:
        return [self.tools[name].to_mcp() for name in TOOL_ORDER]

    def validate(self) -> None:
        """Check the set is internally coherent and matches the implementations."""
        for name in TOOL_ORDER:
            if name not in REGISTRY:
                raise SchemaError(f"{name} has a schema but no implementation")
            schema = self.tools[name]
            if schema.name != name:
                raise SchemaError(f"{name}.json declares name {schema.name!r}")
            body = schema.input_schema
            if body.get("type") != "object":
                raise SchemaError(f"{name}: input_schema.type must be 'object'")
            properties = body.get("properties")
            if not isinstance(properties, dict):
                raise SchemaError(f"{name}: input_schema.properties must be an object")
            for parameter in body.get("required", []):
                if parameter not in properties:
                    raise SchemaError(f"{name}: required parameter {parameter!r} is not declared")
            for parameter, spec in properties.items():
                if "type" not in spec and "anyOf" not in spec:
                    raise SchemaError(f"{name}.{parameter}: no type declared")
            for entry in schema.error_returns:
                if not {"code", "when"} <= set(entry):
                    raise SchemaError(f"{name}: error_returns entries need 'code' and 'when'")
        for name in REGISTRY:
            if name not in self.tools:
                raise SchemaError(f"{name} is implemented but has no schema")
