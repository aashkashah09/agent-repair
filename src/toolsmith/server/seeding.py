"""Controlled fault injection into a schema set.

There is no way to score a repair against a real server: nobody can say which
of its interfaces were wrong to begin with, so a fix and a no-op look the same
afterwards. The evaluation server is therefore a working one that we break on
purpose, in ways we have written down. Each entry in
``data/defects/seeded_defects.json`` names the pattern it instantiates, its
severity, the tools it touches, and a list of mutations.

Most patterns are purely declarative -- they change what the schema says
without changing what the tool does. Two patterns cannot be expressed that way:
a tool that swallows an error condition and a tool that truncates its result
must actually behave that way at call time. Those carry a ``runtime`` mutation
naming a wrapper in ``BEHAVIOURS``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import DomainError
from .schemas import SchemaSet, ToolSchema

PATTERNS = (
    "undocumented-enum",
    "silent-empty-return",
    "ambiguous-datetime",
    "overloaded-tool-mode",
    "unstated-precondition",
    "near-duplicate-tool-naming",
    "untyped-passthrough",
    "silent-pagination-truncation",
)

SEVERITIES = ("mild", "moderate", "severe")


class SeedingError(Exception):
    pass


@dataclass
class Defect:
    defect_id: str
    pattern: str
    severity: str
    tools: list[str]
    parameter: str | None
    summary: str
    mutations: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Defect:
        if payload["pattern"] not in PATTERNS:
            raise SeedingError(f"{payload['defect_id']}: unknown pattern {payload['pattern']!r}")
        if payload["severity"] not in SEVERITIES:
            raise SeedingError(f"{payload['defect_id']}: unknown severity {payload['severity']!r}")
        return cls(
            defect_id=payload["defect_id"],
            pattern=payload["pattern"],
            severity=payload["severity"],
            tools=list(payload["tools"]),
            parameter=payload.get("parameter"),
            summary=payload["summary"],
            mutations=list(payload["mutations"]),
        )


def load_defects(path: str | Path) -> list[Defect]:
    payload = json.loads(Path(path).read_text())
    return [Defect.from_dict(entry) for entry in payload["defects"]]


# -- schema mutations ----------------------------------------------------


def _resolve(schema: ToolSchema, path: str) -> dict[str, Any]:
    """Walk a dotted parameter path, where ``[]`` steps into an array's items."""
    node: dict[str, Any] = schema.input_schema
    for step in path.split("."):
        if step.endswith("[]"):
            node = node.get("properties", {})[step[:-2]]["items"]
        else:
            node = node.get("properties", {})[step]
    return node


def _apply_mutation(schemas: SchemaSet, mutation: dict[str, Any]) -> None:
    op = mutation["op"]
    if op == "runtime":
        return
    tool = schemas[mutation["tool"]]

    if op == "set_tool_description":
        tool.description = mutation["text"]
    elif op == "set_param_description":
        _resolve(tool, mutation["path"])["description"] = mutation["text"]
    elif op == "drop_enum":
        _resolve(tool, mutation["path"]).pop("enum", None)
    elif op == "set_enum":
        _resolve(tool, mutation["path"])["enum"] = list(mutation["values"])
    elif op == "set_type":
        node = _resolve(tool, mutation["path"])
        node["type"] = mutation["type"]
        for key in ("enum", "properties", "required", "items", "minimum", "maximum"):
            node.pop(key, None)
    elif op == "replace_param_schema":
        parent = tool.input_schema["properties"]
        *head, leaf = mutation["path"].split(".")
        for step in head:
            parent = parent[step]["properties"] if not step.endswith("[]") else parent[step[:-2]]["items"]["properties"]
        parent[leaf] = json.loads(json.dumps(mutation["schema"]))
    elif op == "drop_param":
        tool.input_schema.get("properties", {}).pop(mutation["path"], None)
        required = tool.input_schema.get("required", [])
        if mutation["path"] in required:
            required.remove(mutation["path"])
    elif op == "drop_error_return":
        tool.error_returns = [e for e in tool.error_returns if e["code"] != mutation["code"]]
    elif op == "set_error_return":
        for entry in tool.error_returns:
            if entry["code"] == mutation["code"]:
                entry["when"] = mutation["when"]
                break
        else:
            tool.error_returns.append({"code": mutation["code"], "when": mutation["when"]})
    else:
        raise SeedingError(f"unknown mutation op {op!r}")


def apply_defects(clean: SchemaSet, defects: list[Defect]) -> SchemaSet:
    """Return a copy of ``clean`` with every defect's schema mutations applied."""
    seeded = clean.copy()
    for defect in defects:
        for mutation in defect.mutations:
            _apply_mutation(seeded, mutation)
    seeded.validate()
    return seeded


def runtime_behaviours(defects: list[Defect]) -> set[str]:
    return {
        mutation["behaviour"]
        for defect in defects
        for mutation in defect.mutations
        if mutation["op"] == "runtime"
    }


# -- runtime behaviours --------------------------------------------------
#
# Each wrapper takes the implementation's normal (result, error) outcome and
# returns the outcome the defective server actually produces.


def _search_swallow_unknown_category(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return call()
    except DomainError as exc:
        if exc.code == "invalid_argument" and "category" in exc.message:
            return {"results": [], "total_matches": 0}
        raise


def _track_empty_when_unshipped(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return call()
    except DomainError as exc:
        if exc.code == "not_found" and "has not shipped" in exc.message:
            return {"shipment_id": None, "status": None, "events": []}
        raise


def _list_orders_empty_for_unknown_customer(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return call()
    except DomainError as exc:
        if exc.code == "not_found" and "customer" in exc.message:
            return {"orders": [], "total_matches": 0, "next_cursor": None}
        raise


def _list_orders_drop_pagination(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    result = call()
    return {"orders": result["orders"]}


def _search_truncate_to_ten(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    result = call()
    return {"results": result["results"][:10]}


def _track_truncate_events(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    result = call()
    if result.get("events"):
        result = dict(result)
        result["events"] = result["events"][-2:]
    return result


BEHAVIOURS: dict[str, tuple[str, Callable[[Callable[[], dict[str, Any]]], dict[str, Any]]]] = {
    "search_swallow_unknown_category": ("search_products", _search_swallow_unknown_category),
    "track_empty_when_unshipped": ("track_shipment", _track_empty_when_unshipped),
    "list_orders_empty_for_unknown_customer": (
        "list_orders",
        _list_orders_empty_for_unknown_customer,
    ),
    "list_orders_drop_pagination": ("list_orders", _list_orders_drop_pagination),
    "search_truncate_to_ten": ("search_products", _search_truncate_to_ten),
    "track_truncate_events": ("track_shipment", _track_truncate_events),
}


def behaviours_for(names: set[str]) -> dict[str, list[Callable[..., dict[str, Any]]]]:
    """Group the named wrappers by the tool they wrap, in registration order."""
    grouped: dict[str, list[Callable[..., dict[str, Any]]]] = {}
    for name in BEHAVIOURS:
        if name not in names:
            continue
        tool_name, wrapper = BEHAVIOURS[name]
        grouped.setdefault(tool_name, []).append(wrapper)
    return grouped
