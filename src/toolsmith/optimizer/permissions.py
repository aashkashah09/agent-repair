"""Permission-surface analysis of a schema revision.

A revision that makes a tool clearer is what the loop is for. A revision that
makes a tool able to do more than it could before is a different thing, and the
gate rejects it on that ground alone regardless of what it does to the numbers:
a system that edits its own configuration should not be able to widen its own
authority as a side effect of improving a description.

The surface is what a caller is permitted to ask for. It expands when a closed
set of values opens, when a bound relaxes, when a required field becomes
optional, when a new parameter appears, or when a stated restriction is dropped
from the description or the documented error returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..server.schemas import ToolSchema

# Phrases that carry a restriction. Losing one between revisions means the
# schema no longer tells the caller it cannot do something.
RESTRICTION_PATTERNS = (
    r"\bmust (?:be|already|first|not)\b",
    r"\bcannot\b",
    r"\bcan(?:not| no longer) be\b",
    r"\bonly\b",
    r"\bnot (?:supported|permitted|allowed)\b",
    r"\brejected\b",
    r"\bis required when\b",
    r"\bare immutable\b",
    r"\bnever\b",
)


BOUND_KEYS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
SIZE_KEYS = ("minItems", "maxItems", "minLength", "maxLength")


@dataclass(frozen=True)
class Expansion:
    kind: str
    where: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "where": self.where, "detail": self.detail}


def _restrictions(text: str) -> set[str]:
    """Which restriction markers the description states at all."""
    lowered = text.lower()
    return {pattern for pattern in RESTRICTION_PATTERNS if re.search(pattern, lowered)}


def _walk(node: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    """Flatten a JSON Schema's properties into path -> subschema."""
    flat: dict[str, dict[str, Any]] = {}
    for key, value in (node.get("properties") or {}).items():
        path = f"{prefix}{key}"
        flat[path] = value
        if value.get("type") == "object":
            flat.update(_walk(value, f"{path}."))
        elif value.get("type") == "array" and isinstance(value.get("items"), dict):
            flat[f"{path}[]"] = value["items"]
            flat.update(_walk(value["items"], f"{path}[]."))
    return flat


def diff(before: ToolSchema, after: ToolSchema) -> list[Expansion]:
    """Every way ``after`` permits more than ``before``."""
    expansions: list[Expansion] = []
    old = _walk(before.input_schema)
    new = _walk(after.input_schema)

    for path in sorted(set(new) - set(old)):
        expansions.append(
            Expansion("new_parameter", f"{after.name}.{path}", "parameter did not exist before")
        )

    for path in sorted(set(old) & set(new)):
        old_node, new_node = old[path], new[path]
        where = f"{after.name}.{path}"

        old_enum, new_enum = old_node.get("enum"), new_node.get("enum")
        if old_enum is not None and new_enum is None:
            expansions.append(
                Expansion("enum_removed", where, f"enum {sorted(old_enum)} no longer declared")
            )
        elif old_enum is not None and new_enum is not None:
            added = sorted(set(new_enum) - set(old_enum))
            if added:
                expansions.append(Expansion("enum_widened", where, f"values added: {added}"))

        for key in BOUND_KEYS:
            if key in old_node and key not in new_node:
                expansions.append(Expansion("bound_removed", where, f"{key} dropped"))
            elif key in old_node and key in new_node:
                relaxed = (
                    new_node[key] < old_node[key]
                    if key.endswith(("inimum",))
                    else new_node[key] > old_node[key]
                )
                if relaxed:
                    expansions.append(
                        Expansion(
                            "bound_relaxed", where, f"{key} {old_node[key]} -> {new_node[key]}"
                        )
                    )

        for key in SIZE_KEYS:
            if key in old_node and key not in new_node:
                expansions.append(Expansion("bound_removed", where, f"{key} dropped"))

    old_required = set(before.input_schema.get("required", []))
    new_required = set(after.input_schema.get("required", []))
    for parameter in sorted(old_required - new_required):
        expansions.append(
            Expansion("required_dropped", f"{after.name}.{parameter}", "no longer required")
        )

    lost = _restrictions(before.description) - _restrictions(after.description)
    for pattern in sorted(lost):
        expansions.append(
            Expansion(
                "restriction_removed",
                after.name,
                f"description no longer states a restriction matching /{pattern}/",
            )
        )

    old_codes = {entry["code"] for entry in before.error_returns}
    new_codes = {entry["code"] for entry in after.error_returns}
    for code in sorted(old_codes - new_codes):
        expansions.append(
            Expansion("error_return_removed", after.name, f"{code} no longer documented")
        )

    return expansions


def expands_permissions(before: ToolSchema, after: ToolSchema) -> bool:
    return bool(diff(before, after))


def report(before: ToolSchema, after: ToolSchema) -> dict[str, Any]:
    expansions = diff(before, after)
    return {
        "tool": after.name,
        "expands": bool(expansions),
        "expansions": [expansion.to_dict() for expansion in expansions],
    }
