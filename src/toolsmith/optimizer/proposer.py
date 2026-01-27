"""Propose a schema revision for one tool.

The proposer is given the tool's current schema, the failures attributed to it,
and the implementation's observable contract as a reference. It returns a whole
replacement schema rather than a patch, because a description and the parameter
descriptions under it have to stay consistent with each other and patching them
independently does not.

Drafting a revision is the easy half of this system. The proposer is allowed to
be optimistic; the gate decides whether it was right.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..llm import Client
from ..server.schemas import ToolSchema
from .permissions import diff

SYSTEM = """\
You improve the published interface of one tool on an MCP server. You are \
given the tool's current declaration and a set of episodes in which an agent \
using that declaration failed.

You may change:
- the tool description
- any parameter description
- the "error_returns" list, which documents what the tool returns when a call \
does not succeed
- a parameter's declared "type" or "enum", but only to make it match what the \
tool already accepts

You may not:
- rename the tool or any parameter
- add a parameter that does not already exist
- add a value to an enum that the tool does not already accept
- remove a parameter from "required"
- remove or weaken any restriction the current description states
- describe behaviour the implementation does not have

The last four are hard limits. A revision that widens what a caller may ask the \
tool to do is rejected outright, however much it would help.

Write for an agent that will read this declaration and nothing else. Say what \
the tool does, when to use it rather than a neighbouring tool, what each \
parameter means and what form it takes, which preconditions must already hold, \
and what a failure looks like. Be concrete: give the literal accepted values, \
the exact timestamp format, the identifier shape.

Reply with a JSON object and nothing else:
{"description": "<tool description>", "parameters": {"<name>": "<description>", \
...}, "enums": {"<name>": ["<value>", ...]}, "types": {"<name>": "<json type>"}, \
"error_returns": [{"code": "<code>", "when": "<condition>"}], \
"rationale": "<one sentence on what was wrong>"}

Include "parameters" entries only for parameters you are changing. Omit "enums" \
and "types" entirely unless a declared value or type is factually wrong. Nested \
parameters are addressed as "address.line1" or "items[].sku"."""

PROMPT = """\
CURRENT DECLARATION
{schema}

WHAT THE IMPLEMENTATION ACTUALLY ACCEPTS AND RETURNS
{contract}

FAILURES ATTRIBUTED TO THIS TOOL ({count} runs across {tasks} tasks)
{failures}"""


@dataclass
class Proposal:
    tool: str
    schema: ToolSchema
    rationale: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "rationale": self.rationale,
            "schema": self.schema.to_dict(),
        }


def _resolve(node: dict[str, Any], path: str) -> dict[str, Any] | None:
    current = node
    for step in path.split("."):
        properties = current.get("properties") or {}
        if step.endswith("[]"):
            parent = properties.get(step[:-2])
            if parent is None:
                return None
            current = parent.get("items") or {}
        else:
            if step not in properties:
                return None
            current = properties[step]
    return current


class Proposer:
    def __init__(self, client: Client):
        self.client = client

    def propose(
        self,
        current: ToolSchema,
        contract: str,
        failures: list[dict[str, Any]],
        run_count: int,
        task_count: int,
    ) -> Proposal:
        prompt = PROMPT.format(
            schema=json.dumps(current.to_dict(), indent=2),
            contract=contract,
            count=run_count,
            tasks=task_count,
            failures="\n".join(
                f"- [{item['failure_class']}] {item['reason']}" for item in failures[:24]
            )
            or "(no reasons recorded)",
        )
        payload = self.client.complete_json(SYSTEM, [{"role": "user", "content": prompt}])
        return self._apply(current, payload)

    @staticmethod
    def _apply(current: ToolSchema, payload: dict[str, Any]) -> Proposal:
        revised = ToolSchema.from_dict(current.to_dict())
        revised.revision = current.revision + 1

        if isinstance(payload.get("description"), str) and payload["description"].strip():
            revised.description = payload["description"].strip()

        for path, text in (payload.get("parameters") or {}).items():
            node = _resolve(revised.input_schema, path)
            if node is not None and isinstance(text, str) and text.strip():
                node["description"] = text.strip()

        for path, values in (payload.get("enums") or {}).items():
            node = _resolve(revised.input_schema, path)
            if node is not None and isinstance(values, list) and values:
                node["enum"] = [str(value) for value in values]

        for path, declared in (payload.get("types") or {}).items():
            node = _resolve(revised.input_schema, path)
            if node is not None and isinstance(declared, str) and declared:
                node["type"] = declared

        returns = payload.get("error_returns")
        if isinstance(returns, list) and returns:
            cleaned = [
                {"code": str(entry["code"]), "when": str(entry["when"])}
                for entry in returns
                if isinstance(entry, dict) and "code" in entry and "when" in entry
            ]
            if cleaned:
                revised.error_returns = cleaned

        return Proposal(
            tool=current.name,
            schema=revised,
            rationale=str(payload.get("rationale", "")).strip(),
            raw=payload,
        )


def contract_of(tool_name: str) -> str:
    """The implementation's own signature and docstring, as a reference."""
    import inspect

    from ..server.tools import REGISTRY

    implementation = REGISTRY[tool_name]
    signature = inspect.signature(implementation)
    parameters = ", ".join(
        f"{name}: {parameter.annotation}"
        + ("" if parameter.default is inspect.Parameter.empty else f" = {parameter.default!r}")
        for name, parameter in signature.parameters.items()
        if name != "db"
    )
    source = inspect.getsource(implementation)
    return f"{tool_name}({parameters})\n\n{source}"


def permission_check(current: ToolSchema, proposal: Proposal) -> list[dict[str, str]]:
    return [expansion.to_dict() for expansion in diff(current, proposal.schema)]
