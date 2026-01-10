"""The tool server the agent talks to.

``ToolServer`` binds a schema set, a database and whatever runtime behaviours
the active defect set installs, and exposes a single ``call`` entry point. The
harness drives it in-process; ``serve_stdio`` exposes the same server over MCP
so a schema set can be pointed at an external client unchanged.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .db import Database, DomainError
from .schemas import TOOL_ORDER, SchemaSet
from .seeding import behaviours_for, load_defects, runtime_behaviours
from .tools import REGISTRY


@dataclass
class ToolResult:
    ok: bool
    payload: dict[str, Any]
    error_code: str | None = None

    def to_content(self) -> str:
        if self.ok:
            return json.dumps(self.payload, separators=(",", ": "), indent=None)
        return json.dumps({"error": self.error_code, "message": self.payload["message"]})


@dataclass
class ToolServer:
    schemas: SchemaSet
    db: Database
    behaviours: dict[str, list[Any]] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        schema_dir: str | Path,
        domain_dir: str | Path,
        defects_path: str | Path | None = None,
    ) -> ToolServer:
        schemas = SchemaSet.load(schema_dir)
        schemas.validate()
        behaviours: dict[str, list[Any]] = {}
        if defects_path is not None:
            defects = load_defects(defects_path)
            behaviours = behaviours_for(runtime_behaviours(defects))
        return cls(schemas=schemas, db=Database.load(domain_dir), behaviours=behaviours)

    # -- declarations ----------------------------------------------------

    def declarations(self) -> list[dict[str, Any]]:
        return self.schemas.declarations()

    # -- dispatch --------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in REGISTRY:
            return self._fail(name, arguments, "unknown_tool", f"No tool named {name!r}.")

        implementation = REGISTRY[name]
        signature = inspect.signature(implementation)
        accepted = {p for p in signature.parameters if p != "db"}
        unexpected = sorted(set(arguments) - accepted)
        if unexpected:
            return self._fail(
                name,
                arguments,
                "invalid_argument",
                f"Unexpected argument(s): {', '.join(unexpected)}.",
            )

        def invoke() -> dict[str, Any]:
            return implementation(self.db, **arguments)

        wrapped = invoke
        for behaviour in self.behaviours.get(name, []):
            wrapped = _bind(behaviour, wrapped)

        try:
            payload = wrapped()
        except DomainError as exc:
            return self._fail(name, arguments, exc.code, exc.message)
        except TypeError as exc:
            # Wrong argument type or a missing required argument reaches the
            # implementation as a TypeError; surface it the way the server's
            # own validation would.
            return self._fail(name, arguments, "invalid_argument", str(exc))
        except (KeyError, ValueError) as exc:
            return self._fail(name, arguments, "invalid_argument", str(exc))

        self.call_log.append({"tool": name, "arguments": arguments, "ok": True})
        return ToolResult(ok=True, payload=payload)

    def _fail(
        self, name: str, arguments: dict[str, Any], code: str, message: str
    ) -> ToolResult:
        self.call_log.append(
            {"tool": name, "arguments": arguments, "ok": False, "error_code": code}
        )
        return ToolResult(ok=False, payload={"message": message}, error_code=code)

    # -- episode lifecycle -----------------------------------------------

    def reset(self, snapshot: dict[str, Any]) -> None:
        self.db.restore(snapshot)
        self.call_log = []


def _bind(behaviour: Any, inner: Any) -> Any:
    def wrapped() -> dict[str, Any]:
        return behaviour(inner)

    return wrapped


def serve_stdio(
    schema_dir: str | Path,
    domain_dir: str | Path,
    defects_path: str | Path | None = None,
) -> None:
    """Expose the server over MCP on stdio."""
    import asyncio

    import mcp.server.stdio
    from mcp import types
    from mcp.server import Server

    server_state = ToolServer.build(schema_dir, domain_dir, defects_path)
    app: Server = Server("toolsmith-commerce")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=declaration["name"],
                description=declaration["description"],
                inputSchema=declaration["input_schema"],
            )
            for declaration in server_state.declarations()
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        result = server_state.call(name, arguments)
        return [types.TextContent(type="text", text=result.to_content())]

    async def run() -> None:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(run())


__all__ = ["ToolServer", "ToolResult", "serve_stdio", "TOOL_ORDER"]
