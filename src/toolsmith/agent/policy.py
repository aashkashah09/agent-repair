"""The agent policy.

Deliberately plain: a system prompt describing the role, the tool declarations
exactly as the active schema set publishes them, and a tool-calling loop. The
prompt says nothing about any individual tool. Anything the agent knows about
how a tool behaves has to come from that tool's schema, which is what makes the
schema the thing under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm import Client, ToolCall
from ..server.mcp_server import ToolServer

SYSTEM = """\
You are a customer support agent for an online retailer. You talk to customers \
and you act on their accounts using the tools you have been given.

How to work:
- Use the tools to establish facts. Do not tell the customer something about \
their account, an order, stock or a shipment that you have not read from a tool.
- Read each tool's description and its parameter descriptions before calling \
it. They tell you what the tool expects and what it returns when something \
goes wrong.
- When a tool returns an error, read it and adjust. Do not retry an identical \
call that has already failed.
- Take actions that change the customer's account -- placing, changing, \
cancelling, refunding -- only when the customer has asked for that action. If \
you are unsure which of two actions they want, ask.
- Ask the customer for information you cannot obtain with a tool. Do not ask \
them for something a tool would tell you.

When you are finished, reply to the customer in plain language: what you did, \
and any figures, identifiers or dates they asked for. Include the actual values \
rather than referring to them."""


@dataclass
class AgentTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Agent:
    client: Client
    server: ToolServer
    max_tool_calls: int = 32
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": declaration["name"],
                "description": declaration["description"],
                "input_schema": declaration["input_schema"],
            }
            for declaration in self.server.declarations()
        ]

    def step(self, user_message: str) -> str:
        """Advance one customer turn, running tool calls until the agent speaks."""
        self.messages.append({"role": "user", "content": user_message})
        tools = self._tools()

        while True:
            reply = self.client.complete_tools(SYSTEM, self.messages, tools)
            self.messages.append({"role": "assistant", "content": reply.raw_content})

            if not reply.tool_calls:
                return reply.text

            if self.tool_call_count + len(reply.tool_calls) > self.max_tool_calls:
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": "Tool call budget for this conversation is "
                                "exhausted. Reply to the customer with what you have.",
                                "is_error": True,
                            }
                            for call in reply.tool_calls
                        ],
                    }
                )
                self.tool_call_count += len(reply.tool_calls)
                continue

            results = []
            for call in reply.tool_calls:
                outcome = self.server.call(call.name, call.arguments)
                self.tool_call_count += 1
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": outcome.to_content(),
                        "is_error": not outcome.ok,
                    }
                )
            self.messages.append({"role": "user", "content": results})

    def reset(self) -> None:
        self.messages = []
        self.tool_call_count = 0
