"""Thin wrapper over the Anthropic Messages API.

Everything in the loop -- agent, user simulator, judge, optimizer -- goes
through ``complete`` or ``complete_tools`` so retries, token accounting, and
JSON coercion live in one place.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: Any) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += 1


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw_content: list[Any] = field(default_factory=list)


class Client:
    """Retrying Messages API client with per-role model selection."""

    def __init__(self, model: str, max_tokens: int = 4096, effort: str = "medium"):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.usage = Usage()
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _send(self, **kwargs: Any) -> Any:
        delay = 2.0
        last: Exception | None = None
        for _ in range(6):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    output_config={"effort": self.effort},
                    **kwargs,
                )
                self.usage.add(response.usage)
                return response
            except _RETRYABLE as exc:
                last = exc
                time.sleep(delay + random.uniform(0, 1.0))
                delay = min(delay * 2, 45.0)
        raise RuntimeError(f"messages.create failed after retries: {last}")

    def complete(self, system: str, messages: list[dict[str, Any]]) -> Reply:
        response = self._send(system=system, messages=messages)
        return _to_reply(response)

    def complete_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Reply:
        response = self._send(system=system, messages=messages, tools=tools)
        return _to_reply(response)

    def complete_json(self, system: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Complete and parse the reply as a JSON object.

        The judge and the optimizer both want structured output; a single
        reformat attempt covers the occasional fenced or prefixed response.
        """
        reply = self.complete(system, messages)
        parsed = _extract_json(reply.text)
        if parsed is not None:
            return parsed
        repair = messages + [
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": "Return only the JSON object, with no surrounding text."},
        ]
        parsed = _extract_json(self.complete(system, repair).text)
        if parsed is None:
            raise ValueError("model did not return a JSON object")
        return parsed


def _to_reply(response: Any) -> Reply:
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    return Reply(
        text="\n".join(text_parts).strip(),
        tool_calls=calls,
        stop_reason=response.stop_reason,
        raw_content=list(response.content),
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    fenced = _JSON_BLOCK.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    brace = text.find("{")
    if brace != -1:
        candidates.append(text[brace : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
