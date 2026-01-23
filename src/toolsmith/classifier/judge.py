"""LLM judge over failed episodes.

The judge sees the transcript, the tool calls with their arguments and returns,
the task's grading checks and which of them failed. It returns a class from the
taxonomy, the tool it holds responsible when the class is tool-attributable,
and a one-line reason. It is not asked whether the run failed -- grading already
settled that -- only why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm import Client
from .taxonomy import BY_KEY, KEYS, definitions_block, is_valid, tool_attributable

SYSTEM = """\
You are analysing a failed episode of a customer support agent working against \
a set of tools. Grading has already determined the episode failed. Your job is \
to say why, in one label.

Assign exactly one class:
{definitions}

Rules:
- Label the earliest point at which the episode was already lost, not the last \
thing that went wrong.
- Prefer a tool-attributable class only when the tool's published description, \
parameters or documented error returns are what misled the agent. If the schema \
said what the agent needed and the agent still went wrong, the class is \
agent_attributable.
- When the class is tool-attributable, name the single tool whose interface is \
responsible in "tool". Use the tool the agent was misled *about*, which is not \
always the tool it called. When the class is agent_attributable, set "tool" to \
null.
- "confidence" is your own, between 0 and 1.

Reply with a JSON object and nothing else:
{{"class": "<one of {keys}>", "tool": "<tool name or null>", \
"reason": "<one sentence>", "confidence": <number>}}"""

PROMPT = """\
TASK
{instruction}

GRADING CHECKS THAT FAILED
{failed}

TOOL CALLS
{calls}

CONVERSATION
{transcript}"""


@dataclass
class Verdict:
    task_id: str
    sample: int
    failure_class: str
    tool: str | None
    reason: str
    confidence: float
    raw: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample": self.sample,
            "failure_class": self.failure_class,
            "tool": self.tool,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


def _render_calls(call_log: list[dict[str, Any]], limit: int = 40) -> str:
    lines = []
    for index, entry in enumerate(call_log[:limit], start=1):
        status = "ok" if entry.get("ok") else f"error:{entry.get('error_code')}"
        arguments = json.dumps(entry.get("arguments", {}), separators=(",", ":"))
        lines.append(f"{index}. {entry['tool']}({arguments}) -> {status}")
    if len(call_log) > limit:
        lines.append(f"... {len(call_log) - limit} further calls omitted")
    return "\n".join(lines) or "(no tool calls)"


def _render_transcript(transcript: list[dict[str, str]], limit: int = 24) -> str:
    lines = [f"{turn['role'].upper()}: {turn['content']}" for turn in transcript[:limit]]
    if len(transcript) > limit:
        lines.append(f"... {len(transcript) - limit} further turns omitted")
    return "\n\n".join(lines)


class Judge:
    def __init__(self, client: Client):
        self.client = client

    def classify(
        self,
        task_id: str,
        sample: int,
        instruction: str,
        failed_checks: list[str],
        call_log: list[dict[str, Any]],
        transcript: list[dict[str, str]],
        known_tools: set[str],
    ) -> Verdict:
        system = SYSTEM.format(definitions=definitions_block(), keys=", ".join(KEYS))
        prompt = PROMPT.format(
            instruction=instruction,
            failed=", ".join(failed_checks) or "(none recorded)",
            calls=_render_calls(call_log),
            transcript=_render_transcript(transcript),
        )
        payload = self.client.complete_json(system, [{"role": "user", "content": prompt}])
        return self._coerce(task_id, sample, payload, known_tools)

    @staticmethod
    def _coerce(
        task_id: str, sample: int, payload: dict[str, Any], known_tools: set[str]
    ) -> Verdict:
        label = str(payload.get("class", "")).strip()
        if not is_valid(label):
            label = "agent_attributable"
        tool = payload.get("tool")
        if isinstance(tool, str):
            tool = tool.strip() or None
        if tool is not None and tool not in known_tools:
            tool = None
        if not tool_attributable(label):
            tool = None
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return Verdict(
            task_id=task_id,
            sample=sample,
            failure_class=label,
            tool=tool,
            reason=str(payload.get("reason", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            raw=payload,
        )


def class_distribution(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: 0 for key in KEYS}
    for verdict in verdicts:
        counts[verdict["failure_class"]] = counts.get(verdict["failure_class"], 0) + 1
    total = sum(counts.values()) or 1
    return {
        "total": sum(counts.values()),
        "counts": counts,
        "share": {key: round(value / total, 4) for key, value in counts.items()},
        "labels": {key: BY_KEY[key].label for key in KEYS},
    }
