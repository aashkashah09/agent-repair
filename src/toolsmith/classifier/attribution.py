"""From classified failures to a repair queue.

Attribution answers a narrower question than classification: given every failed
run in a round, which tool interface should the optimizer be pointed at, and
which tasks would a fix have to move for the gate to accept it?

A tool's priority is the number of distinct tasks whose failures are attributed
to it, not the number of runs. A single flaky task failing eight times is one
task; eight tasks failing once each is a broader problem and gets worked first.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .taxonomy import tool_attributable


@dataclass
class ToolAttribution:
    tool: str
    tasks: list[str] = field(default_factory=list)
    runs: int = 0
    classes: dict[str, int] = field(default_factory=dict)
    mean_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "tasks": sorted(self.tasks),
            "task_count": len(self.tasks),
            "runs": self.runs,
            "classes": dict(sorted(self.classes.items(), key=lambda kv: (-kv[1], kv[0]))),
            "mean_confidence": round(self.mean_confidence, 3),
        }


def attribute(
    verdicts: list[dict[str, Any]],
    min_confidence: float = 0.5,
) -> list[ToolAttribution]:
    """Group verdicts by the tool they blame, strongest signal first."""
    tasks_by_tool: dict[str, set[str]] = defaultdict(set)
    runs_by_tool: dict[str, int] = defaultdict(int)
    classes_by_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    confidence_by_tool: dict[str, list[float]] = defaultdict(list)

    for verdict in verdicts:
        tool = verdict.get("tool")
        if not tool or not tool_attributable(verdict["failure_class"]):
            continue
        if float(verdict.get("confidence", 0.0)) < min_confidence:
            continue
        tasks_by_tool[tool].add(verdict["task_id"])
        runs_by_tool[tool] += 1
        classes_by_tool[tool][verdict["failure_class"]] += 1
        confidence_by_tool[tool].append(float(verdict["confidence"]))

    attributions = [
        ToolAttribution(
            tool=tool,
            tasks=sorted(tasks_by_tool[tool]),
            runs=runs_by_tool[tool],
            classes=dict(classes_by_tool[tool]),
            mean_confidence=sum(confidence_by_tool[tool]) / len(confidence_by_tool[tool]),
        )
        for tool in tasks_by_tool
    ]
    attributions.sort(key=lambda a: (-len(a.tasks), -a.runs, a.tool))
    return attributions


def target_tasks(attribution: ToolAttribution) -> list[str]:
    """The tasks a revision to this tool is expected to move."""
    return sorted(attribution.tasks)


def unattributed_share(verdicts: list[dict[str, Any]]) -> float:
    """Fraction of failures the judge did not pin on any tool."""
    if not verdicts:
        return 0.0
    loose = sum(
        1
        for verdict in verdicts
        if not verdict.get("tool") or not tool_attributable(verdict["failure_class"])
    )
    return loose / len(verdicts)
