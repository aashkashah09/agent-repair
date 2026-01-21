"""One episode, start to finish.

Restores the database, runs the conversation, grades the final state, and
returns the row the harness writes to ``runs.jsonl``. Full transcripts are
written separately and are not part of the row: at k=8 over a hundred tasks
they are large, and everything the analysis needs is in the row itself.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..eval.tasks import Task, grade
from ..llm import Client
from ..server.mcp_server import ToolServer
from ..simulator.graph import Episode
from ..simulator.personas import resolve
from ..simulator.user_sim import UserSimulator
from .policy import Agent


@dataclass
class RunOutcome:
    task_id: str
    sample: int
    success: bool
    turns: int
    tool_calls: int
    failed_checks: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    ended_by: str = "user"
    error: str | None = None
    wall_s: float = 0.0

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["wall_s"] = round(self.wall_s, 2)
        return row


def run_episode(
    task: Task,
    sample: int,
    server: ToolServer,
    baseline: dict[str, Any],
    agent_client: Client,
    user_client: Client,
    user_mode: str = "adversarial",
    max_turns: int = 24,
    max_tool_calls: int = 32,
) -> tuple[RunOutcome, list[dict[str, str]]]:
    started = time.monotonic()
    server.reset(baseline)

    agent = Agent(client=agent_client, server=server, max_tool_calls=max_tool_calls)
    user = UserSimulator(
        client=user_client,
        task=task,
        persona=resolve(task.persona, user_mode),
    )
    episode = Episode(agent=agent, user=user, max_turns=max_turns)

    try:
        state = episode.run()
    except Exception as exc:  # an episode that crashes is a failed episode
        return (
            RunOutcome(
                task_id=task.task_id,
                sample=sample,
                success=False,
                turns=len(episode.transcript) // 2,
                tool_calls=agent.tool_call_count,
                failed_checks=["episode_error"],
                tools_used=sorted({entry["tool"] for entry in server.call_log}),
                ended_by="error",
                error=f"{type(exc).__name__}: {exc}",
                wall_s=time.monotonic() - started,
            ),
            episode.transcript,
        )

    result = grade(task, server.db, baseline, episode.last_agent_message())
    return (
        RunOutcome(
            task_id=task.task_id,
            sample=sample,
            success=result.success,
            turns=int(state.get("turns", 0)),
            tool_calls=agent.tool_call_count,
            failed_checks=result.failed_checks,
            tools_used=sorted({entry["tool"] for entry in server.call_log}),
            ended_by=str(state.get("ended_by", "user")),
            wall_s=time.monotonic() - started,
        ),
        episode.transcript,
    )
