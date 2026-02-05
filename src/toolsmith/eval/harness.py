"""Run the task suite at k samples per task and write the result files.

Every evaluation writes the same two files into its own directory:

  runs.jsonl   one row per (task, sample), in task then sample order
  summary.json the aggregate statistics plus the metadata identifying the run

The rows are the record; the summary is derived from them and can be rebuilt at
any time with ``toolsmith summarise``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent.runner import RunOutcome, run_episode
from ..config import Config
from ..llm import Client
from ..server.db import Database
from ..server.mcp_server import ToolServer
from ..server.schemas import SchemaSet
from .metrics import group_by_task, summarise
from .tasks import Task, load_tasks

ROW_ORDER = (
    "task_id",
    "sample",
    "success",
    "turns",
    "tool_calls",
    "failed_checks",
    "tools_used",
    "ended_by",
    "error",
    "wall_s",
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ordered(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ROW_ORDER if key in row}


@dataclass
class Harness:
    config: Config
    schema_dir: Path
    defects_path: Path | None = None

    def build_server(self) -> ToolServer:
        return ToolServer.build(
            self.schema_dir,
            self.config.resolve(self.config.domain_path),
            self.defects_path,
        )

    def run_suite(
        self,
        out_dir: str | Path,
        run_name: str,
        tasks: list[Task] | None = None,
        write_traces: bool = True,
    ) -> dict[str, Any]:
        tasks = tasks or load_tasks(self.config.resolve(self.config.tasks_path))
        server = self.build_server()
        baseline = server.db.snapshot()

        agent_client = Client(
            self.config.models.agent,
            self.config.models.max_tokens,
            self.config.models.effort,
        )
        user_client = Client(
            self.config.models.user_sim,
            self.config.models.max_tokens,
            self.config.models.effort,
        )

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        traces_dir = out / "traces"
        if write_traces:
            traces_dir.mkdir(exist_ok=True)

        started_at = _utc_now()
        clock = time.monotonic()
        rows: list[RunOutcome] = []

        with (out / "runs.jsonl").open("w") as handle:
            for task in tasks:
                for sample in range(self.config.eval.k):
                    outcome, transcript = run_episode(
                        task=task,
                        sample=sample,
                        server=server,
                        baseline=baseline,
                        agent_client=agent_client,
                        user_client=user_client,
                        user_mode=self.config.eval.user_mode,
                        max_turns=self.config.eval.max_turns,
                        max_tool_calls=self.config.eval.max_tool_calls,
                    )
                    rows.append(outcome)
                    handle.write(json.dumps(_ordered(outcome.to_row())) + "\n")
                    handle.flush()
                    if write_traces:
                        trace_path = traces_dir / f"{task.task_id}-{sample}.json"
                        trace_path.write_text(json.dumps(transcript, indent=2))

        elapsed = time.monotonic() - clock
        summary = self.build_summary(
            run_name=run_name,
            rows=[row.to_row() for row in rows],
            started_at=started_at,
            finished_at=_utc_now(),
            wall_clock_s=elapsed,
            token_usage={
                "agent_input": agent_client.usage.input_tokens,
                "agent_output": agent_client.usage.output_tokens,
                "user_input": user_client.usage.input_tokens,
                "user_output": user_client.usage.output_tokens,
            },
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def build_summary(
        self,
        run_name: str,
        rows: list[dict[str, Any]],
        started_at: str,
        finished_at: str,
        wall_clock_s: float,
        token_usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        outcomes = group_by_task(rows)
        stats = summarise(outcomes, self.config.eval.k)
        summary = {
            "run": run_name,
            "schema_set": str(Path(self.schema_dir).relative_to(self.config.resolve("."))),
            "schema_set_digest": SchemaSet.load(self.schema_dir).digest(),
            "defects": (
                str(Path(self.defects_path).relative_to(self.config.resolve(".")))
                if self.defects_path
                else None
            ),
            "config": self.config.name,
            "config_fingerprint": self.config.fingerprint(),
            "seed": self.config.seed,
            "user_mode": self.config.eval.user_mode,
            "models": {
                "agent": self.config.models.agent,
                "user_sim": self.config.models.user_sim,
                "effort": self.config.models.effort,
            },
            **stats,
            "ended_by": _tally(rows, "ended_by"),
            "mean_turns": round(_mean(rows, "turns"), 3),
            "mean_tool_calls": round(_mean(rows, "tool_calls"), 3),
            "token_usage": token_usage or {},
            "wall_clock_s": round(wall_clock_s, 1),
            "started_at": started_at,
            "finished_at": finished_at,
        }
        return summary


def _tally(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[field]] = counts.get(row[field], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(row[field] for row in rows) / len(rows)


def load_runs(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def load_outcomes(path: str | Path) -> dict[str, list[bool]]:
    return group_by_task(load_runs(path))


def load_database(config: Config) -> Database:
    return Database.load(config.resolve(config.domain_path))
