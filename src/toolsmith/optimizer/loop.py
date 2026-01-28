"""The repair loop.

One round is: evaluate the current schema set, classify what failed, attribute
each failure to a tool, propose a revision for the worst offenders, and put each
revision through the gate against a fresh full-suite evaluation. Accepted
revisions are written into the next round's schema set; rejected ones are
recorded and discarded.

Repairs apply between rounds. Nothing here edits a schema while an episode is
in flight, so every evaluation runs against one fixed interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..classifier.attribution import ToolAttribution, attribute
from ..classifier.judge import Judge
from ..config import Config
from ..eval.harness import Harness, load_outcomes
from ..eval.tasks import load_tasks
from ..llm import Client
from ..server.schemas import SchemaSet
from .gate import evaluate as gate_evaluate
from .proposer import Proposer, contract_of


@dataclass
class RoundResult:
    round: int
    schema_set_in: str
    schema_set_out: str
    proposed: int
    accepted: int
    decisions: list[dict[str, Any]]
    pass_1_before: float
    pass_1_after: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "schema_set_in": self.schema_set_in,
            "schema_set_out": self.schema_set_out,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "pass_1_before": round(self.pass_1_before, 5),
            "pass_1_after": round(self.pass_1_after, 5),
            "decisions": self.decisions,
        }


class RepairLoop:
    def __init__(self, config: Config, defects_path: Path | None, results_root: Path):
        self.config = config
        self.defects_path = defects_path
        self.results_root = results_root
        self.judge = Judge(
            Client(config.models.judge, config.models.max_tokens, config.models.effort)
        )
        self.proposer = Proposer(
            Client(config.models.optimizer, config.models.max_tokens, config.models.effort)
        )

    # -- one round -------------------------------------------------------

    def run_round(
        self,
        round_index: int,
        schema_dir: Path,
        out_schema_dir: Path,
        max_revisions: int = 10,
    ) -> RoundResult:
        tasks = load_tasks(self.config.resolve(self.config.tasks_path))
        by_id = {task.task_id: task for task in tasks}

        baseline_dir = self.results_root / f"round{round_index}" / "baseline"
        harness = Harness(self.config, schema_dir, self.defects_path)
        harness.run_suite(baseline_dir, run_name=f"round{round_index}-baseline", tasks=tasks)
        before = load_outcomes(baseline_dir / "runs.jsonl")

        verdicts = self._classify(baseline_dir, by_id)
        (self.results_root / f"round{round_index}" / "verdicts.jsonl").write_text(
            "\n".join(json.dumps(verdict) for verdict in verdicts) + "\n"
        )

        attributions = attribute(verdicts)[:max_revisions]
        current = SchemaSet.load(schema_dir)
        accepted_set = current.copy()

        decisions: list[dict[str, Any]] = []
        for index, attribution in enumerate(attributions, start=1):
            revision_id = f"R{round_index}-{index:02d}"
            decision = self._evaluate_revision(
                revision_id=revision_id,
                round_index=round_index,
                attribution=attribution,
                current=current,
                accepted_set=accepted_set,
                before=before,
                verdicts=verdicts,
                tasks=tasks,
            )
            decisions.append(decision)

        accepted_set.save(out_schema_dir)
        after_dir = self.results_root / f"round{round_index}" / "after"
        Harness(self.config, out_schema_dir, self.defects_path).run_suite(
            after_dir, run_name=f"round{round_index}", tasks=tasks
        )
        after = load_outcomes(after_dir / "runs.jsonl")

        from ..eval.metrics import pass_1

        return RoundResult(
            round=round_index,
            schema_set_in=str(schema_dir),
            schema_set_out=str(out_schema_dir),
            proposed=len(decisions),
            accepted=sum(1 for d in decisions if d["decision"] == "accepted"),
            decisions=decisions,
            pass_1_before=pass_1(before),
            pass_1_after=pass_1(after),
        )

    # -- pieces ----------------------------------------------------------

    def _classify(self, run_dir: Path, by_id: dict[str, Any]) -> list[dict[str, Any]]:
        from ..eval.harness import load_runs

        known_tools = set(SchemaSet.load(self.config.resolve("data/schemas/clean")).tools)
        verdicts: list[dict[str, Any]] = []
        for row in load_runs(run_dir / "runs.jsonl"):
            if row["success"]:
                continue
            trace_path = run_dir / "traces" / f"{row['task_id']}-{row['sample']}.json"
            transcript = json.loads(trace_path.read_text()) if trace_path.exists() else []
            verdict = self.judge.classify(
                task_id=row["task_id"],
                sample=row["sample"],
                instruction=by_id[row["task_id"]].instruction,
                failed_checks=row["failed_checks"],
                call_log=[{"tool": tool, "arguments": {}, "ok": True} for tool in row["tools_used"]],
                transcript=transcript,
                known_tools=known_tools,
            )
            verdicts.append(verdict.to_row())
        return verdicts

    def _evaluate_revision(
        self,
        revision_id: str,
        round_index: int,
        attribution: ToolAttribution,
        current: SchemaSet,
        accepted_set: SchemaSet,
        before: dict[str, list[bool]],
        verdicts: list[dict[str, Any]],
        tasks: list[Any],
    ) -> dict[str, Any]:
        tool = attribution.tool
        failures = [v for v in verdicts if v.get("tool") == tool]
        proposal = self.proposer.propose(
            current=current[tool],
            contract=contract_of(tool),
            failures=failures,
            run_count=attribution.runs,
            task_count=len(attribution.tasks),
        )

        candidate = accepted_set.copy()
        candidate.replace(proposal.schema)
        candidate_dir = self.results_root / f"round{round_index}" / revision_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        schema_dir = candidate_dir / "schemas"
        candidate.save(schema_dir)

        Harness(self.config, schema_dir, self.defects_path).run_suite(
            candidate_dir, run_name=revision_id, tasks=tasks, write_traces=False
        )
        after = load_outcomes(candidate_dir / "runs.jsonl")

        decision = gate_evaluate(
            revision_id=revision_id,
            round_index=round_index,
            tool=tool,
            current_schema=current[tool],
            revised_schema=proposal.schema,
            before=before,
            after=after,
            target_tasks=attribution.tasks,
            config=self.config.gate,
            k=self.config.eval.k,
            resamples=self.config.eval.bootstrap_resamples,
            ci_level=self.config.eval.ci_level,
            seed=self.config.seed,
        )
        if decision.accepted:
            accepted_set.replace(proposal.schema)

        payload = decision.to_dict()
        payload["rationale"] = proposal.rationale
        return payload
