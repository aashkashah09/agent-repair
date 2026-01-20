"""pass^k and the per-task success rates everything downstream is built on.

pass^1 is the mean success rate over all (task, sample) pairs. pass^k is the
fraction of tasks that succeed on all k samples, which is the quantity that
actually tracks whether a suite is usable: an agent that solves a task five
times out of eight has not solved it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    task_id: str
    sample: int
    success: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunRecord:
        return cls(
            task_id=payload["task_id"],
            sample=int(payload["sample"]),
            success=bool(payload["success"]),
        )


def group_by_task(records: Iterable[RunRecord | dict[str, Any]]) -> dict[str, list[bool]]:
    """Collapse run records into per-task outcome lists, ordered by sample index."""
    buckets: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for record in records:
        if isinstance(record, dict):
            record = RunRecord.from_dict(record)
        buckets[record.task_id].append((record.sample, record.success))
    return {
        task_id: [success for _, success in sorted(samples)]
        for task_id, samples in sorted(buckets.items())
    }


def per_task_rate(outcomes: dict[str, list[bool]]) -> dict[str, float]:
    return {
        task_id: (sum(results) / len(results) if results else 0.0)
        for task_id, results in outcomes.items()
    }


def pass_1(outcomes: dict[str, list[bool]]) -> float:
    total = sum(len(results) for results in outcomes.values())
    if total == 0:
        return 0.0
    return sum(sum(results) for results in outcomes.values()) / total


def pass_k(outcomes: dict[str, list[bool]], k: int) -> float:
    """Fraction of tasks solved on all k samples.

    Tasks with fewer than k samples are excluded rather than counted as
    failures, so a partial run reports the statistic it can actually support.
    """
    eligible = [results for results in outcomes.values() if len(results) >= k]
    if not eligible:
        return 0.0
    solved = sum(1 for results in eligible if all(results[:k]))
    return solved / len(eligible)


def solved_all(outcomes: dict[str, list[bool]], k: int) -> list[str]:
    return sorted(
        task_id
        for task_id, results in outcomes.items()
        if len(results) >= k and all(results[:k])
    )


def failing_tasks(outcomes: dict[str, list[bool]], threshold: float = 1.0) -> list[str]:
    """Tasks whose success rate is below ``threshold``."""
    rates = per_task_rate(outcomes)
    return sorted(task_id for task_id, rate in rates.items() if rate < threshold)


def summarise(outcomes: dict[str, list[bool]], k: int) -> dict[str, Any]:
    p1 = pass_1(outcomes)
    pk = pass_k(outcomes, k)
    return {
        "tasks": len(outcomes),
        "samples_per_task": k,
        "runs": sum(len(results) for results in outcomes.values()),
        "successes": sum(sum(results) for results in outcomes.values()),
        "pass_1": round(p1, 5),
        "pass_k": round(pk, 5),
        "pass_k_over_pass_1": round(pk / p1, 5) if p1 else 0.0,
    }


def paired_deltas(
    before: dict[str, list[bool]],
    after: dict[str, list[bool]],
    task_ids: Sequence[str] | None = None,
) -> list[float]:
    """Per-task change in success rate, in percentage points."""
    before_rate = per_task_rate(before)
    after_rate = per_task_rate(after)
    keys = list(task_ids) if task_ids is not None else sorted(
        set(before_rate) & set(after_rate)
    )
    return [100.0 * (after_rate[key] - before_rate[key]) for key in keys]
