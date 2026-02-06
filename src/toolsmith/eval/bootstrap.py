"""Paired bootstrap over tasks.

Two evaluations of the same suite share their tasks, so comparisons are paired:
resample tasks with replacement, take the mean of the per-task deltas within
each resample, and read the interval off the resampled distribution. Pairing
removes between-task variance, which dominates the between-run variance we
actually want to measure.

Resampling is over tasks, not over runs. The k samples of a task are not
independent of each other -- they share the task, the tools and the persona --
so the task is the unit that gets resampled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Comparison:
    n: int
    mean: float
    ci_low: float
    ci_high: float
    ci_level: float
    resamples: int
    p_two_sided: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["significant"] = self.significant
        return payload


def paired_bootstrap(
    deltas: Sequence[float],
    resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 20260105,
) -> Comparison:
    """Percentile bootstrap CI on the mean of paired per-task deltas."""
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        return Comparison(0, 0.0, 0.0, 0.0, ci_level, resamples, 1.0)

    rng = np.random.default_rng(seed)
    n = values.size
    draws = rng.integers(0, n, size=(resamples, n))
    means = values[draws].mean(axis=1)

    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    observed = float(values.mean())

    # Two-sided bootstrap p-value: the proportion of resampled means on the
    # far side of zero, doubled and clipped to 1.
    if observed >= 0:
        tail = float((means <= 0).mean())
    else:
        tail = float((means >= 0).mean())
    p_value = min(1.0, 2.0 * tail)

    return Comparison(
        n=n,
        mean=round(observed, 4),
        ci_low=round(float(low), 4),
        ci_high=round(float(high), 4),
        ci_level=ci_level,
        resamples=resamples,
        p_two_sided=round(p_value, 5),
    )


def level_interval(
    outcomes: dict[str, list[bool]],
    statistic: str = "pass_1",
    k: int = 8,
    resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 20260105,
) -> tuple[float, float, float]:
    """Bootstrap interval on one evaluation's own pass rate.

    Unpaired, so these are much wider than the paired intervals a comparison
    produces: most of the spread here is between-task difficulty, which pairing
    removes. Reported as ``(observed, low, high)`` in percentage points.
    """
    task_ids = sorted(outcomes)
    per_task = np.array(
        [
            [sum(outcomes[t]) / len(outcomes[t]) for t in task_ids],
            [1.0 if len(outcomes[t]) >= k and all(outcomes[t][:k]) else 0.0 for t in task_ids],
        ]
    )
    row = per_task[0] if statistic == "pass_1" else per_task[1]

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, row.size, size=(resamples, row.size))
    means = row[draws].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return (
        round(100.0 * float(row.mean()), 4),
        round(100.0 * float(low), 4),
        round(100.0 * float(high), 4),
    )


def compare(
    before: dict[str, list[bool]],
    after: dict[str, list[bool]],
    task_ids: Sequence[str] | None = None,
    resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 20260105,
) -> Comparison:
    from .metrics import paired_deltas

    return paired_bootstrap(
        paired_deltas(before, after, task_ids),
        resamples=resamples,
        ci_level=ci_level,
        seed=seed,
    )
