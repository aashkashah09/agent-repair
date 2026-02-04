"""The regression gate.

A revision is deployed only if the full suite, re-run at k samples per task,
says it helped where it was meant to and did not quietly cost anything
elsewhere. Three things are checked, in this order:

1. Permission surface. If the revision lets a caller ask for anything it could
   not ask for before, it is blocked. This is checked first and decided on its
   own terms: no measured improvement overrides it.
2. Target improvement. The tasks whose failures were attributed to this tool
   must improve, with a paired bootstrap interval that excludes zero.
3. Collateral. On every other task, the paired interval's lower bound must stay
   above the tolerance. The tolerance is not zero: sampling alone moves a
   handful of tasks in either direction between any two evaluations of an
   unchanged suite, so a rule that rejected on a single task moving down would
   reject almost everything. The criterion has to be an interval.

Each decision also records what two weaker gates would have concluded from the
same evidence: one comparing point estimates only, and one with a single sample
per task. Those counterfactuals are what make the gate's cost legible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import GateConfig
from ..eval.bootstrap import Comparison, paired_bootstrap
from ..eval.metrics import paired_deltas, pass_1, pass_k, per_task_rate
from ..server.schemas import ToolSchema
from .permissions import diff

ACCEPTED = "accepted"
REJECTED_REGRESSION = "rejected_regression"
REJECTED_NO_GAIN = "rejected_no_gain"
BLOCKED_PERMISSION = "blocked_permission_expansion"


@dataclass
class GateDecision:
    revision_id: str
    round: int
    tool: str
    decision: str
    reason: str
    target_tasks: list[str]
    target: dict[str, Any]
    collateral: dict[str, Any]
    overall: dict[str, Any]
    pass_1_before: float
    pass_1_after: float
    pass_k_before: float
    pass_k_after: float
    permission_expansions: list[dict[str, str]] = field(default_factory=list)
    counterfactual: dict[str, bool] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision == ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "round": self.round,
            "tool": self.tool,
            "decision": self.decision,
            "reason": self.reason,
            "target_task_count": len(self.target_tasks),
            "target_tasks": self.target_tasks,
            "target": self.target,
            "collateral": self.collateral,
            "overall": self.overall,
            "pass_1_before": round(self.pass_1_before, 5),
            "pass_1_after": round(self.pass_1_after, 5),
            "pass_k_before": round(self.pass_k_before, 5),
            "pass_k_after": round(self.pass_k_after, 5),
            "permission_expansions": self.permission_expansions,
            "counterfactual": self.counterfactual,
        }


def _counterfactuals(
    before: dict[str, list[bool]],
    after: dict[str, list[bool]],
    target_tasks: list[str],
) -> dict[str, bool]:
    """What two weaker gates would have concluded from the same evidence.

    The point-estimate gate promotes whenever aggregate reliability went up,
    with no interval and no separate collateral test -- the comparison anyone
    makes by reading two summary numbers side by side. The k=1 gate applies the
    same rule to a single sample per task, which is what an evaluation without
    repeated sampling would have seen.
    """
    before_rate = per_task_rate(before)
    after_rate = per_task_rate(after)
    shared = sorted(set(before_rate) & set(after_rate))

    point_estimate = sum(after_rate[t] - before_rate[t] for t in shared) > 0

    k1_before = per_task_rate({task: results[:1] for task, results in before.items()})
    k1_after = per_task_rate({task: results[:1] for task, results in after.items()})
    k1 = sum(k1_after[t] - k1_before[t] for t in shared) > 0

    return {"point_estimate_gate": bool(point_estimate), "k1_gate": bool(k1)}


def evaluate(
    revision_id: str,
    round_index: int,
    tool: str,
    current_schema: ToolSchema,
    revised_schema: ToolSchema,
    before: dict[str, list[bool]],
    after: dict[str, list[bool]],
    target_tasks: list[str],
    config: GateConfig,
    k: int,
    resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 20260105,
) -> GateDecision:
    expansions = [expansion.to_dict() for expansion in diff(current_schema, revised_schema)]

    shared = sorted(set(before) & set(after))
    targets = [task for task in target_tasks if task in set(shared)]
    others = [task for task in shared if task not in set(targets)]

    target_cmp: Comparison = paired_bootstrap(
        paired_deltas(before, after, targets), resamples, ci_level, seed
    )
    collateral_cmp: Comparison = paired_bootstrap(
        paired_deltas(before, after, others), resamples, ci_level, seed + 1
    )
    overall_cmp: Comparison = paired_bootstrap(
        paired_deltas(before, after, shared), resamples, ci_level, seed + 2
    )

    decision, reason = _decide(target_cmp, collateral_cmp, expansions, config)

    return GateDecision(
        revision_id=revision_id,
        round=round_index,
        tool=tool,
        decision=decision,
        reason=reason,
        target_tasks=targets,
        target=target_cmp.to_dict(),
        collateral=collateral_cmp.to_dict(),
        overall=overall_cmp.to_dict(),
        pass_1_before=pass_1(before),
        pass_1_after=pass_1(after),
        pass_k_before=pass_k(before, k),
        pass_k_after=pass_k(after, k),
        permission_expansions=expansions,
        counterfactual=_counterfactuals(before, after, targets),
    )


def _decide(
    target: Comparison,
    collateral: Comparison,
    expansions: list[dict[str, str]],
    config: GateConfig,
) -> tuple[str, str]:
    if config.block_permission_expansion and expansions:
        kinds = ", ".join(sorted({expansion["kind"] for expansion in expansions}))
        return (
            BLOCKED_PERMISSION,
            f"revision expands the tool's permission surface ({kinds})",
        )

    min_target = config.min_target_delta * 100.0
    if target.mean < min_target or target.ci_low <= 0.0:
        return (
            REJECTED_NO_GAIN,
            f"target tasks moved {target.mean:+.2f} points, 95% CI "
            f"[{target.ci_low:+.2f}, {target.ci_high:+.2f}]; "
            f"required at least {min_target:+.2f} with an interval excluding zero",
        )

    tolerance = config.max_collateral_regression * 100.0
    if collateral.ci_low < tolerance:
        return (
            REJECTED_REGRESSION,
            f"non-target tasks moved {collateral.mean:+.2f} points, 95% CI "
            f"[{collateral.ci_low:+.2f}, {collateral.ci_high:+.2f}]; "
            f"lower bound is below the {tolerance:+.2f} point tolerance",
        )

    return (
        ACCEPTED,
        f"target tasks {target.mean:+.2f} points, 95% CI "
        f"[{target.ci_low:+.2f}, {target.ci_high:+.2f}]; non-target tasks "
        f"{collateral.mean:+.2f}, 95% CI [{collateral.ci_low:+.2f}, {collateral.ci_high:+.2f}]",
    )


def replay(decision: dict[str, Any], config: GateConfig) -> str:
    """Re-decide a recorded revision under a different set of thresholds.

    The interval a revision produced is a property of the evaluation, not of
    the thresholds, so a decision can be replayed against other settings
    without re-running anything.
    """
    if config.block_permission_expansion and decision["permission_expansions"]:
        return BLOCKED_PERMISSION
    target, collateral = decision["target"], decision["collateral"]
    if target["mean"] < config.min_target_delta * 100.0 or target["ci_low"] <= 0.0:
        return REJECTED_NO_GAIN
    if collateral["ci_low"] < config.max_collateral_regression * 100.0:
        return REJECTED_REGRESSION
    return ACCEPTED


def sensitivity(
    decisions: list[dict[str, Any]],
    tolerances: Sequence[float] = (-0.01, -0.02, -0.03, -0.04, -0.05, -0.08),
    minimum_gains: Sequence[float] = (0.02, 0.05, 0.10),
) -> dict[str, Any]:
    """How the ledger moves as the two thresholds are swept.

    Reported so that the headline split can be read against its neighbours
    rather than taken on its own.
    """
    grid = []
    for tolerance in tolerances:
        for gain in minimum_gains:
            config = GateConfig(min_target_delta=gain, max_collateral_regression=tolerance)
            outcomes = [replay(decision, config) for decision in decisions]
            grid.append(
                {
                    "max_collateral_regression": tolerance,
                    "min_target_delta": gain,
                    "accepted": outcomes.count(ACCEPTED),
                    "rejected_regression": outcomes.count(REJECTED_REGRESSION),
                    "rejected_no_gain": outcomes.count(REJECTED_NO_GAIN),
                    "blocked_permission_expansion": outcomes.count(BLOCKED_PERMISSION),
                }
            )
    return {"proposed": len(decisions), "grid": grid}


def tally(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision["decision"]] = counts.get(decision["decision"], 0) + 1
    rejected = [d for d in decisions if d["decision"] == REJECTED_REGRESSION]
    return {
        "proposed": len(decisions),
        "accepted": counts.get(ACCEPTED, 0),
        "rejected_regression": counts.get(REJECTED_REGRESSION, 0),
        "rejected_no_gain": counts.get(REJECTED_NO_GAIN, 0),
        "blocked_permission_expansion": counts.get(BLOCKED_PERMISSION, 0),
        "regression_rejections_a_point_estimate_gate_would_have_shipped": sum(
            1 for d in rejected if d["counterfactual"]["point_estimate_gate"]
        ),
        "regression_rejections_a_k1_gate_would_have_shipped": sum(
            1 for d in rejected if d["counterfactual"]["k1_gate"]
        ),
    }
