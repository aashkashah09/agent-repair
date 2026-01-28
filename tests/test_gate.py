import pytest
from helpers import outcomes

from toolsmith.config import GateConfig
from toolsmith.optimizer.gate import (
    ACCEPTED,
    BLOCKED_PERMISSION,
    REJECTED_NO_GAIN,
    REJECTED_REGRESSION,
    evaluate,
    tally,
)
from toolsmith.server.schemas import ToolSchema

CONFIG = GateConfig()
TARGETS = [f"T{i:03d}" for i in range(1, 21)]
OTHERS = [f"T{i:03d}" for i in range(21, 101)]


def schema(description="Cancel an order. The order must not have shipped.") -> ToolSchema:
    return ToolSchema.from_dict(
        {
            "name": "cancel_order",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Order."}},
                "required": ["order_id"],
            },
            "error_returns": [{"code": "not_found", "when": "No such order."}],
        }
    )


def suite(target_hits: int, other_hits: int) -> dict[str, list[bool]]:
    spec = {task: [1] * target_hits + [0] * (8 - target_hits) for task in TARGETS}
    spec.update({task: [1] * other_hits + [0] * (8 - other_hits) for task in OTHERS})
    return outcomes(spec)


def run(before, after, current=None, revised=None):
    return evaluate(
        revision_id="R1-01",
        round_index=1,
        tool="cancel_order",
        current_schema=current or schema(),
        revised_schema=revised or schema(),
        before=before,
        after=after,
        target_tasks=TARGETS,
        config=CONFIG,
        k=8,
        resamples=2000,
        seed=17,
    )


def test_a_clear_win_is_accepted():
    decision = run(suite(2, 6), suite(7, 6))
    assert decision.decision == ACCEPTED
    assert decision.accepted


def test_target_gain_with_collateral_damage_is_rejected():
    decision = run(suite(2, 6), suite(7, 4))
    assert decision.decision == REJECTED_REGRESSION
    assert decision.collateral["ci_low"] < CONFIG.max_collateral_regression * 100


def test_no_movement_on_target_is_rejected():
    decision = run(suite(2, 6), suite(2, 6))
    assert decision.decision == REJECTED_NO_GAIN


def test_a_gain_below_the_floor_is_rejected():
    # A third of a sample per task is under the five-point minimum.
    before = outcomes({task: [1, 1, 0, 0, 0, 0, 0, 0] for task in TARGETS + OTHERS})
    after = dict(before)
    after = {task: list(results) for task, results in before.items()}
    for task in TARGETS[:5]:
        after[task][2] = True
    decision = run(before, after)
    assert decision.decision == REJECTED_NO_GAIN


def test_permission_expansion_blocks_even_a_large_win():
    widened = schema(description="Cancel an order.")
    decision = run(suite(1, 6), suite(8, 7), revised=widened)
    assert decision.decision == BLOCKED_PERMISSION
    assert decision.permission_expansions


def test_permission_check_runs_before_the_statistics():
    widened = schema(description="Cancel an order.")
    decision = run(suite(2, 6), suite(2, 3), revised=widened)
    assert decision.decision == BLOCKED_PERMISSION


def test_blocking_can_be_turned_off():
    widened = schema(description="Cancel an order.")
    decision = evaluate(
        revision_id="R1-02", round_index=1, tool="cancel_order",
        current_schema=schema(), revised_schema=widened,
        before=suite(2, 6), after=suite(7, 6), target_tasks=TARGETS,
        config=GateConfig(block_permission_expansion=False), k=8, resamples=2000, seed=17,
    )
    assert decision.decision == ACCEPTED


def test_decision_records_both_counterfactuals():
    decision = run(suite(2, 6), suite(7, 4))
    assert set(decision.counterfactual) == {"point_estimate_gate", "k1_gate"}


def test_the_point_estimate_gate_reads_the_aggregate_only():
    # Twenty target tasks gain five samples each; eighty others lose two. The
    # gate rejects it on the collateral interval, but the totals still net out
    # negative, so the weaker gate would reject it too.
    heavy = run(suite(2, 6), suite(7, 4))
    assert heavy.decision == REJECTED_REGRESSION
    assert heavy.counterfactual["point_estimate_gate"] is False

    # The same target gain against a lighter collateral cost nets out positive:
    # the aggregate improved, so the weaker gate ships a revision this one
    # still rejects.
    before = suite(2, 7)
    after = {task: list(results) for task, results in before.items()}
    for task in TARGETS:
        after[task] = [True] * 7 + [False]
    for task in OTHERS[:60]:
        after[task] = [True] * 6 + [False] * 2
    light = run(before, after)
    assert light.decision == REJECTED_REGRESSION
    assert light.counterfactual["point_estimate_gate"] is True


def test_reason_is_populated_for_every_outcome():
    for before, after, revised in (
        (suite(2, 6), suite(7, 6), None),
        (suite(2, 6), suite(7, 4), None),
        (suite(2, 6), suite(2, 6), None),
        (suite(2, 6), suite(7, 6), schema(description="Cancel an order.")),
    ):
        assert run(before, after, revised=revised).reason


def test_pass_rates_are_carried_on_the_decision():
    decision = run(suite(2, 6), suite(7, 6))
    assert decision.pass_1_after > decision.pass_1_before
    assert 0.0 <= decision.pass_k_before <= 1.0


def test_tally_counts_outcomes_and_counterfactuals():
    decisions = [
        {"decision": ACCEPTED, "counterfactual": {"point_estimate_gate": True, "k1_gate": True}},
        {"decision": REJECTED_REGRESSION,
         "counterfactual": {"point_estimate_gate": True, "k1_gate": True}},
        {"decision": REJECTED_REGRESSION,
         "counterfactual": {"point_estimate_gate": True, "k1_gate": False}},
        {"decision": BLOCKED_PERMISSION,
         "counterfactual": {"point_estimate_gate": False, "k1_gate": False}},
    ]
    counts = tally(decisions)
    assert counts["proposed"] == 4
    assert counts["accepted"] == 1
    assert counts["rejected_regression"] == 2
    assert counts["blocked_permission_expansion"] == 1
    assert counts["regression_rejections_a_point_estimate_gate_would_have_shipped"] == 2
    assert counts["regression_rejections_a_k1_gate_would_have_shipped"] == 1


def test_target_tasks_absent_from_the_runs_are_dropped():
    before = suite(2, 6)
    after = suite(7, 6)
    decision = evaluate(
        revision_id="R1-03", round_index=1, tool="cancel_order",
        current_schema=schema(), revised_schema=schema(),
        before=before, after=after, target_tasks=TARGETS + ["T999"],
        config=CONFIG, k=8, resamples=1000, seed=17,
    )
    assert "T999" not in decision.target_tasks
    assert decision.target["n"] == len(TARGETS)


def test_collateral_and_target_partition_the_suite():
    decision = run(suite(2, 6), suite(7, 6))
    assert decision.target["n"] + decision.collateral["n"] == decision.overall["n"]
    assert decision.overall["n"] == pytest.approx(100)
