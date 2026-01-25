import pytest

from toolsmith.classifier.attribution import attribute, unattributed_share
from toolsmith.classifier.calibration import cohens_kappa, evaluate
from toolsmith.classifier.judge import Judge, class_distribution
from toolsmith.classifier.taxonomy import BY_KEY, CLASSES, KEYS, is_valid, tool_attributable

TOOLS = {"get_order", "cancel_order", "process_refund"}


def verdict(task_id, sample, failure_class, tool, confidence=0.8):
    return {
        "task_id": task_id,
        "sample": sample,
        "failure_class": failure_class,
        "tool": tool,
        "confidence": confidence,
        "reason": "",
    }


# -- taxonomy ------------------------------------------------------------


def test_taxonomy_keys_are_unique_and_stable():
    assert len(KEYS) == len(set(KEYS)) == 6
    assert set(BY_KEY) == set(KEYS)


def test_exactly_one_class_is_not_tool_attributable():
    assert sum(1 for c in CLASSES if not c.tool_attributable) == 1
    assert not tool_attributable("agent_attributable")


def test_every_class_has_a_definition():
    for failure in CLASSES:
        assert failure.definition.strip()
        assert is_valid(failure.key)


# -- judge coercion ------------------------------------------------------


def test_an_unknown_class_falls_back_to_agent_attributable():
    result = Judge._coerce("T001", 0, {"class": "gremlins", "tool": "get_order"}, TOOLS)
    assert result.failure_class == "agent_attributable"
    assert result.tool is None


def test_an_unknown_tool_is_dropped():
    result = Judge._coerce(
        "T001", 0, {"class": "wrong_tool_selection", "tool": "teleport"}, TOOLS
    )
    assert result.tool is None


def test_a_tool_on_an_agent_attributable_verdict_is_dropped():
    result = Judge._coerce(
        "T001", 0, {"class": "agent_attributable", "tool": "get_order"}, TOOLS
    )
    assert result.tool is None


def test_confidence_is_clamped():
    high = Judge._coerce("T001", 0, {"class": "loop", "tool": "get_order", "confidence": 4}, TOOLS)
    low = Judge._coerce("T001", 0, {"class": "loop", "tool": "get_order", "confidence": -1}, TOOLS)
    junk = Judge._coerce(
        "T001", 0, {"class": "loop", "tool": "get_order", "confidence": "high"}, TOOLS
    )
    assert high.confidence == 1.0 and low.confidence == 0.0 and junk.confidence == 0.0


def test_distribution_sums_to_one():
    verdicts = [
        verdict("T001", 0, "loop", "get_order"),
        verdict("T001", 1, "loop", "get_order"),
        verdict("T002", 0, "agent_attributable", None),
    ]
    distribution = class_distribution(verdicts)
    assert distribution["total"] == 3
    assert sum(distribution["share"].values()) == pytest.approx(1.0)
    assert distribution["counts"]["loop"] == 2


# -- attribution ---------------------------------------------------------


def test_attribution_ranks_by_distinct_tasks_not_runs():
    verdicts = [
        *[verdict("T001", i, "loop", "cancel_order") for i in range(8)],
        *[verdict(f"T0{i:02d}", 0, "wrong_tool_selection", "get_order") for i in range(10, 14)],
    ]
    ranked = attribute(verdicts)
    assert ranked[0].tool == "get_order"
    assert len(ranked[0].tasks) == 4
    assert ranked[1].tool == "cancel_order"
    assert ranked[1].runs == 8


def test_low_confidence_verdicts_are_ignored():
    verdicts = [verdict("T001", 0, "loop", "cancel_order", confidence=0.2)]
    assert attribute(verdicts) == []
    assert attribute(verdicts, min_confidence=0.1)


def test_agent_attributable_verdicts_never_produce_a_repair_target():
    verdicts = [verdict("T001", 0, "agent_attributable", "cancel_order")]
    assert attribute(verdicts) == []


def test_unattributed_share():
    verdicts = [
        verdict("T001", 0, "loop", "cancel_order"),
        verdict("T002", 0, "agent_attributable", None),
    ]
    assert unattributed_share(verdicts) == pytest.approx(0.5)
    assert unattributed_share([]) == 0.0


# -- calibration ---------------------------------------------------------


def test_perfect_agreement_is_one():
    labels = ["loop", "loop", "context_loss", "wrong_tool_selection"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_discounts_chance_agreement():
    human = ["loop"] * 90 + ["context_loss"] * 10
    judge = ["loop"] * 100
    # 90% raw agreement, but the judge never distinguishes anything.
    assert cohens_kappa(human, judge) == pytest.approx(0.0, abs=1e-9)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        cohens_kappa(["loop"], ["loop", "loop"])


def test_evaluate_reports_a_square_confusion_matrix():
    pairs = [("loop", "loop"), ("loop", "malformed_arguments"), ("context_loss", "context_loss")]
    calibration = evaluate(pairs)
    assert calibration.n == 3
    assert calibration.agreement == pytest.approx(2 / 3)
    assert set(calibration.confusion) == set(KEYS)
    assert calibration.confusion["loop"]["malformed_arguments"] == 1
    assert sum(sum(row.values()) for row in calibration.confusion.values()) == 3


def test_per_class_support_matches_the_human_labels():
    pairs = [("loop", "loop"), ("loop", "context_loss"), ("context_loss", "context_loss")]
    calibration = evaluate(pairs)
    assert calibration.per_class["loop"]["support"] == 2
    assert calibration.per_class["loop"]["recall"] == pytest.approx(0.5)
