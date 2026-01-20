import pytest
from helpers import outcomes

from toolsmith.eval.metrics import (
    failing_tasks,
    group_by_task,
    paired_deltas,
    pass_1,
    pass_k,
    per_task_rate,
    solved_all,
    summarise,
)


def test_group_by_task_orders_by_sample():
    rows = [
        {"task_id": "T002", "sample": 1, "success": True},
        {"task_id": "T001", "sample": 2, "success": True},
        {"task_id": "T001", "sample": 0, "success": False},
        {"task_id": "T001", "sample": 1, "success": True},
    ]
    grouped = group_by_task(rows)
    assert list(grouped) == ["T001", "T002"]
    assert grouped["T001"] == [False, True, True]


def test_pass_1_is_the_run_level_mean():
    data = outcomes({"T001": [1, 1, 0, 0], "T002": [1, 0, 0, 0]})
    assert pass_1(data) == pytest.approx(3 / 8)


def test_pass_k_requires_every_sample():
    data = outcomes({"T001": [1, 1, 1, 1], "T002": [1, 1, 1, 0]})
    assert pass_k(data, 4) == pytest.approx(0.5)
    assert solved_all(data, 4) == ["T001"]


def test_pass_k_never_exceeds_pass_1():
    data = outcomes({f"T{i:03d}": [1, 1, 0, 1] for i in range(10)})
    assert pass_k(data, 4) <= pass_1(data)


def test_pass_k_excludes_tasks_with_too_few_samples():
    data = outcomes({"T001": [1, 1], "T002": [1, 1, 1, 1]})
    # T001 has two samples, so it cannot answer a k=4 question either way.
    assert pass_k(data, 4) == pytest.approx(1.0)


def test_pass_k_of_empty_suite_is_zero():
    assert pass_k({}, 8) == 0.0
    assert pass_1({}) == 0.0


def test_per_task_rate_and_failing_tasks():
    data = outcomes({"T001": [1, 1], "T002": [1, 0], "T003": [0, 0]})
    assert per_task_rate(data) == {"T001": 1.0, "T002": 0.5, "T003": 0.0}
    assert failing_tasks(data) == ["T002", "T003"]


def test_paired_deltas_are_percentage_points_in_task_order():
    before = outcomes({"T001": [0, 0, 0, 0], "T002": [1, 1, 1, 1]})
    after = outcomes({"T001": [1, 1, 0, 0], "T002": [1, 1, 1, 0]})
    assert paired_deltas(before, after) == pytest.approx([50.0, -25.0])


def test_paired_deltas_honours_an_explicit_task_list():
    before = outcomes({"T001": [0, 0], "T002": [0, 0]})
    after = outcomes({"T001": [1, 1], "T002": [0, 0]})
    assert paired_deltas(before, after, ["T002"]) == pytest.approx([0.0])


def test_summarise_reports_counts_and_the_ratio():
    data = outcomes({"T001": [1] * 8, "T002": [1, 1, 1, 1, 0, 0, 0, 0]})
    summary = summarise(data, 8)
    assert summary["runs"] == 16
    assert summary["successes"] == 12
    assert summary["pass_1"] == pytest.approx(0.75)
    assert summary["pass_k"] == pytest.approx(0.5)
    assert summary["pass_k_over_pass_1"] == pytest.approx(2 / 3, rel=1e-4)
