import pytest

from toolsmith.config import REPO_ROOT
from toolsmith.eval.tasks import (
    CHECKS,
    DOMAINS,
    PERSONAS,
    Task,
    TaskError,
    grade,
    load_tasks,
    validate_tasks,
)
from toolsmith.server.schemas import TOOL_ORDER

TASKS = load_tasks(REPO_ROOT / "data" / "tasks" / "tasks.jsonl")


def test_suite_size_and_identifiers():
    assert len(TASKS) == 100
    assert [task.task_id for task in TASKS] == [f"T{i:03d}" for i in range(1, 101)]


def test_every_persona_and_domain_is_used():
    assert {task.persona for task in TASKS} == set(PERSONAS)
    assert {task.domain for task in TASKS} <= set(DOMAINS)


def test_personas_are_evenly_spread():
    counts = {persona: 0 for persona in PERSONAS}
    for task in TASKS:
        counts[task.persona] += 1
    assert max(counts.values()) - min(counts.values()) <= 2


def test_expected_tools_exist():
    for task in TASKS:
        assert task.tools_expected
        for tool in task.tools_expected:
            assert tool in TOOL_ORDER, f"{task.task_id}: {tool}"


def test_every_tool_is_exercised_by_some_task():
    used = {tool for task in TASKS for tool in task.tools_expected}
    assert used == set(TOOL_ORDER)


def test_checks_reference_known_entities(db):
    for task in TASKS:
        for check in task.checks:
            if "order_id" in check:
                assert check["order_id"] in db.orders, task.task_id
            if "customer_id" in check:
                assert check["customer_id"] in db.customers, task.task_id
            if "ticket_id" in check:
                assert check["ticket_id"] in db.tickets, task.task_id
            if check["type"] == "inventory_available":
                assert check["sku"] in db.inventory, task.task_id


def test_every_check_type_in_use_is_implemented():
    used = {check["type"] for task in TASKS for check in task.checks}
    assert used <= set(CHECKS)


def test_read_only_tasks_assert_that_nothing_changed():
    for task in TASKS:
        types = {check["type"] for check in task.checks}
        if "communicated" in types and not (types & {"order_status", "refund", "order_created"}):
            assert types & {"state_unchanged", "no_new_orders"}, task.task_id


def test_duplicate_task_ids_are_rejected():
    duplicated = [TASKS[0], Task.from_dict(TASKS[0].to_dict())]
    with pytest.raises(TaskError):
        validate_tasks(duplicated)


def test_unknown_persona_is_rejected():
    payload = TASKS[0].to_dict()
    payload["persona"] = "cheerful"
    with pytest.raises(TaskError):
        validate_tasks([Task.from_dict(payload)])


def test_unknown_check_type_is_rejected():
    payload = TASKS[0].to_dict()
    payload["checks"] = [{"type": "vibes"}]
    with pytest.raises(TaskError):
        validate_tasks([Task.from_dict(payload)])


# -- grading -------------------------------------------------------------


def task_with(check) -> Task:
    return Task(
        task_id="TX",
        domain="commerce",
        persona="underspecified",
        instruction="x",
        known={},
        checks=[check],
    )


def test_state_unchanged_holds_on_an_untouched_database(db):
    baseline = db.snapshot()
    assert grade(task_with({"type": "state_unchanged"}), db, baseline, "").success


def test_state_unchanged_fails_after_a_status_change(db):
    baseline = db.snapshot()
    order_id = next(o for o, r in db.orders.items() if r["status"] == "pending")
    db.orders[order_id]["status"] = "cancelled"
    assert not grade(task_with({"type": "state_unchanged"}), db, baseline, "").success


def test_order_created_ignores_orders_present_in_the_baseline(db):
    baseline = db.snapshot()
    existing = next(iter(db.orders.values()))
    check = {
        "type": "order_created",
        "customer_id": existing["customer_id"],
        "items": [{"sku": i["sku"], "quantity": i["quantity"]} for i in existing["items"]],
    }
    assert not grade(task_with(check), db, baseline, "").success


def test_refund_only_counts_refunds_issued_during_the_episode(db):
    baseline = db.snapshot()
    order_id = next(iter(db.refunds.values()))["order_id"]
    check = {"type": "refund", "order_id": order_id}
    assert not grade(task_with(check), db, baseline, "").success


def test_communicated_is_whitespace_and_case_insensitive(db):
    baseline = db.snapshot()
    check = {"type": "communicated", "any_of": ["w00042"]}
    assert grade(task_with(check), db, baseline, "Your order   W00042,  is on its way.").success
    assert not grade(task_with(check), db, baseline, "I could not find it.").success


def test_grade_names_the_failing_checks(db):
    baseline = db.snapshot()
    task = Task(
        task_id="TX", domain="commerce", persona="underspecified", instruction="x",
        known={},
        checks=[
            {"type": "state_unchanged"},
            {"type": "communicated", "any_of": ["nothing like this"]},
        ],
    )
    result = grade(task, db, baseline, "")
    assert not result.success
    assert result.failed_checks == ["1:communicated"]
