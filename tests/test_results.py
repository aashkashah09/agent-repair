"""Checks over the committed run artefacts.

These guard the invariants that make the results readable: every run has one row
per (task, sample), the summary matches the rows it was derived from, and the
gate ledger accounts for every revision.
"""

import json

import pytest

from toolsmith.config import REPO_ROOT, load_config
from toolsmith.eval.harness import load_outcomes, load_runs
from toolsmith.eval.metrics import summarise
from toolsmith.eval.tasks import load_tasks
from toolsmith.server.schemas import TOOL_ORDER

RESULTS = REPO_ROOT / "results"
RUN_DIRS = sorted(path.parent for path in RESULTS.glob("*/runs.jsonl"))
TASK_IDS = [task.task_id for task in load_tasks(REPO_ROOT / "data" / "tasks" / "tasks.jsonl")]
K = load_config(REPO_ROOT / "configs" / "default.yaml").eval.k


def summary(run_dir):
    return json.loads((run_dir / "summary.json").read_text())


def test_there_are_runs_to_check():
    assert RUN_DIRS


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_one_row_per_task_and_sample(run_dir):
    rows = load_runs(run_dir / "runs.jsonl")
    assert len(rows) == len(TASK_IDS) * K
    seen = {(row["task_id"], row["sample"]) for row in rows}
    assert seen == {(task_id, sample) for task_id in TASK_IDS for sample in range(K)}


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_rows_have_a_stable_field_order(run_dir):
    with (run_dir / "runs.jsonl").open() as handle:
        keys = [list(json.loads(line)) for line in handle if line.strip()]
    assert len({tuple(k) for k in keys}) == 1


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_summary_agrees_with_the_rows(run_dir):
    recomputed = summarise(load_outcomes(run_dir / "runs.jsonl"), K)
    stored = summary(run_dir)
    for field in ("tasks", "runs", "successes", "pass_1", "pass_k", "pass_k_over_pass_1"):
        assert stored[field] == pytest.approx(recomputed[field]), f"{run_dir.name}.{field}"


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_summary_carries_its_provenance(run_dir):
    stored = summary(run_dir)
    for field in ("run", "schema_set", "config", "config_fingerprint", "seed",
                  "user_mode", "models", "wall_clock_s", "started_at", "finished_at"):
        assert field in stored, f"{run_dir.name} is missing {field}"
    assert (REPO_ROOT / stored["schema_set"]).is_dir()
    assert stored["finished_at"] >= stored["started_at"]


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_failed_rows_name_a_failing_check_and_successes_do_not(run_dir):
    for row in load_runs(run_dir / "runs.jsonl"):
        if row["success"]:
            assert row["failed_checks"] == []
        else:
            assert row["failed_checks"]


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_tools_used_are_real_tools(run_dir):
    for row in load_runs(run_dir / "runs.jsonl"):
        assert set(row["tools_used"]) <= set(TOOL_ORDER)


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_pass_k_never_exceeds_pass_1(run_dir):
    stored = summary(run_dir)
    assert stored["pass_k"] <= stored["pass_1"]


def test_repair_rounds_are_monotone():
    rounds = ["seeded", "round1", "round2", "round3", "round4"]
    values = [summary(RESULTS / name)["pass_1"] for name in rounds]
    assert values == sorted(values)


def test_the_hand_tuned_set_is_the_ceiling():
    ceiling = summary(RESULTS / "hand_tuned")["pass_1"]
    for name in ("seeded", "round1", "round2", "round3", "round4"):
        assert summary(RESULTS / name)["pass_1"] <= ceiling


def test_scripted_users_score_above_adversarial_ones():
    for adversarial, scripted in (("seeded", "scripted_seeded"), ("round4", "scripted_round4")):
        assert summary(RESULTS / scripted)["pass_1"] > summary(RESULTS / adversarial)["pass_1"]


# -- verdicts ------------------------------------------------------------


VERDICT_FILES = sorted(RESULTS.glob("*/verdicts.jsonl"))


@pytest.mark.parametrize("path", VERDICT_FILES, ids=lambda p: p.parent.name)
def test_a_verdict_exists_for_every_failed_run(path):
    rows = load_runs(path.parent / "runs.jsonl")
    failures = {(row["task_id"], row["sample"]) for row in rows if not row["success"]}
    verdicts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert {(v["task_id"], v["sample"]) for v in verdicts} == failures


@pytest.mark.parametrize("path", VERDICT_FILES, ids=lambda p: p.parent.name)
def test_verdicts_are_well_formed(path):
    from toolsmith.classifier.taxonomy import KEYS, tool_attributable

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        verdict = json.loads(line)
        assert verdict["failure_class"] in KEYS
        assert 0.0 <= verdict["confidence"] <= 1.0
        assert verdict["reason"]
        if verdict["tool"] is not None:
            assert verdict["tool"] in TOOL_ORDER
            assert tool_attributable(verdict["failure_class"])


# -- gate ledger ---------------------------------------------------------


def gate_decisions():
    path = RESULTS / "gate" / "decisions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_gate_summary_agrees_with_the_decisions():
    from toolsmith.optimizer.gate import tally

    decisions = gate_decisions()
    stored = json.loads((RESULTS / "gate" / "summary.json").read_text())
    recomputed = tally(decisions)
    for field, value in recomputed.items():
        assert stored[field] == value


def test_revision_ids_are_unique_and_name_their_round():
    decisions = gate_decisions()
    ids = [decision["revision_id"] for decision in decisions]
    assert len(set(ids)) == len(ids)
    for decision in decisions:
        assert decision["revision_id"].startswith(f"R{decision['round']}-")


def test_every_decision_targets_a_real_tool_and_real_tasks():
    for decision in gate_decisions():
        assert decision["tool"] in TOOL_ORDER
        assert set(decision["target_tasks"]) <= set(TASK_IDS)


def test_blocked_decisions_name_their_expansion_and_others_do_not():
    for decision in gate_decisions():
        if decision["decision"] == "blocked_permission_expansion":
            assert decision["permission_expansions"]
        else:
            assert not decision["permission_expansions"]


def test_accepted_decisions_cleared_both_criteria():
    from toolsmith.config import GateConfig

    config = GateConfig()
    for decision in gate_decisions():
        if decision["decision"] != "accepted":
            continue
        assert decision["target"]["ci_low"] > 0
        assert decision["target"]["mean"] >= config.min_target_delta * 100
        assert decision["collateral"]["ci_low"] >= config.max_collateral_regression * 100


def test_regression_rejections_did_move_their_target():
    for decision in gate_decisions():
        if decision["decision"] == "rejected_regression":
            assert decision["target"]["mean"] > 0
            assert decision["collateral"]["ci_low"] < 0


# -- defect ledger -------------------------------------------------------


def test_defect_ledger_accounts_for_every_seeded_defect():
    ledger = json.loads((RESULTS / "defects" / "recovery.json").read_text())
    catalogue = json.loads(
        (REPO_ROOT / "data" / "defects" / "seeded_defects.json").read_text()
    )["defects"]
    assert ledger["seeded"] == len(catalogue)
    assert {row["defect_id"] for row in ledger["defects"]} == {
        defect["defect_id"] for defect in catalogue
    }
    assert ledger["resolved"] + ledger["unresolved"] == ledger["seeded"]
    assert sum(ledger["by_round"].values()) == ledger["resolved"]


def test_resolved_defects_name_the_revision_that_cleared_them():
    ledger = json.loads((RESULTS / "defects" / "recovery.json").read_text())
    accepted = {
        decision["revision_id"]
        for decision in gate_decisions()
        if decision["decision"] == "accepted"
    }
    for row in ledger["defects"]:
        if row["resolved"]:
            assert row["resolved_by"] in accepted
            assert row["resolved_in_round"] in (1, 2, 3, 4)
        else:
            assert row["resolved_by"] is None


# -- schema provenance ---------------------------------------------------


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_summary_pins_the_schema_set_by_content(run_dir):
    from toolsmith.server.schemas import SchemaSet

    stored = summary(run_dir)
    assert stored["schema_set_digest"] == SchemaSet.load(REPO_ROOT / stored["schema_set"]).digest()


def test_distinct_schema_sets_have_distinct_digests():
    from toolsmith.server.schemas import SchemaSet

    digests = {
        d.name: SchemaSet.load(d).digest()
        for d in sorted((REPO_ROOT / "data" / "schemas").iterdir())
        if d.is_dir()
    }
    assert len(set(digests.values())) == len(digests)


# -- the accept-all ablation ---------------------------------------------


def test_the_ablation_ships_every_proposal():
    from toolsmith.optimizer.permissions import diff
    from toolsmith.server.schemas import TOOL_ORDER, SchemaSet

    gated = SchemaSet.load(REPO_ROOT / "data" / "schemas" / "round4")
    everything = SchemaSet.load(REPO_ROOT / "data" / "schemas" / "accept_all")
    changed = [t for t in TOOL_ORDER if gated[t].to_dict() != everything[t].to_dict()]
    assert changed, "the ablation must differ from the gated set"
    expansions = [e for t in TOOL_ORDER for e in diff(gated[t], everything[t])]
    assert expansions, "accepting everything deploys the permission-expanding revisions"


def test_the_ablation_trades_consistency_for_mean():
    gated = summary(RESULTS / "round4")
    everything = summary(RESULTS / "ablation_accept_all")
    seeded = summary(RESULTS / "seeded")
    assert everything["pass_1"] < gated["pass_1"]
    assert everything["pass_k"] < gated["pass_k"]
    # Most of the mean gain survives; the variance gain does not.
    assert everything["pass_1"] > seeded["pass_1"] + 0.10
    assert everything["pass_k_over_pass_1"] < gated["pass_k_over_pass_1"]


def test_committed_levels_agree_with_a_recomputation():
    from toolsmith.eval.bootstrap import level_interval

    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    for entry in json.loads((RESULTS / "comparisons" / "levels.json").read_text()):
        outcomes = load_outcomes(RESULTS / entry["run"] / "runs.jsonl")
        for statistic in ("pass_1", "pass_k"):
            value, low, high = level_interval(
                outcomes, statistic, config.eval.k, entry["resamples"],
                entry["ci_level"], config.seed,
            )
            assert entry[statistic] == {"value": value, "ci_low": low, "ci_high": high}


def test_committed_sensitivity_agrees_with_a_replay():
    from toolsmith.optimizer.gate import sensitivity

    decisions = gate_decisions()
    stored = json.loads((RESULTS / "gate" / "sensitivity.json").read_text())
    assert stored == sensitivity(decisions)


def test_the_shipped_thresholds_reproduce_the_recorded_ledger():
    from toolsmith.config import GateConfig
    from toolsmith.optimizer.gate import replay

    config = GateConfig()
    for decision in gate_decisions():
        assert replay(decision, config) == decision["decision"], decision["revision_id"]
