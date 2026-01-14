import pytest

from toolsmith.config import REPO_ROOT
from toolsmith.server.schemas import TOOL_ORDER, SchemaSet
from toolsmith.server.seeding import (
    BEHAVIOURS,
    PATTERNS,
    SEVERITIES,
    apply_defects,
    behaviours_for,
    runtime_behaviours,
)

SEEDED_DIR = REPO_ROOT / "data" / "schemas" / "seeded"


def test_catalogue_is_eight_patterns_at_three_severities(defects):
    assert len(defects) == 24
    by_pattern = {}
    for defect in defects:
        by_pattern.setdefault(defect.pattern, []).append(defect.severity)
    assert set(by_pattern) == set(PATTERNS)
    for pattern, severities in by_pattern.items():
        assert sorted(severities) == sorted(SEVERITIES), pattern


def test_defect_ids_are_unique_and_ordered(defects):
    ids = [defect.defect_id for defect in defects]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_every_defect_names_real_tools(defects):
    for defect in defects:
        assert defect.tools
        for tool in defect.tools:
            assert tool in TOOL_ORDER, f"{defect.defect_id}: {tool}"


def test_every_mutation_names_a_real_tool_or_behaviour(defects):
    for defect in defects:
        assert defect.mutations, defect.defect_id
        for mutation in defect.mutations:
            if mutation["op"] == "runtime":
                assert mutation["behaviour"] in BEHAVIOURS
            else:
                assert mutation["tool"] in TOOL_ORDER


def test_seeding_is_deterministic(clean_schemas, defects):
    first = apply_defects(clean_schemas, defects)
    second = apply_defects(clean_schemas, defects)
    for name in TOOL_ORDER:
        assert first[name].to_dict() == second[name].to_dict()


def test_seeding_does_not_touch_the_clean_set(clean_schemas, defects):
    before = clean_schemas["cancel_order"].to_dict()
    apply_defects(clean_schemas, defects)
    assert clean_schemas["cancel_order"].to_dict() == before


def test_committed_seeded_set_matches_the_catalogue(clean_schemas, defects):
    expected = apply_defects(clean_schemas, defects)
    committed = SchemaSet.load(SEEDED_DIR)
    for name in TOOL_ORDER:
        assert committed[name].to_dict() == expected[name].to_dict(), name


def test_seeding_actually_degrades_the_schemas(clean_schemas, defects):
    seeded = apply_defects(clean_schemas, defects)
    # D01 opens a closed value set.
    assert "enum" in clean_schemas["cancel_order"].parameters["reason"]
    assert "enum" not in seeded["cancel_order"].parameters["reason"]
    # D19 flattens a structured parameter.
    assert clean_schemas["update_customer_profile"].parameters["address"]["type"] == "object"
    assert seeded["update_customer_profile"].parameters["address"]["type"] == "string"
    # D22 removes the pagination cursor.
    assert "cursor" in clean_schemas["list_orders"].parameters
    assert "cursor" not in seeded["list_orders"].parameters


def test_seeding_leaves_a_valid_schema_set(clean_schemas, defects):
    apply_defects(clean_schemas, defects).validate()


def test_runtime_behaviours_are_grouped_by_tool(defects):
    names = runtime_behaviours(defects)
    assert names
    grouped = behaviours_for(names)
    for tool, wrappers in grouped.items():
        assert tool in TOOL_ORDER
        assert wrappers


def test_a_subset_installs_only_its_own_behaviours(defects):
    subset = [d for d in defects if d.defect_id == "D22"]
    grouped = behaviours_for(runtime_behaviours(subset))
    assert set(grouped) == {"list_orders"}
    assert len(grouped["list_orders"]) == 1


@pytest.mark.parametrize("defect_id", ["D04", "D05", "D06", "D22", "D23", "D24"])
def test_behavioural_defects_carry_a_runtime_mutation(defects, defect_id):
    defect = next(d for d in defects if d.defect_id == defect_id)
    assert any(m["op"] == "runtime" for m in defect.mutations)
