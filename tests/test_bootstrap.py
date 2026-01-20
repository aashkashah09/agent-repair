import pytest
from helpers import outcomes

from toolsmith.eval.bootstrap import compare, paired_bootstrap


def test_interval_brackets_the_observed_mean():
    deltas = [12.5, 0.0, -12.5, 25.0, 12.5, 0.0, 0.0, 37.5, -12.5, 12.5]
    result = paired_bootstrap(deltas, resamples=4000, seed=11)
    assert result.ci_low <= result.mean <= result.ci_high
    assert result.n == len(deltas)


def test_identical_runs_give_a_zero_width_interval():
    result = paired_bootstrap([0.0] * 40, resamples=2000, seed=3)
    assert result.mean == 0.0
    assert result.ci_low == 0.0 and result.ci_high == 0.0
    assert not result.significant


def test_a_uniform_shift_is_significant():
    result = paired_bootstrap([12.5] * 60, resamples=4000, seed=5)
    assert result.significant
    assert result.ci_low > 0


def test_noise_around_zero_is_not_significant():
    deltas = [12.5, -12.5] * 30
    result = paired_bootstrap(deltas, resamples=6000, seed=7)
    assert not result.significant
    assert result.p_two_sided > 0.05


def test_same_seed_is_reproducible():
    deltas = [5.0, -3.0, 12.0, 0.0, -8.0, 4.0, 9.0]
    first = paired_bootstrap(deltas, resamples=3000, seed=42)
    second = paired_bootstrap(deltas, resamples=3000, seed=42)
    assert first == second


def test_a_wider_level_gives_a_wider_interval():
    deltas = [10.0, -5.0, 20.0, 0.0, 5.0, -10.0, 15.0, 5.0]
    narrow = paired_bootstrap(deltas, resamples=6000, ci_level=0.80, seed=9)
    wide = paired_bootstrap(deltas, resamples=6000, ci_level=0.99, seed=9)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_empty_input_is_handled():
    result = paired_bootstrap([], resamples=100, seed=1)
    assert result.n == 0 and result.mean == 0.0 and not result.significant


def test_compare_pairs_on_task_id_not_position():
    before = outcomes({"T001": [0, 0], "T002": [1, 1], "T003": [1, 0]})
    after = outcomes({"T003": [1, 1], "T001": [1, 1], "T002": [1, 1]})
    result = compare(before, after, resamples=2000, seed=13)
    assert result.n == 3
    assert result.mean == pytest.approx((100.0 + 0.0 + 50.0) / 3)


def test_compare_ignores_tasks_missing_from_either_side():
    before = outcomes({"T001": [1, 0], "T002": [0, 0]})
    after = outcomes({"T001": [1, 1]})
    result = compare(before, after, resamples=1000, seed=2)
    assert result.n == 1
