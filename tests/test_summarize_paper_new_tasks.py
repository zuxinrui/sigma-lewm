from __future__ import annotations

import pytest

from scripts.summarize_paper_new_tasks import (
    exact_paired_binomial,
    paired_bootstrap_delta,
    progress_values,
    wilson,
)


def test_wilson_interval_contains_observed_rate():
    low, high = wilson(21, 30)
    assert low < 0.7 < high
    assert (low, high) == pytest.approx(
        (0.5212421254128504, 0.8333525173175619)
    )


def test_exact_paired_test_counts_discordant_cases():
    result = exact_paired_binomial(
        [False, False, True, True],
        [True, True, False, True],
    )
    assert result == {
        "a_only": 1,
        "b_only": 2,
        "discordant": 3,
        "two_sided_exact_p": 1.0,
    }


def test_case_bootstrap_preserves_pairing_and_direction():
    result = paired_bootstrap_delta(
        [False] * 10,
        [True] * 10,
        seed=3,
        draws=100,
    )
    assert result["delta"] == 1.0
    assert result["ci95"] == [1.0, 1.0]
    assert result["probability_delta_positive"] == 1.0


def test_gear_progress_is_initial_violation_minus_best_and_final():
    record = {
        "paired_source": {
            "init_height_mm": 6.0,
            "init_xy_mm": 3.0,
            "init_orientation_deg": 5.0,
        },
        "best": {"normalized_violation": 1.5},
        "final_height_mm": -3.0,
        "final_xy_mm": 3.0,
        "final_orientation_deg": 5.0,
    }
    # initial=max(2,1,1)=2; final=max(0,1,1)+1+1=3.
    assert progress_values("gear", record) == pytest.approx((0.5, -1.0))


def test_franka_progress_uses_goal_distance_progress_fields():
    record = {
        "best": {
            "active_cube_goal_progress_best": 0.04,
            "active_cube_goal_progress_final": -0.01,
        }
    }
    assert progress_values("franka", record) == pytest.approx((0.04, -0.01))
