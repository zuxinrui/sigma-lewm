from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from scripts.analyze_paper_new_tasks_final import (
    ACTION_BRANCH_NEGATIVES,
    action_branch_causality_gate,
    catastrophic_failure,
    cluster_interval,
    cluster_paired_effect,
    final_analysis_provenance,
    frozen_factorial_fairness_gate,
    holm_adjust,
    old_gear_table,
    plot_conditional_gear_horizons,
)
from scripts import analyze_paper_new_tasks_final as analysis


def _branch_records(successes):
    return [
        {"variant": variant, "case_id": f"case-{case}", "success": case < count}
        for variant, count in successes.items()
        for case in range(30)
    ]


def test_catastrophic_tail_definition_is_task_specific_and_success_safe():
    assert catastrophic_failure("gear", "out_of_bounds") == 1
    assert catastrophic_failure("gear", "wrong_direction") == 0
    assert catastrophic_failure("franka", "dropped_or_scattered") == 1
    assert catastrophic_failure("franka", "never_opened") == 1
    assert catastrophic_failure("franka", "stacked_wrong_pose") == 0


def test_cluster_bootstrap_keeps_repeated_planner_seeds_in_case():
    values = np.asarray([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=float)
    result = cluster_interval(values)
    assert result["cluster_count"] == 4
    assert result["repeats_per_cluster"] == 2
    assert result["mean"] == 0.5
    assert result["mean_ci95"][0] <= 0.5 <= result["mean_ci95"][1]


def test_cluster_paired_effect_has_registered_direction_method_minus_baseline():
    baseline = np.zeros((30, 2))
    method = np.ones((30, 2))
    result = cluster_paired_effect(baseline, method)
    assert result["mean"] == 1.0
    assert result["mean_ci95"] == [1.0, 1.0]
    assert result["case_win_fraction"] == 1.0


def test_final_analysis_provenance_binds_complete_audit_manifest_and_generator(
    tmp_path: Path,
):
    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps({"manifest_sha256": "frozen-main"}), encoding="utf-8"
    )
    generator = tmp_path / "generator.py"
    generator.write_text("print('paper')\n", encoding="utf-8")
    formal = tmp_path / "formal.json"
    formal.write_text(
        json.dumps(
            {
                "schema_version": "paper-formal-artifact-integrity-v1",
                "require_complete": True,
                "manifest": str(experiment),
                "manifest_file_sha256": analysis.sha256(experiment),
                "manifest_sha256": "frozen-main",
            }
        ),
        encoding="utf-8",
    )
    provenance = final_analysis_provenance(formal, generator)
    assert provenance["generator_sha256"] == analysis.sha256(generator)
    assert provenance["formal_artifact_audit_sha256"] == analysis.sha256(formal)
    assert provenance["experiment_manifest_sha256"] == analysis.sha256(experiment)
    assert provenance["experiment_manifest_internal_sha256"] == "frozen-main"


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_action_branch_gate_requires_expert_to_beat_every_destructive_variant():
    successes = {"expert": 25, "small_perturb": 24}
    successes.update({variant: 5 for variant in ACTION_BRANCH_NEGATIVES["gear"]})
    gate = action_branch_causality_gate("gear", _branch_records(successes))
    assert gate["passes"] is True
    assert set(gate["comparisons"]) == set(ACTION_BRANCH_NEGATIVES["gear"])
    assert gate["comparisons"]["zero"]["success_delta"] == 20

    successes["delay"] = 25
    gate = action_branch_causality_gate("gear", _branch_records(successes))
    assert gate["passes"] is False
    assert gate["comparisons"]["delay"]["expert_strictly_better"] is False


def test_action_branch_gate_keeps_small_perturb_as_robustness_only():
    successes = {"expert": 20, "small_perturb": 20}
    successes.update({variant: 10 for variant in ACTION_BRANCH_NEGATIVES["franka"]})
    gate = action_branch_causality_gate("franka", _branch_records(successes))
    assert gate["passes"] is True
    assert "small_perturb" not in gate["comparisons"]


def test_conditional_horizon_plot_preserves_clustered_denominators(tmp_path: Path):
    rows = []
    for horizon, count in ((3, 10), (5, 30), (10, 10), (15, 10)):
        for planner_label, planner_seed in (("primary", 42), ("secondary", 10042)):
            for arm_index, arm in enumerate(("L0", "L1", "C0", "C1")):
                rows.append(
                    {
                        "horizon": horizon,
                        "arm": arm,
                        "planner_seed_label": planner_label,
                        "planner_seed_base": planner_seed,
                        "successes": min(count, arm_index + 1),
                        "case_count": count,
                    }
                )
    audit = {"tasks": {"gear": {"trigger": {"triggered": True}}}}
    destination = tmp_path / "conditional"
    plot_conditional_gear_horizons(rows, audit, destination)
    assert destination.with_suffix(".png").stat().st_size > 0
    assert destination.with_suffix(".pdf").stat().st_size > 0


def test_training_selection_table_verifies_the_complete_checkpoint_chain(
    tmp_path: Path, monkeypatch
):
    training = [
        {
            "epoch": 1,
            "checkpoint_sha256": "checkpoint-1",
            "loss_control_decision": {
                "validation_ema": 0.5,
                "train_factor": 0.4,
                "generalization_gap": 0.1,
            },
        },
        {
            "epoch": 2,
            "checkpoint_sha256": "checkpoint-2",
            "loss_control_decision": {
                "validation_ema": 0.6,
                "train_factor": 0.3,
                "generalization_gap": 0.3,
            },
        },
    ]
    h5 = [
        {"epoch": item["epoch"], "checkpoint_sha256": item["checkpoint_sha256"]}
        for item in training
    ]
    monkeypatch.setattr(analysis, "OUTPUT", tmp_path)
    monkeypatch.setattr(analysis, "load_training", lambda task, arm: training)
    monkeypatch.setattr(analysis, "load_h5", lambda task, arm: h5)
    for task in analysis.TASKS:
        for arm in analysis.ARMS:
            path = tmp_path / task / arm.lower() / "selection.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "paper-loss-selected-checkpoint-v1",
                "selection_authority": "validation-loss composite EMA only",
                "h5_used_for_selection": False,
                "final_epoch": 2,
                "stop_reason": "plateau",
                "best_epoch": 1,
                "best_validation_ema": 0.5,
                "checkpoint_sha256": "checkpoint-1",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            path.with_suffix(".json.done").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "result_schema_version": "paper-loss-selected-checkpoint-v1",
                        "record_count": 1,
                        "result_sha256": analysis.sha256(path),
                    }
                ),
                encoding="utf-8",
            )
    rows = analysis.training_selection_table()
    assert len(rows) == 8
    assert all(row["checkpoint_chain_verified"] for row in rows)
    assert all(row["selected_epoch"] == 1 for row in rows)
    assert all(row["h5_dev30_epoch_count"] == 2 for row in rows)
