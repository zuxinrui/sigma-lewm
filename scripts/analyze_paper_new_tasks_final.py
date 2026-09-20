#!/usr/bin/env python3
"""Generate final paired statistics, paper figures, source tables, and report."""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from scripts.summarize_paper_new_tasks import (
    ARMS,
    COLORS,
    committed_json,
    exact_paired_binomial,
    load_h5,
    load_training,
    progress_values,
    sha256,
    wilson,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/paper_new_tasks_v1"
SEEDS = (("primary", 42), ("secondary", 10042))
TASKS = ("gear", "franka")
DISPLAY = {
    "L0": "LeWM",
    "L1": "LeWM + task constraints",
    "C0": "Champion",
    "C1": "Champion + task constraints",
}
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260906
ACTION_BRANCH_NEGATIVES = {
    "gear": (
        "cross",
        "delay",
        "zero",
        "persistence",
        "xyz_cross",
        "yaw_cross",
    ),
    "franka": (
        "cross",
        "delay",
        "zero",
        "persistence",
        "arm_cross_phase_matched",
        "phase_delay",
        "phase_advance",
    ),
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def final_analysis_provenance(
    formal_audit_path: Path, generator_path: Path
) -> dict:
    """Bind derived paper artifacts to their audited inputs and exact generator."""

    formal_audit_path = formal_audit_path.expanduser().resolve()
    generator_path = generator_path.expanduser().resolve()
    formal = json.loads(formal_audit_path.read_text(encoding="utf-8"))
    if (
        formal.get("schema_version") != "paper-formal-artifact-integrity-v1"
        or formal.get("require_complete") is not True
    ):
        raise ValueError("final analysis requires the complete formal artifact audit")
    experiment_manifest = Path(formal["manifest"]).expanduser().resolve()
    if sha256(experiment_manifest) != formal["manifest_file_sha256"]:
        raise ValueError("experiment manifest changed after the complete formal audit")
    experiment_payload = json.loads(
        experiment_manifest.read_text(encoding="utf-8")
    )
    if experiment_payload.get("manifest_sha256") != formal["manifest_sha256"]:
        raise ValueError("formal audit and experiment manifest identities differ")
    return {
        "command": [sys.executable, str(generator_path)],
        "generator": str(generator_path),
        "generator_sha256": sha256(generator_path),
        "formal_artifact_audit": str(formal_audit_path),
        "formal_artifact_audit_sha256": sha256(formal_audit_path),
        "experiment_manifest": str(experiment_manifest),
        "experiment_manifest_sha256": sha256(experiment_manifest),
        "experiment_manifest_internal_sha256": experiment_payload[
            "manifest_sha256"
        ],
    }


def _quantile(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def cluster_interval(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> dict:
    """Case-cluster bootstrap for [case, repeated planner seed] values."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"cluster values must be finite [case,repeat], got {values.shape}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(BOOTSTRAP_DRAWS, values.shape[0]))
    per_case = values.mean(axis=1)
    mean_distribution = per_case[indices].mean(axis=1)
    median_distribution = np.median(values[indices].reshape(BOOTSTRAP_DRAWS, -1), axis=1)
    return {
        "cluster_count": int(values.shape[0]),
        "repeats_per_cluster": int(values.shape[1]),
        "mean": float(values.mean()),
        "mean_ci95": _quantile(mean_distribution),
        "median": float(np.median(values)),
        "median_ci95": _quantile(median_distribution),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
    }


def cluster_paired_effect(
    baseline: np.ndarray, method: np.ndarray, *, seed: int = BOOTSTRAP_SEED
) -> dict:
    baseline = np.asarray(baseline, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    if baseline.shape != method.shape:
        raise ValueError("paired cluster arrays differ")
    delta = method - baseline
    report = cluster_interval(delta, seed=seed)
    per_case = delta.mean(axis=1)
    report.update(
        {
            "case_win_fraction": float(np.mean(per_case > 0)),
            "case_tie_fraction": float(np.mean(per_case == 0)),
            "probability_mean_positive": None,
        }
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(per_case), size=(BOOTSTRAP_DRAWS, len(per_case)))
    distribution = per_case[indices].mean(axis=1)
    report["probability_mean_positive"] = float(np.mean(distribution > 0))
    return report


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * float(p_values[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def action_branch_causality_gate(task: str, records: list[dict]) -> dict:
    """Require expert actions to beat every frozen destructive branch.

    The simulator branching experiment is model-independent and starts every
    variant from the same restored state. It complements the model-specific
    offline action-contract gate: the latter shows that the world model uses
    the aligned action, while this gate shows that the corresponding action
    corruptions actually change task success in the simulator. Small
    perturbations are a robustness control, not a destructive negative, so
    they are reported elsewhere but intentionally excluded from this gate.
    """

    if task not in ACTION_BRANCH_NEGATIVES:
        raise ValueError(f"unknown action-branch task: {task}")
    by_variant: dict[str, dict[str, int]] = {}
    for record in records:
        variant = str(record["variant"])
        case_id = str(record["case_id"])
        values = by_variant.setdefault(variant, {})
        if case_id in values:
            raise ValueError(f"duplicate action branch: {task}/{variant}/{case_id}")
        values[case_id] = int(bool(record["success"]))

    required = ("expert", *ACTION_BRANCH_NEGATIVES[task])
    missing = [variant for variant in required if variant not in by_variant]
    if missing:
        raise ValueError(f"missing action branches for {task}: {missing}")
    expert_ids = list(by_variant["expert"])
    if len(expert_ids) != 30:
        raise ValueError(f"{task} expert branch must contain 30 cases")

    comparisons = {}
    for variant in ACTION_BRANCH_NEGATIVES[task]:
        if list(by_variant[variant]) != expert_ids:
            raise ValueError(f"{task}/{variant} case order differs from expert")
        expert = np.asarray(
            [by_variant["expert"][case_id] for case_id in expert_ids], dtype=float
        )
        negative = np.asarray(
            [by_variant[variant][case_id] for case_id in expert_ids], dtype=float
        )
        paired = exact_paired_binomial(negative, expert)
        expert_successes = int(expert.sum())
        negative_successes = int(negative.sum())
        comparisons[variant] = {
            "expert_successes": expert_successes,
            "negative_successes": negative_successes,
            "success_delta": expert_successes - negative_successes,
            "expert_strictly_better": expert_successes > negative_successes,
            "rescues": int(np.sum((negative == 0) & (expert == 1))),
            "harms": int(np.sum((negative == 1) & (expert == 0))),
            "mcnemar": paired,
        }
    return {
        "criterion": (
            "same-init expert strict-success count must exceed every frozen "
            "destructive action branch; small_perturb is robustness-only"
        ),
        "passes": all(
            item["expert_strictly_better"] for item in comparisons.values()
        ),
        "comparisons": comparisons,
    }


def frozen_factorial_fairness_gate(lock: dict) -> dict:
    """Expose the frozen anti-confound checks used by the strongest claim."""

    tasks = {}
    for task in TASKS:
        initialization = lock["initialization_audit"][task]
        arms = initialization["arms"]
        config = lock["config_audit"][task]
        deployable_counts = {
            arm: int(arms[arm]["deployable_planner_parameters"]) for arm in ARMS
        }
        training_only_counts = {
            arm: int(arms[arm]["training_only_auxiliary_parameters"])
            for arm in ARMS
        }
        total_trainable_counts = {
            arm: int(arms[arm]["total_trainable_parameters"]) for arm in ARMS
        }
        optimizer_steps = int(config["optimizer_steps_per_complete_epoch"])
        sampling = config["epoch_sampling_contract"]["per_epoch"]
        planner = lock["planner_contracts"][task]
        planner_expected = (
            {
                "mode": "projected_cem",
                "horizon_macro": 5,
                "full_commit_macro": 5,
                "action_block": 5,
                "budget_raw": 50,
                "num_samples": 300,
                "iterations": 30,
                "topk": 30,
            }
            if task == "gear"
            else {
                "mode": "train_terminal_wm_rerank",
                "horizon_macro": 5,
                "full_commit_macro": 5,
                "action_block": 5,
                "budget_raw": 50,
                "retrieval_topk": 300,
                "continuation": "abstain",
            }
        )
        checks = {
            "equal_deployable_parameter_count": len(set(deployable_counts.values())) == 1,
            "released_champion_initial_deployable_state_bit_exact": all(
                initialization["released_vs_champion_deployable_bit_exact"].values()
            ),
            "task_factor_shared_initial_tensors_bit_exact": bool(
                initialization["task_factor_shared_encoder_predictor_action_bit_exact"]
            ),
            "single_task_common_data_optimizer_contract": len(
                str(config["common_contract_sha256"])
            ) == 64,
            "fixed_100_epoch_scheduler_horizon": int(config["scheduler_max_steps"])
            == optimizer_steps * 100,
            "fixed_one_percent_scheduler_warmup": int(config["scheduler_warmup_steps"])
            == optimizer_steps,
            "all_100_epoch_orders_precommitted": set(sampling)
            == {str(epoch) for epoch in range(1, 101)},
            "single_task_level_planner_candidate_contract": all(
                planner.get(key) == value for key, value in planner_expected.items()
            ),
        }
        tasks[task] = {
            "passes": all(checks.values()),
            "checks": checks,
            "deployable_planner_parameters": deployable_counts,
            "training_only_auxiliary_parameters": training_only_counts,
            "total_trainable_parameters": total_trainable_counts,
            "optimizer_steps_per_complete_epoch": optimizer_steps,
            "scheduler_max_steps": int(config["scheduler_max_steps"]),
            "planner_contract": planner,
        }
    return {
        "passes": all(item["passes"] for item in tasks.values()),
        "interpretation": (
            "deployable planner size, task-local data/order, per-epoch optimizer "
            "steps, 100-epoch scheduler horizon, and candidate support are frozen; "
            "champion-only auxiliary parameters are training-only and reported"
        ),
        "tasks": tasks,
    }


def _final_path(task: str, arm: str, seed_label: str) -> Path:
    return OUTPUT / task / arm.lower() / "final_test" / seed_label / "eval.json"


def load_final(task: str, arm: str, seed_label: str) -> dict:
    path = _final_path(task, arm, seed_label)
    payload = committed_json(
        path, schema="paper-h5-checkpoint-aggregate-v1", count=30
    )
    if payload is None:
        raise ValueError(f"missing committed final result: {path}")
    return payload


def failure_category(task: str, record: dict) -> str:
    if bool(record["success"]):
        return "success"
    best_progress, final_progress = progress_values(task, record)
    if task == "gear":
        if int(record.get("max_success_streak", 0)) > 0:
            return "edge_jitter"
        if float(record["final_xy_mm"]) > 30 or abs(float(record["final_height_mm"])) > 30:
            return "out_of_bounds"
        if best_progress <= 0:
            return "wrong_direction"
        if final_progress < 0:
            return "approach_then_reverse"
        return "stalled_near_target"
    final = record["final"]
    best = record["best"]
    if not bool(best["active_cube_lifted"]):
        return "never_lifted"
    if not bool(final["gripper_open"]):
        return "never_opened"
    if bool(final["structure_ok"]) and not bool(final["goal_position_ok"]):
        return "stacked_wrong_pose"
    if best_progress <= 0:
        return "wrong_direction"
    if final_progress < 0:
        return "approach_then_reverse"
    return "dropped_or_scattered"


def catastrophic_failure(task: str, category: str) -> int:
    """Frozen severe-tail definition used by the comprehensive claim gate."""

    if task == "gear":
        return int(category == "out_of_bounds")
    return int(category in {"dropped_or_scattered", "never_opened"})


def collect_final_rows() -> tuple[list[dict], dict]:
    rows = []
    sources = {}
    for task in TASKS:
        sources[task] = {}
        canonical = None
        for arm in ARMS:
            sources[task][arm] = {}
            for seed_label, planner_seed in SEEDS:
                payload = load_final(task, arm, seed_label)
                path = _final_path(task, arm, seed_label)
                sources[task][arm][seed_label] = {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
                case_ids = [record["case_id"] for record in payload["records"]]
                if canonical is None:
                    canonical = case_ids
                if case_ids != canonical:
                    raise ValueError(f"{task} final case order changed")
                if int(payload["planner_seed_base"]) != planner_seed:
                    raise ValueError(f"{task}/{arm}/{seed_label} planner seed changed")
                for position, record in enumerate(payload["records"]):
                    best, final = progress_values(task, record)
                    success_step = (
                        record.get("success_step")
                        if task == "gear"
                        else record.get("t_success")
                    )
                    cost = None
                    if task == "franka":
                        diagnostics = [
                            item for item in record.get("plan_diagnostics", [])
                            if not item.get("abstain")
                        ]
                        if diagnostics:
                            costs = diagnostics[0].get("selected_model_costs", [])
                            cost = float(costs[0]) if costs else None
                    category = failure_category(task, record)
                    rows.append(
                        {
                            "task": task,
                            "arm": arm,
                            "arm_display": DISPLAY[arm],
                            "planner_seed_label": seed_label,
                            "planner_seed_base": planner_seed,
                            "case_position": position,
                            "case_id": record["case_id"],
                            "success": int(bool(record["success"])),
                            "success_step": (
                                None
                                if success_step is None or int(success_step) < 0
                                else int(success_step)
                            ),
                            "best_progress": best,
                            "final_progress": final,
                            "selected_model_cost": cost,
                            "failure_category": category,
                            "catastrophic_failure": catastrophic_failure(task, category),
                        }
                    )
    return rows, sources


def _matrix(rows: list[dict], task: str, arm: str, field: str) -> np.ndarray:
    selected = [row for row in rows if row["task"] == task and row["arm"] == arm]
    by = {(row["case_position"], row["planner_seed_label"]): row for row in selected}
    return np.asarray(
        [
            [by[(case, seed)][field] for seed, _ in SEEDS]
            for case in range(30)
        ],
        dtype=np.float64,
    )


def final_statistics(rows: list[dict], sources: dict) -> dict:
    report = {
        "schema_version": "paper-new-tasks-final-statistics-v1",
        "generated_unix": time.time(),
        "cluster_definition": "frozen case; two planner seeds are repeated measures",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "tasks": {},
        "sources": sources,
    }
    primary_p = {}
    contrasts = {
        "L1_minus_L0": ("L0", "L1"),
        "C0_minus_L0": ("L0", "C0"),
        "C1_minus_C0": ("C0", "C1"),
        "C1_minus_L0": ("L0", "C1"),
    }
    for task in TASKS:
        task_result = {"arms": {}, "contrasts": {}}
        for arm in ARMS:
            success = _matrix(rows, task, arm, "success")
            arm_result = {"clustered_success": cluster_interval(success)}
            if task == "franka":
                selected_rows = [
                    row for row in rows
                    if row["task"] == task
                    and row["arm"] == arm
                    and row["selected_model_cost"] is not None
                ]
                costs = [row["selected_model_cost"] for row in selected_rows]
                progress = [row["best_progress"] for row in selected_rows]
                correlation = spearmanr(costs, progress)
                arm_result["cost_calibration"] = {
                    "count": len(costs),
                    "spearman_cost_vs_best_progress": float(correlation.statistic),
                    "two_sided_p": float(correlation.pvalue),
                    "expected_direction": "negative",
                }
            for seed_index, (seed_label, _) in enumerate(SEEDS):
                count = int(success[:, seed_index].sum())
                arm_result[seed_label] = {
                    "successes": count,
                    "count": 30,
                    "rate": count / 30,
                    "wilson95": list(wilson(count, 30)),
                }
            task_result["arms"][arm] = arm_result
        for name, (baseline, method) in contrasts.items():
            item = {"primary_contrast": name == "C1_minus_L0", "metrics": {}}
            for field in (
                "success", "best_progress", "final_progress", "catastrophic_failure"
            ):
                item["metrics"][field] = cluster_paired_effect(
                    _matrix(rows, task, baseline, field),
                    _matrix(rows, task, method, field),
                )
            for seed_index, (seed_label, _) in enumerate(SEEDS):
                baseline_values = _matrix(rows, task, baseline, "success")[:, seed_index]
                method_values = _matrix(rows, task, method, "success")[:, seed_index]
                item[seed_label] = {
                    "mcnemar": exact_paired_binomial(baseline_values, method_values),
                    "rescues": int(np.sum((baseline_values == 0) & (method_values == 1))),
                    "harms": int(np.sum((baseline_values == 1) & (method_values == 0))),
                    "ties": int(np.sum(baseline_values == method_values)),
                }
            task_result["contrasts"][name] = item
        p_value = task_result["contrasts"]["C1_minus_L0"]["primary"]["mcnemar"][
            "two_sided_exact_p"
        ]
        primary_p[task] = p_value
        report["tasks"][task] = task_result
    adjusted = holm_adjust(primary_p)
    for task in TASKS:
        report["tasks"][task]["contrasts"]["C1_minus_L0"]["primary"][
            "holm_adjusted_p_across_tasks"
        ] = adjusted[task]
    return report


def collect_confirmation_statistics() -> tuple[dict, list[dict]]:
    """Summarize the frozen conditional fresh-training-seed replication.

    This is intentionally kept separate from the preregistered main contrast:
    the extra training seeds are run only after a positive descriptive trigger.
    """

    audit_path = OUTPUT / "audit/confirmation_artifact_integrity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "paper-confirmation-artifact-integrity-v1":
        raise ValueError("confirmation audit schema drifted")
    result = {
        "schema_version": "paper-confirmation-statistics-v1",
        "conditional_on_positive_main_trigger": True,
        "inference_note": (
            "fresh-seed results are conditional replication evidence and are not "
            "pooled into the preregistered main-test p-value"
        ),
        "source": str(audit_path.resolve()),
        "source_sha256": sha256(audit_path),
        "tasks": {},
    }
    rows = []
    for task in TASKS:
        audited = audit["tasks"][task]
        trigger = audited["trigger"]
        task_result = {"trigger": trigger, "training_seeds": {}, "clustered_fresh_seed_effect": None}
        if not trigger["triggered"]:
            result["tasks"][task] = task_result
            continue
        seed_differences = []
        canonical_ids = None
        for seed_key, arms in sorted(audited["runs"].items(), key=lambda item: int(item[0])):
            seed = int(seed_key)
            values = {}
            for arm in ("L0", "C1"):
                run = arms[arm]
                path = Path(run["evaluation"]["record"])
                payload = committed_json(
                    path, schema="paper-h5-checkpoint-aggregate-v1", count=30
                )
                if payload is None:
                    raise ValueError(f"confirmation evaluation is not committed: {path}")
                case_ids = [str(record["case_id"]) for record in payload["records"]]
                if canonical_ids is None:
                    canonical_ids = case_ids
                elif case_ids != canonical_ids:
                    raise ValueError(f"confirmation case order drifted: {task}/{seed}/{arm}")
                values[arm] = np.asarray(
                    [int(bool(record["success"])) for record in payload["records"]],
                    dtype=np.float64,
                )
                selection = run["selection"]
                row = {
                    "task": task,
                    "training_seed": seed,
                    "arm": arm,
                    "successes": int(values[arm].sum()),
                    "count": 30,
                    "rate": float(values[arm].mean()),
                    "stopped_epoch": int(run["training_epochs"]),
                    "selected_epoch": int(selection["best_epoch"]),
                    "selected_validation_ema": float(selection["best_validation_ema"]),
                    "checkpoint_sha256": selection["checkpoint_sha256"],
                    "source": str(path.resolve()),
                    "source_sha256": sha256(path),
                }
                rows.append(row)
            delta = values["C1"] - values["L0"]
            seed_differences.append(delta)
            paired = exact_paired_binomial(values["L0"], values["C1"])
            task_result["training_seeds"][seed_key] = {
                "L0_successes": int(values["L0"].sum()),
                "C1_successes": int(values["C1"].sum()),
                "delta_successes": int(delta.sum()),
                "delta_rate": float(delta.mean()),
                "rescues": int(np.sum(delta == 1)),
                "harms": int(np.sum(delta == -1)),
                "mcnemar": paired,
            }
        # [case, fresh training seed] keeps both trained models for a frozen
        # case inside the same bootstrap cluster.
        matrix = np.stack(seed_differences, axis=1)
        task_result["clustered_fresh_seed_effect"] = cluster_interval(
            matrix, seed=BOOTSTRAP_SEED + 101
        )
        task_result["all_fresh_seeds_positive"] = all(
            item["delta_rate"] > 0 for item in task_result["training_seeds"].values()
        )
        result["tasks"][task] = task_result
    return result, rows


def collect_gif_summary() -> dict:
    audit_path = OUTPUT / "audit/gif_artifact_integrity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "paper-gif-artifact-integrity-v1" or audit.get("status") != "complete":
        raise ValueError("representative GIF audit is not complete")
    return {
        "source": str(audit_path.resolve()),
        "source_sha256": sha256(audit_path),
        "tasks": audit["tasks"],
    }


def comprehensive_claim_gate(statistics: dict, confirmation: dict) -> dict:
    """Evaluate every pre-registered condition for the strongest wording."""

    audit_paths = {
        "formal": OUTPUT / "audit/formal_artifact_integrity.json",
        "supplementary": OUTPUT / "audit/supplementary_artifact_integrity.json",
        "confirmation": OUTPUT / "audit/confirmation_artifact_integrity.json",
        "gifs": OUTPUT / "audit/gif_artifact_integrity.json",
    }
    audit_schemas = {
        "formal": "paper-formal-artifact-integrity-v1",
        "supplementary": "paper-supplementary-artifact-integrity-v1",
        "confirmation": "paper-confirmation-artifact-integrity-v1",
        "gifs": "paper-gif-artifact-integrity-v1",
    }
    protocol_checks = {}
    audit_sources = {}
    audit_payloads = {}
    for name, path in audit_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit_payloads[name] = payload
        valid = payload.get("schema_version") == audit_schemas[name]
        if name == "formal":
            valid = valid and payload.get("require_complete") is True
        if name == "gifs":
            valid = valid and payload.get("status") == "complete"
        protocol_checks[name] = bool(valid)
        audit_sources[name] = {"path": str(path.resolve()), "sha256": sha256(path)}

    formal = audit_payloads["formal"]
    lock_path = Path(formal["manifest"])
    if sha256(lock_path) != formal["manifest_file_sha256"]:
        raise ValueError("formal manifest changed after artifact audit")
    fairness = frozen_factorial_fairness_gate(
        json.loads(lock_path.read_text(encoding="utf-8"))
    )
    protocol_checks["frozen_factorial_fairness"] = bool(fairness["passes"])

    result = {
        "definition": "docs master plan section 2.1, evaluated without post-hoc substitutions",
        "task_results": {},
        "protocol_integrity": {
            "checks": protocol_checks,
            "all_pass": all(protocol_checks.values()),
            "sources": audit_sources,
            "factorial_fairness": fairness,
        },
    }
    for task in TASKS:
        contrast = statistics["tasks"][task]["contrasts"]["C1_minus_L0"]
        success = contrast["metrics"]["success"]
        best = contrast["metrics"]["best_progress"]
        final = contrast["metrics"]["final_progress"]
        tail = contrast["metrics"]["catastrophic_failure"]

        correct = load_final(task, "C1", "primary")
        wrong_path = OUTPUT / task / "c1/wrong_goal_test/eval.json"
        wrong = committed_json(
            wrong_path, schema="paper-h5-goal-ablation-aggregate-v1", count=30
        )
        if wrong is None:
            raise ValueError(f"missing C1 wrong-goal result: {task}")
        correct_by = _record_map_for_claim(correct["records"])
        wrong_by = _record_map_for_claim(wrong["records"])
        if list(correct_by) != list(wrong_by):
            raise ValueError(f"wrong-goal case order changed: {task}")
        correct_values = np.asarray(
            [int(bool(record["success"])) for record in correct_by.values()], dtype=float
        )
        wrong_values = np.asarray(
            [int(bool(record["success"])) for record in wrong_by.values()], dtype=float
        )
        goal_effect = cluster_paired_effect(
            wrong_values[:, None], correct_values[:, None], seed=BOOTSTRAP_SEED + 211
        )
        goal_effect["mcnemar"] = exact_paired_binomial(wrong_values, correct_values)

        action_path = OUTPUT / "audit/action_contract" / task / "c1/first_success_h5.json"
        action_count = 44 if task == "gear" else 100
        action = committed_json(
            action_path, schema="lewm-action-contract-audit-v6", count=action_count
        )
        if action is None:
            raise ValueError(f"missing C1 terminal action audit: {task}")
        action_gate = bool(action["strict_action_gate"]["closed_loop_eligible"])

        branch_path = OUTPUT / "action_branching" / task / "eval.json"
        branch = committed_json(
            branch_path,
            schema="paper-action-branching-aggregate-v1",
            count=(8 if task == "gear" else 9) * 30,
        )
        if branch is None:
            raise ValueError(f"missing same-init action branching result: {task}")
        branch_gate = action_branch_causality_gate(task, branch["records"])

        planner_deltas = {
            seed: int(contrast[seed]["rescues"] - contrast[seed]["harms"])
            for seed, _ in SEEDS
        }
        fresh = confirmation["tasks"][task]
        confirmation_direction = bool(
            fresh["trigger"]["triggered"]
            and fresh.get("all_fresh_seeds_positive", False)
        )
        checks = {
            "success_ci_strictly_positive": bool(
                success["mean"] > 0 and success["mean_ci95"][0] > 0
            ),
            "best_and_final_progress_nonnegative_with_one_positive_ci": bool(
                best["mean"] >= 0
                and final["mean"] >= 0
                and (best["mean_ci95"][0] > 0 or final["mean_ci95"][0] > 0)
            ),
            "catastrophic_failure_not_increased": bool(tail["mean"] <= 0),
            "correct_goal_clearly_better": bool(
                goal_effect["mean"] > 0 and goal_effect["mean_ci95"][0] > 0
            ),
            "strict_action_causality_gate": action_gate,
            "simulator_action_branch_causality": bool(branch_gate["passes"]),
            "planner_seed_direction_never_reverses": all(
                delta >= 0 for delta in planner_deltas.values()
            ),
            "fresh_training_seed_direction_confirmed": confirmation_direction,
            "frozen_factorial_fairness": bool(fairness["tasks"][task]["passes"]),
        }
        result["task_results"][task] = {
            "supports_comprehensive_wording": all(checks.values()),
            "checks": checks,
            "success_effect": success,
            "best_progress_effect": best,
            "final_progress_effect": final,
            "catastrophic_failure_effect": tail,
            "goal_effect_correct_minus_wrong": goal_effect,
            "planner_seed_success_deltas": planner_deltas,
            "action_gate": {
                "closed_loop_eligible": action_gate,
                "checks": action["strict_action_gate"]["checks"],
                "source": str(action_path.resolve()),
                "source_sha256": sha256(action_path),
            },
            "simulator_action_branch_gate": {
                **branch_gate,
                "source": str(branch_path.resolve()),
                "source_sha256": sha256(branch_path),
            },
        }
    result["supports_across_task_comprehensive_wording"] = bool(
        result["protocol_integrity"]["all_pass"]
        and all(
            item["supports_comprehensive_wording"]
            for item in result["task_results"].values()
        )
    )
    return result


def _record_map_for_claim(records: list[dict]) -> dict[str, dict]:
    by_id = {str(record["case_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("claim input contains duplicate case IDs")
    return by_id


def collect_supplementary_rows() -> tuple[list[dict], dict]:
    rows = []
    sources = {}
    for task in TASKS:
        sources[task] = {}
        for policy in (("zero", "expert") if task == "gear" else ("zero", "drop", "expert")):
            path = OUTPUT / "baselines" / task / policy / "eval.json"
            payload = committed_json(path, schema="paper-h5-baseline-aggregate-v1", count=30)
            if payload is None:
                raise ValueError(f"missing baseline {path}")
            rows.append({"task": task, "family": "baseline", "arm": "", "condition": policy,
                         "successes": int(payload["successes"]), "count": 30,
                         "rate": float(payload["success_rate"])})
            sources[str(path.resolve())] = sha256(path)
        for arm in ARMS:
            correct = load_final(task, arm, "primary")
            rows.append({"task": task, "family": "goal", "arm": arm, "condition": "correct_goal",
                         "successes": int(correct["successes"]), "count": 30,
                         "rate": float(correct["success_rate"])})
            wrong = OUTPUT / task / arm.lower() / "wrong_goal_test/eval.json"
            wp = committed_json(wrong, schema="paper-h5-goal-ablation-aggregate-v1", count=30)
            if wp is None:
                raise ValueError(f"missing wrong-goal result {wrong}")
            rows.append({"task": task, "family": "goal", "arm": arm, "condition": "wrong_goal",
                         "successes": int(wp["successes"]), "count": 30,
                         "rate": float(wp["success_rate"])})
            mode = "stock_cem" if task == "gear" else "behavior_retrieval"
            ablation = OUTPUT / task / arm.lower() / "inference_ablation" / mode / "eval.json"
            ap = committed_json(ablation, schema="paper-inference-ablation-aggregate-v1", count=30)
            if ap is None:
                raise ValueError(f"missing inference ablation {ablation}")
            rows.append({"task": task, "family": "inference", "arm": arm,
                         "condition": mode, "successes": int(ap["successes"]),
                         "count": 30, "rate": float(ap["success_rate"])})
            rows.append({"task": task, "family": "inference", "arm": arm,
                         "condition": "task_aware", "successes": int(correct["successes"]),
                         "count": 30, "rate": float(correct["success_rate"])})
            sources[str(wrong.resolve())] = sha256(wrong)
            sources[str(ablation.resolve())] = sha256(ablation)
        branch = OUTPUT / "action_branching" / task / "eval.json"
        bp = committed_json(
            branch,
            schema="paper-action-branching-aggregate-v1",
            count=(8 if task == "gear" else 9) * 30,
        )
        if bp is None:
            raise ValueError(f"missing action branching {branch}")
        for component in bp["components"]:
            rows.append({"task": task, "family": "action_branch", "arm": "",
                         "condition": component["variant"],
                         "successes": int(component["successes"]), "count": 30,
                         "rate": int(component["successes"]) / 30})
        sources[str(branch.resolve())] = sha256(branch)
    return rows, sources


def collect_action_audits() -> list[dict]:
    rows = []
    for task in TASKS:
        count = 44 if task == "gear" else 100
        for arm in ARMS:
            for clip in ("random", "first_success_h5"):
                path = OUTPUT / "audit/action_contract" / task / arm.lower() / f"{clip}.json"
                payload = committed_json(path, schema="lewm-action-contract-audit-v6",
                                         count=192 if clip == "random" else count)
                if payload is None:
                    raise ValueError(f"missing action audit {path}")
                for negative, value in payload["terminal_win_fraction"].items():
                    rows.append({"task": task, "arm": arm, "clip": clip,
                                 "negative": negative, "terminal_win_fraction": float(value),
                                 "closed_loop_eligible": int(payload["strict_action_gate"]["closed_loop_eligible"]),
                                 "source": str(path.resolve()), "source_sha256": sha256(path)})
                for negative, value in payload["component_ablation_terminal_win_fraction"].items():
                    rows.append({"task": task, "arm": arm, "clip": clip,
                                 "negative": negative, "terminal_win_fraction": float(value),
                                 "closed_loop_eligible": int(payload["strict_action_gate"]["closed_loop_eligible"]),
                                 "source": str(path.resolve()), "source_sha256": sha256(path)})
    return rows


def collect_residual_audits() -> list[dict]:
    rows = []
    for task in TASKS:
        count = 44 if task == "gear" else 100
        for arm in ("L1", "C1"):
            for clip in ("random", "first_success_h5"):
                path = OUTPUT / "audit/residual_dynamics" / task / arm.lower() / f"{clip}.json"
                payload = committed_json(
                    path,
                    schema="gear-residual-dynamics-audit-v1",
                    count=192 if clip == "random" else count,
                )
                if payload is None:
                    raise ValueError(f"missing residual audit {path}")
                cumulative = payload["free_cumulative_effect_vs_true_from_initial"]
                ratio = cumulative["norm_ratio"]["aggregate_rms_ratio"]
                for horizon, (cosine, rms_ratio, mse) in enumerate(
                    zip(cumulative["cosine_mean"], ratio, cumulative["mse"]), start=1
                ):
                    rows.append(
                        {
                            "task": task,
                            "arm": arm,
                            "clip": clip,
                            "horizon": horizon,
                            "cumulative_cosine": float(cosine),
                            "cumulative_rms_ratio": float(rms_ratio),
                            "cumulative_mse": float(mse),
                            "source": str(path.resolve()),
                            "source_sha256": sha256(path),
                        }
                    )
    return rows


def loss_h5_correlations() -> list[dict]:
    rows = []
    for task in TASKS:
        for arm in ARMS:
            training = {
                int(item["epoch"]): float(item["loss_control_decision"]["validation_ema"])
                for item in load_training(task, arm)
            }
            h5 = {
                int(item["epoch"]): float(item["success_rate"])
                for item in load_h5(task, arm)
            }
            common = sorted(set(training) & set(h5))
            x = np.asarray([training[epoch] for epoch in common])
            y = np.asarray([h5[epoch] for epoch in common])
            current = spearmanr(x, y) if len(common) > 2 else None
            lag_epochs = [epoch for epoch in common if epoch + 1 in h5]
            lag_x = np.asarray([training[epoch] for epoch in lag_epochs])
            lag_y = np.asarray([h5[epoch + 1] for epoch in lag_epochs])
            lag = spearmanr(lag_x, lag_y) if len(lag_epochs) > 2 else None
            rows.append(
                {
                    "task": task,
                    "arm": arm,
                    "paired_epochs": len(common),
                    "spearman_same_epoch": None if current is None else float(current.statistic),
                    "same_epoch_p": None if current is None else float(current.pvalue),
                    "lag1_pairs": len(lag_epochs),
                    "spearman_loss_e_vs_h5_eplus1": None if lag is None else float(lag.statistic),
                    "lag1_p": None if lag is None else float(lag.pvalue),
                }
            )
    return rows


def old_gear_table() -> list[dict]:
    audit_path = (
        OUTPUT / "comparisons/historical_gear_planner_seed_repeat_audit.json"
    )
    summary = json.loads(audit_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "gear_planner_seed_repeat_summary_v1":
        raise ValueError("historical Gear repeat audit schema drifted")
    if summary.get("interpretation", {}).get("planner_attempts_per_case") != 2:
        raise ValueError("historical Gear repeats lost their clustered interpretation")
    rows = []
    for horizon in (3, 5, 10, 15):
        models = summary["horizons"][str(horizon)]["models"]
        for model in ("original", "four_constraint_matched", "no_rank", "champion"):
            item = models[model]
            if int(item["case_count"]) != 10 or int(item["attempt_count"]) != 20:
                raise ValueError(f"historical Gear denominator drifted: H{horizon}/{model}")
            for source in item["inputs"]:
                path = Path(source["path"])
                if not path.is_file() or sha256(path) != source["sha256"]:
                    raise ValueError(f"historical Gear source changed: {path}")
            first, second = map(int, item["batch_successes"])
            total = int(item["cumulative_successes"])
            if first + second != total:
                raise ValueError(f"historical Gear success sum drifted: H{horizon}/{model}")
            rows.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "seed42_successes": first,
                    "seed10042_successes": second,
                    "cumulative_successes": total,
                    "attempts": 20,
                    "rate": total / 20,
                    "primary_source": item["inputs"][0]["path"],
                    "primary_sha256": item["inputs"][0]["sha256"],
                    "secondary_source": item["inputs"][1]["path"],
                    "secondary_sha256": item["inputs"][1]["sha256"],
                    "repeat_audit": str(audit_path.resolve()),
                    "repeat_audit_sha256": sha256(audit_path),
                }
            )
    return rows


def conditional_horizon_table(statistics: dict) -> tuple[dict, list[dict]]:
    """Load the frozen conditional Gear extension plus the matched H5 anchor."""

    audit_path = OUTPUT / "audit/conditional_horizon_artifact_integrity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version")
        != "paper-conditional-horizon-artifact-integrity-v1"
        or audit.get("status") != "complete"
        or set(audit.get("tasks", {})) != set(TASKS)
    ):
        raise ValueError("conditional-horizon integrity audit is incomplete")
    gear_triggered = bool(audit["tasks"]["gear"]["trigger"]["triggered"])
    rows = []
    for planner_label, planner_seed in SEEDS:
        for arm in ARMS:
            path = OUTPUT / "gear" / arm.lower() / "final_test" / planner_label / "eval.json"
            payload = committed_json(
                path, schema="paper-h5-checkpoint-aggregate-v1", count=30
            )
            if payload is None:
                raise ValueError(f"missing H5 anchor for conditional horizon table: {path}")
            rows.append(
                {
                    "task": "gear",
                    "horizon": 5,
                    "arm": arm,
                    "planner_seed_label": planner_label,
                    "planner_seed_base": planner_seed,
                    "successes": int(payload["successes"]),
                    "case_count": 30,
                    "rate": float(payload["success_rate"]),
                    "conditional_triggered": gear_triggered,
                    "registered_extension": False,
                    "source": str(path.resolve()),
                    "source_sha256": sha256(path),
                }
            )
    if gear_triggered:
        results = audit["tasks"]["gear"]["results"]
        for horizon in audit["tasks"]["gear"]["registered_additional_horizons"]:
            for planner_label, planner_seed in SEEDS:
                for arm in ARMS:
                    item = results[str(horizon)][planner_label][arm]
                    path = Path(item["result"])
                    payload = committed_json(
                        path,
                        schema="paper-gear-conditional-horizon-aggregate-v1",
                        count=10,
                    )
                    if payload is None or sha256(path) != item["result_sha256"]:
                        raise ValueError(f"conditional Gear source changed: {path}")
                    rows.append(
                        {
                            "task": "gear",
                            "horizon": horizon,
                            "arm": arm,
                            "planner_seed_label": planner_label,
                            "planner_seed_base": planner_seed,
                            "successes": int(payload["successes"]),
                            "case_count": 10,
                            "rate": float(payload["success_rate"]),
                            "conditional_triggered": True,
                            "registered_extension": True,
                            "source": str(path.resolve()),
                            "source_sha256": sha256(path),
                        }
                    )
    elif audit["tasks"]["gear"].get("results") != {}:
        raise ValueError("non-triggered Gear extension contains results")
    return audit, rows


def training_selection_table() -> list[dict]:
    """Expose the loss-only stop/selection chain in a paper-readable table."""

    rows = []
    for task in TASKS:
        for arm in ARMS:
            training = load_training(task, arm)
            h5 = load_h5(task, arm)
            selection_path = OUTPUT / task / arm.lower() / "selection.json"
            selection = committed_json(
                selection_path,
                schema="paper-loss-selected-checkpoint-v1",
                count=1,
            )
            if not training or selection is None:
                raise ValueError(f"missing completed training selection: {task}/{arm}")
            train_by_epoch = {int(item["epoch"]): item for item in training}
            h5_by_epoch = {int(item["epoch"]): item for item in h5}
            best_epoch = int(selection["best_epoch"])
            final = training[-1]
            chain_verified = (
                set(train_by_epoch) == set(h5_by_epoch)
                and all(
                    train_by_epoch[epoch]["checkpoint_sha256"]
                    == h5_by_epoch[epoch]["checkpoint_sha256"]
                    for epoch in train_by_epoch
                )
                and best_epoch in train_by_epoch
                and train_by_epoch[best_epoch]["checkpoint_sha256"]
                == selection["checkpoint_sha256"]
                and selection["h5_used_for_selection"] is False
                and selection["selection_authority"]
                == "validation-loss composite EMA only"
            )
            if not chain_verified:
                raise ValueError(f"training/selection checkpoint chain failed: {task}/{arm}")
            decision = final["loss_control_decision"]
            rows.append(
                {
                    "task": task,
                    "arm": arm,
                    "stopped_epoch": int(selection["final_epoch"]),
                    "stop_reason": selection["stop_reason"],
                    "selected_epoch": best_epoch,
                    "best_validation_ema": float(selection["best_validation_ema"]),
                    "final_validation_ema": float(decision["validation_ema"]),
                    "final_train_composite": float(decision["train_factor"]),
                    "final_generalization_gap": float(decision["generalization_gap"]),
                    "h5_dev30_epoch_count": len(h5),
                    "h5_used_for_selection": False,
                    "checkpoint_chain_verified": True,
                    "checkpoint_sha256": selection["checkpoint_sha256"],
                    "selection_source": str(selection_path.resolve()),
                    "selection_source_sha256": sha256(selection_path),
                }
            )
    return rows


def plot_main_success(statistics: dict, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    for axis, task in zip(axes, TASKS):
        for index, arm in enumerate(ARMS):
            item = statistics["tasks"][task]["arms"][arm]["primary"]
            low, high = item["wilson95"]
            axis.bar(index, item["rate"], color=COLORS[arm], width=0.68)
            axis.errorbar(index, item["rate"],
                          yerr=[[item["rate"] - low], [high - item["rate"]]],
                          fmt="none", color="black", capsize=3)
            axis.text(index, item["rate"] + 0.035, f"{item['successes']}/30",
                      ha="center", va="bottom", fontsize=9)
        axis.set_title(task.title())
        axis.set_xticks(range(4), ARMS)
        axis.set_xlabel("Model arm")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Strict H5-test success rate (seed 42)")
    axes[0].set_ylim(0, 1.08)
    figure.suptitle("Main A — Frozen H5-test30 strict success")
    _save(figure, destination)


def plot_conditional_gear_horizons(
    rows: list[dict], audit: dict, destination: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    triggered = bool(audit["tasks"]["gear"]["trigger"]["triggered"])
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        horizons = sorted({int(row["horizon"]) for row in arm_rows})
        rates = []
        labels = []
        for horizon in horizons:
            selected = [row for row in arm_rows if int(row["horizon"]) == horizon]
            successes = sum(int(row["successes"]) for row in selected)
            attempts = sum(int(row["case_count"]) for row in selected)
            rates.append(successes / attempts)
            labels.append(f"{successes}/{attempts}")
        axis.plot(
            horizons,
            rates,
            marker="o",
            linewidth=2,
            color=COLORS[arm],
            label=f"{arm} — {DISPLAY[arm]}",
        )
        for horizon, rate, label in zip(horizons, rates, labels):
            axis.annotate(
                label,
                (horizon, rate),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=COLORS[arm],
            )
    axis.set(
        xlabel="macro horizon / goal distance (5 raw steps each)",
        ylabel="strict success rate across two planner seeds",
        ylim=(-0.02, 1.08),
        xticks=sorted({int(row["horizon"]) for row in rows}),
        title=(
            "Conditional Gear distance extension — trigger active"
            if triggered
            else "Conditional Gear distance extension — H5 trigger inactive"
        ),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    axis.text(
        0.01,
        0.01,
        (
            "H5: 30 fixed cases × 2 seeds; H3/H10: 10 fixed cases × 2 seeds; "
            "H15 excluded (0 train-supported windows)"
            if triggered
            else "H5: 30 fixed cases × 2 seeds; distance extension not triggered"
        ),
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    _save(figure, destination)


def plot_effect_forest(statistics: dict, destination: Path) -> None:
    metrics = (
        ("success", "Success Δ", 100.0),
        ("best_progress", "Best progress Δ", 1.0),
        ("final_progress", "Final progress Δ", 1.0),
        ("catastrophic_failure", "Catastrophic failure Δ", 100.0),
    )
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, (field, title, scale) in zip(axes, metrics):
        for y, task in enumerate(TASKS):
            item = statistics["tasks"][task]["contrasts"]["C1_minus_L0"]["metrics"][field]
            point = item["mean"] * scale
            low, high = np.asarray(item["mean_ci95"]) * scale
            axis.errorbar(point, y, xerr=[[point - low], [high - point]],
                          fmt="o", color=COLORS["C1"], capsize=4)
        axis.axvline(0, color="#777777", linewidth=1)
        axis.set_yticks(range(2), [task.title() for task in TASKS])
        axis.set_title(title)
        axis.set_xlabel(
            "percentage points"
            if field in {"success", "catastrophic_failure"}
            else "task progress units"
        )
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Main B — Champion+constraints minus LeWM (case-cluster bootstrap; lower tail Δ is better)"
    )
    _save(figure, destination)


def plot_main_learning(destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, task in enumerate(TASKS):
        for arm in ARMS:
            training = load_training(task, arm)
            h5 = load_h5(task, arm)
            if training:
                axes[0, column].plot(
                    [int(item["epoch"]) for item in training],
                    [float(item["loss_control_decision"]["validation_ema"]) for item in training],
                    color=COLORS[arm], label=arm,
                )
            if h5:
                axes[1, column].plot(
                    [int(item["epoch"]) for item in h5],
                    [float(item["success_rate"]) for item in h5],
                    color=COLORS[arm], marker=".", label=arm,
                )
        axes[0, column].set_title(f"{task.title()} loss-selected composite")
        axes[0, column].set_ylabel("3-epoch EMA (lower better)")
        axes[1, column].set_title(f"{task.title()} H5-dev30 observation")
        axes[1, column].set_ylabel("strict success rate")
        axes[1, column].set_ylim(0, 1)
        axes[1, column].set_xlabel("epoch")
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=2)
    figure.suptitle("Main C — Training controller and non-selecting control observations")
    _save(figure, destination)


def plot_success_ecdf(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for axis, task in zip(axes, TASKS):
        for arm in ARMS:
            chosen = [row for row in rows if row["task"] == task and row["arm"] == arm]
            times = np.asarray([51 if row["success_step"] is None else row["success_step"] + 1 for row in chosen])
            x = np.arange(0, 51)
            y = np.asarray([np.mean(times <= value) for value in x])
            axis.step(x, y, where="post", color=COLORS[arm], label=arm)
        axis.set_title(task.title())
        axis.set_xlabel("control steps")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("cumulative strict-success fraction")
    axes[0].legend(ncol=2)
    figure.suptitle("First strict-success step (failures remain censored after step 50)")
    _save(figure, destination)


def plot_progress_pair(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
    for row_index, task in enumerate(TASKS):
        for column, field in enumerate(("best_progress", "final_progress")):
            baseline = _matrix(rows, task, "L0", field).mean(axis=1)
            champion = _matrix(rows, task, "C1", field).mean(axis=1)
            low = min(baseline.min(), champion.min())
            high = max(baseline.max(), champion.max())
            axes[row_index, column].plot([low, high], [low, high], color="#888888", linewidth=1)
            axes[row_index, column].scatter(baseline, champion, color=COLORS["C1"], alpha=0.75)
            axes[row_index, column].set(title=f"{task.title()} {field.replace('_', ' ')}",
                                        xlabel="L0 case mean", ylabel="C1 case mean")
            axes[row_index, column].grid(alpha=0.2)
    figure.suptitle("Paired case progress across two planner seeds")
    _save(figure, destination)


def plot_failure_taxonomy(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, task in zip(axes, TASKS):
        categories = sorted({row["failure_category"] for row in rows if row["task"] == task})
        bottoms = np.zeros(4)
        for category in categories:
            values = np.asarray([
                sum(row["failure_category"] == category for row in rows
                    if row["task"] == task and row["arm"] == arm)
                for arm in ARMS
            ])
            axis.bar(range(4), values, bottom=bottoms, label=category)
            bottoms += values
        axis.set(title=task.title(), xticks=range(4), xticklabels=ARMS,
                 xlabel="model arm", ylabel="episodes across two planner seeds")
        axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    figure.suptitle("Deterministic diagnostic failure taxonomy (all outcomes included)")
    _save(figure, destination)


def plot_rescue_harm(statistics: dict, destination: Path) -> None:
    labels = [f"{task}\n{seed}" for task in TASKS for seed, _ in SEEDS]
    matrix = []
    for task in TASKS:
        contrast = statistics["tasks"][task]["contrasts"]["C1_minus_L0"]
        for seed, _ in SEEDS:
            matrix.append([contrast[seed]["rescues"], contrast[seed]["harms"], contrast[seed]["ties"]])
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues", aspect="auto")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            axis.text(x, y, str(values[y, x]), ha="center", va="center")
    axis.set(xticks=range(3), xticklabels=("C1 rescue", "C1 harm", "tie"),
             yticks=range(4), yticklabels=labels,
             title="C1 versus L0 paired outcomes")
    figure.colorbar(image, ax=axis, label="case count")
    _save(figure, destination)


def plot_calibration(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    source_rows = [row for row in rows if row["task"] == "franka" and row["selected_model_cost"] is not None]
    for arm in ARMS:
        selected = [row for row in source_rows if row["arm"] == arm]
        cost = np.asarray([row["selected_model_cost"] for row in selected])
        progress = np.asarray([row["best_progress"] for row in selected])
        success = np.asarray([row["success"] for row in selected])
        axes[0].scatter(cost, progress, label=arm, color=COLORS[arm], alpha=0.65)
        if len(np.unique(cost)) > 1:
            order = np.argsort(cost)
            bins = np.array_split(order, 5)
            axes[1].plot([cost[index].mean() for index in bins],
                         [success[index].mean() for index in bins], marker="o",
                         color=COLORS[arm], label=arm)
    axes[0].set(title="Predicted terminal cost vs best progress",
                xlabel="selected model cost (lower preferred)", ylabel="best progress (m)")
    axes[1].set(title="Quintile calibration",
                xlabel="mean predicted cost", ylabel="strict success rate", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2)
    figure.suptitle("Franka planner exploitation calibration")
    _save(figure, destination)


def plot_supplementary(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, task in enumerate(TASKS):
        axis = axes[0, column]
        for index, arm in enumerate(ARMS):
            correct = next(row for row in rows if row["task"] == task and row["family"] == "goal"
                           and row["arm"] == arm and row["condition"] == "correct_goal")
            wrong = next(row for row in rows if row["task"] == task and row["family"] == "goal"
                         and row["arm"] == arm and row["condition"] == "wrong_goal")
            axis.bar(index - 0.18, correct["rate"], width=0.36, color=COLORS[arm], label="correct" if index == 0 else None)
            axis.bar(index + 0.18, wrong["rate"], width=0.36, color=COLORS[arm], alpha=0.35,
                     label="wrong" if index == 0 else None)
        axis.set(title=f"{task.title()} goal causality", xticks=range(4), xticklabels=ARMS,
                 ylabel="strict success rate", ylim=(0, 1))
        axis.legend()
        axis = axes[1, column]
        branches = [row for row in rows if row["task"] == task and row["family"] == "action_branch"]
        axis.barh(range(len(branches)), [row["rate"] for row in branches], color="#4c78a8")
        axis.set(yticks=range(len(branches)), yticklabels=[row["condition"] for row in branches],
                 xlim=(0, 1), xlabel="strict success rate", title=f"{task.title()} same-init action branches")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Goal and action causal controls")
    _save(figure, destination)


def plot_inference_and_baselines(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, task in zip(axes, TASKS):
        mode = "stock_cem" if task == "gear" else "behavior_retrieval"
        for index, arm in enumerate(ARMS):
            aware = next(
                row for row in rows
                if row["task"] == task and row["family"] == "inference"
                and row["arm"] == arm and row["condition"] == "task_aware"
            )
            ablated = next(
                row for row in rows
                if row["task"] == task and row["family"] == "inference"
                and row["arm"] == arm and row["condition"] == mode
            )
            axis.bar(index - 0.18, aware["rate"], width=0.36,
                     color=COLORS[arm], label="task-aware" if index == 0 else None)
            axis.bar(index + 0.18, ablated["rate"], width=0.36,
                     color=COLORS[arm], alpha=0.35,
                     label=mode if index == 0 else None)
        baselines = [
            row for row in rows
            if row["task"] == task and row["family"] == "baseline"
        ]
        for line_index, baseline in enumerate(baselines):
            axis.axhline(
                baseline["rate"],
                linestyle=("--", ":", "-.")[line_index % 3],
                linewidth=1,
                label=f"{baseline['condition']} baseline {baseline['successes']}/30",
            )
        axis.set(
            title=task.title(),
            xticks=range(4),
            xticklabels=ARMS,
            xlabel="model arm",
            ylabel="strict success rate",
            ylim=(0, 1),
        )
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Task-aware inference versus registered compatibility controls")
    _save(figure, destination)


def plot_action_contract(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, task in zip(axes, TASKS):
        selected = [row for row in rows if row["task"] == task and row["clip"] == "first_success_h5"]
        negatives = sorted({row["negative"] for row in selected})
        width = 0.18
        x = np.arange(len(negatives))
        for offset, arm in enumerate(ARMS):
            by = {row["negative"]: row["terminal_win_fraction"] for row in selected if row["arm"] == arm}
            axis.bar(x + (offset - 1.5) * width, [by[name] for name in negatives],
                     width=width, color=COLORS[arm], label=arm)
        axis.axhline(0.5, color="#555555", linewidth=1, linestyle="--")
        axis.set(title=task.title(), xticks=x, xticklabels=negatives, ylim=(0, 1),
                 ylabel="true-action terminal win fraction")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=2)
    figure.suptitle("Held-out first-success H5 action sensitivity")
    _save(figure, destination)


def plot_residual_dynamics(rows: list[dict], destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row_index, task in enumerate(TASKS):
        selected = [row for row in rows if row["task"] == task and row["clip"] == "first_success_h5"]
        for arm in ("L1", "C1"):
            values = [row for row in selected if row["arm"] == arm]
            axes[row_index, 0].plot(
                [row["horizon"] for row in values],
                [row["cumulative_cosine"] for row in values],
                marker="o", color=COLORS[arm], label=arm,
            )
            axes[row_index, 1].plot(
                [row["horizon"] for row in values],
                [row["cumulative_rms_ratio"] for row in values],
                marker="o", color=COLORS[arm], label=arm,
            )
        axes[row_index, 0].axhline(0, color="#777777", linewidth=1)
        axes[row_index, 1].axhline(1, color="#777777", linewidth=1)
        axes[row_index, 0].set(title=f"{task.title()} cumulative direction",
                               xlabel="macro horizon", ylabel="mean cosine")
        axes[row_index, 1].set(title=f"{task.title()} cumulative magnitude",
                               xlabel="macro horizon", ylabel="predicted / target RMS")
        for axis in axes[row_index]:
            axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle("Residual-transition dynamics on exact first-success H5 clips")
    _save(figure, destination)


def plot_loss_h5_scatter(destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, task in zip(axes, TASKS):
        for arm in ARMS:
            training = {int(item["epoch"]): item for item in load_training(task, arm)}
            h5 = {int(item["epoch"]): item for item in load_h5(task, arm)}
            epochs = sorted(set(training) & set(h5))
            axis.scatter(
                [float(training[epoch]["loss_control_decision"]["validation_ema"]) for epoch in epochs],
                [float(h5[epoch]["success_rate"]) for epoch in epochs],
                color=COLORS[arm], label=arm, alpha=0.7,
            )
        axis.set(title=task.title(), xlabel="validation composite EMA (lower better)",
                 ylabel="same-epoch H5-dev30 success rate", ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2)
    figure.suptitle("Loss-to-control association (diagnostic only; never selects checkpoints)")
    _save(figure, destination)


def plot_training_seed_confirmation(
    statistics: dict, confirmation: dict, destination: Path
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True
    )
    for axis, task in zip(axes, TASKS):
        fresh = confirmation["tasks"][task]
        labels = ["main\n3072"]
        l0 = [statistics["tasks"][task]["arms"]["L0"]["primary"]["rate"]]
        c1 = [statistics["tasks"][task]["arms"]["C1"]["primary"]["rate"]]
        for seed, item in sorted(
            fresh["training_seeds"].items(), key=lambda pair: int(pair[0])
        ):
            labels.append(f"confirm\n{seed}")
            l0.append(item["L0_successes"] / 30)
            c1.append(item["C1_successes"] / 30)
        x = np.arange(len(labels), dtype=float)
        width = 0.34
        axis.bar(x - width / 2, l0, width, color=COLORS["L0"], label="L0")
        axis.bar(x + width / 2, c1, width, color=COLORS["C1"], label="C1")
        for xpos, left, right in zip(x, l0, c1):
            axis.text(xpos - width / 2, left + 0.02, f"{round(left*30):.0f}", ha="center", fontsize=8)
            axis.text(xpos + width / 2, right + 0.02, f"{round(right*30):.0f}", ha="center", fontsize=8)
        axis.set(
            title=(
                f"{task.title()} — trigger {'on' if fresh['trigger']['triggered'] else 'off'}"
            ),
            xticks=x,
            xticklabels=labels,
            xlabel="training seed (planner seed 42)",
            ylim=(0, 1.08),
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("strict H5-test30 success rate")
    axes[0].legend(ncol=2)
    figure.suptitle(
        "Conditional fresh-training-seed confirmation (reported separately from main inference)"
    )
    _save(figure, destination)


def write_report(
    statistics: dict,
    training_rows: list[dict],
    confirmation: dict,
    conditional_audit: dict,
    conditional_rows: list[dict],
    gif_summary: dict,
    old_rows: list[dict],
    destination: Path,
) -> None:
    lines = [
        "# Gear and Franka Stack formal results",
        "",
        "All checkpoints below were selected only by the frozen validation-loss composite. H5-dev30 was observational and H5-test30 remained sealed until selection.",
        "",
        "## Main H5-test30",
        "",
        "| Task | Arm | Seed 42 | Seed 10042 | Clustered mean [95% CI] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        for arm in ARMS:
            item = statistics["tasks"][task]["arms"][arm]
            cluster = item["clustered_success"]
            lines.append(
                f"| {task.title()} | {arm} ({DISPLAY[arm]}) | "
                f"{item['primary']['successes']}/30 | {item['secondary']['successes']}/30 | "
                f"{100*cluster['mean']:.1f}% [{100*cluster['mean_ci95'][0]:.1f}, {100*cluster['mean_ci95'][1]:.1f}] |"
            )
    lines += [
        "",
        "## Loss-only stopping and checkpoint selection",
        "",
        "H5-dev30 is present for every completed epoch but is excluded from both "
        "stopping and checkpoint selection. The checkpoint-chain column requires "
        "every H5 epoch to match its training checkpoint and the selected SHA to "
        "match the minimum-validation-EMA epoch.",
        "",
        "| Task | Arm | Stop epoch/reason | Loss-best epoch | Best/final validation EMA | H5-dev epochs | Checkpoint chain |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in training_rows:
        lines.append(
            f"| {row['task'].title()} | {row['arm']} | "
            f"{row['stopped_epoch']} / {row['stop_reason']} | "
            f"{row['selected_epoch']} | {row['best_validation_ema']:.6g} / "
            f"{row['final_validation_ema']:.6g} | {row['h5_dev30_epoch_count']} | "
            f"{'pass' if row['checkpoint_chain_verified'] else 'fail'} |"
        )
    fairness = statistics["comprehensive_claim_gate"]["protocol_integrity"][
        "factorial_fairness"
    ]
    lines += [
        "",
        "## Factorial fairness and parameter accounting",
        "",
        "All four arms share one task-local data/order, optimizer-step, scheduler, and planner-candidate contract. Champion auxiliary parameters are used only during training and are excluded from the deployed planner; both counts are disclosed below.",
        "",
        "| Task | Arm | Deployable planner params | Training-only auxiliary params | Total trainable params | Steps/epoch | Scheduler max steps |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        item = fairness["tasks"][task]
        for arm in ARMS:
            lines.append(
                f"| {task.title()} | {arm} | "
                f"{item['deployable_planner_parameters'][arm]:,} | "
                f"{item['training_only_auxiliary_parameters'][arm]:,} | "
                f"{item['total_trainable_parameters'][arm]:,} | "
                f"{item['optimizer_steps_per_complete_epoch']:,} | "
                f"{item['scheduler_max_steps']:,} |"
            )
    lines += ["", "## Pre-registered C1 − L0 conclusion", ""]
    supported_all = True
    for task in TASKS:
        contrast = statistics["tasks"][task]["contrasts"]["C1_minus_L0"]
        success = contrast["metrics"]["success"]
        primary = contrast["primary"]
        supported = success["mean"] > 0 and success["mean_ci95"][0] > 0
        supported_all &= supported
        lines.append(
            f"- {task.title()}: clustered success effect {100*success['mean']:+.1f} pp "
            f"(95% CI {100*success['mean_ci95'][0]:+.1f} to {100*success['mean_ci95'][1]:+.1f}); "
            f"primary rescues/harms={primary['rescues']}/{primary['harms']}; exact McNemar "
            f"p={primary['mcnemar']['two_sided_exact_p']:.4g}, Holm-adjusted p={primary['holm_adjusted_p_across_tasks']:.4g}."
        )
    lines += [
        "",
        (
            "The frozen evidence supports a consistent C1 advantage over L0 on both tasks."
            if supported_all
            else "The frozen evidence does not establish an across-task C1 victory; the per-task effects and uncertainty above are the authoritative conclusion."
        ),
        "",
        "## Comprehensive-advantage claim gate",
        "",
        "The strongest wording requires every frozen criterion below; success rate alone is insufficient.",
        "",
        "| Task | Success CI | Progress | Failure tail | Goal causal | Offline action | Simulator action | Planner seeds | Training seeds | Fairness | Overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    claim = statistics["comprehensive_claim_gate"]
    check_order = (
        "success_ci_strictly_positive",
        "best_and_final_progress_nonnegative_with_one_positive_ci",
        "catastrophic_failure_not_increased",
        "correct_goal_clearly_better",
        "strict_action_causality_gate",
        "simulator_action_branch_causality",
        "planner_seed_direction_never_reverses",
        "fresh_training_seed_direction_confirmed",
        "frozen_factorial_fairness",
    )
    for task in TASKS:
        item = claim["task_results"][task]
        symbols = ["pass" if item["checks"][key] else "fail" for key in check_order]
        lines.append(
            f"| {task.title()} | " + " | ".join(symbols) + " | "
            + ("pass" if item["supports_comprehensive_wording"] else "fail") + " |"
        )
    lines += [
        "",
        (
            "All frozen gates pass on both tasks; the comprehensive-advantage wording is supported."
            if claim["supports_across_task_comprehensive_wording"]
            else "At least one frozen gate fails; the report must use narrower per-metric wording and must not claim comprehensive victory."
        ),
        "",
        "## Conditional fresh-training-seed confirmation",
        "",
        "These runs were triggered only when C1 exceeded L0 in the two-seed main result. They are reported as conditional replication and are not pooled into the preregistered p-value.",
        "",
        "| Task | Trigger | Training seed | L0 | C1 | C1−L0 | Rescues/harms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        item = confirmation["tasks"][task]
        if not item["trigger"]["triggered"]:
            lines.append(f"| {task.title()} | no | — | — | — | — | — |")
            continue
        for seed, seed_result in sorted(
            item["training_seeds"].items(), key=lambda pair: int(pair[0])
        ):
            lines.append(
                f"| {task.title()} | yes | {seed} | "
                f"{seed_result['L0_successes']}/30 | {seed_result['C1_successes']}/30 | "
                f"{100*seed_result['delta_rate']:+.1f} pp | "
                f"{seed_result['rescues']}/{seed_result['harms']} |"
            )
        clustered = item["clustered_fresh_seed_effect"]
        lines.append(
            f"| {task.title()} fresh-seed clustered summary | yes | case-clustered | — | — | "
            f"{100*clustered['mean']:+.1f} pp "
            f"[{100*clustered['mean_ci95'][0]:+.1f}, {100*clustered['mean_ci95'][1]:+.1f}] | — |"
        )
    gear_extension = conditional_audit["tasks"]["gear"]
    gear_triggered = bool(gear_extension["trigger"]["triggered"])
    lines += [
        "",
        "## Conditional current-checkpoint distance extension",
        "",
        (
            "Gear passed the frozen positive-H5 trigger, so the loss-selected current "
            "checkpoints were additionally evaluated on the train-supported H3/H10 "
            "cases. H5 uses 30 fixed cases per planner seed; every additional horizon "
            "uses 10 fixed cases per planner seed. H15 is not run because it needs "
            "75 consecutive real actions while the frozen train episodes contain at "
            "most 74 (zero supported windows)."
            if gear_triggered
            else "Gear did not pass the frozen positive-H5 trigger; no post-H5 distance "
            "extension was run. The H5 anchor remains reported below."
        ),
        "",
        "| Horizon | L0 | L1 | C0 | C1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for horizon in sorted({int(row["horizon"]) for row in conditional_rows}):
        cells = []
        for arm in ARMS:
            selected = [
                row
                for row in conditional_rows
                if int(row["horizon"]) == horizon and row["arm"] == arm
            ]
            successes = sum(int(row["successes"]) for row in selected)
            attempts = sum(int(row["case_count"]) for row in selected)
            cells.append(f"{successes}/{attempts}")
        lines.append(f"| H{horizon} | " + " | ".join(cells) + " |")
    franka_extension = conditional_audit["tasks"]["franka"]
    if franka_extension["trigger"]["triggered"]:
        boundary = franka_extension["support_boundary"]
        lines += [
            "",
            "Franka Stack remains H5-only under the matched frozen planner: "
            f"`{boundary['frozen_main_planner']}` is registered for "
            f"{boundary['frozen_horizon_macro']} macro steps / "
            f"{boundary['frozen_raw_plan_steps']} raw steps. A shorter projected-CEM "
            "run would change the proposal family, while longer extrapolation is "
            "forbidden; neither is presented as a matched distance result.",
        ]
    else:
        lines += [
            "",
            "Franka Stack did not pass the frozen positive-H5 trigger, so no "
            "conditional distance extension was authorized.",
        ]
    lines += [
        "",
        "## Deterministic rollout gallery",
        "",
        "Cases were selected by frozen outcome rules, never manually: lowest case ID within each paired outcome class, worst C1 progress for the catastrophic tail, and lowest case ID for wrong-goal. Replays were checked against the formal success labels.",
        "",
    ]
    for task in TASKS:
        manifest = Path(gif_summary["tasks"][task]["manifest"])
        selection = Path(gif_summary["tasks"][task]["selection"])
        lines.append(
            f"- {task.title()}: [selection]({selection}) · [render manifest]({manifest}) · "
            f"{gif_summary['tasks'][task]['gif_count']} audited GIFs"
        )
    lines += [
        "",
        "## Historical Gear 10-case × 2-planner-seed diagnostic",
        "",
        "These are repeated planner-seed measurements of the same ten cases and are not part of the new L0/L1/C0/C1 test30 matrix.",
        "",
        "| Horizon | Original | Four constraints | No ranking | Formal champion |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for horizon in (3, 5, 10, 15):
        by = {row["model"]: row for row in old_rows if row["horizon"] == horizon}
        lines.append(
            f"| H{horizon} | {by['original']['cumulative_successes']}/20 | "
            f"{by['four_constraint_matched']['cumulative_successes']}/20 | "
            f"{by['no_rank']['cumulative_successes']}/20 | "
            f"{by['champion']['cumulative_successes']}/20 |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Planner seeds are repeated measurements, not independent episodes; all combined uncertainty resamples cases as clusters.",
        "- Failure labels are deterministic diagnostics from recorded states, not ground-truth causal annotations.",
        "- Wrong-goal, inference, offline action ranking, and same-init action branches are reported as controls and do not alter checkpoint selection.",
        "- Claims must follow the confidence intervals and paired tests; a desired champion outcome is not itself evidence.",
        "",
        "## Evidence tiers",
        "",
        "- **Formal/confirmed:** frozen L0/L1/C0/C1 training, loss-only selections, dual-planner-seed H5-test30, frozen controls, and any trigger-authorized fresh training seeds.",
        "- **Diagnostic:** H5-dev30 learning curves, loss-to-control correlations, failure labels, GIF appearance, and the historical Gear 10-case × 2-seed table.",
        "- **Invalidated/excluded:** manifest v1/v2 runs and every Snellius result; none contribute to the tables or claim gate.",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_executive_summary(statistics: dict, destination: Path) -> None:
    claim = statistics["comprehensive_claim_gate"]
    verdict = (
        "SUPPORTED: every frozen comprehensive-advantage gate passed on both tasks."
        if claim["supports_across_task_comprehensive_wording"]
        else "NOT SUPPORTED AS A BLANKET CLAIM: use the per-metric results below because at least one frozen gate failed."
    )
    lines = [
        "# Gear + Franka Stack: executive summary",
        "",
        f"**Verdict — {verdict}**",
        "",
        "All eight main runs used fresh seed 3072, the same task-local data and optimizer schedule, at least 50 epochs, validation-loss-only checkpoint selection, and sealed H5-test30. H5-dev30 was observational only.",
        "",
        "| Task | L0 seeds 42/10042 | C1 seeds 42/10042 | Clustered C1−L0 success [95% CI] | Comprehensive gate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        task_stats = statistics["tasks"][task]
        l0 = task_stats["arms"]["L0"]
        c1 = task_stats["arms"]["C1"]
        effect = task_stats["contrasts"]["C1_minus_L0"]["metrics"]["success"]
        gate = claim["task_results"][task]["supports_comprehensive_wording"]
        lines.append(
            f"| {task.title()} | {l0['primary']['successes']}/30, {l0['secondary']['successes']}/30 | "
            f"{c1['primary']['successes']}/30, {c1['secondary']['successes']}/30 | "
            f"{100*effect['mean']:+.1f} pp [{100*effect['mean_ci95'][0]:+.1f}, {100*effect['mean_ci95'][1]:+.1f}] | "
            f"{'pass' if gate else 'fail'} |"
        )
    lines += [
        "",
        "## What is safe to claim",
        "",
    ]
    for task in TASKS:
        item = claim["task_results"][task]
        passed = [key for key, value in item["checks"].items() if value]
        failed = [key for key, value in item["checks"].items() if not value]
        lines.append(f"- **{task.title()}** — passed: {', '.join(passed) or 'none'}; failed: {', '.join(failed) or 'none'}.")
    lines += [
        "",
        "## Evidence and limits",
        "",
        "- Formal evidence: dual planner seeds, case-cluster bootstrap, exact McNemar, goal/action controls, and conditional fresh-training-seed confirmation.",
        "- Diagnostic evidence: per-epoch H5-dev30, loss correlations, failure taxonomy, representative GIFs, and historical Gear repeats.",
        "- Excluded evidence: invalidated v1/v2 artifacts and all Snellius experiments.",
        "- Planner seeds are repeated measurements of 30 cases, not 60 independent cases; extra training seeds are conditional on the frozen positive trigger.",
        "",
        f"Full report: {OUTPUT / 'reports/paper_new_tasks_results.md'}",
        f"Machine-readable statistics: {OUTPUT / 'comparisons/final_statistics.json'}",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, sources = collect_final_rows()
    statistics = final_statistics(rows, sources)
    confirmation, confirmation_rows = collect_confirmation_statistics()
    comprehensive = comprehensive_claim_gate(statistics, confirmation)
    fairness_rows = []
    for task in TASKS:
        item = comprehensive["protocol_integrity"]["factorial_fairness"]["tasks"][task]
        for arm in ARMS:
            fairness_rows.append(
                {
                    "task": task,
                    "arm": arm,
                    "deployable_planner_parameters": item[
                        "deployable_planner_parameters"
                    ][arm],
                    "training_only_auxiliary_parameters": item[
                        "training_only_auxiliary_parameters"
                    ][arm],
                    "total_trainable_parameters": item["total_trainable_parameters"][arm],
                    "optimizer_steps_per_complete_epoch": item[
                        "optimizer_steps_per_complete_epoch"
                    ],
                    "scheduler_max_steps": item["scheduler_max_steps"],
                    "fairness_gate_passed": int(item["passes"]),
                }
            )
    gif_summary = collect_gif_summary()
    supplementary, supplementary_sources = collect_supplementary_rows()
    action_audits = collect_action_audits()
    residual_audits = collect_residual_audits()
    correlation_rows = loss_h5_correlations()
    old_rows = old_gear_table()
    conditional_audit, conditional_rows = conditional_horizon_table(statistics)
    training_rows = training_selection_table()
    tables = OUTPUT / "tables"
    figures = OUTPUT / "figures"
    _write_csv(tables / "final_per_case.csv", rows)
    _write_csv(tables / "supplementary_success.csv", supplementary)
    _write_csv(tables / "action_contract_terminal_wins.csv", action_audits)
    _write_csv(tables / "residual_dynamics.csv", residual_audits)
    _write_csv(tables / "loss_h5_correlations.csv", correlation_rows)
    _write_csv(tables / "historical_gear_20_attempt.csv", old_rows)
    _write_csv(tables / "confirmation_training_seeds.csv", confirmation_rows)
    _write_csv(tables / "factorial_fairness.csv", fairness_rows)
    _write_csv(tables / "conditional_gear_horizons.csv", conditional_rows)
    _write_csv(tables / "training_selection_audit.csv", training_rows)
    statistics["supplementary_sources"] = supplementary_sources
    statistics["confirmation"] = confirmation
    statistics["conditional_horizons"] = conditional_audit
    statistics["training_selection"] = training_rows
    statistics["comprehensive_claim_gate"] = comprehensive
    statistics["representative_gifs"] = gif_summary
    stats_path = OUTPUT / "comparisons/final_statistics.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_main_success(statistics, figures / "main_a_h5_test_success")
    plot_effect_forest(statistics, figures / "main_b_c1_minus_l0_forest")
    plot_main_learning(figures / "main_c_learning_curves")
    plot_success_ecdf(rows, figures / "final_success_step_ecdf")
    plot_progress_pair(rows, figures / "final_paired_progress")
    plot_failure_taxonomy(rows, figures / "final_failure_taxonomy")
    plot_rescue_harm(statistics, figures / "final_rescues_harms")
    plot_calibration(rows, figures / "franka_cost_calibration")
    plot_supplementary(supplementary, figures / "supplementary_causal_controls")
    plot_inference_and_baselines(
        supplementary, figures / "supplementary_inference_baselines"
    )
    plot_action_contract(action_audits, figures / "supplementary_action_contract")
    plot_residual_dynamics(
        residual_audits, figures / "supplementary_residual_dynamics"
    )
    plot_loss_h5_scatter(figures / "supplementary_loss_h5_scatter")
    plot_training_seed_confirmation(
        statistics, confirmation, figures / "confirmation_training_seeds"
    )
    plot_conditional_gear_horizons(
        conditional_rows, conditional_audit, figures / "conditional_gear_horizons"
    )
    write_report(
        statistics,
        training_rows,
        confirmation,
        conditional_audit,
        conditional_rows,
        gif_summary,
        old_rows,
        OUTPUT / "reports/paper_new_tasks_results.md",
    )
    write_executive_summary(
        statistics, OUTPUT / "reports/paper_new_tasks_executive_summary.md"
    )

    artifact_paths = [
        stats_path,
        tables / "final_per_case.csv",
        tables / "supplementary_success.csv",
        tables / "action_contract_terminal_wins.csv",
        tables / "residual_dynamics.csv",
        tables / "loss_h5_correlations.csv",
        tables / "historical_gear_20_attempt.csv",
        tables / "confirmation_training_seeds.csv",
        tables / "factorial_fairness.csv",
        tables / "conditional_gear_horizons.csv",
        tables / "training_selection_audit.csv",
        OUTPUT / "reports/paper_new_tasks_results.md",
        OUTPUT / "reports/paper_new_tasks_executive_summary.md",
    ]
    for stem in (
        "main_a_h5_test_success",
        "main_b_c1_minus_l0_forest",
        "main_c_learning_curves",
        "final_success_step_ecdf",
        "final_paired_progress",
        "final_failure_taxonomy",
        "final_rescues_harms",
        "franka_cost_calibration",
        "supplementary_causal_controls",
        "supplementary_inference_baselines",
        "supplementary_action_contract",
        "supplementary_residual_dynamics",
        "supplementary_loss_h5_scatter",
        "confirmation_training_seeds",
        "conditional_gear_horizons",
    ):
        artifact_paths.extend((figures / f"{stem}.png", figures / f"{stem}.pdf"))
    manifest = {
        "schema_version": "paper-new-tasks-final-analysis-manifest-v1",
        "generated_unix": time.time(),
        **final_analysis_provenance(
            OUTPUT / "audit/formal_artifact_integrity.json", Path(__file__)
        ),
        "artifacts": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(artifact_paths))
        ],
    }
    manifest_path = OUTPUT / "audit/final_analysis_manifest.json"
    temporary = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps({"status": "complete", "statistics": str(stats_path),
                      "report": str(OUTPUT / 'reports/paper_new_tasks_results.md'),
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
