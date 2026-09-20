#!/usr/bin/env python3
"""Run the frozen final analysis under the authorized training-only amendment.

The numerical/statistical implementation remains the frozen
``analyze_paper_new_tasks_final`` module.  This adapter changes only audit
provenance, the permitted training/H5 checkpoint-chain shape, and prose that
describes the now-truncated observational H5-dev curves.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import scripts.analyze_paper_new_tasks_final as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/paper_new_tasks_v1"
FORMAL_AUDIT = OUTPUT / "audit/formal_artifact_integrity.json"
AMENDMENT = OUTPUT / "manifest/training_only_user_amendment.json"


def final_analysis_provenance(_formal_audit_path: Path, _generator_path: Path) -> dict:
    formal = json.loads(FORMAL_AUDIT.read_text(encoding="utf-8"))
    if (
        formal.get("schema_version")
        != "paper-training-only-artifact-integrity-v1"
        or formal.get("require_complete") is not True
    ):
        raise ValueError("final analysis requires the complete training-only audit")
    manifest = Path(formal["manifest"]).expanduser().resolve()
    if base.sha256(manifest) != formal["manifest_file_sha256"]:
        raise ValueError("experiment manifest changed after training-only audit")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("manifest_sha256") != formal["manifest_sha256"]:
        raise ValueError("training-only audit and manifest identities differ")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment.get("frozen_manifest_sha256") != formal["manifest_sha256"]:
        raise ValueError("training-only amendment references another manifest")
    return {
        "command": [sys.executable, str(Path(base.__file__).resolve())],
        "generator": str(Path(base.__file__).resolve()),
        "generator_sha256": base.sha256(Path(base.__file__).resolve()),
        "actual_command": [sys.executable, str(Path(__file__).resolve())],
        "training_only_adapter": str(Path(__file__).resolve()),
        "training_only_adapter_sha256": base.sha256(Path(__file__).resolve()),
        "formal_artifact_audit": str(FORMAL_AUDIT.resolve()),
        "formal_artifact_audit_sha256": base.sha256(FORMAL_AUDIT),
        "training_only_amendment": str(AMENDMENT.resolve()),
        "training_only_amendment_sha256": base.sha256(AMENDMENT),
        "experiment_manifest": str(manifest),
        "experiment_manifest_sha256": base.sha256(manifest),
        "experiment_manifest_internal_sha256": manifest_payload["manifest_sha256"],
    }


def training_selection_table() -> list[dict]:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    rows = []
    for task in base.TASKS:
        cutoff = int(
            amendment["per_epoch_h5_dev30"][task]["preserved_l0_epochs"][1]
        )
        for arm in base.ARMS:
            training = base.load_training(task, arm)
            h5 = base.load_h5(task, arm)
            selection_path = OUTPUT / task / arm.lower() / "selection.json"
            selection = base.committed_json(
                selection_path,
                schema="paper-loss-selected-checkpoint-v1",
                count=1,
            )
            if not training or selection is None:
                raise ValueError(f"missing completed training selection: {task}/{arm}")
            train_by_epoch = {int(item["epoch"]): item for item in training}
            h5_by_epoch = {int(item["epoch"]): item for item in h5}
            expected_h5 = set(range(1, cutoff + 1)) if arm == "L0" else set()
            best_epoch = int(selection["best_epoch"])
            chain_verified = (
                set(h5_by_epoch) == expected_h5
                and set(h5_by_epoch).issubset(train_by_epoch)
                and all(
                    train_by_epoch[epoch]["checkpoint_sha256"]
                    == h5_by_epoch[epoch]["checkpoint_sha256"]
                    for epoch in h5_by_epoch
                )
                and best_epoch in train_by_epoch
                and train_by_epoch[best_epoch]["checkpoint_sha256"]
                == selection["checkpoint_sha256"]
                and selection["h5_used_for_selection"] is False
                and selection["selection_authority"]
                == "validation-loss composite EMA only"
            )
            if not chain_verified:
                raise ValueError(
                    f"training-only selection checkpoint chain failed: {task}/{arm}"
                )
            decision = training[-1]["loss_control_decision"]
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
                    "selection_source_sha256": base.sha256(selection_path),
                }
            )
    return rows


def comprehensive_claim_gate(statistics: dict, confirmation: dict) -> dict:
    audit_paths = {
        "formal": FORMAL_AUDIT,
        "supplementary": OUTPUT / "audit/supplementary_artifact_integrity.json",
        "confirmation": OUTPUT / "audit/confirmation_artifact_integrity.json",
        "gifs": OUTPUT / "audit/gif_artifact_integrity.json",
    }
    audit_schemas = {
        "formal": "paper-training-only-artifact-integrity-v1",
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
        audit_sources[name] = {
            "path": str(path.resolve()),
            "sha256": base.sha256(path),
        }

    formal = audit_payloads["formal"]
    lock_path = Path(formal["manifest"])
    if base.sha256(lock_path) != formal["manifest_file_sha256"]:
        raise ValueError("formal manifest changed after training-only artifact audit")
    fairness = base.frozen_factorial_fairness_gate(
        json.loads(lock_path.read_text(encoding="utf-8"))
    )
    protocol_checks["frozen_factorial_fairness"] = bool(fairness["passes"])

    result = {
        "definition": (
            "master plan section 2.1 claim gates under the user-authorized "
            "training-only observational amendment"
        ),
        "task_results": {},
        "protocol_integrity": {
            "checks": protocol_checks,
            "all_pass": all(protocol_checks.values()),
            "sources": audit_sources,
            "factorial_fairness": fairness,
        },
    }
    for task in base.TASKS:
        contrast = statistics["tasks"][task]["contrasts"]["C1_minus_L0"]
        success = contrast["metrics"]["success"]
        best = contrast["metrics"]["best_progress"]
        final = contrast["metrics"]["final_progress"]
        tail = contrast["metrics"]["catastrophic_failure"]

        correct = base.load_final(task, "C1", "primary")
        wrong_path = OUTPUT / task / "c1/wrong_goal_test/eval.json"
        wrong = base.committed_json(
            wrong_path, schema="paper-h5-goal-ablation-aggregate-v1", count=30
        )
        if wrong is None:
            raise ValueError(f"missing C1 wrong-goal result: {task}")
        correct_by = base._record_map_for_claim(correct["records"])
        wrong_by = base._record_map_for_claim(wrong["records"])
        if list(correct_by) != list(wrong_by):
            raise ValueError(f"wrong-goal case order changed: {task}")
        correct_values = base.np.asarray(
            [int(bool(record["success"])) for record in correct_by.values()],
            dtype=float,
        )
        wrong_values = base.np.asarray(
            [int(bool(record["success"])) for record in wrong_by.values()],
            dtype=float,
        )
        goal_effect = base.cluster_paired_effect(
            wrong_values[:, None],
            correct_values[:, None],
            seed=base.BOOTSTRAP_SEED + 211,
        )
        goal_effect["mcnemar"] = base.exact_paired_binomial(
            wrong_values, correct_values
        )

        action_path = (
            OUTPUT / "audit/action_contract" / task / "c1/first_success_h5.json"
        )
        action_count = 44 if task == "gear" else 100
        action = base.committed_json(
            action_path, schema="lewm-action-contract-audit-v6", count=action_count
        )
        if action is None:
            raise ValueError(f"missing C1 terminal action audit: {task}")
        action_gate = bool(action["strict_action_gate"]["closed_loop_eligible"])

        branch_path = OUTPUT / "action_branching" / task / "eval.json"
        branch = base.committed_json(
            branch_path,
            schema="paper-action-branching-aggregate-v1",
            count=(8 if task == "gear" else 9) * 30,
        )
        if branch is None:
            raise ValueError(f"missing same-init action branching result: {task}")
        branch_gate = base.action_branch_causality_gate(task, branch["records"])

        planner_deltas = {
            seed: int(contrast[seed]["rescues"] - contrast[seed]["harms"])
            for seed, _ in base.SEEDS
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
                "source_sha256": base.sha256(action_path),
            },
            "simulator_action_branch_gate": {
                **branch_gate,
                "source": str(branch_path.resolve()),
                "source_sha256": base.sha256(branch_path),
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


_write_report = base.write_report
_write_executive_summary = base.write_executive_summary


def write_report(*args, **kwargs) -> None:
    _write_report(*args, **kwargs)
    destination = kwargs.get("destination") or args[-1]
    path = Path(destination)
    text = path.read_text(encoding="utf-8")
    replacements = {
        "All checkpoints below were selected only by the frozen validation-loss composite. H5-dev30 was observational and H5-test30 remained sealed until selection.":
            "All checkpoints below were selected only by the frozen validation-loss composite. Under the latest user-authorized fixed-budget amendment, Gear L0 closes at epoch 28 and the remaining Gear arms run 30 epochs; Franka L0 closes at epoch 45 and the remaining Franka arms run 20 epochs. Early stopping is disabled for these main runs. Gear L0 H5-dev30 ends at epoch 23 and Franka L0 at epoch 40; later training and all L1/C0/C1 training omit per-epoch H5. H5-test30 remained sealed until selection.",
        "H5-dev30 is present for every completed epoch but is excluded from both stopping and checkpoint selection. The checkpoint-chain column requires every H5 epoch to match its training checkpoint and the selected SHA to match the minimum-validation-EMA epoch.":
            "H5-dev30 is present only for the authorized L0 cutoff ranges and is excluded from checkpoint selection. Main-run termination follows the user-authorized fixed epoch endpoints, while the checkpoint-chain column requires every preserved H5 epoch to match its training checkpoint and the selected SHA to match the minimum-validation-EMA epoch.",
        "- **Diagnostic:** H5-dev30 learning curves, loss-to-control correlations, failure labels, GIF appearance, and the historical Gear 10-case × 2-seed table.":
            "- **Diagnostic:** truncated L0 H5-dev30 learning curves through the authorized cutoff, loss-to-control correlations on those paired epochs, failure labels, GIF appearance, and the historical Gear 10-case × 2-seed table.",
    }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"expected report sentence missing: {old}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def write_executive_summary(*args, **kwargs) -> None:
    _write_executive_summary(*args, **kwargs)
    destination = kwargs.get("destination") or args[-1]
    path = Path(destination)
    text = path.read_text(encoding="utf-8")
    replacements = {
        "All eight main runs used fresh seed 3072, the same task-local data and optimizer schedule, at least 50 epochs, validation-loss-only checkpoint selection, and sealed H5-test30. H5-dev30 was observational only.":
            "All eight main runs used fresh seed 3072, the same task-local data and optimizer schedule, validation-loss-only checkpoint selection, and sealed H5-test30. Under the latest user-authorized fixed-budget amendment, Gear L0 closes at epoch 28, Gear L1/C0/C1 at epoch 30, Franka L0 at epoch 45, and Franka L1/C0/C1 at epoch 20; early stopping is disabled for the main matrix. Training-period H5-dev30 was retained only for Gear L0 epochs 1–23 and Franka L0 epochs 1–40.",
        "- Diagnostic evidence: per-epoch H5-dev30, loss correlations, failure taxonomy, representative GIFs, and historical Gear repeats.":
            "- Diagnostic evidence: cutoff-limited L0 H5-dev30, loss correlations on paired epochs, failure taxonomy, representative GIFs, and historical Gear repeats.",
    }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"expected executive-summary sentence missing: {old}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    base.final_analysis_provenance = final_analysis_provenance
    base.training_selection_table = training_selection_table
    base.comprehensive_claim_gate = comprehensive_claim_gate
    base.write_report = write_report
    base.write_executive_summary = write_executive_summary
    base.main()


if __name__ == "__main__":
    main()
