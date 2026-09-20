#!/usr/bin/env python3
"""Create the only evaluator-eligible receipt for one formal e10 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trim.evaluation.manifest import (  # noqa: E402
    _formal_method_name,
    _load_inference_core,
    _method_signature,
    _validate_frozen_training_config,
)
from trim.protocols.formal_training_completion import (  # noqa: E402
    FORMAL_TASK_DATASETS,
    FORMAL_COMPLETION_ARTIFACTS,
    FORMAL_COMPLETION_SCHEMA_VERSION,
    EPISODE_SPLIT_SCHEMA_VERSION,
    FORMAL_TRAINING_SEEDS,
    canonical_attempt_registry_path,
    canonical_sha256,
    file_sha256,
    formal_study_id,
    is_sha256,
    verify_smoke_attestation,
    verify_stage_authorization,
    write_self_hashed_json,
)
from trim.training.utilities import git_tracked_tree_sha256, runtime_environment_fingerprint  # noqa: E402


MILESTONES = (3, 6, 8, 10)
REFERENCE_TRAINING_SEED = 3072
SUPPORTED_TRAINING_SEEDS = frozenset(FORMAL_TRAINING_SEEDS)
SPLIT_SEED = 20260822


def _git_provenance(repo: Path):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo,
        text=True,
    ).strip()
    if status:
        raise RuntimeError(f"formal audit requires a clean worktree: {status[:500]}")
    return commit, git_tracked_tree_sha256(repo)


def _sha_episode_ids(values):
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()


def _finite_normalizer_stats(stats):
    if not isinstance(stats, dict) or not stats:
        return False
    for payload in stats.values():
        if not isinstance(payload, dict) or set(payload) != {"mean", "std"}:
            return False
        mean = torch.as_tensor(payload["mean"], dtype=torch.float64)
        std = torch.as_tensor(payload["std"], dtype=torch.float64)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            return False
        if not bool((std > 0).all()):
            return False
    return True


def _validated_training_seed(training_seed):
    training_seed = int(training_seed)
    if training_seed not in SUPPORTED_TRAINING_SEEDS:
        raise ValueError(
            "formal transition-triangle training_seed must be one of "
            f"{sorted(SUPPORTED_TRAINING_SEEDS)}, received {training_seed}"
        )
    return training_seed


def _audit_manifest(
    manifest, *, task, training_seed, expected_commit, expected_tree
):
    training_seed = _validated_training_seed(training_seed)
    expected_dataset = FORMAL_TASK_DATASETS[task]
    expected = {
        "schema_version": EPISODE_SPLIT_SCHEMA_VERSION,
        "training_seed": training_seed,
        "split_seed": SPLIT_SEED,
        "train_fraction": 0.9,
        "dataset_name": expected_dataset,
        "source_git_commit": expected_commit,
        "source_tracked_tree_sha256": expected_tree,
        "normalizer_fit": "train_episodes_only",
        "loader_generator_seed": training_seed,
        "manager_seed": training_seed,
        "worker_seed_mode": "lightning_seed_everything_workers",
        "transition_triangle_task": task,
    }
    mismatches = {
        key: (manifest.get(key), wanted)
        for key, wanted in expected.items()
        if manifest.get(key) != wanted
    }
    train_ids = [int(value) for value in manifest.get("train_internal_episode_ids", [])]
    val_ids = [
        int(value) for value in manifest.get("validation_internal_episode_ids", [])
    ]
    if not train_ids or not val_ids or set(train_ids).intersection(val_ids):
        mismatches["episode_split"] = (
            (len(train_ids), len(val_ids), len(set(train_ids).intersection(val_ids))),
            "two nonempty disjoint episode sets",
        )
    if manifest.get("train_sha256") != _sha_episode_ids(train_ids):
        mismatches["train_sha256"] = (manifest.get("train_sha256"), "recomputed")
    if manifest.get("validation_sha256") != _sha_episode_ids(val_ids):
        mismatches["validation_sha256"] = (
            manifest.get("validation_sha256"), "recomputed"
        )
    train_windows = int(manifest.get("train_windows", 0))
    val_windows = int(manifest.get("validation_windows", 0))
    dataset_windows = int(manifest.get("dataset_windows", 0))
    if min(train_windows, val_windows) <= 0 or train_windows + val_windows != dataset_windows:
        mismatches["window_partition"] = (
            (train_windows, val_windows, dataset_windows),
            "positive train+validation=dataset",
        )
    if not _finite_normalizer_stats(manifest.get("normalizer_stats")):
        mismatches["normalizer_stats"] = ("invalid", "finite positive-std stats")
    if not is_sha256(manifest.get("dataset_file_sha256")):
        mismatches["dataset_file_sha256"] = (
            manifest.get("dataset_file_sha256"), "lowercase SHA256"
        )
    runtime = manifest.get("runtime_environment")
    if canonical_sha256(runtime) != manifest.get("runtime_environment_sha256"):
        mismatches["runtime_environment_sha256"] = (
            manifest.get("runtime_environment_sha256"), canonical_sha256(runtime)
        )
    if mismatches:
        raise RuntimeError(f"formal episode manifest mismatch: {mismatches}")


def _audit_initialization_contract(
    *, manifest, smoke, authorization, arm, training_seed
):
    """Verify the preregistered initialization pairing for one formal seed.

    Seed 3072 is the discovery/reference run and must exactly reproduce the
    source-matched smoke initialization, including each method's training-only
    inverse head.  Confirmation seeds 1/2 intentionally have fresh
    initialization and therefore must not be compared with the seed-3072 smoke.
    For a confirmation candidate, its shared core is instead paired to the
    already-completed champion of the same seed through the authorization.
    """
    training_seed = _validated_training_seed(training_seed)
    if training_seed == REFERENCE_TRAINING_SEED:
        if manifest["initial_shared_core_sha256"] != smoke[
            "initial_shared_core_sha256"
        ]:
            raise RuntimeError(
                "formal shared-core initialization differs from smoke"
            )
        if manifest["initial_full_model_sha256"] != smoke["arms"][arm][
            "initial_full_model_sha256"
        ]:
            raise RuntimeError(
                f"formal {arm} full initialization differs from smoke"
            )
        return

    if arm == "champion":
        # The smoke remains the resource/topology/runtime gate, but its fixed
        # seed-3072 parameter hashes are not an initialization target for fresh
        # confirmation champions.
        return

    prior_shared = authorization.get("prior_initial_shared_core_sha256")
    if not isinstance(prior_shared, str) or not prior_shared:
        raise RuntimeError(
            f"formal seed{training_seed} {arm} authorization lacks the prior "
            "champion initial shared-core SHA256"
        )
    if manifest["initial_shared_core_sha256"] != prior_shared:
        raise RuntimeError(
            f"formal seed{training_seed} {arm} shared-core initialization "
            "differs from its champion predecessor"
        )


def _audit_receipt(receipt, *, arm, task, manifest):
    num_batches = int(receipt.get("num_training_batches", -1))
    expected_steps = num_batches * 10
    expected = {
        "schema_version": 2,
        "method_name": arm,
        "task": task,
        "world_size": 2,
        "distributed_ranks": [0, 1],
        "per_device_batch_size": 64,
        "accumulate_grad_batches": 1,
        "effective_global_batch_size": 128,
        "optimizer_steps": expected_steps,
        "completed_epochs": list(range(1, 11)),
        "max_epochs": 10,
        "estimated_stepping_batches": expected_steps,
        "initial_full_model_sha256": manifest["initial_full_model_sha256"],
        "initial_shared_core_sha256": manifest["initial_shared_core_sha256"],
        "active_gradient_audit_passed": True,
        "active_update_audit_passed": True,
        "formal_execution": True,
        "formal_stage_authorization_sha256": manifest[
            "formal_stage_authorization_sha256"
        ],
        "formal_smoke_attestation_sha256": manifest[
            "formal_smoke_attestation_sha256"
        ],
        "formal_slurm_job_id": manifest["formal_slurm_job_id"],
        "formal_slurm_restart_count": 0,
        "formal_spt_cache_dir_sha256": manifest[
            "formal_spt_cache_dir_sha256"
        ],
    }
    mismatches = {
        key: (receipt.get(key), wanted)
        for key, wanted in expected.items()
        if receipt.get(key) != wanted
    }
    # DDP drops the incomplete final per-rank batch in the formal DataLoader.
    if num_batches != int(manifest["train_windows"]) // 128:
        mismatches["num_training_batches"] = (
            num_batches, int(manifest["train_windows"]) // 128
        )
    if "DDP" not in str(receipt.get("strategy_class", "")):
        mismatches["strategy_class"] = (receipt.get("strategy_class"), "*DDP*")
    for key in (
        "fit_wall_time_seconds",
        "optimizer_steps_per_second",
        "peak_cuda_memory_allocated_bytes",
    ):
        value = receipt.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            mismatches[key] = (value, "finite and >0")
    expected_groups = {"backward_state", "inverse_action"}
    for key in ("gradient_min_l2_by_group", "update_min_max_abs_by_group"):
        values = receipt.get(key)
        if (
            not isinstance(values, dict)
            or set(values) != expected_groups
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
                for value in values.values()
            )
        ):
            mismatches[key] = (values, "two positive finite active groups")
    expected_source = {
        "source_git_commit": manifest["source_git_commit"],
        "source_tracked_tree_sha256": manifest["source_tracked_tree_sha256"],
        "source_worktree_clean": True,
    }
    for key in ("source_at_fit_start", "source_at_fit_end"):
        if receipt.get(key) != expected_source:
            mismatches[key] = (receipt.get(key), expected_source)
    expected_optimizer = [{
        "class_name": "AdamW",
        "initial_lr": 5e-5,
        "weight_decay": 1e-3,
    }]
    if receipt.get("optimizer_contracts") != expected_optimizer:
        mismatches["optimizer_contracts"] = (
            receipt.get("optimizer_contracts"), expected_optimizer
        )
    schedulers = receipt.get("scheduler_contracts")
    if (
        not isinstance(schedulers, list)
        or len(schedulers) != 1
        or schedulers[0].get("class_name") != "LinearWarmupCosineAnnealingLR"
        or schedulers[0].get("interval") != "epoch"
        or int(schedulers[0].get("last_epoch", -1)) < 9
    ):
        mismatches["scheduler_contracts"] = (
            schedulers, "one completed epoch-interval cosine scheduler"
        )
    if mismatches:
        raise RuntimeError(f"formal execution receipt mismatch: {mismatches}")
    return expected_steps


def _audit_metrics(path: Path, *, arm, optimizer_steps):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    inverse = "dcl_idm_loss" if arm == "champion" else "idm_loss"
    required = {
        f"{stage}_epoch/{metric}"
        for stage in ("fit", "validate")
        for metric in (
            "loss", "pred_loss", "tf_loss", "ar_loss", "fmca_loss",
            "standalone_sigreg_loss", "rankme", inverse,
            "forward_loss_contribution", "backward_loss_contribution",
            "inverse_loss_contribution", "sigreg_loss_contribution",
            "transition_triangle_loss",
        )
    }
    observed_epochs = {key: set() for key in required}
    epoch_values = {key: {} for key in required}
    steps = []
    for row in rows:
        raw_epoch = row.get("epoch")
        epoch = None if raw_epoch in (None, "") else int(float(raw_epoch))
        raw_step = row.get("step")
        if raw_step not in (None, ""):
            steps.append(int(float(raw_step)))
        for key, raw in row.items():
            if raw in (None, ""):
                continue
            if "loss" in key or "rankme" in key:
                value = float(raw)
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite persisted metric {key}={value}")
            if key in required:
                if epoch is None:
                    raise RuntimeError(f"persisted epoch metric {key} lacks epoch")
                observed_epochs[key].add(epoch)
                epoch_values[key][epoch] = float(raw)
    expected_epochs = set(range(10))
    incomplete = {
        key: sorted(epochs)
        for key, epochs in observed_epochs.items()
        if epochs != expected_epochs
    }
    if incomplete:
        raise RuntimeError(f"formal metrics lack exact epochs 0..9: {incomplete}")
    contribution_names = (
        "forward_loss_contribution",
        "backward_loss_contribution",
        "inverse_loss_contribution",
        "sigreg_loss_contribution",
    )
    for stage in ("fit", "validate"):
        for epoch in sorted(expected_epochs):
            prefix = f"{stage}_epoch/"
            core = epoch_values[prefix + "transition_triangle_loss"][epoch]
            total = epoch_values[prefix + "loss"][epoch]
            recomputed = sum(
                epoch_values[prefix + name][epoch]
                for name in contribution_names
            )
            if not math.isclose(core, recomputed, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(
                    f"{stage} epoch {epoch} transition-triangle contributions "
                    f"do not sum: logged={core}, recomputed={recomputed}"
                )
            if not math.isclose(total, core, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(
                    f"{stage} epoch {epoch} formal loss contains an unaccounted "
                    f"term: total={total}, transition_triangle={core}"
                )
    if not steps or steps != sorted(steps) or max(steps) not in {
        optimizer_steps - 1, optimizer_steps
    }:
        raise RuntimeError(
            f"formal metric steps are incompatible with optimizer steps: "
            f"last={max(steps) if steps else None}, optimizer={optimizer_steps}"
        )


def audit_formal_training(
    *, run_dir: Path, arm: str, task: str, training_seed: int,
    expected_commit: str,
    smoke_attestation_path: Path, stage_authorization_path: Path,
):
    training_seed = _validated_training_seed(training_seed)
    run_dir = Path(run_dir).resolve()
    repo = Path(__file__).resolve().parents[1]
    commit, tree = _git_provenance(repo)
    if commit != expected_commit:
        raise RuntimeError(f"formal source commit {commit} != {expected_commit}")
    smoke = verify_smoke_attestation(
        smoke_attestation_path,
        expected_commit=commit,
        expected_tree=tree,
        expected_task=task,
    )
    authorization = verify_stage_authorization(
        stage_authorization_path,
        expected_commit=commit,
        expected_arm=arm,
        expected_task=task,
        expected_training_seed=training_seed,
        require_registry=True,
        expected_registry_status="AUTHORIZED",
    )
    if authorization.get("training_seed") != training_seed:
        raise RuntimeError(
            "stage authorization training seed differs from the requested run: "
            f"authorization={authorization.get('training_seed')!r}, "
            f"requested={training_seed}"
        )
    if authorization.get("smoke_attestation_sha256") != smoke["attestation_sha256"]:
        raise RuntimeError("stage authorization does not bind the supplied smoke")
    if authorization.get("slurm_job_id") != os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("stage authorization is not bound to this Slurm job")
    if arm == "champion" and authorization.get("prior_completion_sha256") is not None:
        raise RuntimeError("champion authorization unexpectedly has a predecessor")
    if arm in {"concat_mse", "delta_mse"} and not authorization.get(
        "prior_completion_sha256"
    ):
        raise RuntimeError(f"{arm} authorization lacks the champion predecessor")

    manifest_path = run_dir / "episode_split_manifest.json"
    config_path = run_dir / "config.yaml"
    manifest = json.loads(manifest_path.read_text())
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    _audit_manifest(
        manifest,
        task=task,
        training_seed=training_seed,
        expected_commit=commit,
        expected_tree=tree,
    )
    formal_binding_expected = {
        "transition_triangle_formal_execution": True,
        "formal_stage_authorization_sha256": authorization[
            "authorization_sha256"
        ],
        "formal_smoke_attestation_sha256": smoke["attestation_sha256"],
        "formal_slurm_job_id": authorization["slurm_job_id"],
        "formal_slurm_restart_count": 0,
    }
    formal_binding_mismatches = {
        key: (manifest.get(key), value)
        for key, value in formal_binding_expected.items()
        if manifest.get(key) != value
    }
    if formal_binding_mismatches:
        raise RuntimeError(
            f"formal manifest stage binding mismatch: {formal_binding_mismatches}"
        )
    _validate_frozen_training_config(config, manifest, run_dir)
    method_name = _formal_method_name(_method_signature(config))
    if method_name != arm:
        raise RuntimeError(f"formal arm mismatch: config={method_name}, expected={arm}")
    if manifest["dataset_fingerprint_sha256"] != smoke["dataset_fingerprint_sha256"]:
        raise RuntimeError("formal dataset fingerprint differs from smoke")
    expected_dataset_file_sha256 = authorization["dataset_file_sha256"]
    dataset_hashes = {
        "manifest": manifest.get("dataset_file_sha256"),
        "smoke": smoke.get("dataset_file_sha256"),
        "config": config.get("formal_dataset_file_sha256"),
    }
    dataset_hash_mismatches = {
        name: (observed, expected_dataset_file_sha256)
        for name, observed in dataset_hashes.items()
        if observed != expected_dataset_file_sha256
    }
    if dataset_hash_mismatches:
        raise RuntimeError(
            "formal dataset file SHA256 binding mismatch: "
            f"{dataset_hash_mismatches}"
        )
    _audit_initialization_contract(
        manifest=manifest,
        smoke=smoke,
        authorization=authorization,
        arm=arm,
        training_seed=training_seed,
    )
    staged_dataset_path = Path(manifest["dataset_path"])
    staged_dataset_file_sha256 = file_sha256(staged_dataset_path)
    if staged_dataset_file_sha256 != expected_dataset_file_sha256:
        raise RuntimeError("formal staged dataset bytes differ from authorized smoke data")
    current_runtime_sha256 = canonical_sha256(runtime_environment_fingerprint())
    if current_runtime_sha256 != manifest["runtime_environment_sha256"]:
        raise RuntimeError("runtime environment changed before post-training audit")
    if current_runtime_sha256 != smoke["runtime_environment_sha256"]:
        raise RuntimeError("formal runtime environment differs from smoke")

    resumable = sorted(run_dir.rglob("*.ckpt"))
    if resumable:
        raise RuntimeError(f"formal run contains resumable checkpoints: {resumable[:8]}")
    raw_weights = {path.name for path in run_dir.glob("weights_epoch_*.pt")}
    expected_weights = {f"weights_epoch_{epoch}.pt" for epoch in MILESTONES}
    if raw_weights != expected_weights:
        raise RuntimeError(
            f"formal raw checkpoint set mismatch: {sorted(raw_weights)} != "
            f"{sorted(expected_weights)}"
        )
    if list(run_dir.glob("weights_step_*.pt")):
        raise RuntimeError("formal run contains forbidden step weights")

    checkpoint_hashes = {}
    for epoch in MILESTONES:
        checkpoint = run_dir / f"weights_epoch_{epoch}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        ):
            raise RuntimeError(f"epoch {epoch} is not a raw tensor state_dict")
        nonfinite = [
            key for key, value in state.items()
            if (value.is_floating_point() or value.is_complex())
            and not bool(torch.isfinite(value).all())
        ]
        if nonfinite:
            raise RuntimeError(f"epoch {epoch} has non-finite state: {nonfinite[:8]}")
        _load_inference_core(
            checkpoint,
            method_signature=_method_signature(config),
            enforce_canonical_architecture=True,
        )
        checkpoint_hashes[checkpoint.name] = file_sha256(checkpoint)
    if len(set(checkpoint_hashes.values())) != len(MILESTONES):
        raise RuntimeError("formal milestone checkpoints are not all distinct")

    receipt_path = run_dir / "transition_triangle_execution_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    optimizer_steps = _audit_receipt(
        receipt, arm=arm, task=task, manifest=manifest
    )
    metrics_path = run_dir / "training_metrics" / "metrics.csv"
    _audit_metrics(metrics_path, arm=arm, optimizer_steps=optimizer_steps)

    run_authorization = run_dir / "formal_stage_authorization.json"
    if run_authorization.exists():
        raise RuntimeError(f"refusing pre-existing run authorization: {run_authorization}")
    shutil.copyfile(stage_authorization_path, run_authorization)
    run_smoke = run_dir / "formal_smoke_attestation.json"
    if run_smoke.exists():
        raise RuntimeError(f"refusing pre-existing run smoke attestation: {run_smoke}")
    shutil.copyfile(smoke_attestation_path, run_smoke)
    artifacts = {
        relative: file_sha256(run_dir / relative)
        for relative in sorted(FORMAL_COMPLETION_ARTIFACTS)
    }
    payload = {
        "schema_version": FORMAL_COMPLETION_SCHEMA_VERSION,
        "status": "PASS",
        "task": task,
        "method_name": arm,
        "training_seed": training_seed,
        "split_seed": SPLIT_SEED,
        "run_dir_name": run_dir.name,
        "checkpoint_relative_path": "weights_epoch_10.pt",
        "source_git_commit": commit,
        "source_tracked_tree_sha256": tree,
        "runtime_environment_sha256": current_runtime_sha256,
        "dataset_file_sha256": staged_dataset_file_sha256,
        "initial_shared_core_sha256": manifest["initial_shared_core_sha256"],
        "smoke_attestation_sha256": smoke["attestation_sha256"],
        "stage_authorization_sha256": authorization["authorization_sha256"],
        "formal_slurm_job_id": authorization["slurm_job_id"],
        "artifacts": artifacts,
    }
    output = run_dir / "formal_training_complete.json"
    if output.exists():
        raise RuntimeError(f"refusing pre-existing completion receipt: {output}")
    result = write_self_hashed_json(
        payload, output, hash_key="completion_sha256"
    )
    registry_path = canonical_attempt_registry_path(
        task=task, arm=arm, training_seed=training_seed
    )
    registry_completion = registry_path / "formal_training_complete.json"
    if registry_completion.exists():
        raise RuntimeError(
            f"refusing pre-existing registry completion: {registry_completion}"
        )
    shutil.copyfile(output, registry_completion)
    write_self_hashed_json(
        {
            "schema_version": 1,
            "status": "COMPLETED",
            "study_id": formal_study_id(task),
            "task": task,
            "arm": arm,
            "training_seed": training_seed,
            "authorization_sha256": authorization["authorization_sha256"],
            "completion_sha256": result["completion_sha256"],
        },
        registry_path / "attempt_status.json",
        hash_key="status_sha256",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("champion", "concat_mse", "delta_mse"),
    )
    parser.add_argument("--task", required=True, choices=("pusht", "tworoom"))
    parser.add_argument(
        "--training-seed",
        required=True,
        type=int,
        choices=tuple(sorted(SUPPORTED_TRAINING_SEEDS)),
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--smoke-attestation", required=True, type=Path)
    parser.add_argument("--stage-authorization", required=True, type=Path)
    args = parser.parse_args()
    result = audit_formal_training(
        run_dir=args.run_dir,
        arm=args.arm,
        task=args.task,
        training_seed=args.training_seed,
        expected_commit=args.expected_commit,
        smoke_attestation_path=args.smoke_attestation,
        stage_authorization_path=args.stage_authorization,
    )
    print(
        f"FORMAL_TRAINING_AUDIT PASS arm={result['method_name']} "
        f"training_seed={result['training_seed']} "
        f"sha256={result['completion_sha256']}"
    )


if __name__ == "__main__":
    main()
