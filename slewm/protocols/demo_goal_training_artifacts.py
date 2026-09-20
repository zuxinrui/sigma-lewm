"""Study-scoped provenance chain for formal Demo-Goal training runs."""

from __future__ import annotations
from repro_paths import resolve_path as _resolve_artifact

import json
from pathlib import Path

from slewm.protocols.demo_goal_formal_artifacts import (
    file_sha256,
    verify_demo_goal_method_lock,
    verify_self_hashed_json,
)
from slewm.protocols.demo_goal_formal_specs import (
    DEMO_GOAL_FORMAL_ARMS,
    DEMO_GOAL_STUDY_IDS,
    DEMO_GOAL_TRAINING_SEEDS,
)
from slewm.protocols.tworoom_replay_attestation import verify_tworoom_replay_attestation


DEMO_GOAL_SMOKE_SCHEMA_VERSION = 1
DEMO_GOAL_AUTHORIZATION_SCHEMA_VERSION = 1
DEMO_GOAL_COMPLETION_SCHEMA_VERSION = 1
DEMO_GOAL_COMPLETION_STATUS_SCHEMA_VERSION = 2
DEMO_GOAL_ATTEMPT_REGISTRY_ROOT = Path(
    _resolve_artifact('runs:lewm_demo_goal_formal_attempts')
)

DEMO_GOAL_COMMON_COMPLETION_ARTIFACTS = {
    "config.yaml",
    "config.json",
    "episode_split_manifest.json",
    "transition_triangle_execution_receipt.json",
    "training_metrics/metrics.csv",
    "weights_epoch_3.pt",
    "weights_epoch_6.pt",
    "weights_epoch_8.pt",
    "weights_epoch_10.pt",
    "demo_goal_stage_authorization.json",
    "demo_goal_smoke_attestation.json",
    "demo_goal_method_lock.json",
}


def demo_goal_phase(training_seed: int) -> str:
    training_seed = int(training_seed)
    if training_seed not in DEMO_GOAL_TRAINING_SEEDS:
        raise ValueError(
            f"unsupported Demo-Goal training seed {training_seed}; "
            f"want one of {DEMO_GOAL_TRAINING_SEEDS}"
        )
    return "discovery" if training_seed == 3072 else "confirmation"


def demo_goal_stage_name(*, task: str, arm: str, training_seed: int) -> str:
    if task not in DEMO_GOAL_STUDY_IDS:
        raise ValueError(f"unsupported Demo-Goal task: {task!r}")
    if arm not in DEMO_GOAL_FORMAL_ARMS:
        raise ValueError(f"unsupported Demo-Goal arm: {arm!r}")
    phase = demo_goal_phase(training_seed)
    base = f"{task}_{arm}"
    return base if phase == "discovery" else f"{base}_seed{int(training_seed)}"


def demo_goal_attempt_registry_path(
    *, task: str, arm: str, training_seed: int
) -> Path:
    demo_goal_stage_name(task=task, arm=arm, training_seed=training_seed)
    return (
        DEMO_GOAL_ATTEMPT_REGISTRY_ROOT
        / DEMO_GOAL_STUDY_IDS[task]
        / task
        / arm
        / f"seed{int(training_seed)}"
    )


def demo_goal_attempt_registry_key(
    *, task: str, arm: str, training_seed: int
) -> str:
    demo_goal_attempt_registry_path(
        task=task, arm=arm, training_seed=training_seed
    )
    return (
        f"{DEMO_GOAL_STUDY_IDS[task]}:{task}:{arm}:seed{int(training_seed)}"
    )


def verify_demo_goal_smoke_attestation(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    expected_task: str | None = None,
    expected_method_lock_sha256: str | None = None,
):
    value = verify_self_hashed_json(
        path, hash_key="smoke_attestation_sha256"
    )
    expected = {
        "schema_version": DEMO_GOAL_SMOKE_SCHEMA_VERSION,
        "status": "PASS",
        "contains_performance_outcomes": False,
        "formal_arms": list(DEMO_GOAL_FORMAL_ARMS),
    }
    if expected_commit is not None:
        expected["source_git_commit"] = expected_commit
    if expected_tree is not None:
        expected["source_tracked_tree_sha256"] = expected_tree
    if expected_task is not None:
        expected["task"] = expected_task
        expected["study_id"] = DEMO_GOAL_STUDY_IDS[expected_task]
    if expected_method_lock_sha256 is not None:
        expected["method_lock_sha256"] = expected_method_lock_sha256
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    arms = value.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(DEMO_GOAL_FORMAL_ARMS):
        mismatches["arms"] = (
            sorted(arms) if isinstance(arms, dict) else arms,
            list(DEMO_GOAL_FORMAL_ARMS),
        )
    else:
        predecessor_hashes = set()
        for arm, receipt in arms.items():
            required = {
                "method_name": arm,
                "world_size": 2,
                "effective_global_batch_size": 128,
                "active_gradient_audit_passed": True,
                "active_update_audit_passed": True,
            }
            for key, wanted in required.items():
                if receipt.get(key) != wanted:
                    mismatches[f"arms.{arm}.{key}"] = (
                        receipt.get(key), wanted
                    )
            predecessor_hashes.add(
                receipt.get("initial_champion_predecessor_sha256")
            )
        if len(predecessor_hashes) != 1 or None in predecessor_hashes:
            mismatches["paired_champion_predecessor_initialization"] = (
                sorted(str(value) for value in predecessor_hashes),
                "one shared non-null hash",
            )
    method_lock_path = value.get("method_lock_path")
    try:
        lock = verify_demo_goal_method_lock(
            Path(method_lock_path),
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
        )
        if file_sha256(Path(method_lock_path)) != value.get(
            "method_lock_file_sha256"
        ):
            mismatches["method_lock_file_sha256"] = ("recomputed", "bound")
        if lock.get("method_lock_sha256") != value.get("method_lock_sha256"):
            mismatches["method_lock_sha256"] = (
                value.get("method_lock_sha256"), lock.get("method_lock_sha256")
            )
    except (TypeError, OSError, RuntimeError) as error:
        mismatches["method_lock"] = (str(error), "valid bound lock")
    if value.get("task") == "tworoom":
        try:
            replay_path = Path(value["tworoom_replay_attestation_path"])
            replay = verify_tworoom_replay_attestation(
                replay_path,
                expected_commit=value.get("source_git_commit"),
                expected_tree=value.get("source_tracked_tree_sha256"),
                expected_dataset_sha256=value.get("dataset_file_sha256"),
                expected_train_sha256=value.get("train_sha256"),
                expected_method_lock_sha256=value.get("method_lock_sha256"),
                replay_current_environment=True,
            )
            if file_sha256(replay_path) != value.get(
                "tworoom_replay_attestation_file_sha256"
            ):
                mismatches["tworoom_replay_file_sha256"] = (
                    "recomputed", "bound"
                )
            if replay.get("replay_attestation_sha256") != value.get(
                "tworoom_replay_attestation_sha256"
            ):
                mismatches["tworoom_replay_attestation_sha256"] = (
                    value.get("tworoom_replay_attestation_sha256"),
                    replay.get("replay_attestation_sha256"),
                )
        except (KeyError, OSError, RuntimeError) as error:
            mismatches["tworoom_replay_attestation"] = (
                str(error), "valid bound replay gate"
            )
    if mismatches:
        raise RuntimeError(
            f"Demo-Goal smoke attestation mismatch: {mismatches}"
        )
    return value


def verify_demo_goal_stage_authorization(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    expected_task: str,
    expected_arm: str,
    expected_training_seed: int,
    expected_method_lock_sha256: str | None = None,
    require_registry: bool = False,
    expected_registry_status: str | None = None,
):
    value = verify_self_hashed_json(path, hash_key="authorization_sha256")
    expected_task = str(expected_task)
    expected_arm = str(expected_arm)
    expected_training_seed = int(expected_training_seed)
    expected = {
        "schema_version": DEMO_GOAL_AUTHORIZATION_SCHEMA_VERSION,
        "status": "AUTHORIZED",
        "study_id": DEMO_GOAL_STUDY_IDS[expected_task],
        "task": expected_task,
        "arm": expected_arm,
        "training_seed": expected_training_seed,
        "phase": demo_goal_phase(expected_training_seed),
        "stage": demo_goal_stage_name(
            task=expected_task,
            arm=expected_arm,
            training_seed=expected_training_seed,
        ),
        "split_seed": 20260822,
    }
    if expected_commit is not None:
        expected["source_git_commit"] = expected_commit
    if expected_tree is not None:
        expected["source_tracked_tree_sha256"] = expected_tree
    if expected_method_lock_sha256 is not None:
        expected["method_lock_sha256"] = expected_method_lock_sha256
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    try:
        smoke_path = Path(value["smoke_attestation_path"])
        smoke = verify_demo_goal_smoke_attestation(
            smoke_path,
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
            expected_task=expected_task,
            expected_method_lock_sha256=value.get("method_lock_sha256"),
        )
        if file_sha256(smoke_path) != value.get("smoke_attestation_file_sha256"):
            mismatches["smoke_attestation_file_sha256"] = (
                "recomputed", "bound"
            )
        if smoke.get("smoke_attestation_sha256") != value.get(
            "smoke_attestation_sha256"
        ):
            mismatches["smoke_attestation_sha256"] = (
                value.get("smoke_attestation_sha256"),
                smoke.get("smoke_attestation_sha256"),
            )
        smoke_copied_bindings = {
            "runtime_environment_sha256": smoke.get(
                "runtime_environment_sha256"
            ),
            "dataset_fingerprint_sha256": smoke.get(
                "dataset_fingerprint_sha256"
            ),
            "dataset_file_sha256": smoke.get("dataset_file_sha256"),
            "train_sha256": smoke.get("train_sha256"),
            "validation_sha256": smoke.get("validation_sha256"),
        }
        for key, wanted in smoke_copied_bindings.items():
            if value.get(key) != wanted:
                mismatches[key] = (value.get(key), wanted)
        if expected_task == "tworoom":
            replay_bindings = {
                "tworoom_replay_attestation_sha256": smoke.get(
                    "tworoom_replay_attestation_sha256"
                ),
                "tworoom_replay_attestation_file_sha256": smoke.get(
                    "tworoom_replay_attestation_file_sha256"
                ),
            }
            for key, wanted in replay_bindings.items():
                if value.get(key) != wanted:
                    mismatches[key] = (value.get(key), wanted)
    except (KeyError, OSError, RuntimeError) as error:
        mismatches["smoke_attestation"] = (str(error), "valid bound smoke")
    if expected_arm == "champion":
        for key in (
            "prior_completion_sha256",
            "prior_initial_champion_predecessor_sha256",
            "prior_slurm_job_id",
        ):
            if value.get(key) is not None:
                mismatches[key] = (value.get(key), None)
    if expected_arm == "dg_champion":
        candidate_prior_fields = {
            "prior_completion_sha256": "champion completion",
            "prior_initial_champion_predecessor_sha256": (
                "champion predecessor initialization"
            ),
            "prior_slurm_job_id": "champion Slurm job",
        }
        for key, wanted in candidate_prior_fields.items():
            if not value.get(key):
                mismatches[key] = (value.get(key), wanted)
    expected_discovery_tasks = []
    if expected_task == "tworoom" or demo_goal_phase(expected_training_seed) == "confirmation":
        expected_discovery_tasks.append("pusht")
    if demo_goal_phase(expected_training_seed) == "confirmation":
        expected_discovery_tasks.append("tworoom")
    discovery_reports = value.get("passed_discovery_reports")
    if expected_discovery_tasks:
        if (
            not isinstance(discovery_reports, dict)
            or set(discovery_reports) != set(expected_discovery_tasks)
        ):
            mismatches["passed_discovery_reports"] = (
                sorted(discovery_reports)
                if isinstance(discovery_reports, dict)
                else discovery_reports,
                sorted(expected_discovery_tasks),
            )
        else:
            try:
                from scripts.compare_formal_eval_results import (
                    verify_comparison_report,
                )

                for report_task in expected_discovery_tasks:
                    binding = discovery_reports[report_task]
                    report_path = Path(binding["path"])
                    report = verify_comparison_report(
                        report_path,
                        expected_task=report_task,
                        expected_mode="discovery",
                        expected_baseline_method="champion",
                        expected_candidate_method="dg_champion",
                        expected_commit=value.get("source_git_commit"),
                        expected_tree=value.get(
                            "source_tracked_tree_sha256"
                        ),
                    )
                    expected_binding = {
                        "file_sha256": file_sha256(report_path),
                        "report_sha256": report["report_sha256"],
                    }
                    for key, wanted in expected_binding.items():
                        if binding.get(key) != wanted:
                            mismatches[
                                f"passed_discovery_reports.{report_task}.{key}"
                            ] = (binding.get(key), wanted)
                    if not report.get("gate_pass"):
                        mismatches[
                            f"passed_discovery_reports.{report_task}.gate_pass"
                        ] = (report.get("gate_pass"), True)
                    if report.get("method_lock_sha256") != value.get(
                        "method_lock_sha256"
                    ):
                        mismatches[
                            f"passed_discovery_reports.{report_task}.method_lock_sha256"
                        ] = (
                            report.get("method_lock_sha256"),
                            value.get("method_lock_sha256"),
                        )
            except (KeyError, OSError, RuntimeError) as error:
                mismatches["passed_discovery_reports.binding"] = (
                    str(error), "valid passed discovery reports"
                )
    elif discovery_reports is not None:
        mismatches["passed_discovery_reports"] = (
            discovery_reports, "absent"
        )
    needs_pusht_confirmation = (
        expected_task == "tworoom"
        and demo_goal_phase(expected_training_seed) == "confirmation"
    )
    confirmation_reports = value.get("passed_confirmation_reports")
    if needs_pusht_confirmation:
        if not isinstance(confirmation_reports, dict) or set(
            confirmation_reports
        ) != {"pusht"}:
            mismatches["passed_confirmation_reports"] = (
                confirmation_reports, {"pusht": "passed confirmation"}
            )
        else:
            try:
                from scripts.compare_formal_eval_results import (
                    verify_comparison_report,
                )

                binding = confirmation_reports["pusht"]
                report_path = Path(binding["path"])
                report = verify_comparison_report(
                    report_path,
                    expected_task="pusht",
                    expected_mode="confirmation",
                    expected_baseline_method="champion",
                    expected_candidate_method="dg_champion",
                    expected_commit=value.get("source_git_commit"),
                    expected_tree=value.get("source_tracked_tree_sha256"),
                )
                expected_binding = {
                    "file_sha256": file_sha256(report_path),
                    "report_sha256": report["report_sha256"],
                }
                for key, wanted in expected_binding.items():
                    if binding.get(key) != wanted:
                        mismatches[
                            f"passed_confirmation_reports.pusht.{key}"
                        ] = (binding.get(key), wanted)
                if not report.get("gate_pass"):
                    mismatches[
                        "passed_confirmation_reports.pusht.gate_pass"
                    ] = (report.get("gate_pass"), True)
                if report.get("method_lock_sha256") != value.get(
                    "method_lock_sha256"
                ):
                    mismatches[
                        "passed_confirmation_reports.pusht.method_lock_sha256"
                    ] = (
                        report.get("method_lock_sha256"),
                        value.get("method_lock_sha256"),
                    )
            except (KeyError, OSError, RuntimeError) as error:
                mismatches["passed_confirmation_reports.binding"] = (
                    str(error), "valid passed PushT confirmation"
                )
    elif confirmation_reports is not None:
        mismatches["passed_confirmation_reports"] = (
            confirmation_reports, "absent"
        )
    qualification_binding = value.get("discovery_qualification")
    if demo_goal_phase(expected_training_seed) == "confirmation":
        if not isinstance(qualification_binding, dict):
            mismatches["discovery_qualification"] = (
                qualification_binding, "immutable qualification binding"
            )
        else:
            try:
                from slewm.protocols.demo_goal_final_attestation import (
                    verify_discovery_qualification,
                )

                qualification_path = Path(qualification_binding["path"])
                qualification = verify_discovery_qualification(
                    qualification_path,
                    expected_commit=value.get("source_git_commit"),
                    expected_tree=value.get(
                        "source_tracked_tree_sha256"
                    ),
                    expected_method_lock_sha256=value.get(
                        "method_lock_sha256"
                    ),
                )
                wanted = {
                    "path": str(qualification_path.resolve()),
                    "file_sha256": file_sha256(qualification_path),
                    "qualification_sha256": qualification[
                        "qualification_sha256"
                    ],
                }
                if qualification_binding != wanted:
                    mismatches["discovery_qualification"] = (
                        qualification_binding, wanted
                    )
            except (KeyError, OSError, RuntimeError) as error:
                mismatches["discovery_qualification.binding"] = (
                    str(error), "valid two-task discovery qualification"
                )
    elif qualification_binding is not None:
        mismatches["discovery_qualification"] = (
            qualification_binding, "absent during discovery"
        )
    if require_registry:
        registry = demo_goal_attempt_registry_path(
            task=expected_task,
            arm=expected_arm,
            training_seed=expected_training_seed,
        )
        if value.get("attempt_registry_path") != str(registry):
            mismatches["attempt_registry_path"] = (
                value.get("attempt_registry_path"), str(registry)
            )
        if value.get("attempt_registry_key") != demo_goal_attempt_registry_key(
            task=expected_task,
            arm=expected_arm,
            training_seed=expected_training_seed,
        ):
            mismatches["attempt_registry_key"] = (
                value.get("attempt_registry_key"), "canonical"
            )
        try:
            registry_authorization = registry / "authorization.json"
            if file_sha256(registry_authorization) != file_sha256(path):
                mismatches["registry.authorization_file"] = (
                    "different", "byte-identical"
                )
            status_name = (
                "completion_status.json"
                if expected_registry_status == "COMPLETED"
                else "attempt_status.json"
            )
            status = verify_self_hashed_json(
                registry / status_name, hash_key="status_sha256"
            )
            if expected_registry_status is not None and status.get(
                "status"
            ) != expected_registry_status:
                mismatches["registry.status"] = (
                    status.get("status"), expected_registry_status
                )
            if status.get("authorization_sha256") != value.get(
                "authorization_sha256"
            ):
                mismatches["registry.authorization_sha256"] = (
                    status.get("authorization_sha256"),
                    value.get("authorization_sha256"),
                )
        except (OSError, RuntimeError) as error:
            mismatches["registry"] = (str(error), "valid attempt registry")
    if mismatches:
        raise RuntimeError(
            f"Demo-Goal stage authorization mismatch: {mismatches}"
        )
    return value


def verify_demo_goal_training_completion(
    path: Path,
    *,
    run_dir: Path,
    checkpoint: Path,
    expected_method_name: str,
    expected_task: str,
    expected_training_seed: int,
    expected_method_lock_sha256: str,
    require_registry: bool = False,
):
    run_dir = Path(run_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    path = Path(path).resolve()
    expected_path = run_dir / "demo_goal_training_complete.json"
    if path != expected_path:
        raise RuntimeError(
            f"Demo-Goal completion must be {expected_path}, received {path}"
        )
    value = verify_self_hashed_json(path, hash_key="completion_sha256")
    expected_keys = {
        "schema_version",
        "status",
        "study_id",
        "task",
        "method_name",
        "training_seed",
        "split_seed",
        "run_dir_name",
        "checkpoint_relative_path",
        "source_git_commit",
        "source_tracked_tree_sha256",
        "runtime_environment_sha256",
        "dataset_file_sha256",
        "initial_shared_core_sha256",
        "initial_full_model_sha256",
        "initial_champion_predecessor_sha256",
        "method_lock_sha256",
        "smoke_attestation_sha256",
        "stage_authorization_sha256",
        "formal_slurm_job_id",
        "artifacts",
        "completion_sha256",
    }
    if expected_task == "tworoom":
        expected_keys.update({
            "tworoom_replay_attestation_sha256",
            "tworoom_replay_attestation_file_sha256",
        })
    expected = {
        "schema_version": DEMO_GOAL_COMPLETION_SCHEMA_VERSION,
        "status": "PASS",
        "study_id": DEMO_GOAL_STUDY_IDS[expected_task],
        "task": expected_task,
        "method_name": expected_method_name,
        "training_seed": int(expected_training_seed),
        "split_seed": 20260822,
        "run_dir_name": run_dir.name,
        "checkpoint_relative_path": "weights_epoch_10.pt",
        "method_lock_sha256": expected_method_lock_sha256,
    }
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if set(value) != expected_keys:
        mismatches["completion_keys"] = (
            sorted(value), sorted(expected_keys)
        )
    expected_artifacts = set(DEMO_GOAL_COMMON_COMPLETION_ARTIFACTS)
    if expected_task == "tworoom":
        expected_artifacts.add("demo_goal_tworoom_replay_attestation.json")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        mismatches["artifacts"] = (
            sorted(artifacts) if isinstance(artifacts, dict) else artifacts,
            sorted(expected_artifacts),
        )
    else:
        for relative, claimed in artifacts.items():
            artifact_path = run_dir / relative
            try:
                artifact_path.resolve(strict=True).relative_to(run_dir)
                inside_run = True
            except (OSError, ValueError):
                inside_run = False
            if (
                artifact_path.is_symlink()
                or not artifact_path.is_file()
                or not inside_run
            ):
                mismatches[f"artifacts.{relative}"] = (
                    str(artifact_path), "regular in-run file"
                )
                continue
            try:
                actual = file_sha256(artifact_path)
            except OSError as error:
                mismatches[f"artifacts.{relative}"] = (str(error), claimed)
                continue
            if claimed != actual:
                mismatches[f"artifacts.{relative}"] = (claimed, actual)
    try:
        authorization = verify_demo_goal_stage_authorization(
            run_dir / "demo_goal_stage_authorization.json",
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
            expected_task=expected_task,
            expected_arm=expected_method_name,
            expected_training_seed=expected_training_seed,
            expected_method_lock_sha256=expected_method_lock_sha256,
            require_registry=require_registry,
            expected_registry_status=("COMPLETED" if require_registry else None),
        )
        smoke = verify_demo_goal_smoke_attestation(
            run_dir / "demo_goal_smoke_attestation.json",
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
            expected_task=expected_task,
            expected_method_lock_sha256=expected_method_lock_sha256,
        )
        lock = verify_demo_goal_method_lock(
            run_dir / "demo_goal_method_lock.json",
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
        )
        chain = {
            "stage_authorization_sha256": authorization.get(
                "authorization_sha256"
            ),
            "smoke_attestation_sha256": smoke.get(
                "smoke_attestation_sha256"
            ),
            "method_lock_sha256": lock.get("method_lock_sha256"),
        }
        for key, wanted in chain.items():
            if value.get(key) != wanted:
                mismatches[key] = (value.get(key), wanted)
        if expected_task == "tworoom":
            replay_path = (
                run_dir / "demo_goal_tworoom_replay_attestation.json"
            )
            replay = verify_tworoom_replay_attestation(
                replay_path,
                expected_commit=value.get("source_git_commit"),
                expected_tree=value.get("source_tracked_tree_sha256"),
                expected_dataset_sha256=value.get("dataset_file_sha256"),
                expected_train_sha256=smoke.get("train_sha256"),
                expected_method_lock_sha256=expected_method_lock_sha256,
                replay_current_environment=True,
            )
            replay_bindings = {
                "tworoom_replay_attestation_sha256": replay.get(
                    "replay_attestation_sha256"
                ),
                "tworoom_replay_attestation_file_sha256": file_sha256(
                    replay_path
                ),
            }
            for key, wanted in replay_bindings.items():
                if value.get(key) != wanted:
                    mismatches[key] = (value.get(key), wanted)
            original_replay_path = Path(
                smoke["tworoom_replay_attestation_path"]
            )
            if file_sha256(original_replay_path) != file_sha256(replay_path):
                mismatches["tworoom_replay_archive"] = (
                    "different", "byte-identical to smoke-bound replay"
                )
    except (OSError, RuntimeError) as error:
        mismatches["embedded_chain"] = (str(error), "valid embedded chain")
    if checkpoint != run_dir / "weights_epoch_10.pt" or not checkpoint.is_file():
        mismatches["checkpoint"] = (
            str(checkpoint), str(run_dir / "weights_epoch_10.pt")
        )
    elif isinstance(artifacts, dict) and artifacts.get(
        "weights_epoch_10.pt"
    ) != file_sha256(checkpoint):
        mismatches["checkpoint_sha256"] = (
            artifacts.get("weights_epoch_10.pt"), file_sha256(checkpoint)
        )
    if require_registry:
        registry = demo_goal_attempt_registry_path(
            task=expected_task,
            arm=expected_method_name,
            training_seed=expected_training_seed,
        )
        try:
            registry_completion = (
                registry / "demo_goal_training_complete.json"
            )
            if file_sha256(registry_completion) != file_sha256(path):
                mismatches["registry.completion_file"] = (
                    "different", "byte-identical"
                )
            status = verify_self_hashed_json(
                registry / "completion_status.json",
                hash_key="status_sha256",
            )
            registry_expected = {
                "schema_version": DEMO_GOAL_COMPLETION_STATUS_SCHEMA_VERSION,
                "status": "COMPLETED",
                "study_id": DEMO_GOAL_STUDY_IDS[expected_task],
                "task": expected_task,
                "arm": expected_method_name,
                "training_seed": int(expected_training_seed),
                "run_dir_path": str(run_dir),
                "completion_sha256": value.get("completion_sha256"),
                "authorization_sha256": value.get(
                    "stage_authorization_sha256"
                ),
            }
            expected_status_keys = set(registry_expected) | {"status_sha256"}
            if set(status) != expected_status_keys:
                mismatches["registry.status.keys"] = (
                    sorted(status), sorted(expected_status_keys)
                )
            for key, wanted in registry_expected.items():
                if status.get(key) != wanted:
                    mismatches[f"registry.status.{key}"] = (
                        status.get(key), wanted
                    )
        except (OSError, RuntimeError) as error:
            mismatches["registry.completion"] = (
                str(error), "valid immutable completion registry"
            )
    if mismatches:
        raise RuntimeError(
            f"Demo-Goal training completion mismatch: {mismatches}"
        )
    return value
