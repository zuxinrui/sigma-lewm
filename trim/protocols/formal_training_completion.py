"""Self-hashed attestations for the transition-triangle formal workflow.

This module deliberately contains no training or evaluation imports.  It is
shared by the smoke audit, stage gate, post-training audit, and evaluator so a
PASS cannot be represented by an unaudited boolean or an unbound file path.
"""

from __future__ import annotations
from repro_paths import resolve_path as _resolve_artifact

import hashlib
import json
import os
from pathlib import Path


SMOKE_SCHEMA_VERSION = 2
STAGE_AUTHORIZATION_SCHEMA_VERSION = 2
EPISODE_SPLIT_SCHEMA_VERSION = 2
FORMAL_COMPLETION_SCHEMA_VERSION = 2
FORMAL_STUDY_IDS = {
    "pusht": "delta-triangle-pusht-v1",
    "tworoom": "simple-transition-triangle-tworoom-v1",
}
# Backwards-compatible name for the already-frozen PushT study.  New code must
# use ``formal_study_id(task)`` so a TwoRoom artifact can never resolve into the
# PushT attempt registry by omission.
FORMAL_STUDY_ID = FORMAL_STUDY_IDS["pusht"]
FORMAL_TASK_DATASETS = {
    "pusht": "pusht_expert_train.h5",
    "tworoom": "tworoom.h5",
}
FORMAL_ARMS = {"champion", "concat_mse", "delta_mse"}
FORMAL_TRAINING_SEEDS = (3072, 1, 2)
ATTEMPT_REGISTRY_ROOT = Path(_resolve_artifact('runs:lewm_formal_attempts'))

FORMAL_COMPLETION_ARTIFACTS = {
    "config.yaml",
    "config.json",
    "episode_split_manifest.json",
    "transition_triangle_execution_receipt.json",
    "training_metrics/metrics.csv",
    "weights_epoch_3.pt",
    "weights_epoch_6.pt",
    "weights_epoch_8.pt",
    "weights_epoch_10.pt",
    "formal_stage_authorization.json",
    "formal_smoke_attestation.json",
}


def canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def formal_study_id(task: str = "pusht") -> str:
    try:
        return FORMAL_STUDY_IDS[str(task)]
    except KeyError as error:
        raise ValueError(f"unsupported formal task: {task!r}") from error


def formal_phase(training_seed: int) -> str:
    training_seed = int(training_seed)
    if training_seed not in FORMAL_TRAINING_SEEDS:
        raise ValueError(
            f"unsupported formal training seed {training_seed}; "
            f"want one of {FORMAL_TRAINING_SEEDS}"
        )
    return "discovery" if training_seed == 3072 else "confirmation"


def formal_stage_name(*, task: str, arm: str, training_seed: int) -> str:
    task = str(task)
    formal_study_id(task)
    if arm not in FORMAL_ARMS:
        raise ValueError(f"unsupported formal arm: {arm!r}")
    phase = formal_phase(training_seed)
    base = f"{task}_{arm}"
    return base if phase == "discovery" else f"{base}_seed{int(training_seed)}"


def canonical_attempt_registry_path(
    *, task: str = "pusht", arm: str, training_seed: int = 3072
) -> Path:
    """Return the one immutable attempt slot for this preregistered study.

    The path intentionally excludes smoke hashes and git commits.  A cheap new
    smoke or a documentation-only commit therefore cannot create another
    seed-3072 performance attempt under the same frozen study identity.
    """
    task = str(task)
    study_id = formal_study_id(task)
    if arm not in FORMAL_ARMS:
        raise ValueError(f"unsupported formal arm: {arm!r}")
    formal_phase(training_seed)
    return (
        ATTEMPT_REGISTRY_ROOT
        / study_id
        / task
        / arm
        / f"seed{int(training_seed)}"
    )


def canonical_attempt_registry_key(
    *, task: str = "pusht", arm: str, training_seed: int = 3072
) -> str:
    canonical_attempt_registry_path(
        task=task, arm=arm, training_seed=training_seed
    )
    return f"{formal_study_id(task)}:{task}:{arm}:seed{int(training_seed)}"


def write_self_hashed_json(payload, path: Path, *, hash_key: str):
    """Atomically write ``payload`` after adding its canonical self hash."""
    path = Path(path)
    value = dict(payload)
    value.pop(hash_key, None)
    value[hash_key] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return value


def verify_self_hashed_json(path: Path, *, hash_key: str):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    claimed = value.get(hash_key)
    unsigned = dict(value)
    unsigned.pop(hash_key, None)
    actual = canonical_sha256(unsigned)
    if claimed != actual:
        raise RuntimeError(
            f"self-hash mismatch for {path}: claimed={claimed}, actual={actual}"
        )
    return value


def verify_pusht_selection_receipt(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
):
    """Verify and recompute the PushT winner that unlocks TwoRoom.

    A bare self-hash is not evidence: both comparison reports are reopened,
    their exact result inputs are rehashed, and the deterministic selection
    rule is rerun before the receipt is accepted.
    """
    path = Path(path).resolve()
    value = verify_self_hashed_json(path, hash_key="selection_sha256")
    expected = {
        "schema_version": 1,
        "status": "SELECTED",
        "study_id": "simple-transition-triangle-pusht-selection-v1",
        "task": "pusht",
        "training_seed": 3072,
        "selected_before_tworoom_authorization": True,
    }
    if expected_commit is not None:
        expected["source_git_commit"] = str(expected_commit)
    if expected_tree is not None:
        expected["source_tracked_tree_sha256"] = str(expected_tree)
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatches:
        raise RuntimeError(f"PushT selection receipt mismatch: {mismatches}")

    from scripts.compare_formal_eval_results import verify_comparison_report

    reports = {}
    for method, prefix in (("delta_mse", "delta"), ("concat_mse", "concat")):
        report_path = Path(value[f"{prefix}_comparison_path"]).resolve()
        report = verify_comparison_report(
            report_path,
            expected_task="pusht",
            expected_mode="discovery",
            expected_baseline_method="champion",
            expected_candidate_method=method,
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
        )
        binding_expected = {
            f"{prefix}_comparison_file_sha256": file_sha256(report_path),
            f"{prefix}_comparison_sha256": report["report_sha256"],
        }
        for key, wanted in binding_expected.items():
            if value.get(key) != wanted:
                mismatches[key] = (value.get(key), wanted)
        reports[method] = report
    if mismatches:
        raise RuntimeError(f"PushT selection receipt mismatch: {mismatches}")

    delta = reports["delta_mse"]
    concat = reports["concat_mse"]
    if not bool(delta.get("gate_pass")):
        raise RuntimeError("Delta-MSE failed its frozen PushT discovery gate")
    if delta.get("formal_manifest_sha256") != concat.get("formal_manifest_sha256"):
        raise RuntimeError("PushT simple arms use different formal manifests")
    if delta["inputs"]["baseline"] != concat["inputs"]["baseline"]:
        raise RuntimeError("PushT simple arms do not share the exact champion result")
    delta_primary = delta["protocols"][delta["primary_protocol"]]
    concat_primary = concat["protocols"][concat["primary_protocol"]]
    delta_successes = int(delta_primary["candidate_successes"])
    concat_successes = int(concat_primary["candidate_successes"])
    selected = (
        "concat_mse"
        if bool(concat.get("gate_pass")) and concat_successes > delta_successes
        else "delta_mse"
    )
    deterministic_expected = {
        "formal_manifest_sha256": delta["formal_manifest_sha256"],
        "selected_method": selected,
        "delta_gate_pass": True,
        "concat_gate_pass": bool(concat.get("gate_pass")),
        "delta_primary_successes": delta_successes,
        "concat_primary_successes": concat_successes,
        "num_primary_pairs_per_arm": int(delta_primary["num_episode_pairs"]),
        "tie_break": "delta_mse_has_fewer_training_only_inverse_parameters",
    }
    deterministic_mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in deterministic_expected.items()
        if value.get(key) != wanted
    }
    if deterministic_mismatches:
        raise RuntimeError(
            "PushT winner differs from deterministic recomputation: "
            f"{deterministic_mismatches}"
        )
    return value


def verify_smoke_attestation(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    expected_task: str | None = None,
):
    value = verify_self_hashed_json(path, hash_key="attestation_sha256")
    mismatches = {}
    expected = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "status": "PASS",
    }
    if expected_commit is not None:
        expected["source_git_commit"] = str(expected_commit)
    if expected_tree is not None:
        expected["source_tracked_tree_sha256"] = str(expected_tree)
    if expected_task is not None:
        formal_study_id(expected_task)
        # Schema-2 attestations created before task parameterization were all
        # PushT.  Preserve those exact artifacts while still failing closed if
        # one is ever supplied to the TwoRoom chain.
        expected["task"] = str(expected_task)
    for key, wanted in expected.items():
        observed = value.get(key)
        if key == "task" and observed is None:
            observed = "pusht"
        if observed != wanted:
            mismatches[key] = (observed, wanted)
    task = value.get("task", "pusht")
    try:
        expected_dataset = FORMAL_TASK_DATASETS[str(task)]
    except KeyError:
        mismatches["task"] = (task, sorted(FORMAL_TASK_DATASETS))
    else:
        dataset_name = value.get("dataset_name")
        if dataset_name is not None and dataset_name != expected_dataset:
            mismatches["dataset_name"] = (dataset_name, expected_dataset)
    observed_arms = set(value.get("arms", {}))
    observed_task = str(value.get("task", "pusht"))
    if observed_task == "pusht":
        wanted_arms = {"champion", "concat_mse", "delta_mse"}
    else:
        selected = value.get("selected_method")
        if selected not in {"concat_mse", "delta_mse"}:
            mismatches["selected_method"] = (
                selected, "concat_mse or delta_mse"
            )
            wanted_arms = {"champion"}
        else:
            wanted_arms = {"champion", selected}
    if observed_arms != wanted_arms:
        mismatches["arms"] = (sorted(observed_arms), sorted(wanted_arms))
    dataset_hash = value.get("dataset_file_sha256")
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        mismatches["dataset_file_sha256"] = (dataset_hash, "64-char SHA256")
    if not str(value.get("slurm_job_id", "")).isdigit():
        mismatches["slurm_job_id"] = (value.get("slurm_job_id"), "numeric job id")
    if mismatches:
        raise RuntimeError(f"smoke attestation is incompatible: {mismatches}")
    return value


def verify_stage_authorization(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_arm: str | None = None,
    expected_task: str = "pusht",
    expected_training_seed: int = 3072,
    require_registry: bool = False,
    expected_registry_status: str | None = None,
):
    value = verify_self_hashed_json(path, hash_key="authorization_sha256")
    expected_task = str(expected_task)
    expected_training_seed = int(expected_training_seed)
    phase = formal_phase(expected_training_seed)
    study_id = formal_study_id(expected_task)
    expected = {
        "schema_version": STAGE_AUTHORIZATION_SCHEMA_VERSION,
        "status": "AUTHORIZED",
        "task": expected_task,
        "training_seed": expected_training_seed,
        "split_seed": 20260822,
        "phase": phase,
    }
    if expected_task != "pusht" or value.get("study_id") is not None:
        expected["study_id"] = study_id
    if expected_commit is not None:
        expected["source_git_commit"] = str(expected_commit)
    if expected_arm is not None:
        expected["arm"] = str(expected_arm)
        expected["stage"] = formal_stage_name(
            task=expected_task,
            arm=str(expected_arm),
            training_seed=expected_training_seed,
        )
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatches:
        raise RuntimeError(f"formal stage authorization is incompatible: {mismatches}")
    if expected_task == "tworoom" or phase == "confirmation":
        selection_path_value = value.get("pusht_selection_path")
        if not isinstance(selection_path_value, str) or not selection_path_value:
            raise RuntimeError(
                f"{expected_task} {phase} authorization lacks its PushT "
                "selection path"
            )
        selection_path = Path(selection_path_value).resolve()
        selection = verify_pusht_selection_receipt(
            selection_path,
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
        )
        selection_mismatches = {}
        selection_expected = {
            "pusht_selection_file_sha256": file_sha256(selection_path),
            "pusht_selection_sha256": selection["selection_sha256"],
            "selected_method": selection["selected_method"],
        }
        for key, wanted in selection_expected.items():
            if value.get(key) != wanted:
                selection_mismatches[key] = (value.get(key), wanted)
        observed_arm = value.get("arm")
        if observed_arm not in {"champion", selection["selected_method"]}:
            selection_mismatches["arm"] = (
                observed_arm,
                f"champion or selected {selection['selected_method']}",
            )
        if selection_mismatches:
            raise RuntimeError(
                f"{expected_task} {phase} authorization selection binding mismatch: "
                f"{selection_mismatches}"
            )
    if phase == "confirmation":
        discovery_value = value.get("discovery_comparison_path")
        if not isinstance(discovery_value, str) or not discovery_value:
            raise RuntimeError(
                "confirmation authorization lacks its passed discovery comparison"
            )
        from scripts.compare_formal_eval_results import verify_comparison_report

        discovery_path = Path(discovery_value).resolve()
        discovery = verify_comparison_report(
            discovery_path,
            expected_task=expected_task,
            expected_mode="discovery",
            expected_baseline_method="champion",
            expected_candidate_method=selection["selected_method"],
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
        )
        discovery_expected = {
            "discovery_comparison_file_sha256": file_sha256(discovery_path),
            "discovery_comparison_sha256": discovery["report_sha256"],
            "discovery_gate_pass": True,
        }
        discovery_mismatches = {
            key: (value.get(key), wanted)
            for key, wanted in discovery_expected.items()
            if value.get(key) != wanted
        }
        if not bool(discovery.get("gate_pass")):
            discovery_mismatches["recomputed_discovery_gate_pass"] = (
                False, True
            )
        if discovery_mismatches:
            raise RuntimeError(
                "confirmation predecessor gate mismatch: "
                f"{discovery_mismatches}"
            )
    if (
        expected_task == "pusht"
        and phase == "discovery"
        and value.get("arm") == "concat_mse"
    ):
        delta_report_value = value.get("pusht_delta_comparison_path")
        if not isinstance(delta_report_value, str) or not delta_report_value:
            raise RuntimeError(
                "PushT concat authorization lacks the passed Delta-MSE comparison"
            )
        from scripts.compare_formal_eval_results import verify_comparison_report

        delta_report_path = Path(delta_report_value).resolve()
        delta_report = verify_comparison_report(
            delta_report_path,
            expected_task="pusht",
            expected_mode="discovery",
            expected_baseline_method="champion",
            expected_candidate_method="delta_mse",
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
        )
        delta_mismatches = {}
        delta_expected = {
            "pusht_delta_comparison_file_sha256": file_sha256(delta_report_path),
            "pusht_delta_comparison_sha256": delta_report["report_sha256"],
            "pusht_delta_gate_pass": True,
        }
        for key, wanted in delta_expected.items():
            observed = value.get(key)
            if observed != wanted:
                delta_mismatches[key] = (observed, wanted)
        if not bool(delta_report.get("gate_pass")):
            delta_mismatches["recomputed_delta_gate_pass"] = (False, True)
        if delta_mismatches:
            raise RuntimeError(
                "PushT concat predecessor gate mismatch: "
                f"{delta_mismatches}"
            )
    if require_registry:
        arm = str(value.get("arm"))
        training_seed = int(value.get("training_seed"))
        registry_path = canonical_attempt_registry_path(
            task=expected_task, arm=arm, training_seed=training_seed
        )
        expected_key = canonical_attempt_registry_key(
            task=expected_task, arm=arm, training_seed=training_seed
        )
        registry_mismatches = {}
        if value.get("study_id") != study_id:
            registry_mismatches["study_id"] = (
                value.get("study_id"), study_id
            )
        if value.get("attempt_registry_key") != expected_key:
            registry_mismatches["attempt_registry_key"] = (
                value.get("attempt_registry_key"), expected_key
            )
        if value.get("attempt_registry_path") != str(registry_path):
            registry_mismatches["attempt_registry_path"] = (
                value.get("attempt_registry_path"), str(registry_path)
            )
        registry_authorization = registry_path / "authorization.json"
        if not registry_authorization.is_file():
            registry_mismatches["registry.authorization"] = (
                "<missing>", str(registry_authorization)
            )
        elif file_sha256(registry_authorization) != file_sha256(path):
            registry_mismatches["registry.authorization"] = (
                file_sha256(registry_authorization), file_sha256(path)
            )
        status_path = registry_path / "attempt_status.json"
        try:
            status = verify_self_hashed_json(
                status_path, hash_key="status_sha256"
            )
        except (FileNotFoundError, RuntimeError) as error:
            registry_mismatches["registry.status"] = (
                str(error), "valid self-hashed status"
            )
            status = {}
        if status.get("authorization_sha256") != value.get(
            "authorization_sha256"
        ):
            registry_mismatches["registry.status.authorization_sha256"] = (
                status.get("authorization_sha256"),
                value.get("authorization_sha256"),
            )
        status_expected = {
            "schema_version": 1,
            "study_id": study_id,
            "task": expected_task,
            "arm": arm,
            "training_seed": training_seed,
        }
        for key, wanted in status_expected.items():
            observed = status.get(key)
            if key == "task" and observed is None and expected_task == "pusht":
                observed = "pusht"
            if observed != wanted:
                registry_mismatches[f"registry.status.{key}"] = (
                    observed, wanted
                )
        if expected_registry_status is not None and status.get(
            "status"
        ) != expected_registry_status:
            registry_mismatches["registry.status.status"] = (
                status.get("status"), expected_registry_status
            )
        if registry_mismatches:
            raise RuntimeError(
                "formal attempt registry is incompatible: "
                f"{registry_mismatches}"
            )
    return value


def verify_formal_training_completion(
    path: Path,
    *,
    run_dir: Path,
    checkpoint: Path,
    expected_method_name: str | None = None,
    expected_task: str | None = None,
    expected_training_seed: int = 3072,
    require_registry: bool = False,
):
    """Verify a PASS sidecar and every run-local artifact it binds."""
    run_dir = Path(run_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    path = Path(path).resolve()
    expected_path = run_dir / "formal_training_complete.json"
    if path != expected_path:
        raise RuntimeError(
            f"formal completion must be {expected_path}, received {path}"
        )
    value = verify_self_hashed_json(path, hash_key="completion_sha256")
    expected = {
        "schema_version": FORMAL_COMPLETION_SCHEMA_VERSION,
        "status": "PASS",
        "run_dir_name": run_dir.name,
        "checkpoint_relative_path": "weights_epoch_10.pt",
        "training_seed": int(expected_training_seed),
        "split_seed": 20260822,
    }
    if expected_method_name is not None:
        expected["method_name"] = str(expected_method_name)
    if expected_task is not None:
        expected["task"] = str(expected_task)
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != FORMAL_COMPLETION_ARTIFACTS:
        mismatches["artifacts"] = (
            sorted(artifacts) if isinstance(artifacts, dict) else artifacts,
            sorted(FORMAL_COMPLETION_ARTIFACTS),
        )
    else:
        for relative, claimed in artifacts.items():
            actual = file_sha256(run_dir / relative)
            if claimed != actual:
                mismatches[f"artifacts.{relative}"] = (claimed, actual)
    authorization_path = run_dir / "formal_stage_authorization.json"
    smoke_path = run_dir / "formal_smoke_attestation.json"
    try:
        completion_task = str(expected_task or value.get("task", "pusht"))
        authorization = verify_stage_authorization(
            authorization_path,
            expected_commit=value.get("source_git_commit"),
            expected_arm=expected_method_name,
            expected_task=completion_task,
            expected_training_seed=int(expected_training_seed),
            require_registry=require_registry,
            expected_registry_status=("COMPLETED" if require_registry else None),
        )
        smoke = verify_smoke_attestation(
            smoke_path,
            expected_commit=value.get("source_git_commit"),
            expected_tree=value.get("source_tracked_tree_sha256"),
            expected_task=completion_task,
        )
    except (FileNotFoundError, RuntimeError) as error:
        mismatches["embedded_stage_chain"] = (str(error), "valid embedded chain")
        authorization = {}
        smoke = {}
    embedded_expected = {
        "stage_authorization_sha256": authorization.get("authorization_sha256"),
        "smoke_attestation_sha256": smoke.get("attestation_sha256"),
        "formal_slurm_job_id": authorization.get("slurm_job_id"),
    }
    for key, wanted in embedded_expected.items():
        if value.get(key) != wanted:
            mismatches[key] = (value.get(key), wanted)
    if authorization.get("smoke_attestation_sha256") != smoke.get(
        "attestation_sha256"
    ):
        mismatches["authorization.smoke_attestation_sha256"] = (
            authorization.get("smoke_attestation_sha256"),
            smoke.get("attestation_sha256"),
        )
    if authorization.get("runtime_environment_sha256") != value.get(
        "runtime_environment_sha256"
    ):
        mismatches["authorization.runtime_environment_sha256"] = (
            authorization.get("runtime_environment_sha256"),
            value.get("runtime_environment_sha256"),
        )
    if authorization.get("source_tracked_tree_sha256") != value.get(
        "source_tracked_tree_sha256"
    ):
        mismatches["authorization.source_tracked_tree_sha256"] = (
            authorization.get("source_tracked_tree_sha256"),
            value.get("source_tracked_tree_sha256"),
        )
    if authorization.get("dataset_fingerprint_sha256") != smoke.get(
        "dataset_fingerprint_sha256"
    ):
        mismatches["authorization.dataset_fingerprint_sha256"] = (
            authorization.get("dataset_fingerprint_sha256"),
            smoke.get("dataset_fingerprint_sha256"),
        )
    if authorization.get("dataset_file_sha256") != smoke.get(
        "dataset_file_sha256"
    ):
        mismatches["authorization.dataset_file_sha256"] = (
            authorization.get("dataset_file_sha256"),
            smoke.get("dataset_file_sha256"),
        )
    if smoke.get("runtime_environment_sha256") != value.get(
        "runtime_environment_sha256"
    ):
        mismatches["smoke.runtime_environment_sha256"] = (
            smoke.get("runtime_environment_sha256"),
            value.get("runtime_environment_sha256"),
        )
    try:
        split_manifest = json.loads(
            (run_dir / "episode_split_manifest.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as error:
        mismatches["episode_split_manifest"] = (
            str(error), "valid JSON training manifest"
        )
        split_manifest = {}
    dataset_file_sha256 = value.get("dataset_file_sha256")
    if not is_sha256(dataset_file_sha256):
        mismatches["dataset_file_sha256"] = (
            dataset_file_sha256, "lowercase SHA256"
        )
    dataset_hash_bindings = {
        "episode_split_manifest.dataset_file_sha256": split_manifest.get(
            "dataset_file_sha256"
        ),
        "authorization.dataset_file_sha256": authorization.get(
            "dataset_file_sha256"
        ),
        "smoke.dataset_file_sha256": smoke.get("dataset_file_sha256"),
    }
    for key, observed in dataset_hash_bindings.items():
        if observed != dataset_file_sha256:
            mismatches[key] = (observed, dataset_file_sha256)
    prior = authorization.get("prior_completion_sha256")
    if expected_method_name == "champion" and prior is not None:
        mismatches["authorization.prior_completion_sha256"] = (prior, None)
    if expected_method_name in {"concat_mse", "delta_mse"} and not prior:
        mismatches["authorization.prior_completion_sha256"] = (
            prior, "champion completion SHA256"
        )
    if (
        require_registry
        and expected_method_name in {"concat_mse", "delta_mse"}
        and prior
    ):
        champion_registry = canonical_attempt_registry_path(
            task=completion_task,
            arm="champion", training_seed=authorization["training_seed"]
        )
        champion_completion_path = (
            champion_registry / "formal_training_complete.json"
        )
        try:
            champion_completion = verify_self_hashed_json(
                champion_completion_path, hash_key="completion_sha256"
            )
        except (FileNotFoundError, RuntimeError) as error:
            mismatches["registry.prior_completion"] = (
                str(error), "unique champion completion"
            )
        else:
            champion_expected = {
                "status": "PASS",
                "method_name": "champion",
                "task": completion_task,
                "training_seed": authorization["training_seed"],
                "completion_sha256": prior,
            }
            for key, wanted in champion_expected.items():
                if champion_completion.get(key) != wanted:
                    mismatches[f"registry.prior_completion.{key}"] = (
                        champion_completion.get(key), wanted
                    )
    if checkpoint != run_dir / "weights_epoch_10.pt":
        mismatches["checkpoint_path"] = (
            str(checkpoint), str(run_dir / "weights_epoch_10.pt")
        )
    elif checkpoint.is_file():
        expected_checkpoint_hash = artifacts.get("weights_epoch_10.pt") if isinstance(artifacts, dict) else None
        actual_checkpoint_hash = file_sha256(checkpoint)
        if expected_checkpoint_hash != actual_checkpoint_hash:
            mismatches["checkpoint_sha256"] = (
                expected_checkpoint_hash, actual_checkpoint_hash
            )
    if require_registry and authorization:
        registry_path = canonical_attempt_registry_path(
            task=completion_task,
            arm=authorization["arm"],
            training_seed=authorization["training_seed"],
        )
        registry_completion = registry_path / "formal_training_complete.json"
        if not registry_completion.is_file():
            mismatches["registry.completion"] = (
                "<missing>", str(registry_completion)
            )
        elif file_sha256(registry_completion) != file_sha256(path):
            mismatches["registry.completion"] = (
                file_sha256(registry_completion), file_sha256(path)
            )
        try:
            registry_status = verify_self_hashed_json(
                registry_path / "attempt_status.json",
                hash_key="status_sha256",
            )
        except (FileNotFoundError, RuntimeError) as error:
            mismatches["registry.completion_status"] = (
                str(error), "valid COMPLETED status"
            )
        else:
            if registry_status.get("completion_sha256") != value.get(
                "completion_sha256"
            ):
                mismatches["registry.status.completion_sha256"] = (
                    registry_status.get("completion_sha256"),
                    value.get("completion_sha256"),
                )
    if mismatches:
        raise RuntimeError(f"formal training completion is incompatible: {mismatches}")
    return value
