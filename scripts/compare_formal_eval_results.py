#!/usr/bin/env python3
"""Paired, manifest-locked comparison for transition-triangle results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slewm.protocols.formal_method_specs import FORMAL_METHOD_SIGNATURES
from slewm.protocols.demo_goal_formal_specs import (
    DEMO_GOAL_CONFIRMATION_GATES,
    DEMO_GOAL_DISCOVERY_GATES,
    DEMO_GOAL_METHOD_SIGNATURES,
    DEMO_GOAL_STUDY_IDS,
)
from slewm.evaluation.protocol import (
    FORMAL_EVAL_MANIFEST_SCHEMA_VERSION,
    FORMAL_EVAL_RESULT_SCHEMA_VERSION,
    FORMAL_TASK_PROTOCOL_NAMES,
    _validate_binary_episode_successes,
    _validate_formal_bank_structure,
    _validate_task_protocols,
)
from slewm.training.utilities import (
    git_tracked_tree_sha256,
    runtime_environment_compatibility_fingerprint,
)


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_bindings(paths):
    rows = []
    for supplied in paths:
        path = Path(supplied).resolve()
        payload = _load_verified(path)
        rows.append({
            "path": str(path),
            "file_sha256": _file_sha256(path),
            "result_sha256": payload["result_sha256"],
        })
    return rows


def seal_comparison_report(
    report,
    *,
    baseline_paths,
    candidate_paths,
    repo: Path | None = None,
):
    """Bind a comparison to its exact inputs and auditable source tree."""
    repo = (
        Path(__file__).resolve().parents[1]
        if repo is None else Path(repo).resolve()
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("formal comparison sealing requires a clean worktree")
    value = dict(report)
    value["schema_version"] = 2
    value["comparison_schema_version"] = int(report["schema_version"])
    value["source_git_commit"] = commit
    value["source_tracked_tree_sha256"] = git_tracked_tree_sha256(repo)
    value["inputs"] = {
        "baseline": _input_bindings(baseline_paths),
        "candidate": _input_bindings(candidate_paths),
    }
    value["report_sha256"] = _canonical_sha256(value)
    return value


def verify_comparison_report(
    path: Path,
    *,
    expected_task: str | None = None,
    expected_mode: str | None = None,
    expected_baseline_method: str | None = None,
    expected_candidate_method: str | None = None,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
):
    """Reopen every result and fully recompute a sealed comparison report."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    claimed = payload.get("report_sha256")
    unsigned = dict(payload)
    unsigned.pop("report_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise RuntimeError(f"comparison report self-hash mismatch: {path}")
    expected = {
        "schema_version": 2,
        "comparison_schema_version": 1,
    }
    optional = {
        "task": expected_task,
        "mode": expected_mode,
        "baseline_method": expected_baseline_method,
        "candidate_method": expected_candidate_method,
        "source_git_commit": expected_commit,
        "source_tracked_tree_sha256": expected_tree,
    }
    expected.update({key: value for key, value in optional.items() if value is not None})
    mismatches = {
        key: (payload.get(key), wanted)
        for key, wanted in expected.items()
        if payload.get(key) != wanted
    }
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"baseline", "candidate"}:
        mismatches["inputs"] = (inputs, "baseline/candidate bindings")
    if mismatches:
        raise RuntimeError(f"comparison report contract mismatch: {mismatches}")

    resolved = {}
    for role in ("baseline", "candidate"):
        rows = inputs[role]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"comparison report has no {role} inputs")
        resolved[role] = []
        for row in rows:
            result_path = Path(row["path"]).resolve()
            result = _load_verified(result_path)
            observed = {
                "path": str(result_path),
                "file_sha256": _file_sha256(result_path),
                "result_sha256": result["result_sha256"],
            }
            if row != observed:
                raise RuntimeError(
                    f"comparison {role} input binding changed: {row} != {observed}"
                )
            resolved[role].append(result_path)
    recomputed = compare(
        resolved["baseline"],
        resolved["candidate"],
        payload["mode"],
        baseline_method=payload["baseline_method"],
        candidate_method=payload["candidate_method"],
    )
    for key, wanted in recomputed.items():
        observed = (
            payload.get("comparison_schema_version")
            if key == "schema_version" else payload.get(key)
        )
        if observed != wanted:
            raise RuntimeError(
                f"comparison report differs from recomputation: {key}"
            )
    return payload


def _load_verified(path: Path):
    payload = json.loads(Path(path).read_text())
    claimed = payload.get("result_sha256")
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    actual = _canonical_sha256(unsigned)
    if claimed != actual:
        raise RuntimeError(f"result hash mismatch for {path}: {claimed} != {actual}")
    _verify_result_against_manifest(payload)
    return payload


def _verify_result_against_manifest(payload):
    if int(payload.get("schema_version", -1)) != FORMAL_EVAL_RESULT_SCHEMA_VERSION:
        raise RuntimeError(
            "formal transition-triangle study requires result schema_version="
            f"{FORMAL_EVAL_RESULT_SCHEMA_VERSION}"
        )
    manifest_path = Path(payload["formal_manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"formal manifest referenced by result is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("schema_version", -1)) != FORMAL_EVAL_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "formal transition-triangle study requires manifest schema_version="
            f"{FORMAL_EVAL_MANIFEST_SCHEMA_VERSION}"
        )
    claimed_manifest = manifest.get("manifest_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    if claimed_manifest != _canonical_sha256(unsigned_manifest):
        raise RuntimeError("formal manifest content hash is invalid")
    task = payload.get("task")
    manifest_task = manifest.get("task")
    if task != manifest_task:
        raise RuntimeError(
            f"formal result task differs from its manifest: {task!r} != {manifest_task!r}"
        )
    if manifest.get("study_id") is not None:
        if payload.get("study_id") != manifest.get("study_id"):
            raise RuntimeError("Demo-Goal result study_id differs from its manifest")
        if payload.get("method_lock_sha256") != manifest.get(
            "method_lock_sha256"
        ):
            raise RuntimeError("Demo-Goal result method lock differs from its manifest")
    _validate_task_protocols(manifest_task, manifest.get("protocols"))
    _validate_formal_bank_structure(manifest)
    if payload.get("source_git_commit") != manifest.get("created_by_git_commit"):
        raise RuntimeError(
            "evaluation result source commit differs from the formal manifest"
        )
    # Results carry the same source tree through their revalidated training
    # contract. Accept that spelling when no explicit evaluation-tree field is
    # present, but never accept a value that differs from the manifest.
    result_source_tree = payload.get(
        "source_tracked_tree_sha256",
        payload.get("training_source_tracked_tree_sha256"),
    )
    if result_source_tree != manifest.get("source_tracked_tree_sha256"):
        raise RuntimeError(
            "evaluation result source tree differs from the formal manifest"
        )
    if claimed_manifest != payload["formal_manifest_sha256"]:
        raise RuntimeError("result references the wrong formal manifest hash")
    if payload["dataset_fingerprint_sha256"] != manifest["dataset_fingerprint_sha256"]:
        raise RuntimeError("result dataset differs from the formal manifest")
    if payload.get("dataset_file_sha256") != manifest.get("dataset_file_sha256"):
        raise RuntimeError(
            "result dataset full-file SHA256 differs from the formal manifest"
        )
    if payload["training_source_git_commit"] != manifest["created_by_git_commit"]:
        raise RuntimeError("training source commit differs from the formal manifest")
    if (
        payload["training_source_tracked_tree_sha256"]
        != manifest["source_tracked_tree_sha256"]
    ):
        raise RuntimeError("training source content differs from the formal manifest")
    if (
        payload["training_runtime_environment_sha256"]
        != manifest["runtime_environment_sha256"]
    ):
        raise RuntimeError("training runtime environment differs from the formal manifest")
    evaluation_runtime = payload.get("evaluation_runtime_environment")
    if not isinstance(evaluation_runtime, dict):
        raise RuntimeError("formal result lacks its full evaluation runtime environment")
    if payload.get("evaluation_runtime_environment_sha256") != _canonical_sha256(
        evaluation_runtime
    ):
        raise RuntimeError("evaluation runtime environment self-hash is invalid")
    evaluation_compatibility_sha256 = _canonical_sha256(
        runtime_environment_compatibility_fingerprint(evaluation_runtime)
    )
    if payload.get(
        "evaluation_runtime_compatibility_sha256"
    ) != evaluation_compatibility_sha256:
        raise RuntimeError("evaluation runtime compatibility self-hash is invalid")
    if evaluation_compatibility_sha256 != manifest.get(
        "runtime_compatibility_sha256"
    ):
        raise RuntimeError(
            "evaluation runtime is not inference-compatible with the formal manifest"
        )
    smoke_sha256 = payload.get("formal_training_smoke_attestation_sha256")
    if not isinstance(smoke_sha256, str) or not smoke_sha256:
        raise RuntimeError("formal result lacks its training smoke attestation hash")
    completion_sha256 = payload.get("formal_training_completion_sha256")
    if not isinstance(completion_sha256, str) or not completion_sha256:
        raise RuntimeError("formal result lacks its training completion hash")

    checkpoint_value = payload.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise RuntimeError("formal result lacks an explicit checkpoint path")
    checkpoint = Path(checkpoint_value).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"formal result checkpoint is missing: {checkpoint}"
        )
    training_manifest_value = payload.get("training_manifest_path")
    if not isinstance(training_manifest_value, str) or not training_manifest_value:
        raise RuntimeError("formal result lacks its training manifest path")
    training_manifest_path = Path(training_manifest_value).resolve()

    # Local import avoids making the lightweight comparison module initialize
    # the evaluator stack until a formal result actually needs verification.
    # `_training_contract` reopens config, split manifest, completion receipt,
    # smoke/stage chain and the immutable attempt registry.
    from slewm.evaluation.manifest import _checkpoint_sha256, _training_contract

    observed_checkpoint_sha256 = _checkpoint_sha256(checkpoint)
    if payload.get("checkpoint_sha256") != observed_checkpoint_sha256:
        raise RuntimeError(
            "formal result checkpoint SHA256 differs from the checkpoint bytes: "
            f"claimed={payload.get('checkpoint_sha256')!r}, "
            f"actual={observed_checkpoint_sha256}"
        )
    recomputed_training_contract = _training_contract(
        checkpoint, manifest, training_manifest_path
    )
    training_contract_mismatches = {
        key: (payload.get(key, "<missing>"), expected)
        for key, expected in recomputed_training_contract.items()
        if payload.get(key, "<missing>") != expected
    }
    if training_contract_mismatches:
        raise RuntimeError(
            "formal result differs from its recomputed training contract: "
            f"{training_contract_mismatches}"
        )
    expected_bank_count = 3 if payload["bank_set"] == "discovery" else 5
    if int(manifest["num_eval_per_bank"]) != 50:
        raise RuntimeError("formal comparison requires exactly 50 episodes per bank")
    if len(manifest["banks"][payload["bank_set"]]) != expected_bank_count:
        raise RuntimeError(
            f"formal {payload['bank_set']} comparison requires {expected_bank_count} banks"
        )

    protocols = {item["name"]: item for item in manifest["protocols"]}
    banks = {
        int(item["bank_index"]): item
        for item in manifest["banks"][payload["bank_set"]]
    }
    expected_record_keys = {
        (protocol_name, bank_index)
        for protocol_name in protocols for bank_index in banks
    }
    actual_record_keys = [
        (record["protocol"], int(record["bank_index"]))
        for record in payload["records"]
    ]
    if len(actual_record_keys) != len(set(actual_record_keys)):
        raise RuntimeError("formal result contains duplicate protocol/bank records")
    if set(actual_record_keys) != expected_record_keys:
        raise RuntimeError("formal result does not cover every frozen protocol/bank")
    for record in payload["records"]:
        protocol = protocols.get(record["protocol"])
        bank = banks.get(int(record["bank_index"]))
        if protocol is None or bank is None:
            raise RuntimeError("result contains a protocol/bank absent from its manifest")
        expected = {
            "bank_set": payload["bank_set"],
            "bank_seed": int(bank["seed"]),
            "horizon": int(protocol["horizon"]),
            "receding_horizon": int(protocol["receding_horizon"]),
            "action_block": int(protocol["action_block"]),
            "eval_budget": int(protocol["eval_budget"]),
            "row_indices": bank["row_indices"],
            "episode_ids": bank["episode_ids"],
            "start_steps": bank["start_steps"],
            "goal_steps": bank["goal_steps"],
        }
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                raise RuntimeError(
                    f"result record differs from formal manifest: {record['protocol']} "
                    f"bank={record['bank_index']} field={field}"
                )
        outcomes = _validate_binary_episode_successes(
            record.get("episode_successes"),
            expected_count=int(manifest["num_eval_per_bank"]),
            context=(
                f"result episode_successes protocol={record['protocol']} "
                f"bank={record['bank_index']}"
            ),
        )
        expected_rate = float(np.mean(outcomes) * 100.0)
        if not math.isclose(float(record["success_rate"]), expected_rate, abs_tol=1e-9):
            raise RuntimeError("result success_rate disagrees with episode outcomes")


def _record_map(payload):
    records = {
        (record["protocol"], record["bank_set"], int(record["bank_index"])): record
        for record in payload["records"]
    }
    if len(records) != len(payload["records"]):
        raise RuntimeError("formal result contains duplicate protocol/bank records")
    return records


def _paired_seed_summary(
    baseline, candidate, *, baseline_method="champion", candidate_method="delta_mse"
):
    if baseline["formal_manifest_sha256"] != candidate["formal_manifest_sha256"]:
        raise RuntimeError("candidate and baseline use different formal manifests")
    if baseline["task"] != candidate["task"]:
        raise RuntimeError("candidate and baseline tasks differ")
    matched_training_fields = (
        "source_git_commit",
        "training_source_git_commit",
        "training_source_tracked_tree_sha256",
        "training_runtime_environment_sha256",
        "training_seed",
        "split_seed",
        "train_split_sha256",
        "validation_split_sha256",
        "dataset_fingerprint_sha256",
        "dataset_file_sha256",
        "training_dataset_file_sha256",
        "training_normalizer_sha256",
        "initial_shared_core_sha256",
        "training_config_comparison_sha256",
        "training_recipe_sha256",
        "formal_training_smoke_attestation_sha256",
        "inference_config_sha256",
        "initial_champion_predecessor_sha256",
        "method_lock_sha256",
    )
    for field in matched_training_fields:
        if baseline.get(field) != candidate.get(field):
            raise RuntimeError(
                f"candidate and baseline training contracts differ: {field}"
            )
    baseline_study_id = baseline.get("study_id")
    if candidate.get("study_id") != baseline_study_id:
        raise RuntimeError("candidate and baseline belong to different studies")
    signatures = (
        DEMO_GOAL_METHOD_SIGNATURES
        if baseline_study_id in set(DEMO_GOAL_STUDY_IDS.values())
        else FORMAL_METHOD_SIGNATURES
    )
    try:
        expected_baseline = signatures[baseline_method]
        expected_candidate = signatures[candidate_method]
    except KeyError as error:
        raise ValueError(f"unknown formal method {error.args[0]!r}") from error
    if baseline.get("method_signature") != expected_baseline:
        raise RuntimeError(
            f"baseline is not frozen {baseline_method}: {baseline.get('method_signature')}"
        )
    if candidate.get("method_signature") != expected_candidate:
        raise RuntimeError(
            f"candidate is not frozen {candidate_method}: {candidate.get('method_signature')}"
        )
    if {baseline_method, candidate_method} <= {"champion", "ibr"}:
        if baseline.get("initial_full_model_sha256") != candidate.get(
            "initial_full_model_sha256"
        ):
            raise RuntimeError(
                "candidate and baseline training contracts differ: "
                "initial_full_model_sha256"
            )
    baseline_records = _record_map(baseline)
    candidate_records = _record_map(candidate)
    if baseline_records.keys() != candidate_records.keys():
        raise RuntimeError("candidate and baseline evaluated different protocol banks")

    protocols = {}
    for key in sorted(baseline_records):
        baseline = baseline_records[key]
        proposed = candidate_records[key]
        for field in (
            "bank_seed",
            "horizon",
            "receding_horizon",
            "action_block",
            "eval_budget",
            "row_indices",
            "episode_ids",
            "start_steps",
            "goal_steps",
        ):
            if baseline[field] != proposed[field]:
                raise RuntimeError(f"pairing mismatch for {key}: {field}")
        expected_count = len(baseline["episode_ids"])
        y0 = _validate_binary_episode_successes(
            baseline.get("episode_successes"),
            expected_count=expected_count,
            context=f"baseline episode_successes for {key}",
        )
        y1 = _validate_binary_episode_successes(
            proposed.get("episode_successes"),
            expected_count=expected_count,
            context=f"candidate episode_successes for {key}",
        )
        protocol = key[0]
        entry = protocols.setdefault(protocol, {"y0": [], "y1": [], "banks": []})
        entry["y0"].append(y0)
        entry["y1"].append(y1)
        entry["banks"].append({
            "bank_index": key[2],
            "num_episode_pairs": int(y0.size),
            "baseline_successes": int(y0.sum()),
            "candidate_successes": int(y1.sum()),
            "success_delta_count": int(y1.sum() - y0.sum()),
            "baseline_rate": float(y0.mean() * 100.0),
            "candidate_rate": float(y1.mean() * 100.0),
            "delta_pp": float((y1.mean() - y0.mean()) * 100.0),
            "rescues": int(((y0 == 0) & (y1 == 1)).sum()),
            "harms": int(((y0 == 1) & (y1 == 0)).sum()),
        })
    for entry in protocols.values():
        entry["y0"] = np.concatenate(entry["y0"])
        entry["y1"] = np.concatenate(entry["y1"])
    return protocols


def _mcnemar_exact_p(rescues: int, harms: int) -> float:
    discordant = int(rescues + harms)
    if discordant == 0:
        return 1.0
    tail = min(rescues, harms)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return float(min(1.0, 2.0 * probability))


def _two_way_cluster_bootstrap(seed_pairs, samples=10000, seed=20260822):
    """Resample training seeds and the shared episode columns independently."""
    rng = np.random.default_rng(int(seed))
    y0 = np.stack([pair[0] for pair in seed_pairs], axis=0)
    y1 = np.stack([pair[1] for pair in seed_pairs], axis=0)
    num_seeds, num_episodes = y0.shape
    deltas = np.empty(int(samples), dtype=np.float64)
    for draw in range(int(samples)):
        chosen_seeds = rng.integers(0, num_seeds, size=num_seeds)
        chosen_episodes = rng.integers(0, num_episodes, size=num_episodes)
        delta = y1[np.ix_(chosen_seeds, chosen_episodes)] \
            - y0[np.ix_(chosen_seeds, chosen_episodes)]
        deltas[draw] = delta.mean() * 100.0
    return [float(x) for x in np.percentile(deltas, [2.5, 97.5])]


def _delta_at_least_pp(report, threshold_pp: int) -> bool:
    """Compare a percentage-point threshold using exact integer counts."""
    return (
        int(report["success_delta_count"]) * 100
        >= int(threshold_pp) * int(report["num_episode_pairs"])
    )


def _demo_goal_gate_checks(*, task, mode, primary_report, secondary_report):
    if mode == "discovery":
        thresholds = dict(DEMO_GOAL_DISCOVERY_GATES[task])
        if (
            int(primary_report["num_episode_pairs"])
            != thresholds["primary_pairs"]
            or int(secondary_report["num_episode_pairs"])
            != thresholds["secondary_pairs"]
        ):
            raise RuntimeError(
                "Demo-Goal discovery pair counts differ from the frozen gate"
            )
        per_seed_banks = primary_report["per_seed_banks"]
        if len(per_seed_banks) != 1:
            raise RuntimeError(
                "Demo-Goal discovery gate requires exactly one training seed"
            )
        checks = {
            "primary_delta": int(primary_report["success_delta_count"])
            >= thresholds["primary_min_delta_count"],
            "primary_positive_banks": sum(
                int(bank["success_delta_count"]) > 0
                for bank in per_seed_banks[0]
            ) >= thresholds["primary_min_positive_banks"],
            "secondary_noninferiority": int(
                secondary_report["success_delta_count"]
            ) >= thresholds["secondary_min_delta_count"],
        }
        return checks, thresholds

    thresholds = dict(DEMO_GOAL_CONFIRMATION_GATES)
    if (
        int(primary_report["num_episode_pairs"])
        != thresholds["primary_pairs"]
        or int(secondary_report["num_episode_pairs"])
        != thresholds["secondary_pairs"]
    ):
        raise RuntimeError(
            "Demo-Goal confirmation pair counts differ from the frozen gate"
        )
    seed_counts = primary_report["seed_success_delta_counts"]
    if len(seed_counts) != 3:
        raise RuntimeError(
            "Demo-Goal confirmation gate requires exactly three training seeds"
        )
    checks = {
        "primary_delta": int(primary_report["success_delta_count"])
        >= thresholds["primary_min_delta_count"],
        "all_training_seeds_positive": all(
            int(delta) > 0 for delta in seed_counts
        ),
        "primary_ci_lower_positive": float(
            primary_report["two_way_cluster_bootstrap_95ci_pp"][0]
        ) > 0,
        "secondary_noninferiority": int(
            secondary_report["success_delta_count"]
        ) >= thresholds["secondary_min_delta_count"],
    }
    return checks, thresholds


def compare(
    champion_paths,
    candidate_paths,
    mode: str,
    *,
    baseline_method="champion",
    candidate_method="delta_mse",
):
    if len(champion_paths) != len(candidate_paths) or not champion_paths:
        raise ValueError("baseline and candidate path counts must match and be nonzero")
    pairs = [
        (_load_verified(base), _load_verified(candidate))
        for base, candidate in zip(champion_paths, candidate_paths)
    ]
    training_seeds = [int(base["training_seed"]) for base, _ in pairs]
    if len(training_seeds) != len(set(training_seeds)):
        raise RuntimeError("formal comparison repeats a training seed")
    expected_training_seeds = {3072} if mode == "discovery" else {3072, 1, 2}
    if set(training_seeds) != expected_training_seeds:
        raise RuntimeError(
            f"{mode} requires training seeds {sorted(expected_training_seeds)}; "
            f"received {sorted(training_seeds)}"
        )
    recipe_hashes = {
        payload["training_recipe_sha256"]
        for pair in pairs for payload in pair
    }
    if len(recipe_hashes) != 1:
        raise RuntimeError("training recipe differs across formal training seeds")
    task = pairs[0][0]["task"]
    if any(base["task"] != task or candidate["task"] != task for base, candidate in pairs):
        raise RuntimeError("all result pairs must use one task")
    formal_hashes = {
        payload["formal_manifest_sha256"]
        for pair in pairs for payload in pair
    }
    if len(formal_hashes) != 1:
        raise RuntimeError("all training seeds must use the same formal manifest")
    expected_set = "discovery" if mode == "discovery" else "confirmation"
    if any(base["bank_set"] != expected_set or candidate["bank_set"] != expected_set
           for base, candidate in pairs):
        raise RuntimeError(f"{mode} comparison requires the {expected_set} bank set")

    per_seed = [
        _paired_seed_summary(
            base,
            candidate,
            baseline_method=baseline_method,
            candidate_method=candidate_method,
        )
        for base, candidate in pairs
    ]
    expected_protocol_names = FORMAL_TASK_PROTOCOL_NAMES[task]
    expected_protocol_set = set(expected_protocol_names)
    if set(per_seed[0]) != expected_protocol_set or any(
        set(summary) != expected_protocol_set for summary in per_seed
    ):
        raise RuntimeError("training seeds have different protocol coverage")
    primary, secondary = expected_protocol_names

    protocol_reports = {}
    for protocol in expected_protocol_names:
        seed_pairs = [(summary[protocol]["y0"], summary[protocol]["y1"]) for summary in per_seed]
        seed_deltas = [float((y1.mean() - y0.mean()) * 100.0) for y0, y1 in seed_pairs]
        seed_success_delta_counts = [
            int(y1.sum() - y0.sum()) for y0, y1 in seed_pairs
        ]
        all_y0 = np.concatenate([pair[0] for pair in seed_pairs])
        all_y1 = np.concatenate([pair[1] for pair in seed_pairs])
        rescues = int(((all_y0 == 0) & (all_y1 == 1)).sum())
        harms = int(((all_y0 == 1) & (all_y1 == 0)).sum())
        protocol_reports[protocol] = {
            "baseline_rate": float(all_y0.mean() * 100.0),
            "candidate_rate": float(all_y1.mean() * 100.0),
            "mean_delta_pp": float((all_y1.sum() - all_y0.sum()) / all_y0.size * 100.0),
            "num_episode_pairs": int(all_y0.size),
            "baseline_successes": int(all_y0.sum()),
            "candidate_successes": int(all_y1.sum()),
            "success_delta_count": int(all_y1.sum() - all_y0.sum()),
            "seed_deltas_pp": seed_deltas,
            "seed_success_delta_counts": seed_success_delta_counts,
            "worst_seed_delta_pp": float(min(seed_deltas)),
            "two_way_cluster_bootstrap_95ci_pp": _two_way_cluster_bootstrap(seed_pairs),
            "rescues": rescues,
            "harms": harms,
            "pooled_mcnemar_exploratory_p": _mcnemar_exact_p(rescues, harms),
            "per_seed_banks": [summary[protocol]["banks"] for summary in per_seed],
        }

    primary_report = protocol_reports[primary]
    secondary_report = protocol_reports[secondary]
    is_demo_goal = pairs[0][0].get("study_id") in set(
        DEMO_GOAL_STUDY_IDS.values()
    )
    gate_thresholds = None
    if is_demo_goal:
        gate_checks, gate_thresholds = _demo_goal_gate_checks(
            task=task,
            mode=mode,
            primary_report=primary_report,
            secondary_report=secondary_report,
        )
    elif mode == "discovery":
        primary_threshold = 4 if task == "pusht" else 3
        gate_checks = {
            "primary_delta": _delta_at_least_pp(
                primary_report, primary_threshold
            ),
            "primary_positive_banks": sum(
                bank["success_delta_count"] > 0
                for bank in primary_report["per_seed_banks"][0]
            ) >= 2,
            "secondary_noninferiority": _delta_at_least_pp(
                secondary_report, -2
            ),
        }
    else:
        gate_checks = {
            "primary_delta": _delta_at_least_pp(primary_report, 3),
            "all_training_seeds_positive": all(
                delta > 0 for delta in primary_report["seed_success_delta_counts"]
            ),
            "primary_ci_lower_positive": (
                primary_report["two_way_cluster_bootstrap_95ci_pp"][0] > 0
            ),
            "secondary_noninferiority": _delta_at_least_pp(secondary_report, -2),
        }
    result = {
        "schema_version": 1,
        "task": task,
        "mode": mode,
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "num_training_seed_pairs": len(pairs),
        "formal_manifest_sha256": pairs[0][0]["formal_manifest_sha256"],
        "primary_protocol": primary,
        "secondary_protocol": secondary,
        "protocols": protocol_reports,
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
        **(
            {
                "study_id": pairs[0][0]["study_id"],
                "method_lock_sha256": pairs[0][0]["method_lock_sha256"],
            }
            if pairs[0][0].get("study_id") is not None
            else {}
        ),
    }
    if gate_thresholds is not None:
        result["gate_thresholds"] = gate_thresholds
    return result


def main():
    parser = argparse.ArgumentParser()
    baseline_group = parser.add_mutually_exclusive_group(required=True)
    baseline_group.add_argument("--baseline", type=Path, action="append")
    baseline_group.add_argument(
        "--champion", dest="baseline", type=Path, action="append",
        help="deprecated alias for --baseline",
    )
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--mode", choices=("discovery", "confirmation"), required=True)
    parser.add_argument(
        "--baseline-method",
        choices=tuple(dict.fromkeys(
            tuple(FORMAL_METHOD_SIGNATURES) + tuple(DEMO_GOAL_METHOD_SIGNATURES)
        )),
        default="champion",
    )
    parser.add_argument(
        "--candidate-method",
        choices=tuple(dict.fromkeys(
            tuple(FORMAL_METHOD_SIGNATURES) + tuple(DEMO_GOAL_METHOD_SIGNATURES)
        )),
        default="delta_mse",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = compare(
        args.baseline,
        args.candidate,
        args.mode,
        baseline_method=args.baseline_method,
        candidate_method=args.candidate_method,
    )
    report = seal_comparison_report(
        analysis,
        baseline_paths=args.baseline,
        candidate_paths=args.candidate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(args.output, flags, 0o600)
    try:
        encoded = (
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        ).encode()
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    print(
        f"FORMAL_COMPARE task={report['task']} mode={report['mode']} "
        f"gate_pass={report['gate_pass']} output={args.output}"
    )
    raise SystemExit(0 if report["gate_pass"] else 4)


if __name__ == "__main__":
    main()
