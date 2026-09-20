"""Immutable, outcome-blind artifacts for the Demo-Goal successor study."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from trim.protocols.demo_goal_formal_specs import (
    DEMO_GOAL_BANKS_PER_SET,
    DEMO_GOAL_BANK_SEED_NAMESPACES,
    DEMO_GOAL_CONFIRMATION_GATES,
    DEMO_GOAL_DISCOVERY_GATES,
    DEMO_GOAL_FINAL_CONFIRMATION_ID,
    DEMO_GOAL_FORMAL_ARMS,
    DEMO_GOAL_FORMAL_TRAINING_JOB_SCHEDULE,
    DEMO_GOAL_FROZEN_TRAINING_CONFIG,
    DEMO_GOAL_METHOD_LOCK_ID,
    DEMO_GOAL_METHOD_SIGNATURES,
    DEMO_GOAL_RESERVED_EPISODES_PER_TASK,
    DEMO_GOAL_STUDY_IDS,
    DEMO_GOAL_TASK_PROTOCOLS,
    DEMO_GOAL_TRAINING_SEEDS,
)


DEMO_GOAL_METHOD_LOCK_SCHEMA_VERSION = 1
DEMO_GOAL_METHOD_LOCK_HASH_KEY = "method_lock_sha256"


def canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_git_object_id(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _exclusive_json_write(path: Path, payload) -> None:
    """Create one immutable JSON slot without symlink following or replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        ).encode()
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def write_exclusive_self_hashed_json(payload, path: Path, *, hash_key: str):
    value = copy.deepcopy(dict(payload))
    value.pop(hash_key, None)
    value[hash_key] = canonical_sha256(value)
    _exclusive_json_write(path, value)
    return value


def verify_self_hashed_json(path: Path, *, hash_key: str):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"artifact must be a regular non-symlink file: {path}")
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


def demo_goal_training_recipe_payload():
    """Return the complete outcome-independent recipe sealed by the lock."""
    return {
        "formal_arms": list(DEMO_GOAL_FORMAL_ARMS),
        "method_signatures": copy.deepcopy(DEMO_GOAL_METHOD_SIGNATURES),
        "frozen_training_config": copy.deepcopy(
            DEMO_GOAL_FROZEN_TRAINING_CONFIG
        ),
        "training_seeds": list(DEMO_GOAL_TRAINING_SEEDS),
        "formal_training_job_schedule": [
            {"task": task, "arm": arm, "training_seed": seed}
            for task, arm, seed in DEMO_GOAL_FORMAL_TRAINING_JOB_SCHEDULE
        ],
        "task_protocols": {
            task: [copy.deepcopy(protocol) for protocol in protocols]
            for task, protocols in DEMO_GOAL_TASK_PROTOCOLS.items()
        },
        "bank_seed_namespaces": copy.deepcopy(
            DEMO_GOAL_BANK_SEED_NAMESPACES
        ),
        "banks_per_set": copy.deepcopy(DEMO_GOAL_BANKS_PER_SET),
        "reserved_episodes_per_task": (
            DEMO_GOAL_RESERVED_EPISODES_PER_TASK
        ),
        "cross_study_episode_exclusion": {
            "required": True,
            "excluded_study_family": "delta_triangle",
            "binding": "full exclusion-manifest file SHA256 plus sorted episode-id SHA256",
        },
        "discovery_gates": copy.deepcopy(DEMO_GOAL_DISCOVERY_GATES),
        "confirmation_gates": copy.deepcopy(DEMO_GOAL_CONFIRMATION_GATES),
        "stage_order": [
            "pusht_discovery",
            "tworoom_replay_attestation",
            "tworoom_discovery",
            "discovery_qualification",
            "pusht_confirmation",
            "tworoom_confirmation",
            "two_task_confirmation",
        ],
        "candidate_policy": {
            "one_locked_candidate_only": True,
            "candidate": "dg_champion",
            "dg_mse_is_separate_exploratory_ablation": True,
            "candidate_rescue_after_outcomes_forbidden": True,
        },
    }


def build_demo_goal_method_lock_payload(
    *, source_git_commit: str, source_tracked_tree_sha256: str
):
    if not is_git_object_id(source_git_commit):
        raise ValueError("source_git_commit must be a lowercase Git object id")
    if not is_sha256(source_tracked_tree_sha256):
        raise ValueError("source_tracked_tree_sha256 must be a lowercase SHA256")
    recipe = demo_goal_training_recipe_payload()
    return {
        "schema_version": DEMO_GOAL_METHOD_LOCK_SCHEMA_VERSION,
        "status": "LOCKED",
        "method_lock_id": DEMO_GOAL_METHOD_LOCK_ID,
        "final_confirmation_id": DEMO_GOAL_FINAL_CONFIRMATION_ID,
        "study_ids": copy.deepcopy(DEMO_GOAL_STUDY_IDS),
        "source_git_commit": source_git_commit,
        "source_tracked_tree_sha256": source_tracked_tree_sha256,
        "source_worktree_clean": True,
        "created_before_performance_outcomes": True,
        "contains_performance_outcomes": False,
        "training_recipe": recipe,
        "training_recipe_sha256": canonical_sha256(recipe),
    }


def write_demo_goal_method_lock(
    *,
    output_path: Path,
    source_git_commit: str,
    source_tracked_tree_sha256: str,
):
    payload = build_demo_goal_method_lock_payload(
        source_git_commit=source_git_commit,
        source_tracked_tree_sha256=source_tracked_tree_sha256,
    )
    return write_exclusive_self_hashed_json(
        payload,
        output_path,
        hash_key=DEMO_GOAL_METHOD_LOCK_HASH_KEY,
    )


def verify_demo_goal_method_lock(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
):
    value = verify_self_hashed_json(
        path, hash_key=DEMO_GOAL_METHOD_LOCK_HASH_KEY
    )
    expected_keys = {
        "schema_version",
        "status",
        "method_lock_id",
        "final_confirmation_id",
        "study_ids",
        "source_git_commit",
        "source_tracked_tree_sha256",
        "source_worktree_clean",
        "created_before_performance_outcomes",
        "contains_performance_outcomes",
        "training_recipe",
        "training_recipe_sha256",
        DEMO_GOAL_METHOD_LOCK_HASH_KEY,
    }
    if set(value) != expected_keys:
        raise RuntimeError(
            "Demo-Goal method lock has unexpected or missing fields: "
            f"actual={sorted(value)} expected={sorted(expected_keys)}"
        )
    exact = {
        "schema_version": DEMO_GOAL_METHOD_LOCK_SCHEMA_VERSION,
        "status": "LOCKED",
        "method_lock_id": DEMO_GOAL_METHOD_LOCK_ID,
        "final_confirmation_id": DEMO_GOAL_FINAL_CONFIRMATION_ID,
        "study_ids": DEMO_GOAL_STUDY_IDS,
        "source_worktree_clean": True,
        "created_before_performance_outcomes": True,
        "contains_performance_outcomes": False,
        "training_recipe": demo_goal_training_recipe_payload(),
    }
    if expected_commit is not None:
        exact["source_git_commit"] = expected_commit
    if expected_tree is not None:
        exact["source_tracked_tree_sha256"] = expected_tree
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in exact.items()
        if value.get(key) != wanted
    }
    recipe_sha256 = canonical_sha256(value.get("training_recipe"))
    if value.get("training_recipe_sha256") != recipe_sha256:
        mismatches["training_recipe_sha256"] = (
            value.get("training_recipe_sha256"),
            recipe_sha256,
        )
    if not is_git_object_id(value.get("source_git_commit")):
        mismatches["source_git_commit_format"] = (
            value.get("source_git_commit"),
            "lowercase 40- or 64-hex Git object id",
        )
    if not is_sha256(value.get("source_tracked_tree_sha256")):
        mismatches["source_tracked_tree_sha256_format"] = (
            value.get("source_tracked_tree_sha256"),
            "lowercase 64-hex",
        )
    if mismatches:
        raise RuntimeError(f"Demo-Goal method lock mismatch: {mismatches}")
    return value
