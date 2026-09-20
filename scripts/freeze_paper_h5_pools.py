#!/usr/bin/env python3
"""Freeze independent Gear and Franka H5 dev30/test30 case manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "config/eval/pools/paper_new_tasks_h5_pools_v2.json"
GEAR_REVALIDATION_DIR = (
    REPO_ROOT / "outputs/paper_new_tasks_v1/pools/gear_restoration_revalidation_v3"
)
GEAR_RESTORATION_LIMITS = {
    "all_cases_passed": True,
    "max_init_frame_mean_abs_error": 12.0,
    "max_init_frame_pixels_over_8": 0.4,
    "scope": "all_expert_eligible_cases_revalidated_without_control_steps",
}
GEAR_SOURCES = (
    (
        REPO_ROOT
        / "outputs/isaaclab_automate/gear_01036_eval_set_v2_seed4242/eval_trajectories.h5",
        REPO_ROOT
        / "outputs/gear_formal_v2_seed4242_h135/expert_gate/eval_expert_goalminus25_budget50_n15.json",
        GEAR_REVALIDATION_DIR / "source_0_all6.json",
    ),
    (
        REPO_ROOT
        / "outputs/isaaclab_automate/gear_01036_eval_set_v2_seed5242_n1000/eval_trajectories.h5",
        REPO_ROOT
        / "outputs/gear_formal_v2_seed5242_n1000_h135/expert_gate/eval_expert_goalminus25_budget50_n27.json",
        GEAR_REVALIDATION_DIR / "source_1_all17.json",
    ),
    (
        REPO_ROOT
        / "outputs/isaaclab_automate/gear_01036_eval_set_v2_seed8242_n1000_steps150/eval_trajectories.h5",
        REPO_ROOT
        / "outputs/paper_new_tasks_v1/pools/gear_seed8242_h5_expert_gate/eval_expert_goalminus25_budget50_n58.json",
        GEAR_REVALIDATION_DIR / "source_2_all41.json",
    ),
)
FRANKA_GATE = (
    REPO_ROOT
    / "outputs/paper_new_tasks_v1/pools/franka_h5_expert_gate/eval_expert_teleport_s3_offset25.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _case_id(task: str, source: str, episode: int, goal: int) -> str:
    raw = f"{task}\0{source}\0{episode}\0{goal}".encode()
    return f"{task}-{hashlib.sha256(raw).hexdigest()[:16]}"


def load_gear() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for source_index, (dataset, gate, revalidation) in enumerate(GEAR_SOURCES):
        dataset = dataset.resolve()
        gate = gate.resolve()
        revalidation = revalidation.resolve()
        payload = json.loads(gate.read_text(encoding="utf-8"))
        expected = {
            "policy": "expert",
            "eval_dataset": str(dataset),
            "goal_offset_steps": 25,
            "eval_budget": 50,
            "seed": 42,
            "paired_goal_selection": "episode_first_strict_frame_and_persists",
        }
        actual = {key: payload.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Gear gate contract mismatch: {gate}: {actual}")
        pool = payload.get("paired_master_pool")
        if not isinstance(pool, dict) or 25 not in pool.get("master_offsets", []):
            raise ValueError(f"Gear gate has no H5 master pool: {gate}")
        if pool.get("goal_contract") != "episode_first_strict_frame_and_persists":
            raise ValueError(f"Gear gate uses a non-first-success goal: {gate}")
        records = payload.get("per_episode")
        if (
            not isinstance(records, list)
            or len(records) != int(payload.get("episodes", -1))
            or len(records) != int(pool.get("candidate_count", -1))
            or len(records) != int(pool.get("unfiltered_candidate_count", -1))
        ):
            raise ValueError(f"Gear gate is not exhaustive: {gate}")
        if payload.get("restoration_gate", {}).get("all_cases_passed") is not True:
            raise ValueError(f"Gear restoration gate failed: {gate}")
        if payload.get("action_contract", {}).get("all_steps_exact") is not True:
            raise ValueError(f"Gear expert actions were not exact: {gate}")
        eligible_count = 0
        eligible_episodes = []
        for record in records:
            if record.get("success") is not True:
                continue
            metadata = record.get("paired_source", {})
            episode = int(metadata["source_episode"])
            goal_row = int(metadata["goal_row"])
            identity = (str(dataset), episode)
            if identity in identities:
                raise ValueError(f"duplicate Gear case identity: {identity}")
            identities.add(identity)
            eligible_count += 1
            eligible_episodes.append(episode)
            cases.append(
                {
                    "case_id": _case_id("gear", str(dataset), episode, goal_row),
                    "source_index": source_index,
                    "source_episode": episode,
                    "goal_step": int(metadata["goal_step"]),
                    "goal_row": goal_row,
                    "init_step": int(metadata["init_step"]),
                    "init_row": int(metadata["init_row"]),
                }
            )
        audit = json.loads(revalidation.read_text(encoding="utf-8"))
        audit_marker = revalidation.with_suffix(revalidation.suffix + ".done")
        marker = json.loads(audit_marker.read_text(encoding="utf-8"))
        if (
            audit.get("schema_version") != "gear-h5-eval-v1"
            or audit.get("policy") != "zero"
            or audit.get("restoration_audit_only") is not True
            or audit.get("eval_dataset") != str(dataset)
            or int(audit.get("goal_offset_steps", -1)) != 25
            or int(audit.get("eval_budget", -1)) != 50
            or int(audit.get("episodes", -1)) != eligible_count
            or audit.get("restoration_gate") != {
                "all_cases_passed": True,
                "max_init_frame_mean_abs_error": 12.0,
                "max_init_frame_pixels_over_8": 0.4,
            }
        ):
            raise ValueError(f"Gear restoration revalidation contract mismatch: {revalidation}")
        audit_episodes = [
            int(record["paired_source"]["source_episode"])
            for record in audit.get("per_episode", [])
        ]
        if set(audit_episodes) != set(eligible_episodes) or len(audit_episodes) != eligible_count:
            raise ValueError(f"Gear restoration revalidation coverage mismatch: {revalidation}")
        for record in audit["per_episode"]:
            if (
                record.get("restoration_audit_only") is not True
                or int(record.get("steps", -1)) != 0
                or float(record["restoration"]["init_frame_mean_abs_error"]) > 12.0
                or float(record["restoration"]["init_frame_pixels_over_8"]) > 0.4
            ):
                raise ValueError(f"Gear restoration revalidation failed: {revalidation}")
        if (
            marker.get("status") != "complete"
            or marker.get("result_schema_version") != "gear-h5-eval-v1"
            or int(marker.get("record_count", -1)) != eligible_count
            or marker.get("result_sha256") != sha256(revalidation)
        ):
            raise ValueError(f"Gear restoration revalidation marker failed: {audit_marker}")
        sources.append(
            {
                "source_index": source_index,
                "dataset": fingerprint(dataset),
                "expert_gate": fingerprint(gate),
                "master_offsets": list(map(int, pool["master_offsets"])),
                "master_pool_sha256": pool["master_pool_sha256"],
                "tested_count": len(records),
                "eligible_count": eligible_count,
                "restoration_limits": dict(GEAR_RESTORATION_LIMITS),
                "restoration_revalidation": fingerprint(revalidation),
                "restoration_revalidation_marker": fingerprint(audit_marker),
            }
        )
    return cases, sources


def load_franka() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate = FRANKA_GATE.resolve()
    payload = json.loads(gate.read_text(encoding="utf-8"))
    expected = {
        "policy": "expert",
        "stage": 3,
        "offset": 25,
        "budget": 50,
        "dataset_split": "eval",
        "split_seed": 3072,
        "init_mode": "teleport",
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Franka gate contract mismatch: {actual}")
    records = payload.get("episodes")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("Franka gate must exhaust all 100 eval episodes")
    source = str(Path(payload["source_dataset"]).resolve())
    cases = []
    seen: set[int] = set()
    for record in records:
        demo = int(record["demo"])
        if demo in seen:
            raise ValueError(f"Franka gate repeats demo {demo}")
        seen.add(demo)
        if record.get("success") is not True:
            continue
        goal = int(record["goal_idx"])
        start = int(record["start"])
        if goal - start != 25 or int(record["t_success"]) < 0:
            raise ValueError(f"Franka eligible case violates H5: demo {demo}")
        endpoint = record.get("init_endpoint", {})
        if float(endpoint.get("cube_max_error_m", np.inf)) > 1.0e-8:
            raise ValueError(f"Franka state restore is not exact: demo {demo}")
        if float(endpoint.get("joint_max_error_rad", np.inf)) > 1.0e-8:
            raise ValueError(f"Franka joint restore is not exact: demo {demo}")
        cases.append(
            {
                "case_id": _case_id("franka", source, demo, goal),
                "demo": demo,
                "goal_idx": goal,
                "start": start,
            }
        )
    return cases, {
        "source_dataset": fingerprint(Path(source)),
        "expert_gate": fingerprint(gate),
        "tested_count": len(records),
        "eligible_count": len(cases),
    }


def freeze(cases: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    identities = [item["case_id"] for item in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("planner repeats or duplicate case identities are forbidden")
    if len(cases) < 60:
        raise ValueError(f"need >=60 eligible independent cases, found {len(cases)}")
    ordered = sorted(cases, key=lambda item: item["case_id"])
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    selected = [ordered[int(index)] for index in permutation[:60]]
    dev = selected[:30]
    test = selected[30:]
    if {item["case_id"] for item in dev} & {item["case_id"] for item in test}:
        raise AssertionError("dev/test pool overlap")
    return {
        "selection_seed": seed,
        "eligible_count": len(cases),
        "dev30": dev,
        "test30": test,
        "reserve": [ordered[int(index)] for index in permutation[60:]],
        "selected_case_sha256": hashlib.sha256(
            "\n".join(item["case_id"] for item in selected).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    gear_cases, gear_sources = load_gear()
    franka_cases, franka_source = load_franka()
    payload = {
        "schema_version": "paper-new-tasks-h5-pools-v2",
        "frozen_date": "2026-09-06",
        "selection_scope": (
            "independent expert-restorable cases with current-launch RGB revalidation; "
            "planner repeats never count as cases"
        ),
        "protocol": {
            "horizon": 5,
            "action_block": 5,
            "goal_offset_raw_steps": 25,
            "budget_raw_steps": 50,
            "dev_usage": "observation only; forbidden for stopping/checkpoint selection",
            "test_usage": "sealed until loss-selected checkpoints are frozen",
        },
        "gear": {
            "goal_contract": "first_ever_strict_and_persists_from_that_frame",
            "sources": gear_sources,
            **freeze(gear_cases, seed=2026090601),
        },
        "franka": {
            "goal_contract": "first_strict_stack_with_goal_pose_and_open_gripper",
            "source": franka_source,
            **freeze(franka_cases, seed=2026090602),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "gear_eligible": len(gear_cases),
        "franka_eligible": len(franka_cases),
        "manifest_sha256": sha256(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
