#!/usr/bin/env python3
"""Evaluate non-learned frozen-pool H5 baselines without a planner server."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.result_commit import commit_json_result, file_sha256
from scripts.run_paper_h5_checkpoint import (
    DEFAULT_POOL,
    ISAAC_PYTHON,
    _load_pool,
    _run_logged,
    _select_cases,
    _verify_committed,
)


POLICIES = {"gear": ("zero", "expert"), "franka": ("zero", "drop", "expert")}


def run_gear(
    *, pool: dict, cases: list[dict], policy: str, output_dir: Path,
    env: dict[str, str], env_seed: int,
) -> tuple[list[dict], list[dict]]:
    sources = {int(item["source_index"]): item for item in pool["gear"]["sources"]}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for case in cases:
        grouped[int(case["source_index"])].append(case)
    by_id = {}
    components = []
    for source_index in sorted(grouped):
        selected = grouped[source_index]
        source = sources[source_index]
        group_dir = output_dir / f"source_{source_index}"
        result_path = group_dir / "eval.json"
        episodes = [int(case["source_episode"]) for case in selected]
        command = [
            str(ISAAC_PYTHON), str(ROOT / "scripts/eval_gear_worldmodel.py"),
            "--eval_dataset", source["dataset"]["path"],
            "--expert_gate_json", source["expert_gate"]["path"],
            "--goal_offset_steps", "25", "--paired_min_goal_step", "25",
            "--paired_required_offsets", *[str(value) for value in source["master_offsets"]],
            "--paired_source_episodes", ",".join(map(str, episodes)),
            "--episodes", str(len(selected)), "--eval_budget", "50",
            "--policy", policy, "--action_protocol", "effective4",
            "--seed", str(env_seed), "--strict_persistence_steps", "3",
            "--max_init_frame_mae", str(source["restoration_limits"]["max_init_frame_mean_abs_error"]),
            "--max_init_pixels_over_8", str(source["restoration_limits"]["max_init_frame_pixels_over_8"]),
            "--action_diagnostics", "full", "--output", str(result_path), "--headless",
        ]
        _run_logged(command, log=group_dir / "client.log", env=env)
        result = _verify_committed(result_path, schema="gear-h5-eval-v1", count=len(selected))
        if result.get("policy") != policy or result.get("paired_source_episodes") != episodes:
            raise RuntimeError("Gear baseline protocol drifted")
        for case, record in zip(selected, result["per_episode"]):
            metadata = record["paired_source"]
            if int(metadata["source_episode"]) != int(case["source_episode"]):
                raise RuntimeError("Gear baseline changed case identity")
            by_id[case["case_id"]] = {**record, "case_id": case["case_id"], "source_index": source_index}
        components.append(
            {"source_index": source_index, "result": str(result_path.resolve()),
             "result_sha256": file_sha256(result_path), "case_count": len(selected)}
        )
    return [by_id[case["case_id"]] for case in cases], components


def run_franka(
    *, pool: dict, cases: list[dict], policy: str, output_dir: Path,
    env: dict[str, str], env_seed: int,
) -> tuple[list[dict], list[dict]]:
    demos = [int(case["demo"]) for case in cases]
    native_dir = output_dir / "native"
    command = [
        str(ISAAC_PYTHON), str(ROOT / "scripts/isaac_stack_eval.py"),
        "--offset", "25", "--budget", "50", "--num-eval", str(len(cases)),
        "--stage", "3", "--init-mode", "teleport", "--dataset-split", "eval",
        "--demos", ",".join(map(str, demos)),
        "--expert-gate-json", pool["franka"]["source"]["expert_gate"]["path"],
        "--policy", policy, "--seed", str(env_seed),
        "--out", str(native_dir), "--headless",
    ]
    _run_logged(command, log=output_dir / "client.log", env=env)
    native = native_dir / f"eval_{policy}_teleport_s3_offset25.json"
    result = _verify_committed(native, schema="franka-stack-h5-eval-v1", count=len(cases))
    if result.get("policy") != policy or result.get("seed") != env_seed:
        raise RuntimeError("Franka baseline protocol drifted")
    by_demo = {int(record["demo"]): record for record in result["episodes"]}
    if set(by_demo) != set(demos):
        raise RuntimeError("Franka baseline cases differ from frozen pool")
    records = []
    for case in cases:
        record = by_demo[int(case["demo"])]
        if int(record["goal_idx"]) != int(case["goal_idx"]) or int(record["start"]) != int(case["start"]):
            raise RuntimeError("Franka baseline changed case identity")
        records.append({**record, "case_id": case["case_id"]})
    return records, [{"result": str(native.resolve()), "result_sha256": file_sha256(native), "case_count": len(cases)}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("gear", "franka"))
    parser.add_argument("--policy", required=True, choices=("zero", "drop", "expert"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--development-case-limit", type=int)
    args = parser.parse_args()
    if args.policy not in POLICIES[args.task]:
        parser.error(f"{args.policy} is not a registered {args.task} baseline")
    if args.split == "test" and not args.allow_sealed_test:
        parser.error("sealed test requires --allow-sealed-test")
    pool_path = args.pool.expanduser().resolve()
    pool, cases = _load_pool(pool_path, task=args.task, split=args.split)
    cases = _select_cases(cases, split=args.split, development_case_limit=args.development_case_limit)
    output_dir = args.output_dir.expanduser().resolve()
    aggregate = output_dir / "eval.json"
    if aggregate.with_suffix(".json.done").is_file():
        existing = _verify_committed(aggregate, schema="paper-h5-baseline-aggregate-v1", count=len(cases))
        print(json.dumps({"status": "already_complete", "successes": existing["successes"], "output": str(aggregate)}))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runner = run_gear if args.task == "gear" else run_franka
    records, components = runner(
        pool=pool, cases=cases, policy=args.policy, output_dir=output_dir,
        env=dict(os.environ), env_seed=args.env_seed,
    )
    successes = sum(bool(record["success"]) for record in records)
    payload = {
        "task": args.task, "policy": args.policy, "split": args.split,
        "pool_manifest": str(pool_path), "pool_manifest_sha256": file_sha256(pool_path),
        "selected_sixty_sha256": pool[args.task]["selected_case_sha256"],
        "evaluated_case_order_sha256": hashlib.sha256("\n".join(case["case_id"] for case in cases).encode()).hexdigest(),
        "env_seed": args.env_seed, "case_count": len(records), "successes": successes,
        "success_rate": successes / len(records), "wall_seconds": time.time() - started,
        "component_results": components, "records": records,
    }
    completion = commit_json_result(
        aggregate, payload, schema_version="paper-h5-baseline-aggregate-v1", record_count=len(records)
    )
    print(json.dumps({"status": "complete", "successes": successes, "output": str(aggregate), "sha256": completion["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
