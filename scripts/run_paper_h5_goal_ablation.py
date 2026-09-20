#!/usr/bin/env python3
"""Run paired wrong-goal H5 ablations on the frozen case pools."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
    _server_command,
    _verify_committed,
    _wait_for_socket,
)


def _run_gear_wrong_goal(
    *, pool: dict, cases: list[dict], socket_path: Path, output_dir: Path,
    env: dict[str, str], env_seed: int, planner_seed_base: int,
) -> tuple[list[dict], list[dict]]:
    sources = {int(item["source_index"]): item for item in pool["gear"]["sources"]}
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for position, case in enumerate(cases):
        grouped[int(case["source_index"])].append((position, case))
    by_id = {}
    components = []
    for source_index in sorted(grouped):
        indexed = grouped[source_index]
        source = sources[source_index]
        group_dir = output_dir / f"source_{source_index}"
        result_path = group_dir / "eval.json"
        source_episodes = [int(case["source_episode"]) for _, case in indexed]
        planner_seeds = [planner_seed_base + position for position, _ in indexed]
        command = [
            str(ISAAC_PYTHON), str(ROOT / "scripts/eval_gear_worldmodel.py"),
            "--eval_dataset", source["dataset"]["path"],
            "--expert_gate_json", source["expert_gate"]["path"],
            "--goal_offset_steps", "25",
            "--paired_min_goal_step", "25",
            "--paired_required_offsets", *[str(value) for value in source["master_offsets"]],
            "--paired_source_episodes", ",".join(map(str, source_episodes)),
            "--paired_planner_seeds", ",".join(map(str, planner_seeds)),
            "--episodes", str(len(indexed)),
            "--eval_budget", "50",
            "--policy", "server",
            "--socket", str(socket_path),
            "--goal_ablation", "self_init",
            "--action_protocol", "effective4",
            "--seed", str(env_seed),
            "--planner_seed", str(planner_seed_base),
            "--strict_persistence_steps", "3",
            "--max_init_frame_mae", str(source["restoration_limits"]["max_init_frame_mean_abs_error"]),
            "--max_init_pixels_over_8", str(source["restoration_limits"]["max_init_frame_pixels_over_8"]),
            "--action_diagnostics", "full",
            "--output", str(result_path),
            "--headless",
        ]
        _run_logged(command, log=group_dir / "client.log", env=env)
        result = _verify_committed(result_path, schema="gear-h5-eval-v1", count=len(indexed))
        if result.get("goal_ablation") != "self_init":
            raise RuntimeError("Gear wrong-goal client did not use self_init")
        if result.get("paired_source_episodes") != source_episodes:
            raise RuntimeError("Gear wrong-goal case order drifted")
        if result.get("paired_planner_seeds") != planner_seeds:
            raise RuntimeError("Gear wrong-goal planner seeds drifted")
        for local_index, ((_, case), record) in enumerate(zip(indexed, result["per_episode"])):
            metadata = record["paired_source"]
            if (
                int(metadata["source_episode"]) != int(case["source_episode"])
                or int(metadata["goal_row"]) != int(case["goal_row"])
                or int(metadata["init_row"]) != int(case["init_row"])
            ):
                raise RuntimeError("Gear wrong-goal result changed case identity")
            by_id[case["case_id"]] = {
                **record,
                "case_id": case["case_id"],
                "source_index": source_index,
                "planner_seed": planner_seeds[local_index],
            }
        components.append(
            {
                "source_index": source_index,
                "result": str(result_path.resolve()),
                "result_sha256": file_sha256(result_path),
                "case_count": len(indexed),
            }
        )
    if set(by_id) != {case["case_id"] for case in cases}:
        raise RuntimeError("Gear wrong-goal aggregate is incomplete")
    return [by_id[case["case_id"]] for case in cases], components


def _run_franka_wrong_goal(
    *, pool: dict, cases: list[dict], socket_path: Path, output_dir: Path,
    env: dict[str, str], env_seed: int, planner_seed_base: int,
) -> tuple[list[dict], list[dict]]:
    demos = [int(case["demo"]) for case in cases]
    native_dir = output_dir / "native"
    command = [
        str(ISAAC_PYTHON), str(ROOT / "scripts/isaac_stack_eval.py"),
        "--offset", "25", "--budget", "50", "--num-eval", str(len(cases)),
        "--stage", "3", "--init-mode", "teleport", "--dataset-split", "eval",
        "--demos", ",".join(map(str, demos)),
        "--expert-gate-json", pool["franka"]["source"]["expert_gate"]["path"],
        "--policy", "lewm", "--socket", str(socket_path),
        "--seed", str(env_seed), "--planner-seed-base", str(planner_seed_base),
        "--planner-seed-per-demo", "--shuffle-goals",
        "--expected-planner-mode", "train_terminal_wm_rerank",
        "--expected-plan-bank-continuation", "abstain",
        "--out", str(native_dir), "--headless",
    ]
    _run_logged(command, log=output_dir / "client.log", env=env)
    native = native_dir / "eval_lewm_teleport_s3_offset25.json"
    result = _verify_committed(native, schema="franka-stack-h5-eval-v1", count=len(cases))
    if result.get("shuffle_goals") is not True:
        raise RuntimeError("Franka wrong-goal client did not shuffle goals")
    if result.get("seed") != env_seed or result.get("planner_seed_base") != planner_seed_base:
        raise RuntimeError("Franka wrong-goal seeds drifted")
    by_demo = {int(record["demo"]): record for record in result["episodes"]}
    if set(by_demo) != set(demos):
        raise RuntimeError("Franka wrong-goal cases differ from frozen pool")
    records = []
    for case in cases:
        record = by_demo[int(case["demo"])]
        if int(record["goal_idx"]) != int(case["goal_idx"]) or int(record["start"]) != int(case["start"]):
            raise RuntimeError("Franka wrong-goal result changed initial case")
        if int(record["goal_demo"]) == int(case["demo"]):
            raise RuntimeError("Franka wrong-goal derangement has a fixed point")
        if int(record["planner_seed"]) != planner_seed_base + int(case["demo"]):
            raise RuntimeError("Franka wrong-goal planner seed drifted")
        records.append({**record, "case_id": case["case_id"]})
    return records, [{"result": str(native.resolve()), "result_sha256": file_sha256(native), "case_count": len(cases)}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("gear", "franka"))
    parser.add_argument("--arm", required=True, choices=("L0", "L1", "C0", "C1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--planner-seed-base", type=int, default=42)
    parser.add_argument("--development-case-limit", type=int)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_sealed_test:
        parser.error("sealed test requires --allow-sealed-test after checkpoint selection")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    pool_path = args.pool.expanduser().resolve()
    pool, cases = _load_pool(pool_path, task=args.task, split=args.split)
    cases = _select_cases(cases, split=args.split, development_case_limit=args.development_case_limit)
    output_dir = args.output_dir.expanduser().resolve()
    aggregate = output_dir / "eval.json"
    if aggregate.with_suffix(".json.done").is_file():
        existing = _verify_committed(
            aggregate, schema="paper-h5-goal-ablation-aggregate-v1", count=len(cases)
        )
        print(json.dumps({"status": "already_complete", "successes": existing["successes"], "output": str(aggregate)}))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    socket_path = Path(f"/tmp/lewm-paper-wrong-goal-{args.task}-{os.getpid()}.sock")
    env = dict(os.environ)
    server_log = output_dir / "server.log"
    handle = server_log.open("w", encoding="utf-8")
    server = subprocess.Popen(
        _server_command(args.task, checkpoint, socket_path, args.planner_seed_base),
        cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
    )
    started = time.time()
    try:
        _wait_for_socket(socket_path, server)
        runner = _run_gear_wrong_goal if args.task == "gear" else _run_franka_wrong_goal
        records, components = runner(
            pool=pool, cases=cases, socket_path=socket_path, output_dir=output_dir,
            env=env, env_seed=args.env_seed, planner_seed_base=args.planner_seed_base,
        )
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        handle.close()
        socket_path.unlink(missing_ok=True)
    successes = sum(bool(record["success"]) for record in records)
    payload = {
        "task": args.task, "arm": args.arm, "epoch": args.epoch,
        "split": args.split, "wrong_goal_mode": "self_init" if args.task == "gear" else "cross_case_derangement",
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "pool_manifest": str(pool_path), "pool_manifest_sha256": file_sha256(pool_path),
        "selected_sixty_sha256": pool[args.task]["selected_case_sha256"],
        "evaluated_case_order_sha256": hashlib.sha256("\n".join(case["case_id"] for case in cases).encode()).hexdigest(),
        "env_seed": args.env_seed, "planner_seed_base": args.planner_seed_base,
        "case_count": len(records), "successes": successes,
        "success_rate": successes / len(records), "wall_seconds": time.time() - started,
        "component_results": components, "records": records,
    }
    completion = commit_json_result(
        aggregate, payload, schema_version="paper-h5-goal-ablation-aggregate-v1", record_count=len(records)
    )
    print(json.dumps({"status": "complete", "successes": successes, "output": str(aggregate), "sha256": completion["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
