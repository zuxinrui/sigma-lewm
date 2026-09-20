#!/usr/bin/env python3
"""T7a proposal-only controls on the sealed H5 pool (frozen cem_ablation driver + planner-mode override).

Copy of scripts/run_paper_h5_cem_ablation.py that only adds ``--planner-mode``
(override of the per-task frozen default) and ``--control-tag``.  Every other
flag (cem-iters 30, budget 50, offsets, strict persistence, restoration limits,
plan-bank top-300 / abstain continuation) is identical to the sealed final_test.
Outputs go to outputs/paper_new_tasks_v1/<task>/<arm>/t7a_controls/<tag>/<primary|secondary>/eval.json.
"""

from __future__ import annotations
from repro_paths import resolve_path as _resolve_artifact, resolve_recorded as _recorded

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

try:
    from result_commit import commit_json_result, file_sha256
except ImportError:  # imported as scripts.run_paper_h5_checkpoint in tests
    from scripts.result_commit import commit_json_result, file_sha256


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv/bin/python"
ISAAC_PYTHON = Path(
    _resolve_artifact('isaac_python:')
)
DEFAULT_POOL = ROOT / "config/eval/pools/paper_new_tasks_h5_pools_v2.json"
TRAIN_DATA = {
    "gear": Path(
        _resolve_artifact('data:gear_01036_effective4_persistent_start_h5_train_v4.h5')
    ),
    "franka": Path(
        _resolve_artifact('data:franka_stack_aligned_train_v2.h5')
    ),
}


def _load_pool(path: Path, *, task: str, split: str) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "paper-new-tasks-h5-pools-v2":
        raise ValueError("frozen H5 pool schema mismatch")
    key = "dev30" if split == "dev" else "test30"
    cases = payload[task][key]
    if len(cases) != 30 or len({case["case_id"] for case in cases}) != 30:
        raise ValueError(f"{task} {key} is not 30 independent cases")
    return payload, cases


def _select_cases(
    cases: list[dict], *, split: str, development_case_limit: int | None
) -> list[dict]:
    """Allow a visibly non-formal subset only for pre-freeze live smokes."""

    if development_case_limit is None:
        return cases
    if split != "dev":
        raise ValueError("development case limits are forbidden on the sealed test split")
    if not 1 <= development_case_limit < len(cases):
        raise ValueError(
            "--development-case-limit must be between 1 and 29; "
            "omit it for the formal dev30"
        )
    return cases[:development_case_limit]


def _verify_committed(path: Path, *, schema: str, count: int) -> dict:
    done_path = path.with_suffix(path.suffix + ".done")
    if not path.is_file() or not done_path.is_file():
        raise RuntimeError(f"result is not atomically complete: {path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if (
        done.get("status") != "complete"
        or done.get("result_schema_version") != schema
        or int(done.get("record_count", -1)) != count
        or done.get("result_sha256") != file_sha256(path)
    ):
        raise RuntimeError(f"result completion marker is invalid: {done_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_for_socket(path: Path, process: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"planner server exited with code {process.returncode}")
        if path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(path))
            except OSError:
                pass
            else:
                probe.close()
                return
            finally:
                probe.close()
        time.sleep(0.1)
    raise TimeoutError(f"planner server did not open {path}")


PLANNER_MODES = {
    "gear": ("projected_cem", "projected_random", "stock_cem"),
    "franka": (
        "train_terminal_wm_rerank",
        "train_terminal_behavior_retrieval",
        "train_terminal_behavior_central",
    ),
}
DEFAULT_PLANNER_MODE = {"gear": "projected_cem", "franka": "train_terminal_wm_rerank"}


def _server_command(
    task: str,
    checkpoint: Path,
    socket_path: Path,
    seed: int,
    cem_iters: int = 30,
    planner_mode: str | None = None,
) -> list[str]:
    planner_mode = planner_mode or DEFAULT_PLANNER_MODE[task]
    if planner_mode not in PLANNER_MODES[task]:
        raise ValueError(f"{planner_mode!r} is not a registered {task} planner mode")
    # projected_random lives in an additive wrapper so the hash-locked frozen
    # server (scripts/lewm_policy_server.py) and projected_cem.py stay untouched.
    server_script = (
        "scripts/lewm_policy_server_t7a.py"
        if planner_mode == "projected_random"
        else "scripts/lewm_policy_server.py"
    )
    common = [
        str(VENV_PYTHON),
        str(ROOT / server_script),
        "--checkpoint",
        str(checkpoint),
        "--socket",
        str(socket_path),
        "--train-dataset",
        str(TRAIN_DATA[task]),
        "--action-bound-mode",
        "data_quantile",
        "--action-bound-quantile",
        "0.001",
        "--temporal-tv-quantile",
        "0.99",
        "--horizon",
        "5",
        "--receding-horizon",
        "5",
        "--action-block",
        "5",
        "--num-samples",
        "300",
        "--n-steps",
        str(cem_iters),
        "--topk",
        "30",
        "--seed",
        str(seed),
    ]
    if task == "gear":
        return common + [
            "--img-size",
            "224",
            "--action-dim",
            "4",
            "--task-action-schema",
            "gear_effective4",
            "--planner-mode",
            planner_mode,
        ]
    return common + [
        "--img-size",
        "196",
        "--action-dim",
        "7",
        "--task-action-schema",
        "franka_stack",
        "--planner-mode",
        planner_mode,
        "--plan-bank-continuation",
        "abstain",
        "--plan-bank-topk",
        "300",
    ]


def _run_logged(command: list[str], *, log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(
            f"evaluation client exited {completed.returncode}: {' '.join(command)}\n{tail}"
        )


def _run_gear(
    *,
    pool: dict,
    cases: list[dict],
    socket_path: Path,
    output_dir: Path,
    env: dict[str, str],
    env_seed: int,
    planner_seed_base: int,
    planner_mode: str = "projected_cem",
) -> tuple[list[dict], list[dict]]:
    del planner_mode  # the Gear client learns the mode from the server's planner_config
    sources = {int(item["source_index"]): item for item in pool["gear"]["sources"]}
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for position, case in enumerate(cases):
        grouped[int(case["source_index"])].append((position, case))
    records_by_id = {}
    group_records = []
    for source_index in sorted(grouped):
        indexed = grouped[source_index]
        source = sources[source_index]
        group_dir = output_dir / f"source_{source_index}"
        result_path = group_dir / "eval.json"
        source_episodes = [int(case["source_episode"]) for _, case in indexed]
        planner_seeds = [planner_seed_base + position for position, _ in indexed]
        command = [
            str(ISAAC_PYTHON),
            str(ROOT / "scripts/eval_gear_worldmodel.py"),
            "--eval_dataset",
            _recorded(source["dataset"]["path"]),
            "--expert_gate_json",
            _recorded(source["expert_gate"]["path"]),
            "--goal_offset_steps",
            "25",
            "--paired_min_goal_step",
            "25",
            "--paired_required_offsets",
            *[str(value) for value in source["master_offsets"]],
            "--paired_source_episodes",
            ",".join(map(str, source_episodes)),
            "--paired_planner_seeds",
            ",".join(map(str, planner_seeds)),
            "--episodes",
            str(len(indexed)),
            "--eval_budget",
            "50",
            "--policy",
            "server",
            "--socket",
            str(socket_path),
            "--action_protocol",
            "effective4",
            "--seed",
            str(env_seed),
            "--planner_seed",
            str(planner_seed_base),
            "--strict_persistence_steps",
            "3",
            "--max_init_frame_mae",
            str(source["restoration_limits"]["max_init_frame_mean_abs_error"]),
            "--max_init_pixels_over_8",
            str(source["restoration_limits"]["max_init_frame_pixels_over_8"]),
            "--action_diagnostics",
            "full",
            "--output",
            str(result_path),
            "--headless",
        ]
        _run_logged(command, log=group_dir / "client.log", env=env)
        result = _verify_committed(
            result_path, schema="gear-h5-eval-v1", count=len(indexed)
        )
        if result.get("paired_source_episodes") != source_episodes:
            raise RuntimeError("Gear evaluator changed explicit case order")
        if result.get("paired_planner_seeds") != planner_seeds:
            raise RuntimeError("Gear evaluator changed per-case planner seeds")
        for local_index, ((_, case), record) in enumerate(
            zip(indexed, result["per_episode"])
        ):
            metadata = record["paired_source"]
            if (
                int(metadata["source_episode"]) != int(case["source_episode"])
                or int(metadata["goal_row"]) != int(case["goal_row"])
                or int(metadata["init_row"]) != int(case["init_row"])
            ):
                raise RuntimeError("Gear result does not match frozen case identity")
            records_by_id[case["case_id"]] = {
                **record,
                "case_id": case["case_id"],
                "source_index": source_index,
                "planner_seed": planner_seeds[local_index],
            }
        group_records.append(
            {
                "source_index": source_index,
                "result": str(result_path.resolve()),
                "result_sha256": file_sha256(result_path),
                "case_count": len(indexed),
            }
        )
    if set(records_by_id) != {case["case_id"] for case in cases}:
        raise RuntimeError("Gear aggregate is incomplete")
    return [records_by_id[case["case_id"]] for case in cases], group_records


def _run_franka(
    *,
    pool: dict,
    cases: list[dict],
    socket_path: Path,
    output_dir: Path,
    env: dict[str, str],
    env_seed: int,
    planner_seed_base: int,
    planner_mode: str = "train_terminal_wm_rerank",
) -> tuple[list[dict], list[dict]]:
    demos = [int(case["demo"]) for case in cases]
    gate = _recorded(pool["franka"]["source"]["expert_gate"]["path"])
    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts/isaac_stack_eval.py"),
        "--offset",
        "25",
        "--budget",
        "50",
        "--num-eval",
        str(len(cases)),
        "--stage",
        "3",
        "--init-mode",
        "teleport",
        "--dataset-split",
        "eval",
        "--demos",
        ",".join(map(str, demos)),
        "--expert-gate-json",
        gate,
        "--policy",
        "lewm",
        "--socket",
        str(socket_path),
        "--seed",
        str(env_seed),
        "--planner-seed-base",
        str(planner_seed_base),
        "--planner-seed-per-demo",
        "--expected-planner-mode",
        planner_mode,
        "--expected-plan-bank-continuation",
        "abstain",
        "--out",
        str(output_dir / "native"),
        "--headless",
    ]
    _run_logged(command, log=output_dir / "client.log", env=env)
    native = output_dir / "native/eval_lewm_teleport_s3_offset25.json"
    result = _verify_committed(
        native, schema="franka-stack-h5-eval-v1", count=len(cases)
    )
    expected_protocol = {
        "seed": env_seed,
        "planner_seed_base": planner_seed_base,
        "planner_seed_per_demo": True,
        "policy": "lewm",
        "dataset_split": "eval",
        "offset": 25,
        "budget": 50,
        "stage": 3,
        "init_mode": "teleport",
        "expected_planner_mode": planner_mode,
        "expected_plan_bank_continuation": "abstain",
    }
    actual_protocol = {key: result.get(key) for key in expected_protocol}
    if actual_protocol != expected_protocol:
        raise RuntimeError(
            f"Franka evaluator protocol drifted: {actual_protocol}"
        )
    by_demo = {int(record["demo"]): record for record in result["episodes"]}
    if set(by_demo) != set(demos):
        raise RuntimeError("Franka result demos differ from frozen pool")
    records = []
    for case in cases:
        record = by_demo[int(case["demo"])]
        if int(record["goal_idx"]) != int(case["goal_idx"]) or int(record["start"]) != int(case["start"]):
            raise RuntimeError("Franka result does not match frozen case identity")
        if int(record["planner_seed"]) != planner_seed_base + int(case["demo"]):
            raise RuntimeError("Franka result changed the per-demo planner seed")
        records.append({**record, "case_id": case["case_id"]})
    return records, [
        {
            "result": str(native.resolve()),
            "result_sha256": file_sha256(native),
            "case_count": len(cases),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("gear", "franka"))
    parser.add_argument("--arm", required=True, choices=("L0", "L1", "C0", "C1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "default: outputs/paper_new_tasks_v1/<task>/<arm>/t7a_controls/"
            "<control-tag>/<primary|secondary>"
        ),
    )
    parser.add_argument("--planner-mode", help="override the frozen per-task planner mode")
    parser.add_argument("--control-tag", required=True, help="T7a cell tag, e.g. F1_proposal_only")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--planner-seed-base", type=int, default=42)
    parser.add_argument(
        "--development-case-limit",
        type=int,
        help=(
            "pre-freeze dev-only live smoke; omitted formal runs always use all 30 cases"
        ),
    )
    parser.add_argument("--cem-iters", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epoch < 1:
        parser.error("--epoch must be positive")
    if args.split == "test" and not args.allow_sealed_test:
        parser.error("sealed test requires --allow-sealed-test after checkpoint selection")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    planner_mode = args.planner_mode or DEFAULT_PLANNER_MODE[args.task]
    if planner_mode not in PLANNER_MODES[args.task]:
        parser.error(f"{planner_mode!r} is not a registered {args.task} planner mode")
    if args.output_dir is None:
        seed_label = {42: "primary", 10042: "secondary"}.get(args.planner_seed_base)
        if seed_label is None:
            parser.error("--output-dir is required for planner seeds other than 42/10042")
        args.output_dir = (
            ROOT / "outputs/paper_new_tasks_v1" / args.task / args.arm.lower()
            / "t7a_controls" / args.control_tag / seed_label
        )
    pool_path = args.pool.expanduser().resolve()
    pool, cases = _load_pool(pool_path, task=args.task, split=args.split)
    cases = _select_cases(
        cases,
        split=args.split,
        development_case_limit=args.development_case_limit,
    )
    case_count = len(cases)
    output_dir = args.output_dir.expanduser().resolve()
    aggregate = output_dir / "eval.json"
    if aggregate.with_suffix(".json.done").is_file():
        existing = _verify_committed(
            aggregate,
            schema="paper-h5-checkpoint-aggregate-v1",
            count=case_count,
        )
        print(json.dumps({"status": "already_complete", "successes": existing["successes"], "output": str(aggregate)}))
        return
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "paper-h5-checkpoint-dry-run-v1",
                    "task": args.task,
                    "arm": args.arm,
                    "split": args.split,
                    "case_count": case_count,
                    "development_subset": args.development_case_limit is not None,
                    "control_tag": args.control_tag,
                    "planner_mode": planner_mode,
                    "output_dir": str(output_dir),
                    "server": _server_command(
                        args.task,
                        checkpoint,
                        Path("/tmp/dry.sock"),
                        args.planner_seed_base,
                        args.cem_iters,
                        planner_mode,
                    ),
                    "case_ids": [case["case_id"] for case in cases],
                },
                indent=2,
            )
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    socket_path = Path(f"/tmp/lewm-t7a-{args.task}-{os.getpid()}.sock")
    env = dict(os.environ)
    server_log = output_dir / "server.log"
    server_command = _server_command(
        args.task, checkpoint, socket_path, args.planner_seed_base, args.cem_iters, planner_mode
    )
    server_handle = server_log.open("w", encoding="utf-8")
    server = subprocess.Popen(
        server_command,
        cwd=ROOT,
        env=env,
        stdout=server_handle,
        stderr=subprocess.STDOUT,
    )
    started = time.time()
    try:
        _wait_for_socket(socket_path, server)
        runner = _run_gear if args.task == "gear" else _run_franka
        records, components = runner(
            pool=pool,
            cases=cases,
            socket_path=socket_path,
            output_dir=output_dir,
            env=env,
            env_seed=args.env_seed,
            planner_seed_base=args.planner_seed_base,
            planner_mode=planner_mode,
        )
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        server_handle.close()
        socket_path.unlink(missing_ok=True)

    successes = sum(bool(record["success"]) for record in records)
    payload = {
        "task": args.task,
        "arm": args.arm,
        "epoch": args.epoch,
        "split": args.split,
        "control_tag": args.control_tag,
        "planner_mode": planner_mode,
        "cem_iters": args.cem_iters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "pool_manifest": str(pool_path),
        "pool_manifest_sha256": file_sha256(pool_path),
        "selected_sixty_sha256": pool[args.task]["selected_case_sha256"],
        "evaluated_case_order_sha256": hashlib.sha256(
            "\n".join(case["case_id"] for case in cases).encode("utf-8")
        ).hexdigest(),
        "env_seed": args.env_seed,
        "planner_seed_base": args.planner_seed_base,
        "case_count": len(records),
        "development_subset": args.development_case_limit is not None,
        "successes": successes,
        "success_rate": successes / len(records),
        "wall_seconds": time.time() - started,
        "component_results": components,
        "records": records,
    }
    completion = commit_json_result(
        aggregate,
        payload,
        schema_version="paper-h5-checkpoint-aggregate-v1",
        record_count=len(records),
    )
    print(json.dumps({"status": "complete", "successes": successes, "output": str(aggregate), "sha256": completion["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
