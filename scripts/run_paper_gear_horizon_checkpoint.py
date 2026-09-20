#!/usr/bin/env python3
"""Evaluate one loss-selected Gear checkpoint on a frozen non-H5 horizon."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.result_commit import commit_json_result, file_sha256  # noqa: E402
from scripts.run_paper_h5_checkpoint import (  # noqa: E402
    ISAAC_PYTHON,
    VENV_PYTHON,
    _run_logged,
    _verify_committed,
    _wait_for_socket,
)


DEFAULT_CONTRACT = (
    ROOT / "outputs/paper_new_tasks_v1/manifest/conditional_horizon_lock.json"
)


def _canonical_sha(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("conditional_horizon_manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_contract(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "paper-conditional-horizon-lock-v1":
        raise ValueError("unsupported conditional-horizon contract")
    if payload.get("conditional_horizon_manifest_sha256") != _canonical_sha(payload):
        raise ValueError("conditional-horizon contract self-hash changed")
    for record in payload["immutable_inputs"].values():
        source = Path(record["path"])
        if (
            not source.is_file()
            or source.stat().st_size != int(record["bytes"])
            or file_sha256(source) != record["sha256"]
        ):
            raise ValueError(f"conditional-horizon input changed: {source}")
    return payload


def server_command(
    *, checkpoint: Path, socket_path: Path, horizon: int, planner_seed: int,
    train_dataset: Path,
) -> list[str]:
    return [
        str(VENV_PYTHON),
        str(ROOT / "scripts/lewm_policy_server.py"),
        "--checkpoint", str(checkpoint),
        "--socket", str(socket_path),
        "--train-dataset", str(train_dataset),
        "--action-bound-mode", "data_quantile",
        "--action-bound-quantile", "0.001",
        "--temporal-tv-quantile", "0.99",
        "--horizon", str(horizon),
        "--receding-horizon", str(horizon),
        "--action-block", "5",
        "--num-samples", "300",
        "--n-steps", "30",
        "--topk", "30",
        "--seed", str(planner_seed),
        "--img-size", "224",
        "--action-dim", "4",
        "--task-action-schema", "gear_effective4",
        "--planner-mode", "projected_cem",
    ]


def client_command(
    *, horizon_contract: dict, socket_path: Path, result: Path,
    planner_seed_base: int,
) -> list[str]:
    source_episodes = [int(case["source_episode"]) for case in horizon_contract["cases"]]
    planner_seeds = [planner_seed_base + index for index in range(len(source_episodes))]
    return [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts/eval_gear_worldmodel.py"),
        "--eval_dataset", horizon_contract["eval_dataset"]["path"],
        "--expert_gate_json", horizon_contract["expert_gate"]["path"],
        "--goal_offset_steps", str(horizon_contract["goal_offset_raw_steps"]),
        "--paired_min_goal_step", str(max(horizon_contract["master_offsets"])),
        "--paired_required_offsets",
        *[str(value) for value in horizon_contract["master_offsets"]],
        "--paired_source_episodes", ",".join(map(str, source_episodes)),
        "--paired_planner_seeds", ",".join(map(str, planner_seeds)),
        "--episodes", str(len(source_episodes)),
        "--eval_budget", str(horizon_contract["budget_raw_steps"]),
        "--policy", "server",
        "--socket", str(socket_path),
        "--action_protocol", "effective4",
        "--seed", str(horizon_contract["environment_seed"]),
        "--planner_seed", str(planner_seed_base),
        "--strict_persistence_steps", "3",
        "--max_init_frame_mae", "12.0",
        "--max_init_pixels_over_8", "0.4",
        "--action_diagnostics", "full",
        "--output", str(result),
        "--headless",
    ]


def _case_order_sha(cases: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(str(case["case_id"]) for case in cases).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("L0", "L1", "C0", "C1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True, choices=(3, 10, 15))
    parser.add_argument("--planner-seed-base", type=int, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epoch < 1:
        parser.error("--epoch must be positive")

    contract_path = args.contract.expanduser().resolve()
    contract = load_contract(contract_path)
    if str(args.horizon) not in contract["task_matrix"]["gear"]["horizons"]:
        rejected = contract["task_matrix"]["gear"].get("rejected_horizons", {}).get(
            str(args.horizon)
        )
        raise ValueError(
            f"Gear H{args.horizon} is not a registered formal horizon: {rejected}"
        )
    horizon_contract = contract["task_matrix"]["gear"]["horizons"][str(args.horizon)]
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    aggregate = output_dir / "eval.json"
    case_count = int(horizon_contract["case_count"])
    if aggregate.with_suffix(".json.done").is_file():
        existing = _verify_committed(
            aggregate,
            schema="paper-gear-conditional-horizon-aggregate-v1",
            count=case_count,
        )
        print(json.dumps({"status": "already_complete", "successes": existing["successes"]}))
        return

    socket_path = Path(f"/tmp/lewm-paper-gear-h{args.horizon}-{os.getpid()}.sock")
    train_dataset = Path(contract["task_matrix"]["gear"]["training_dataset"]["path"])
    server = server_command(
        checkpoint=checkpoint,
        socket_path=socket_path,
        horizon=args.horizon,
        planner_seed=args.planner_seed_base,
        train_dataset=train_dataset,
    )
    native = output_dir / "native/eval.json"
    client = client_command(
        horizon_contract=horizon_contract,
        socket_path=socket_path,
        result=native,
        planner_seed_base=args.planner_seed_base,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "paper-gear-conditional-horizon-dry-run-v1",
                    "arm": args.arm,
                    "epoch": args.epoch,
                    "horizon": args.horizon,
                    "case_count": case_count,
                    "case_order_sha256": _case_order_sha(horizon_contract["cases"]),
                    "server": server,
                    "client": client,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    server_log = output_dir / "server.log"
    server_handle = server_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        server, cwd=ROOT, env=env, stdout=server_handle, stderr=subprocess.STDOUT
    )
    started = time.time()
    try:
        _wait_for_socket(socket_path, process)
        _run_logged(client, log=output_dir / "client.log", env=env)
        native_payload = _verify_committed(
            native, schema="gear-h5-eval-v1", count=case_count
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        server_handle.close()
        socket_path.unlink(missing_ok=True)

    cases = horizon_contract["cases"]
    expected_episodes = [int(case["source_episode"]) for case in cases]
    expected_seeds = [args.planner_seed_base + index for index in range(case_count)]
    if native_payload.get("paired_source_episodes") != expected_episodes:
        raise RuntimeError("conditional Gear evaluator changed frozen case order")
    if native_payload.get("paired_planner_seeds") != expected_seeds:
        raise RuntimeError("conditional Gear evaluator changed per-case planner seeds")
    records = []
    for case, record, planner_seed in zip(
        cases, native_payload["per_episode"], expected_seeds
    ):
        source = record["paired_source"]
        if (
            int(source["source_episode"]) != int(case["source_episode"])
            or int(source["goal_row"]) != int(case["goal_row"])
            or int(source["init_row"]) != int(case["init_row"])
        ):
            raise RuntimeError("conditional Gear result changed a frozen case identity")
        records.append({**record, "case_id": case["case_id"], "planner_seed": planner_seed})
    successes = sum(bool(record["success"]) for record in records)
    payload = {
        "task": "gear",
        "arm": args.arm,
        "epoch": args.epoch,
        "horizon": args.horizon,
        "goal_offset_raw_steps": int(horizon_contract["goal_offset_raw_steps"]),
        "budget_raw_steps": int(horizon_contract["budget_raw_steps"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "conditional_horizon_contract": str(contract_path),
        "conditional_horizon_manifest_sha256": contract[
            "conditional_horizon_manifest_sha256"
        ],
        "planner_mode": "projected_cem",
        "planner_seed_base": args.planner_seed_base,
        "environment_seed": int(horizon_contract["environment_seed"]),
        "case_count": case_count,
        "case_order_sha256": _case_order_sha(cases),
        "successes": successes,
        "success_rate": successes / case_count,
        "wall_seconds": time.time() - started,
        "native_result": str(native.resolve()),
        "native_result_sha256": file_sha256(native),
        "records": records,
    }
    completion = commit_json_result(
        aggregate,
        payload,
        schema_version="paper-gear-conditional-horizon-aggregate-v1",
        record_count=case_count,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "successes": successes,
                "output": str(aggregate),
                "sha256": completion["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
