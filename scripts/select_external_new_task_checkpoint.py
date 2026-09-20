#!/usr/bin/env python3
"""Commit the validation-loss-selected checkpoint for AC-MTM or SMWM."""

from __future__ import annotations
from slewm.paths import resolve_path as _resolve_artifact

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from scripts.result_commit import commit_json_result, file_sha256


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = Path(_resolve_artifact('checkpoints:'))
OUTPUT_ROOT = ROOT / "outputs/paper_new_tasks_v2"
SCHEMA = "paper-external-loss-selected-checkpoint-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("gear", "franka"))
    parser.add_argument("--method", required=True, choices=("acmtm", "smwm"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    expected_final = 30 if args.task == "gear" else 20
    run_name = f"paper_{args.task}_{args.method}_seed3072"
    subdir = f"paper_new_tasks_v2/{args.task}/{args.method}_seed3072"
    run_dir = CHECKPOINT_ROOT / subdir
    model_dir = CHECKPOINT_ROOT / run_name
    config_path = run_dir / "config.yaml"
    ledger_path = run_dir / "loss_control.json"
    if not config_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError("training config or loss-control ledger is missing")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if config["output_model_name"] != run_name or config["subdir"] != subdir:
        raise ValueError("runtime config identity drifted")
    if int(config["trainer"]["max_epochs"]) != expected_final:
        raise ValueError("fixed training budget drifted")
    external = config["loss"]["external_inverse"]
    expected_method = "ac-mtm" if args.method == "acmtm" else "smwm"
    if external.get("method") != expected_method:
        raise ValueError("external objective identity drifted")
    if float(config["loss"]["sigreg"]["weight"]) != 0.0:
        raise ValueError("external baseline unexpectedly enabled SIGReg")

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    decisions = {int(item["epoch"]): item for item in ledger["decisions"]}
    if sorted(decisions) != list(range(1, expected_final + 1)):
        raise ValueError("loss-control ledger is not a complete epoch chain")
    final = decisions[expected_final]
    best_epoch = int(final["best_epoch"])
    checkpoint = model_dir / f"weights_epoch_{best_epoch}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = {
        "task": args.task,
        "method": expected_method,
        "training_seed": 3072,
        "selection_authority": "validation-loss composite EMA only",
        "h5_used_for_selection": False,
        "final_epoch": expected_final,
        "best_epoch": best_epoch,
        "best_validation_ema": float(final["best_validation_ema"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "runtime_config": str(config_path.resolve()),
        "runtime_config_sha256": file_sha256(config_path),
        "upstream": {
            "ac-mtm": "43c96f66bb12c95208be68b0858ee658c7e1d363",
            "smwm": "9d22bcc15c6ad8af8cc93337a950df2b49a866b9",
        }[expected_method],
    }
    destination = OUTPUT_ROOT / args.task / args.method / "selection.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        comparable = {k:v for k,v in previous.items() if k != 'schema_version'}
        if comparable != payload:
            raise ValueError('Refusing to overwrite a different checkpoint selection')
    commit_json_result(
        destination,
        payload,
        schema_version=SCHEMA,
        record_count=1,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
