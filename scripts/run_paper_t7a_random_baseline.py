#!/usr/bin/env python3
"""T7a Gear random-action floor on the sealed H5 pool (frozen baseline driver, policy=random).

``scripts/run_paper_h5_baseline.py`` is hash-locked and registers only zero/expert
for Gear, so this additive wrapper reuses its ``run_gear`` unchanged with
``--policy random``.  The Gear evaluator's ``random`` policy draws
``rng.uniform(-1, 1, size=4)`` per control step from ``np.random.default_rng(--seed)``
(uniform in the normalised effective4 command space, NOT the training-action
distribution; see scripts/eval_gear_worldmodel.py around line 636).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.result_commit import commit_json_result, file_sha256  # noqa: E402
from scripts.run_paper_h5_baseline import run_gear  # noqa: E402
from scripts.run_paper_h5_checkpoint import (  # noqa: E402
    DEFAULT_POOL,
    _load_pool,
    _select_cases,
    _verify_committed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--development-case-limit", type=int)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_sealed_test:
        parser.error("sealed test requires --allow-sealed-test")
    pool_path = args.pool.expanduser().resolve()
    pool, cases = _load_pool(pool_path, task="gear", split=args.split)
    cases = _select_cases(cases, split=args.split, development_case_limit=args.development_case_limit)
    output_dir = args.output_dir.expanduser().resolve()
    aggregate = output_dir / "eval.json"
    if aggregate.with_suffix(".json.done").is_file():
        existing = _verify_committed(aggregate, schema="paper-h5-baseline-aggregate-v1", count=len(cases))
        print(json.dumps({"status": "already_complete", "successes": existing["successes"], "output": str(aggregate)}))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records, components = run_gear(
        pool=pool, cases=cases, policy="random", output_dir=output_dir,
        env=dict(os.environ), env_seed=args.env_seed,
    )
    successes = sum(bool(record["success"]) for record in records)
    payload = {
        "task": "gear", "policy": "random", "split": args.split,
        "random_policy_definition": (
            "per-step rng.uniform(-1, 1, size=4) in the normalised effective4 command space, "
            "np.random.default_rng(env_seed); not the training-action distribution"
        ),
        "pool_manifest": str(pool_path), "pool_manifest_sha256": file_sha256(pool_path),
        "selected_sixty_sha256": pool["gear"]["selected_case_sha256"],
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
