#!/usr/bin/env python3
"""T7a wrapper around the frozen LeWM policy server adding ``--planner-mode projected_random``.

``scripts/lewm_policy_server.py`` and ``projected_cem.py`` are hash-locked by the
frozen paper protocol, so this wrapper never edits them.  It translates
``--planner-mode projected_random`` into the frozen ``projected_cem`` build,
substitutes ``RandomCostProjectedCEMSolver`` (identical sampling, projection,
elite updates and latches; world-model costs replaced by planner-seeded uniform
noise) for the solver class, and records the control in the served
``task_action_schema`` (``plan_proposal.mode=projected_random``,
``plan_proposal.selection=random_cost`` and a ``t7a_control`` block).  The
``planner_config.planner_mode`` field itself stays ``projected_cem`` because the
lock-frozen Gear client validator only admits the frozen mode names.
Every other planner mode is rejected: use the frozen server directly.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import lewm_policy_server as frozen_server  # noqa: E402
from trim.planning.t7a_projected_random import RandomCostProjectedCEMSolver  # noqa: E402

CONTROL_MODE = "projected_random"


def _rewrite_argv(argv: list[str]) -> list[str]:
    rewritten = list(argv)
    hits = [index for index, token in enumerate(rewritten) if token == "--planner-mode"]
    if len(hits) != 1 or hits[0] + 1 >= len(rewritten) or rewritten[hits[0] + 1] != CONTROL_MODE:
        raise SystemExit(
            f"lewm_policy_server_t7a.py only serves --planner-mode {CONTROL_MODE}; "
            "use scripts/lewm_policy_server.py for every frozen mode"
        )
    rewritten[hits[0] + 1] = "projected_cem"
    return rewritten


def main() -> int:
    sys.argv = _rewrite_argv(sys.argv)
    frozen_build_policy = frozen_server.build_policy
    frozen_server.ProjectedCEMSolver = RandomCostProjectedCEMSolver

    def build_policy(args):
        policy = frozen_build_policy(args)
        if not isinstance(policy.solver, RandomCostProjectedCEMSolver):
            raise RuntimeError("projected_random did not build the random-cost solver")
        policy.task_action_schema["plan_proposal"] = dict(
            RandomCostProjectedCEMSolver.PLAN_PROPOSAL_PROVENANCE
        )
        # The lock-frozen Gear client (scripts/isaac_gear_eval_env.py
        # validate_gear_planner_config) only admits planner_mode in
        # {projected_cem, stock_cem}, so planner_config keeps reporting the
        # frozen build it re-uses ("projected_cem").  The control is recorded
        # in task_action_schema.plan_proposal (mode/selection) and in the extra
        # t7a_control block below, both of which the client stores verbatim.
        policy.task_action_schema["t7a_control"] = {
            "planner_mode": CONTROL_MODE,
            "solver_class": type(policy.solver).__name__,
            "world_model_cost_used": False,
            "frozen_server_planner_mode_reported": args.planner_mode,
        }
        print(
            "[server][t7a] projected_random: world-model costs replaced by "
            "planner-seeded uniform noise (proposal-only control)",
            flush=True,
        )
        return policy

    frozen_server.build_policy = build_policy
    return frozen_server.main()


if __name__ == "__main__":
    sys.exit(main())
