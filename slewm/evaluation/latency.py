#!/usr/bin/env python
"""Wall-clock latency of one MPC solve, measured inside the real evaluation pipeline.

The paper's CEM ladder (CEM30/8/4/2) changes the number of CEM iterations. That is a
search-budget ratio, not a speed-up: every solve also pays for observation preprocessing,
two encoder passes (observation + goal), tensor expansion and the CEM bookkeeping. This
script measures what a controller actually waits for.

It wraps the stock planning evaluation (slewm.evaluation.benchmark: same config, same
checkpoint loading, same solver object) and hooks the policy, so nothing about the planner is
re-implemented here.

BENCH_MODE=micro  (run with eval.num_eval=1)
    On the first replan, takes the real single-environment observation and times, with
    CUDA synchronisation, WARMUP + REPS repetitions of
      * preprocessing  (policy._prepare_info: uint8 image -> normalised 224 px tensor)
      * solve(n)       (CEMSolver.solve at n CEM iterations, batch of one environment)
    for every n in LADDER, then a separately-instrumented breakdown at CEM30
    (encoder / predictor roll-outs / CEM bookkeeping). Exits without running the episode.

BENCH_MODE=e2e    (standard protocol, eval.num_eval=50)
    Runs the whole evaluation untouched and accumulates the synchronised time spent in
    replanning, next to the total wall-clock of world.evaluate (simulator, rendering and
    video writing included).

Output: one JSON per run at $BENCH_OUT.
"""
import atexit
import json
import os
import statistics
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.solver import CEMSolver

MODE = os.environ.get("BENCH_MODE", "micro")
OUT = os.environ["BENCH_OUT"]
LADDER = [int(x) for x in os.environ.get("BENCH_LADDER", "30,16,8,4,2,1").split(",")]
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
REPS = int(os.environ.get("BENCH_REPS", "50"))
TAG = os.environ.get("BENCH_TAG", "")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _stats(ms):
    s = sorted(ms)
    return dict(
        n=len(s),
        mean=statistics.fmean(s),
        std=statistics.pstdev(s),
        median=statistics.median(s),
        p05=s[int(0.05 * (len(s) - 1))],
        p95=s[int(round(0.95 * (len(s) - 1)))],
        min=s[0],
        max=s[-1],
    )


def _env_meta():
    meta = dict(mode=MODE, tag=TAG, argv=sys.argv[1:], torch=torch.__version__,
                cuda=torch.version.cuda, ladder=LADDER, warmup=WARMUP, reps=REPS,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta


def _write(payload):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, OUT)
    print(f"[bench] wrote {OUT}", flush=True)


# --------------------------------------------------------------------------- micro
def _micro(policy, raw_info):
    solver, model = policy.solver, policy.solver.model
    n_envs = policy.env.num_envs
    assert n_envs == 1, f"micro mode needs eval.num_eval=1, got {n_envs} environments"

    res = dict(meta=_env_meta(), n_envs=n_envs)
    res["meta"].update(num_samples=solver.num_samples, topk=solver.topk,
                       horizon=solver.horizon, action_dim=solver.action_dim,
                       solver_batch_size=solver.batch_size)

    # -- preprocessing: raw uint8 observation -> model input
    res["prep_ms"] = _stats(_timed(lambda: policy._prepare_info(raw_info)))

    info = policy._prepare_info(raw_info)
    info.pop("_needs_flush", None)
    res["obs_shapes"] = {k: list(v.shape) for k, v in info.items() if torch.is_tensor(v)}

    # -- solve latency per CEM budget, identical input for every budget
    n_default = solver.n_steps
    res["solve_ms"] = {}
    for n in LADDER:
        solver.n_steps = n
        res["solve_ms"][str(n)] = _stats(_timed(lambda: solver.solve(info, init_action=None)))
        print(f"[bench] CEM{n:<3d} solve median {res['solve_ms'][str(n)]['median']:8.2f} ms", flush=True)

    # -- linear model: latency = fixed + per_iteration * n
    xs = np.array(LADDER, dtype=float)
    ys = np.array([res["solve_ms"][str(n)]["median"] for n in LADDER])
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(((ys - pred) ** 2).sum()); ss_tot = float(((ys - ys.mean()) ** 2).sum())
    res["linear_fit"] = dict(fixed_ms=float(intercept), per_iteration_ms=float(slope),
                             r2=1.0 - ss_res / ss_tot if ss_tot else 1.0)

    # -- breakdown at the default budget. The extra synchronisation points make the sum
    #    slightly slower than the headline numbers above; read it as proportions.
    acc = dict(encode=0.0, get_cost=0.0)
    calls = dict(encode=0, get_cost=0)

    def _wrap(name, fn):
        def inner(*a, **k):
            _sync(); t0 = time.perf_counter()
            out = fn(*a, **k)
            _sync(); acc[name] += (time.perf_counter() - t0) * 1e3; calls[name] += 1
            return out
        return inner

    solver.n_steps = 30
    o_enc, o_cost = model.encode, model.get_cost
    model.encode, model.get_cost = _wrap("encode", o_enc), _wrap("get_cost", o_cost)
    try:
        for _ in range(WARMUP):
            solver.solve(info, init_action=None)
        for k in acc:
            acc[k] = 0.0; calls[k] = 0
        reps_b = max(10, REPS // 2)
        total = _timed(lambda: solver.solve(info, init_action=None), reps=reps_b, warmup=0)
    finally:
        model.encode, model.get_cost = o_enc, o_cost
    tot = statistics.fmean(total)
    enc = acc["encode"] / reps_b
    cost = acc["get_cost"] / reps_b
    res["breakdown_cem30_ms"] = dict(
        total=tot, encoder=enc, predictor_rollouts=cost - enc, cem_bookkeeping=tot - cost,
        encoder_calls_per_solve=calls["encode"] / reps_b,
        get_cost_calls_per_solve=calls["get_cost"] / reps_b)

    solver.n_steps = n_default
    res["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    _write(res)


# --------------------------------------------------------------------------- e2e
E2E = dict(replan_ms=[], replan_envs=[], get_action_ms=0.0, get_action_calls=0,
           evaluate_wall_s=None)


def _install():
    orig_get_action = WorldModelPolicy.get_action
    orig_solve = CEMSolver.solve
    state = dict(done=False)

    if MODE == "micro":
        def get_action(self, info_dict, **kw):
            if not state["done"]:
                state["done"] = True
                _micro(self, info_dict)
                # hydra would turn an exception into exit code 1 plus a traceback
                sys.stdout.flush(); sys.stderr.flush()
                os._exit(0)
            return orig_get_action(self, info_dict, **kw)
        WorldModelPolicy.get_action = get_action
        return

    def solve(self, info_dict, init_action=None):
        _sync(); t0 = time.perf_counter()
        out = orig_solve(self, info_dict, init_action=init_action)
        _sync()
        E2E["replan_ms"].append((time.perf_counter() - t0) * 1e3)
        E2E["replan_envs"].append(len(next(iter(info_dict.values()))))
        return out

    def get_action(self, info_dict, **kw):
        t0 = time.perf_counter()
        out = orig_get_action(self, info_dict, **kw)
        E2E["get_action_ms"] += (time.perf_counter() - t0) * 1e3
        E2E["get_action_calls"] += 1
        return out

    CEMSolver.solve = solve
    WorldModelPolicy.get_action = get_action

    orig_eval = swm.World.evaluate

    def evaluate(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_eval(self, *a, **k)
        E2E["evaluate_wall_s"] = time.perf_counter() - t0
        E2E["metrics"] = {kk: (float(v) if np.isscalar(v) else None) for kk, v in out.items()}
        return out

    swm.World.evaluate = evaluate

    def dump():
        if not E2E["replan_ms"]:
            return
        envs = sum(E2E["replan_envs"])
        tot = sum(E2E["replan_ms"])
        _write(dict(meta=_env_meta(),
                    solve_calls=len(E2E["replan_ms"]), env_solves=envs,
                    solve_total_s=tot / 1e3, solve_ms_per_env=tot / envs,
                    get_action_total_s=E2E["get_action_ms"] / 1e3,
                    get_action_calls=E2E["get_action_calls"],
                    evaluate_wall_s=E2E["evaluate_wall_s"], metrics=E2E.get("metrics")))
    atexit.register(dump)


if __name__ == "__main__":
    _install()
    from slewm.evaluation.benchmark import run

    run()
