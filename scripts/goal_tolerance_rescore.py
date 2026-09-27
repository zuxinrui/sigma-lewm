"""Re-score goal-tolerance rollouts (outputs/goal_tolerance/*.npz) under four tolerance levels T0..T3.

Success(T) = native any-time rule: the goal error is within T at some step of the budget.
Also reports the final-step variant. Paired test: exact McNemar (binomial on discordant pairs), same episodes.

  python scripts/goal_tolerance_rescore.py --dir <npz dir> --out docs/results/fig5_tolerance.tsv

Port of le-wm scripts/goal_tolerance_rescore.py (058d998) for Figure 5 at 3 training x 5 planner seeds.
Files are <task>_<arm>_ts<training seed>_s<planner seed>.npz; each arm pools its training seeds
(3072, 1, 2) and planner seeds (0-4) in that fixed order, so every arm covers the same episode keys.
The paired McNemar test compares the first two arms (Σ-LeWM vs LeWM), episode by episode, pairing
training seeds by position.
"""
import argparse, glob, json, math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
# Tolerance levels, loose -> strict. Push-T is a joint (position, angle) criterion; both are tightened together.
THRESHOLDS = {
    "tworoom": {"unit": "px",  "levels": [16.0, 8.0, 4.0, 2.0]},          # native: dist < 16 (224-px frame)
    "reacher": {"unit": "rad", "levels": [0.05, 0.03, 0.02, 0.01]},        # native: max_j |dq_j| < 0.05
    "pusht":   {"unit": "px|deg", "levels": [(20, 20), (15, 15), (10, 10), (7, 7)]},   # native: pos<20 (512 frame) & ang<20 deg
    "cube":    {"unit": "m",   "levels": [0.04, 0.02, 0.01, 0.005]},        # native: |p - p*| <= 0.04
}
TASKS = ["tworoom", "reacher", "pusht", "cube"]
LEVEL_NAMES = ["T0", "T1", "T2", "T3"]


def within(task, err, thr):
    """err: [..., k] native error components -> bool [...]. Mirrors each env's own inequality (strict vs non-strict)."""
    if task == "pusht":
        tp, ta = thr
        return (err[..., 0] < tp) & (err[..., 1] < math.radians(ta))
    if task == "tworoom":
        return err[..., 0] < thr
    if task == "reacher":
        return err[..., 0] < thr
    if task == "cube":
        return err[..., 0] <= thr
    raise ValueError(task)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


TRAINING_SEEDS = ["3072", "1", "2"]


def load(task, arm, d):
    files = []
    for ts in TRAINING_SEEDS:
        files += sorted(glob.glob(str(d / f"{task}_{arm}_ts{ts}_s*.npz")),
                        key=lambda f: int(Path(f).stem.rsplit("_s", 1)[1]))
    errs, nat, keys = [], [], []
    for f in files:
        z = np.load(f)
        errs.append(z["errors"]); nat.append(z["native_terminated"])
        stem = Path(f).stem
        s = int(stem.rsplit("_s", 1)[1]); slot = TRAINING_SEEDS.index(stem.rsplit("_ts", 1)[1].split("_s")[0])
        keys += [(slot, s, int(e), int(t)) for e, t in zip(z["episodes"], z["start_steps"])]
    if not files:
        return None
    return {"errors": np.concatenate(errs), "native": np.concatenate(nat), "keys": keys, "n_files": len(files)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["sigma_lewm", "lewm", "subjepa", "rcaux", "smwm", "acmtm"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs/results/fig5_tolerance.tsv"))
    a = ap.parse_args()
    d = Path(a.dir); rows = []; pairs = []
    for task in TASKS:
        data = {arm: load(task, arm, d) for arm in a.arms}
        missing = [k for k, v in data.items() if v is None]
        if any(k in missing for k in a.arms[:2]):
            print(f"[skip] {task}: missing primary arms {missing}"); continue
        if missing:
            print(f"[note] {task}: optional arms missing {missing}"); data = {k: v for k, v in data.items() if v is not None}
        arms = list(data); base = data[arms[0]]["keys"]
        for arm in arms[1:]:
            assert data[arm]["keys"] == base, f"{task}: episode sets differ between arms"
        n = len(base); spec = THRESHOLDS[task]
        for li, thr in enumerate(spec["levels"]):
            succ = {}
            for arm in arms:
                e = data[arm]["errors"]
                s_any = within(task, e, thr).any(1); s_fin = within(task, e[:, -1], thr)
                succ[arm] = s_any
                rows.append(dict(task=task, level=LEVEL_NAMES[li], threshold=str(thr), unit=spec["unit"], arm=arm,
                                 successes=int(s_any.sum()), episodes=n, success_pct=round(100 * s_any.mean(), 2),
                                 wilson_lo=round(wilson(int(s_any.sum()), n)[0], 2), wilson_hi=round(wilson(int(s_any.sum()), n)[1], 2),
                                 final_step_pct=round(100 * s_fin.mean(), 2),
                                 native_any_pct=round(100 * data[arm]["native"].any(1).mean(), 2) if li == 0 else "",
                                 n_seed_files=data[arm]["n_files"]))
            if len(a.arms) >= 2:
                o, l = succ[a.arms[0]], succ[a.arms[1]]
                b, c = int((o & ~l).sum()), int((~o & l).sum())
                p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
                pairs.append(dict(task=task, level=LEVEL_NAMES[li], threshold=str(thr), delta_pp=round(100 * (o.mean() - l.mean()), 2),
                                  ours_only=b, lewm_only=c, both=int((o & l).sum()), neither=int((~o & ~l).sum()), mcnemar_p=p))
        # retention relative to T0
        for arm in arms:
            t0 = [r for r in rows if r["task"] == task and r["arm"] == arm and r["level"] == "T0"][0]["successes"]
            for r in rows:
                if r["task"] == task and r["arm"] == arm:
                    r["retention"] = round(r["successes"] / t0, 3) if t0 else float("nan")
    import csv
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "level", "threshold", "unit", "arm", "successes", "episodes", "success_pct", "wilson_lo", "wilson_hi",
              "retention", "final_step_pct", "native_any_pct", "n_seed_files"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t" if out.suffix == ".tsv" else ","); w.writeheader(); w.writerows(rows)
    out.with_suffix(".pairs.json").write_text(json.dumps(pairs, indent=1))
    print(f"wrote {out} and {out.with_suffix('.pairs.json')}")
    for r in rows:
        print(f"{r['task']:8s} {r['level']} {r['threshold']:>12s} {r['arm']:10s} {r['successes']:4d}/{r['episodes']} = {r['success_pct']:5.1f}% "
              f"[{r['wilson_lo']:.1f},{r['wilson_hi']:.1f}] ret={r['retention']:.2f} final={r['final_step_pct']:.1f}")
    for p in pairs:
        print(f"{p['task']:8s} {p['level']} delta={p['delta_pp']:+6.1f}pp  ours-only={p['ours_only']} lewm-only={p['lewm_only']} p={p['mcnemar_p']:.3g}")


if __name__ == "__main__":
    main()
