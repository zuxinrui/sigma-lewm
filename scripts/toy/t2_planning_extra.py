"""T2 planning follow-up on the saved models: goal frame with FRESH distractor/noise (as in the main eval,
realistic goal image) versus MATCHED distractor (d_goal = d_0, n fresh). Isolates the distractor-mismatch
failure of latent-distance planning from dynamics-model failure. Also a probe-space planner: CEM cost on
the linear-probe estimate of [o] decoded from the rolled-out latent (uses the same predictor)."""
import csv
import glob
import json
import os

import numpy as np
import torch

from common import ARM_COLORS, ARM_LABEL, ARM_ORDER, INK, LIGHT, MUTED, ICLR_W, plt, ridge_r2, savefig, style
from t2_toy_world import A_MAX, ToyWM, World, cem_plan, flat, make_data, make_planning_episodes

HERE = os.path.dirname(os.path.abspath(__file__))


def plan_eval(model, data, mode, H=8, n_ep=100, probe=None):
    mixer = data["mixer"]
    eps = make_planning_episodes(n_ep)
    W = World(99)
    succ = []
    for k, (s0, sg, o_star) in enumerate(eps):
        sg = sg.copy()
        if mode == "matched":
            sg[4:10] = s0[4:10]
        with torch.no_grad():
            z0 = model.E(torch.tensor(mixer(s0[None]), dtype=torch.float32))
            zg = model.E(torch.tensor(mixer(sg[None]), dtype=torch.float32))

        def cost(acts):
            with torch.no_grad():
                z = z0.repeat(len(acts), 1)
                at = torch.tensor(acts, dtype=torch.float32)
                for t in range(H):
                    z = model.predict(z, at[:, t])
                if mode == "probe":
                    o_hat = probe(z.numpy())[:, 2:4]
                    return ((o_hat - o_star) ** 2).sum(1)
                return ((z - zg) ** 2).sum(1).numpy()

        plan = cem_plan(cost, H, seed=k)
        s = s0[None].copy()
        for t in range(H):
            s = W.step(s, plan[t][None])
        succ.append(np.linalg.norm(s[0, 2:4] - o_star) < 0.15)
    return float(np.mean(succ))


def main():
    style()
    data = make_data()
    s, a, s2, x, x2 = flat(data["S"], data["A"], data["X"])
    sub = np.random.default_rng(0).choice(len(x), 20000, replace=False)
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.pt"))):
        name = os.path.basename(path)[:-3]
        arm, seed = name.rsplit("_s", 1)
        model = ToyWM()
        sd = torch.load(path)
        model.E.load_state_dict(sd["E"]); model.P.load_state_dict(sd["P"]); model.eval()
        with torch.no_grad():
            z = model.E(torch.tensor(x[sub], dtype=torch.float32)).numpy()
        _, Wp, (mu, sdv) = ridge_r2(z, s[sub][:, 0:4], z, s[sub][:, 0:4])
        probe = lambda zz: np.concatenate([(zz - mu) / sdv, np.ones((len(zz), 1))], 1) @ Wp
        r = dict(arm=arm, seed=int(seed))
        r["fresh"] = plan_eval(model, data, "fresh")
        r["matched"] = plan_eval(model, data, "matched")
        r["probe"] = plan_eval(model, data, "probe", probe=probe)
        rows.append(r)
        print(r, flush=True)
    with open(os.path.join(HERE, "t2_planning_protocols.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    arms = [a_ for a_ in ARM_ORDER if any(r["arm"] == a_ for r in rows)]
    fig, ax = plt.subplots(figsize=(ICLR_W, 2.0))
    wd = 0.26
    for j, (mode, hatch, alpha, lab) in enumerate([("fresh", "", 1.0, "goal frame: fresh distractor (realistic)"),
                                                   ("matched", "//", 0.55, "goal frame: matched distractor"),
                                                   ("probe", "..", 0.3, "cost on probed object position")]):
        for i, arm in enumerate(arms):
            v = [r[mode] for r in rows if r["arm"] == arm]
            ax.bar(i + (j - 1) * wd, np.mean(v), width=wd * 0.95, color=ARM_COLORS[arm], alpha=alpha, hatch=hatch,
                   edgecolor=INK, lw=0.3, label=lab if i == 0 else None)
            ax.scatter([i + (j - 1) * wd] * len(v), v, s=5, color=INK, zorder=3)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([ARM_LABEL[a_] for a_ in arms], fontsize=6.2, rotation=12)
    ax.set_ylim(0, 1.05); ax.set_ylabel("success")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("Latent CEM (h=8) under three cost protocols, 100 episodes × 3 seeds", loc="left", fontsize=8)
    savefig(fig, os.path.join(HERE, "t2_planning_protocols"))
    summ = {arm: {m: float(np.mean([r[m] for r in rows if r["arm"] == arm])) for m in ("fresh", "matched", "probe")} for arm in arms}
    json.dump(summ, open(os.path.join(HERE, "t2_planning_protocols.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
