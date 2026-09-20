"""T4: empirical (I_F, I_B, I_A) proxies for the trained T2 models, via fresh InfoNCE critics.

For each frozen encoder: train three critics on training latents (800 steps, batch 512, tau 0.1, proj 32):
  I_F: q = f([z; a]),  k = g(z')      I_B: q = f([z'; a]), k = g(z)      I_A: q = f([z; z']), k = g(a)
and report the held-out InfoNCE lower bound  log N - L  with N = 1024 (cap 6.93 nats), averaged over 10 batches.
Also a Gaussian forward proxy from the trained predictor: 0.5 * sum_i log(Var(z'_i) / MSE_i).
"""
import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from common import (ARM_COLORS, ARM_LABEL, ARM_ORDER, BACKWARD, FORWARD, INK, INVERSE, LIGHT, MUTED, ICLR_W,
                    mlp, plt, proj_head, savefig, style)
from t2_toy_world import RUNS_DIR, Z_DIM, X_DIM, ToyWM, flat, make_data

HERE = os.path.dirname(os.path.abspath(__file__))
N_EVAL = 1024
SUF = "" if Z_DIM == 8 else f"_d{Z_DIM}"


def critic_bound(Q, K, Qt, Kt, steps=800, seed=0):
    torch.manual_seed(seed)
    f, g = proj_head(Q.shape[1]), proj_head(K.shape[1])
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), 1e-3)
    gen = torch.Generator().manual_seed(seed)
    n = len(Q)
    for _ in range(steps):
        idx = torch.randint(0, n, (512,), generator=gen)
        q, k = F.normalize(f(Q[idx]), dim=-1), F.normalize(g(K[idx]), dim=-1)
        logits = q @ k.T / 0.1
        loss = F.cross_entropy(logits, torch.arange(512))
        opt.zero_grad()
        loss.backward()
        opt.step()
    bounds = []
    with torch.no_grad():
        for b in range(10):
            idx = torch.randint(0, len(Qt), (N_EVAL,), generator=gen)
            q, k = F.normalize(f(Qt[idx]), dim=-1), F.normalize(g(Kt[idx]), dim=-1)
            loss = F.cross_entropy(q @ k.T / 0.1, torch.arange(N_EVAL))
            bounds.append(np.log(N_EVAL) - loss.item())
    return float(np.mean(bounds))


def compute():
    data = make_data()
    s, a, s2, x, x2 = flat(data["S"], data["A"], data["X"])
    st, at, st2, xt, xt2 = flat(data["S_te"], data["A_te"], data["X_te"])
    sub = np.random.default_rng(0).choice(len(x), 20000, replace=False)
    T = lambda v: torch.tensor(v, dtype=torch.float32)
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, RUNS_DIR, "*.pt"))):
        name = os.path.basename(path)[:-3]
        arm, seed = name.rsplit("_s", 1)
        model = ToyWM()
        sd = torch.load(path)
        model.E.load_state_dict(sd["E"])
        model.P.load_state_dict(sd["P"])
        model.eval()
        with torch.no_grad():
            z, z2 = model.E(T(x[sub])), model.E(T(x2[sub]))
            zt, zt2 = model.E(T(xt)), model.E(T(xt2))
            A_, At_ = T(a[sub]), T(at)
            pred = model.predict(zt, At_)
            mse = ((pred - zt2) ** 2).mean(0)
            var = zt2.var(0)
            gauss_F = float(0.5 * torch.log(var / mse).clamp(min=0).sum())
        I_F = critic_bound(torch.cat([z, A_], 1), z2, torch.cat([zt, At_], 1), zt2, seed=int(seed))
        I_B = critic_bound(torch.cat([z2, A_], 1), z, torch.cat([zt2, At_], 1), zt, seed=int(seed))
        I_A = critic_bound(torch.cat([z, z2], 1), A_, torch.cat([zt, zt2], 1), At_, seed=int(seed))
        rows.append(dict(arm=arm, seed=int(seed), I_F_nce=I_F, I_B_nce=I_B, I_A_nce=I_A, I_F_gauss=gauss_F))
        print(rows[-1], flush=True)
    # reference: the true state (oracle latent) and the raw observation
    for label, Ztr, Zte in [("oracle state", s[sub, :4], st[:, :4]), ("raw obs x", x[sub], xt)]:
        Ztr, Zte = T(Ztr), T(Zte)
        I_F = critic_bound(torch.cat([Ztr, T(a[sub])], 1), T(s2[sub, :4]) if label == "oracle state" else T(x2[sub]),
                           torch.cat([Zte, T(at)], 1), T(st2[:, :4]) if label == "oracle state" else T(xt2))
        Z2tr = T(s2[sub, :4]) if label == "oracle state" else T(x2[sub])
        Z2te = T(st2[:, :4]) if label == "oracle state" else T(xt2)
        I_B = critic_bound(torch.cat([Z2tr, T(a[sub])], 1), Ztr, torch.cat([Z2te, T(at)], 1), Zte)
        I_A = critic_bound(torch.cat([Ztr, Z2tr], 1), T(a[sub]), torch.cat([Zte, Z2te], 1), T(at))
        rows.append(dict(arm=label, seed=-1, I_F_nce=I_F, I_B_nce=I_B, I_A_nce=I_A, I_F_gauss=float("nan")))
        print(rows[-1], flush=True)
    with open(os.path.join(HERE, f"t4_mi_space{SUF}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    style()
    csv_path = os.path.join(HERE, f"t4_mi_space{SUF}.csv")
    if os.environ.get("T4_FIG_ONLY") and os.path.exists(csv_path):
        rows = [dict(r, **{k: float(r[k]) for k in ("I_F_nce", "I_B_nce", "I_A_nce", "I_F_gauss")}) for r in csv.DictReader(open(csv_path))]
    else:
        rows = compute()

    # ---------------- figure: (left) zoomed I_F vs I_A scatter, (right) I_A per arm with references
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(ICLR_W, 2.1), gridspec_kw=dict(width_ratios=[1, 1.15]))
    cap = np.log(N_EVAL)
    short = lambda a: ARM_LABEL[a].replace(" (LeWM)", "").replace(" (ours)", "").replace(" (ctrl)", "")
    for arm in ARM_ORDER:
        r = [row for row in rows if row["arm"] == arm]
        if not r:
            continue
        xs, ys = np.array([q["I_F_nce"] for q in r]), np.array([q["I_A_nce"] for q in r])
        ax.scatter(xs, ys, s=9, color=ARM_COLORS[arm], alpha=0.6, lw=0)
        ax.scatter(xs.mean(), ys.mean(), s=30, color=ARM_COLORS[arm], edgecolor=INK, lw=0.4, zorder=3)
        _off = {"F+SIG": (-5, 4, "right"), "F+Ishuf+SIG": (5, -8, "left"), "F+I+SIG": (-5, -9, "right"),
                "F+B+I+SIG": (5, -8, "left"), "F+B+SIG": (5, -3, "left"), "F+Bmse+Imse+SIG": (5, -3, "left")}.get(arm, (4, -2, "left"))
        ax.annotate(short(arm), (xs.mean(), ys.mean()), xytext=_off[:2], ha=_off[2], textcoords="offset points", fontsize=5.0, color=INK)
    rx = [row for row in rows if row["arm"] == "raw obs x"]
    if rx:
        ax.scatter(rx[0]["I_F_nce"], rx[0]["I_A_nce"], marker="x", s=30, color=MUTED, zorder=2)
        ax.annotate("raw obs x", (rx[0]["I_F_nce"], rx[0]["I_A_nce"]), xytext=(4, -8), textcoords="offset points", fontsize=5.4, color=MUTED)
    ro = [row for row in rows if row["arm"] == "oracle state"]
    if ro:
        ax.text(0.02, 0.96, f"oracle state (off-axis):\n$\\hat I_F$={ro[0]['I_F_nce']:.2f}, $\\hat I_A$={ro[0]['I_A_nce']:.2f}",
                transform=ax.transAxes, fontsize=5.6, color=MUTED, va="top")
    ax.set_xlabel("$\\hat I_F = \\hat I(Z';Z,A)$ (nats; $\\hat I_B$ is identical to 0.01)")
    ax.set_ylabel("$\\hat I_A = \\hat I(A;Z,Z')$ (nats)")
    ax.set_xlim(5.53, 5.66)
    ax.set_ylim(0.7, 1.5)
    ax.xaxis.grid(True, color=LIGHT, lw=0.5); ax.yaxis.grid(True, color=LIGHT, lw=0.5); ax.set_axisbelow(True)
    arms = [a for a in ARM_ORDER if any(row["arm"] == a for row in rows)]
    for i, arm in enumerate(arms):
        v = [row["I_A_nce"] for row in rows if row["arm"] == arm]
        ax2.bar(i, np.mean(v), color=ARM_COLORS[arm], width=0.66)
        ax2.scatter([i] * len(v), v, s=6, color=INK, zorder=3)
    if ro:
        ax2.axhline(ro[0]["I_A_nce"], color=MUTED, lw=0.7, ls="--")
        ax2.text(len(arms) - 0.55, ro[0]["I_A_nce"] + 0.02, "true state", ha="right", fontsize=6, color=MUTED)
    if rx:
        ax2.axhline(rx[0]["I_A_nce"], color=MUTED, lw=0.7, ls=":")
        ax2.text(len(arms) - 0.55, rx[0]["I_A_nce"] + 0.02, "raw observation", ha="right", fontsize=6, color=MUTED)
    ax2.set_xticks(range(len(arms))); ax2.set_xticklabels([short(a) for a in arms], fontsize=5.0, rotation=25, ha="right")
    ax2.set_ylabel("$\\hat I_A$ (nats)")
    ax2.set_ylim(0, 2.1)
    ax2.yaxis.grid(True, color=LIGHT, lw=0.5); ax2.set_axisbelow(True)
    fig.suptitle(f"InfoNCE bounds from fresh critics on frozen latents (D={Z_DIM}; cap log 1024 = 6.93 nats)", fontsize=7.5, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, f"t4_mi_space{SUF}"))
    summary = {arm: {k: float(np.mean([r[k] for r in rows if r["arm"] == arm])) for k in ("I_F_nce", "I_B_nce", "I_A_nce", "I_F_gauss")}
               for arm in ARM_ORDER + ["oracle state", "raw obs x"]}
    json.dump(summary, open(os.path.join(HERE, f"t4_summary{SUF}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
