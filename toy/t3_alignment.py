"""T3: the PI's alignment example, learned.

Phase 1 (static): an immovable object; the agent's stance/cue c in {-1,+1} determines the action cluster
(push-left vs push-right, a_x = 0.1 c + noise). The object does not move; c is resampled i.i.d. each step,
so c_t carries no forward information (I_F gains nothing from c) but a_t is a function of c_t
(I_A and I_B both gain ~1 bit if z_t encodes c_t).
State s = [c, o (static object pos), u1,u2 (slow random-walk distractor), n1,n2 (iid noise)] -> x in R^16.
Latent D = 4 so the encoder must choose (o,u1,u2 = 3 predictable dims; the 4th slot goes to c or n).

Phase 2 (consequence): the wall disappears and the push displaces the object, o' = clip(o + a_x). Note that
with a_x = 0.1c + N(0,0.02) the sign of a_x equals c in every sampled step, so the gate `sign(a_x) == c` in
the code below is never closed; what the stance actually determines is the DIRECTION of the displacement,
not whether it happens. With frozen phase-1 encoders, fit a predictor and ask whether the latent can predict
what the push will do (scored by phase2_disp_r2; phase2_move_acc is also logged but its label is ~97%
constant, so it is not plotted).
"""
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

from common import (ACTION, ARM_COLORS, ARM_LABEL, ARM_ORDER, BACKWARD, FORWARD, GREY, INK, INVERSE, LIGHT,
                    MUTED, ICLR_W, TEAL, SIGReg, infonce_idm_loss, mlp, plt, proj_head, ridge_r2, savefig,
                    style, symmetric_dcl_loss)

HERE = os.path.dirname(os.path.abspath(__file__))
S_DIM, X_DIM, TAU = 6, 16, 0.1
Z_DIM = int(os.environ.get("T3_ZDIM", "4"))  # 4 = spare capacity, 3 = tight (3 predictable dims)
SUF = "" if Z_DIM == 4 else f"_d{Z_DIM}"
ARMS = {"F+SIG": (0.0, 0.0), "F+B+SIG": (0.10, 0.0), "F+I+SIG": (0.0, 0.05), "F+B+I+SIG": (0.10, 0.05)}
COL = {a: ARM_COLORS[a] for a in ARMS}  # paper palette (common.ARM_COLORS)


class Mixer:
    def __init__(self, seed=777):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(size=(X_DIM, S_DIM))
        W1 /= np.linalg.norm(W1, axis=1, keepdims=True)
        self.W1, self.b1 = 1.5 * W1, rng.uniform(-0.3, 0.3, X_DIM)
        self.W2, _ = np.linalg.qr(rng.normal(size=(X_DIM, X_DIM)))
        self.mu = None

    def __call__(self, s):
        x = np.tanh(s @ self.W1.T + self.b1) @ self.W2.T
        if self.mu is None:
            self.mu, self.sd = x.mean(0), x.std(0) + 1e-8
        return (x - self.mu) / self.sd


def gen(n_ep, T, seed, phase):
    rng = np.random.default_rng(seed)
    S = np.zeros((n_ep, T + 1, S_DIM))
    A = np.zeros((n_ep, T, 2))
    s = np.zeros((n_ep, S_DIM))
    s[:, 0] = rng.choice([-1.0, 1.0], n_ep)
    s[:, 1] = rng.uniform(-0.8, 0.8, n_ep)
    s[:, 2:4] = rng.uniform(-1, 1, (n_ep, 2))
    s[:, 4:6] = np.clip(rng.normal(0, 0.3, (n_ep, 2)), -1, 1)
    S[:, 0] = s
    for t in range(T):
        a = np.stack([0.1 * s[:, 0] + rng.normal(0, 0.02, n_ep), rng.normal(0, 0.02, n_ep)], 1)
        s2 = s.copy()
        s2[:, 0] = rng.choice([-1.0, 1.0], n_ep)  # fresh stance
        if phase == 2:
            moves = np.sign(a[:, 0]) == s[:, 0]
            s2[:, 1] = np.clip(s[:, 1] + np.where(moves, a[:, 0], 0.0), -1, 1)
        u = s[:, 2:4] + rng.normal(0, 0.02, (n_ep, 2))
        u = np.where(u > 1, 2 - u, u)
        u = np.where(u < -1, -2 - u, u)
        s2[:, 2:4] = u
        s2[:, 4:6] = np.clip(rng.normal(0, 0.3, (n_ep, 2)), -1, 1)
        A[:, t], S[:, t + 1] = a, s2
        s = s2
    return S, A


def flat(S, A, X):
    return (S[:, :-1].reshape(-1, S_DIM), A.reshape(-1, 2), S[:, 1:].reshape(-1, S_DIM),
            X[:, :-1].reshape(-1, X_DIM), X[:, 1:].reshape(-1, X_DIM))


def train_encoder(arm, seed, X, X2, Aa, steps=2500, batch=512):
    torch.manual_seed(seed)
    wB, wI = ARMS[arm]
    E = mlp(X_DIM, Z_DIM, hidden=128, depth=2)
    P = mlp(Z_DIM + 2, Z_DIM, hidden=128, depth=2)
    fB, gB, fI, gI = proj_head(Z_DIM + 2), proj_head(Z_DIM), proj_head(2 * Z_DIM), proj_head(2)
    params = list(E.parameters()) + list(P.parameters()) + list(fB.parameters()) + list(gB.parameters()) \
        + list(fI.parameters()) + list(gI.parameters())
    opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    sig = SIGReg(num_proj=64)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        z, z2, ab = E(X[idx]), E(X2[idx]), Aa[idx]
        loss = F.mse_loss(P(torch.cat([z, ab], -1)), z2) + 0.09 * sig(z.unsqueeze(0))
        if wB > 0:
            loss = loss + wB * symmetric_dcl_loss(fB(torch.cat([z2, ab], -1)), gB(z), temperature=TAU)
        if wI > 0:
            loss = loss + wI * infonce_idm_loss(fI(torch.cat([z, z2], -1)), gI(ab), ab, temperature=TAU)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return E


def fit_phase2_predictor(E, X, X2, Aa, steps=1500, batch=512, seed=0):
    torch.manual_seed(seed)
    P = mlp(Z_DIM + 2, Z_DIM, hidden=128, depth=2)
    opt = torch.optim.Adam(P.parameters(), 1e-3)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        Z, Z2 = E(X), E(X2)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        loss = F.mse_loss(P(torch.cat([Z[idx], Aa[idx]], -1)), Z2[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return P


def main():
    style()
    mixer = Mixer()
    S, A = gen(600, 40, 11, phase=1)
    X = mixer(S.reshape(-1, S_DIM)).reshape(*S.shape[:2], X_DIM)
    St, At = gen(150, 40, 12, phase=1)
    Xt = mixer(St.reshape(-1, S_DIM)).reshape(*St.shape[:2], X_DIM)
    s, a, s2, x, x2 = flat(S, A, X)
    st, at, st2, xt, xt2 = flat(St, At, Xt)
    T = lambda v: torch.tensor(v, dtype=torch.float32)
    Xtr, X2tr, Atr = T(x), T(x2), T(a)
    # phase 2 data (consequence)
    S2p, A2p = gen(600, 40, 21, phase=2)
    X2p = mixer(S2p.reshape(-1, S_DIM)).reshape(*S2p.shape[:2], X_DIM)
    S2t, A2t = gen(150, 40, 22, phase=2)
    X2t = mixer(S2t.reshape(-1, S_DIM)).reshape(*S2t.shape[:2], X_DIM)
    ps, pa, ps2, px, px2 = flat(S2p, A2p, X2p)
    qs, qa, qs2, qx, qx2 = flat(S2t, A2t, X2t)

    if os.environ.get("T3_FIG_ONLY"):
        rows = [dict(r, **{k: float(r[k]) for k in r if k not in ("arm",)}) for r in csv.DictReader(open(os.path.join(HERE, f"t3_alignment{SUF}.csv")))]
        npz = np.load(os.path.join(HERE, f"t3_pcs{SUF}.npz"))
        pcs = {arm: (npz[f"{arm}__pc"], npz[f"{arm}__ct"]) for arm in ARMS}
        return figure(rows, pcs)
    rows, pcs = [], {}
    for arm in ARMS:
        for seed in range(3):
            E = train_encoder(arm, seed, Xtr, X2tr, Atr)
            with torch.no_grad():
                z, zt = E(Xtr).numpy(), E(T(xt)).numpy()
            c, ct = (s[:, 0] > 0).astype(int), (st[:, 0] > 0).astype(int)
            clf = LogisticRegression(max_iter=500).fit(z, c)
            acc = clf.score(zt, ct)
            sil = silhouette_score(zt[:4000], ct[:4000])
            r2 = {k: float(ridge_r2(z, s[:, sl], zt, st[:, sl])[0].mean())
                  for k, sl in dict(c=slice(0, 1), o=slice(1, 2), u=slice(2, 4), n=slice(4, 6)).items()}
            # phase-2 consequence: frozen encoder, fresh predictor on phase-2 data, probe-space o' error
            P2 = fit_phase2_predictor(E, T(px), T(px2), T(pa), seed=seed)
            with torch.no_grad():
                zq, zq2 = E(T(qx)), E(T(qx2))
                zpred = P2(torch.cat([zq, T(qa)], -1))
                zp = E(T(px))
            _, W, (mu, sd) = ridge_r2(zp.numpy(), ps[:, 1:2], zp.numpy(), ps[:, 1:2])
            probe = lambda zz: np.concatenate([(zz - mu) / sd, np.ones((len(zz), 1))], 1) @ W
            o_pred = probe(zpred.numpy())[:, 0]
            o_true = qs2[:, 1]
            o_now = qs[:, 1]
            moved = np.sign(qa[:, 0]) == qs[:, 0]
            # did the model predict the displacement sign correctly on pushes?
            disp_pred, disp_true = o_pred - probe(zq.numpy())[:, 0], o_true - o_now
            r2_disp = 1 - ((disp_pred - disp_true) ** 2).sum() / ((disp_true - disp_true.mean()) ** 2).sum()
            move_acc = float(((np.abs(disp_pred) > 0.05) == moved).mean())
            rows.append(dict(arm=arm, seed=seed, cue_probe_acc=acc, silhouette=sil, **{f"r2_{k}": v for k, v in r2.items()},
                             phase2_disp_r2=float(r2_disp), phase2_move_acc=move_acc))
            print(rows[-1], flush=True)
            if seed == 0:
                pcs[arm] = (PCA(2).fit_transform(zt[:3000]), ct[:3000])
    with open(os.path.join(HERE, f"t3_alignment{SUF}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(os.path.join(HERE, f"t3_pcs{SUF}.npz"), **{f"{k}__pc": v[0] for k, v in pcs.items()}, **{f"{k}__ct": v[1] for k, v in pcs.items()})
    return figure(rows, pcs)


ARM_TITLE = {a: ARM_LABEL[a] for a in ARMS}
ARM_SHORT = {"F+SIG": "F", "F+B+SIG": "F+B", "F+I+SIG": "F+I", "F+B+I+SIG": "F+B+I"}
CUE_COL = {0: MUTED, 1: ACTION}  # push-left = slate, push-right = amber (paper palette)


def env_panels(fig, gs):
    """Row 1: the environment itself, drawn from the same gen() the experiment uses.

    Left: the action cloud (two clusters, one per stance). Middle: phase-1 and phase-2 object
    traces, same generator, same colour code. Right: a schematic of the state vector and of the
    latent budget (3 predictable dims, D - 3 spare slots).
    """
    Te = 24
    Se, Ae = gen(400, Te, 101, phase=1)   # env illustration only; independent rng, no arm uses it
    Sp, _ = gen(400, Te, 102, phase=2)
    sub = gs.subgridspec(1, 4, width_ratios=[1.02, 1.12, 1.12, 1.62], wspace=0.42)

    # (a) the action cloud: a_x = 0.1 c + noise -> two clusters, one per stance
    ax = fig.add_subplot(sub[0, 0])
    ce, aa = Se[:, :-1, 0].reshape(-1)[:2500], Ae.reshape(-1, 2)[:2500]
    for cval, sgn in [(0, -1.0), (1, 1.0)]:
        m = ce == sgn
        ax.scatter(aa[m, 0], aa[m, 1], s=1.5, c=CUE_COL[cval], alpha=0.45, lw=0, rasterized=True)
    ax.axvline(0.0, color=LIGHT, lw=0.6, zorder=0)
    ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.095, 0.135)
    ax.set_xticks([-0.1, 0.0, 0.1]); ax.set_yticks([-0.05, 0.0, 0.05])
    ax.tick_params(labelsize=5.8, pad=1.2, length=2)
    ax.set_xlabel("$a_x$", fontsize=6.2, labelpad=1.0)
    ax.set_ylabel("$a_y$", fontsize=6.2, labelpad=1.0)
    ax.set_title("action $a_t=[0.1\\,c+\\epsilon,\\ \\epsilon]$", fontsize=6.2, color=INK, pad=11.5)
    ax.text(0.5, 1.015, "one cluster per stance", transform=ax.transAxes, fontsize=5.8,
            color=MUTED, ha="center", va="bottom")
    ax.text(-0.115, 0.128, "push-left\n$c=-1$", fontsize=5.0, color=MUTED, ha="center", va="top",
            linespacing=1.25)
    ax.text(0.115, 0.128, "push-right\n$c=+1$", fontsize=5.0, color=ACTION, ha="center", va="top",
            linespacing=1.25)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(LIGHT)

    # (b, c) object traces: identical actions, wall on / wall off
    for k, (S_, ttl, sub_ttl) in enumerate([
            (Se, "phase 1: the wall", "$o$ never moves"),
            (Sp, "phase 2: the wall is gone", "$o'=o+a_x$, sign$\\,a_x=c$")]):
        ax = fig.add_subplot(sub[0, 1 + k])
        for ep in range(7):
            o, c = S_[ep, :, 1], S_[ep, :-1, 0]
            ax.plot(np.arange(Te + 1), o, color=INK, lw=0.7, alpha=0.45, zorder=2,
                    solid_capstyle="round")
            ax.scatter(np.arange(Te), o[:-1], s=3.4, lw=0, zorder=3,
                       c=[CUE_COL[int(v > 0)] for v in c])
        ax.set_xlim(-0.8, Te + 0.8); ax.set_ylim(-1.08, 1.08)
        ax.set_yticks([-1, 0, 1]); ax.set_xticks([0, Te])
        ax.tick_params(labelsize=5.8, pad=1.2, length=2)
        ax.set_xlabel("step $t$", fontsize=6.2, labelpad=1.0)
        if k == 0:
            ax.set_ylabel("object $o$", fontsize=6.2, labelpad=1.0)
        ax.set_title(ttl, fontsize=6.2, color=INK, pad=11.5)
        ax.text(0.5, 1.015, sub_ttl, transform=ax.transAxes, fontsize=5.8, color=MUTED,
                ha="center", va="bottom")
        ax.yaxis.grid(True, color=LIGHT, lw=0.5); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(LIGHT)

    # (d) schematic: state vector -> observation -> latent budget
    ax = fig.add_subplot(sub[0, 3]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box = lambda x, y, w, h, **kw: ax.add_patch(plt.Rectangle((x, y), w, h, lw=0.7, **kw))
    names = ["$c$", "$o$", "$u_1$", "$u_2$", "$n_1$", "$n_2$"]
    fills = [ACTION, LIGHT, LIGHT, LIGHT, "white", "white"]
    edges = [ACTION, MUTED, MUTED, MUTED, LIGHT, LIGHT]
    w, gap, y0 = 0.102, 0.022, 0.70
    for i, nm in enumerate(names):
        x = i * (w + gap)
        box(x, y0, w, 0.15, fc=fills[i], ec=edges[i], alpha=0.9 if i else 0.35)
        ax.text(x + w / 2, y0 + 0.075, nm, fontsize=6.0, color=INK, ha="center", va="center")
    ax.text(0.0, y0 + 0.185, "state $s_t$", fontsize=6.2, color=INK, ha="left", va="bottom")
    grp = [(0, 1, "cue", ACTION), (1, 4, "predictable", MUTED), (4, 6, "noise", GREY)]
    for i0, i1, lab, col in grp:
        xa, xb = i0 * (w + gap), (i1 - 1) * (w + gap) + w
        ax.plot([xa, xb], [y0 - 0.035] * 2, color=col, lw=0.7)
        ax.text((xa + xb) / 2, y0 - 0.075, lab, fontsize=4.9, color=col, ha="center", va="top")
    ax.annotate("", xy=(0.30, 0.44), xytext=(0.30, 0.545),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, shrinkA=0, shrinkB=0))
    ax.text(0.34, 0.495, "tanh mix $\\rightarrow x_t\\in\\mathbb{R}^{16}$",
            fontsize=5.8, color=MUTED, ha="left", va="center")
    ax.annotate("", xy=(0.30, 0.235), xytext=(0.30, 0.34),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, shrinkA=0, shrinkB=0))
    ax.text(0.34, 0.29, "encoder $\\rightarrow z_t\\in\\mathbb{R}^{%d}$" % Z_DIM,
            fontsize=5.8, color=MUTED, ha="left", va="center")
    spare = Z_DIM - 3
    for j in range(Z_DIM):
        x = j * (w + gap)
        taken = j < 3
        box(x, 0.075, w, 0.15, fc=LIGHT if taken else "white", ec=MUTED if taken else ACTION,
            alpha=0.9, ls="-" if taken else (0, (1.6, 1.2)))
        ax.text(x + w / 2, 0.15, ["$o$", "$u_1$", "$u_2$"][j] if taken else "?",
                fontsize=6.0, color=INK if taken else ACTION, ha="center", va="center")
    tail = ("1 spare slot:\ncue or noise" if spare == 1 else
            ("no spare slot" if spare == 0 else "%d spare slots" % spare))
    ax.text(Z_DIM * (w + gap) + 0.015, 0.15, tail, fontsize=5.8,
            color=ACTION if spare else MUTED, ha="left", va="center", linespacing=1.35)


def figure(rows, pcs):
    """Left: PCA of the phase-1 latent per arm (ours last). Right: phase-2 consequence accuracy."""
    style()
    arms = [a for a in ARM_ORDER if a in ARMS]  # ablations first, F+B+I+SIG (ours) last
    fig = plt.figure(figsize=(ICLR_W, 4.30))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.16, 1.30, 1.00], hspace=0.38,
                             left=0.075, right=0.985, top=0.958, bottom=0.068)
    env_panels(fig, outer[0, 0])
    left = outer[1, 0].subgridspec(1, len(arms), wspace=0.14)
    axes = [fig.add_subplot(left[0, i]) for i in range(len(arms))]
    for ax, arm in zip(axes, arms):
        pc, ct = pcs[arm]
        for cval, lab in [(0, "push-left"), (1, "push-right")]:
            m = ct == cval
            ax.scatter(pc[m, 0], pc[m, 1], s=1.6, c=CUE_COL[cval], alpha=0.5, lw=0, rasterized=True, label=lab)
        ax.set_title(ARM_TITLE[arm], fontsize=6.2, color=COL[arm] if arm == "F+B+I+SIG" else INK, pad=3.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.margins(0.06)
        for sp in ax.spines.values():
            sp.set_color(LIGHT); sp.set_visible(True)
        acc = np.mean([r["cue_probe_acc"] for r in rows if r["arm"] == arm])
        ax.text(0.5, 0.015, f"cue acc {acc:.2f}", transform=ax.transAxes, fontsize=5.8, color=INK,
                ha="center", va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
    axes[0].set_ylabel(f"PCA of $z_t$  (D={Z_DIM})", fontsize=6.6, labelpad=3)

    bottom = outer[2, 0].subgridspec(1, 2, width_ratios=[1.5, 3.5], wspace=0.06)
    ax = fig.add_subplot(bottom[0, 0])
    xs = np.arange(len(arms))
    for i, arm in enumerate(arms):
        v = [r["phase2_disp_r2"] for r in rows if r["arm"] == arm]
        ax.bar(i, np.mean(v), color=COL[arm], width=0.7, lw=0)
        ax.scatter([i] * len(v), v, s=4.5, c=INK, zorder=3, lw=0, alpha=0.85)
    ax.set_xticks(xs); ax.set_xticklabels([ARM_SHORT[a] for a in arms], fontsize=5.8)
    ax.tick_params(axis="x", pad=1.5, length=0)
    ax.tick_params(axis="y", labelsize=6, pad=1.5, length=2)
    for lab, a in zip(ax.get_xticklabels(), arms):
        lab.set_color(COL[a] if a == "F+B+I+SIG" else INK)
    ax.set_xlim(-0.62, len(arms) - 0.38)
    ax.set_ylim(0.2, 0.9)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.yaxis.grid(True, color=LIGHT, lw=0.5); ax.set_axisbelow(True)
    ax.set_title("phase 2: which way will the push move it?", fontsize=6.2, color=INK, pad=3.5)
    ax.set_ylabel("displacement $R^2$", fontsize=6.6, labelpad=2)

    handles = [plt.Line2D([], [], marker="o", ls="none", ms=2.6, mfc=CUE_COL[c], mec="none", label=t)
               for c, t in [(0, "push-left"), (1, "push-right")]]
    note = fig.add_subplot(bottom[0, 1]); note.axis("off")
    note.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 1.02), ncol=1, frameon=False,
                fontsize=6, handletextpad=0.4, labelcolor=INK, borderpad=0.0,
                title="stance $c$", title_fontsize=5.8)
    note.text(0.44, 0.98,
              "Top: the environment, sampled from the same\n"
              "generator the arms are trained on.\n\n"
              "Middle: PCA of the phase-1 latent, one panel\n"
              "per objective, with the mean cue-probe\n"
              "accuracy inset.\n\n"
              "Bottom left: a fresh predictor on the frozen\n"
              "encoder in a second phase where the wall is\n"
              "removed, scored by the displacement it\n"
              "recovers; bars are the mean over three seeds,\n"
              "dots the individual seeds.",
              transform=note.transAxes, fontsize=5.8, color=MUTED, ha="left", va="top", linespacing=1.45)
    savefig(fig, os.path.join(HERE, f"t3_alignment{SUF}"))
    summary = {arm: {k: float(np.mean([r[k] for r in rows if r["arm"] == arm])) for k in rows[0] if k not in ("arm", "seed")}
               for arm in ARMS}
    json.dump(summary, open(os.path.join(HERE, f"t3_summary{SUF}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return rows, pcs


if __name__ == "__main__":
    main()
