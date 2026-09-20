"""T2: learned-encoder toy world. One (arm, seed) per invocation; results -> runs/.

World state s = [p(2) agent, o(2) pushable object, d(6) three random-walk distractors, n(4) iid noise].
Observation x = W2 tanh(1.5 W1 s + b1) + b2  in R^32 (fixed random injective nonlinear mixing).
Action a = agent displacement in [-0.15, 0.15]^2. Object is carried when |p-o| < 0.25.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import SIGReg, infonce_idm_loss, mlp, mlp_r2, proj_head, ridge_r2, symmetric_dcl_loss

HERE = os.path.dirname(os.path.abspath(__file__))
A_MAX = 0.15
R_CONTACT = 0.25
SIG_D = 0.02
SIG_N = 0.3
DIMS = dict(p=slice(0, 2), o=slice(2, 4), d=slice(4, 10), n=slice(10, 14))
S_DIM = 14
X_DIM = 32
Z_DIM = int(os.environ.get("T2_ZDIM", "8"))  # 8 = no capacity pressure; 4/3 = bottleneck (T2b)
RUNS_DIR = "runs" if Z_DIM == 8 else f"runs_d{Z_DIM}"
RUNS = os.path.join(HERE, RUNS_DIR)
TAU = 0.1


# ------------------------------------------------------------------ world
class World:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, B):
        s = np.zeros((B, S_DIM))
        s[:, 0:2] = self.rng.uniform(-0.9, 0.9, (B, 2))
        s[:, 2:4] = self.rng.uniform(-0.9, 0.9, (B, 2))
        s[:, 4:10] = self.rng.uniform(-1, 1, (B, 6))
        s[:, 10:14] = np.clip(self.rng.normal(0, SIG_N, (B, 4)), -1, 1)
        return s

    def step(self, s, a):
        """Vectorised. a: (B,2) clipped to +-A_MAX."""
        a = np.clip(a, -A_MAX, A_MAX)
        s2 = s.copy()
        p, o = s[:, 0:2], s[:, 2:4]
        contact = np.linalg.norm(p - o, axis=1) < R_CONTACT
        s2[:, 0:2] = np.clip(p + a, -1, 1)
        s2[:, 2:4] = np.where(contact[:, None], np.clip(o + a, -1, 1), o)
        d = s[:, 4:10] + self.rng.normal(0, SIG_D, (len(s), 6))
        # reflect at +-1
        d = np.where(d > 1, 2 - d, d)
        d = np.where(d < -1, -2 - d, d)
        s2[:, 4:10] = d
        s2[:, 10:14] = np.clip(self.rng.normal(0, SIG_N, (len(s), 4)), -1, 1)
        return s2

    def policy(self, s, mode, push_dir):
        """mode: 0 random, 1 approach-then-push. push_dir: (B,2) persistent per-episode."""
        B = len(s)
        rand = self.rng.uniform(-A_MAX, A_MAX, (B, 2))
        p, o = s[:, 0:2], s[:, 2:4]
        vec = o - p
        dist = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
        approach = A_MAX * vec / dist * np.minimum(1.0, dist / A_MAX) + self.rng.normal(0, 0.03, (B, 2))
        contact = (dist[:, 0] < R_CONTACT)[:, None]
        push = A_MAX * push_dir + self.rng.normal(0, 0.03, (B, 2))
        goal_directed = np.where(contact, push, approach)
        a = np.where(mode[:, None] == 1, goal_directed, rand)
        return np.clip(a, -A_MAX, A_MAX)

    def episodes(self, n_ep, T):
        s = self.reset(n_ep)
        mode = (self.rng.uniform(size=n_ep) < 0.6).astype(int)
        ang = self.rng.uniform(0, 2 * np.pi, n_ep)
        push_dir = np.stack([np.cos(ang), np.sin(ang)], 1)
        S = np.zeros((n_ep, T + 1, S_DIM))
        A = np.zeros((n_ep, T, 2))
        S[:, 0] = s
        for t in range(T):
            # occasionally re-randomise the push direction / mode
            flip = self.rng.uniform(size=n_ep) < 0.05
            ang = np.where(flip, self.rng.uniform(0, 2 * np.pi, n_ep), ang)
            push_dir = np.stack([np.cos(ang), np.sin(ang)], 1)
            a = self.policy(s, mode, push_dir)
            s = self.step(s, a)
            A[:, t] = a
            S[:, t + 1] = s
        return S, A


class Mixer:
    """Fixed random injective nonlinear mixing s (14) -> x (32)."""

    def __init__(self, seed=12345):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(size=(X_DIM, S_DIM))
        W1 /= np.linalg.norm(W1, axis=1, keepdims=True)
        self.W1 = 1.5 * W1
        self.b1 = rng.uniform(-0.3, 0.3, X_DIM)
        Q, _ = np.linalg.qr(rng.normal(size=(X_DIM, X_DIM)))
        self.W2 = Q
        self.b2 = rng.normal(0, 0.1, X_DIM)
        self.mu, self.sd = None, None

    def __call__(self, s):
        h = np.tanh(s @ self.W1.T + self.b1)
        x = h @ self.W2.T + self.b2
        if self.mu is None:
            self.mu, self.sd = x.mean(0), x.std(0) + 1e-8
        return (x - self.mu) / self.sd


def make_data(seed_train=1, seed_test=2, n_train_ep=1000, n_test_ep=200, T=50):
    mixer = Mixer()
    Wtr = World(seed_train)
    S, A = Wtr.episodes(n_train_ep, T)
    X = mixer(S.reshape(-1, S_DIM)).reshape(S.shape[0], S.shape[1], X_DIM)
    Wte = World(seed_test)
    S2, A2 = Wte.episodes(n_test_ep, T)
    X2 = mixer(S2.reshape(-1, S_DIM)).reshape(S2.shape[0], S2.shape[1], X_DIM)
    return dict(S=S, A=A, X=X, S_te=S2, A_te=A2, X_te=X2, mixer=mixer)


def flat(S, A, X):
    s = S[:, :-1].reshape(-1, S_DIM)
    s2 = S[:, 1:].reshape(-1, S_DIM)
    x = X[:, :-1].reshape(-1, X_DIM)
    x2 = X[:, 1:].reshape(-1, X_DIM)
    a = A.reshape(-1, 2)
    return s, a, s2, x, x2


# ------------------------------------------------------------------ model
class ToyWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = mlp(X_DIM, Z_DIM, hidden=256, depth=2)
        self.P = mlp(Z_DIM + 2, Z_DIM, hidden=256, depth=2)
        # backward heads: q = f([z'; a]), k = g(z)
        self.fB = proj_head(Z_DIM + 2)
        self.gB = proj_head(Z_DIM)
        # inverse heads: q = fI([z; z']), k = gI(a)
        self.fI = proj_head(2 * Z_DIM)
        self.gI = proj_head(2)
        # regression forms
        self.QB = mlp(Z_DIM + 2, Z_DIM, hidden=128, depth=2)
        self.RI = mlp(2 * Z_DIM, 2, hidden=128, depth=2)

    def predict(self, z, a):
        return self.P(torch.cat([z, a], -1))


ARMS = {
    "F+SIG": dict(B=0.0, I=0.0, Bmse=0.0, Imse=0.0, shuf=False),
    "F+B+SIG": dict(B=0.10, I=0.0, Bmse=0.0, Imse=0.0, shuf=False),
    "F+I+SIG": dict(B=0.0, I=0.05, Bmse=0.0, Imse=0.0, shuf=False),
    "F+B+I+SIG": dict(B=0.10, I=0.05, Bmse=0.0, Imse=0.0, shuf=False),
    "F+Bmse+Imse+SIG": dict(B=0.0, I=0.0, Bmse=0.10, Imse=0.05, shuf=False),
    "F+Ishuf+SIG": dict(B=0.0, I=0.05, Bmse=0.0, Imse=0.0, shuf=True),
}
W_SIG = 0.09


def train(arm, seed, data, steps=4000, batch=512, log_every=500):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = ARMS[arm]
    s, a, s2, x, x2 = flat(data["S"], data["A"], data["X"])
    X = torch.tensor(x, dtype=torch.float32)
    X2 = torch.tensor(x2, dtype=torch.float32)
    Aa = torch.tensor(a, dtype=torch.float32)
    a_std = Aa.std(0, keepdim=True)
    model = ToyWM()
    sig = SIGReg(num_proj=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    hist = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        xb, x2b, ab = X[idx], X2[idx], Aa[idx]
        z = model.E(xb)
        z2 = model.E(x2b)
        L_F = F.mse_loss(model.predict(z, ab), z2)  # no stop-grad on the target, as in LeWM
        L_SIG = sig(z.unsqueeze(0))
        loss = L_F + W_SIG * L_SIG
        parts = dict(F=L_F.item(), SIG=L_SIG.item())
        if cfg["B"] > 0:
            L_B = symmetric_dcl_loss(model.fB(torch.cat([z2, ab], -1)), model.gB(z), temperature=TAU)
            loss = loss + cfg["B"] * L_B
            parts["B"] = L_B.item()
        if cfg["I"] > 0:
            key_a = ab
            if cfg["shuf"]:
                key_a = ab[torch.randperm(batch, generator=g)]
            L_I = infonce_idm_loss(model.fI(torch.cat([z, z2], -1)), model.gI(key_a), key_a, temperature=TAU)
            loss = loss + cfg["I"] * L_I
            parts["I"] = L_I.item()
        if cfg["Bmse"] > 0:
            L_Bm = F.mse_loss(model.QB(torch.cat([z2, ab], -1)), z)
            loss = loss + cfg["Bmse"] * L_Bm
            parts["Bmse"] = L_Bm.item()
        if cfg["Imse"] > 0:
            L_Im = F.mse_loss(model.RI(torch.cat([z, z2], -1)), ab / a_std)
            loss = loss + cfg["Imse"] * L_Im
            parts["Imse"] = L_Im.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            parts["step"] = step
            parts["total"] = loss.item()
            hist.append(parts)
            print(f"[{arm} s{seed}] step {step:5d} " + " ".join(f"{k}={v:.4f}" for k, v in parts.items() if k != "step"),
                  flush=True)
    print(f"[{arm} s{seed}] train time {time.time() - t0:.1f}s", flush=True)
    return model, hist


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def encode(model, x):
    return model.E(torch.tensor(x, dtype=torch.float32)).numpy()


def eval_probes(model, data):
    s, a, s2, x, x2 = flat(data["S"], data["A"], data["X"])
    st, at, st2, xt, xt2 = flat(data["S_te"], data["A_te"], data["X_te"])
    sub = np.random.default_rng(0).choice(len(x), 20000, replace=False)
    z, z2 = encode(model, x[sub]), encode(model, x2[sub])
    zt, zt2 = encode(model, xt), encode(model, xt2)
    out = {}
    for k, sl in DIMS.items():
        r2, _, _ = ridge_r2(z, s[sub][:, sl], zt, st[:, sl])
        out[f"lin_r2_{k}"] = float(np.clip(r2, -1, 1).mean())
        r2m = mlp_r2(z, s[sub][:, sl], zt, st[:, sl], steps=500)
        out[f"mlp_r2_{k}"] = float(np.clip(r2m, -1, 1).mean())
    zz = np.concatenate([z, z2], 1)
    zzt = np.concatenate([zt, zt2], 1)
    r2, _, _ = ridge_r2(zz, a[sub], zzt, at)
    out["lin_r2_action"] = float(np.clip(r2, -1, 1).mean())
    out["mlp_r2_action"] = float(np.clip(mlp_r2(zz, a[sub], zzt, at, steps=500), -1, 1).mean())
    # action sensitivity
    rng = np.random.default_rng(1)
    i = rng.choice(len(zt), 4096, replace=False)
    zt_t = torch.tensor(zt[i], dtype=torch.float32)
    a1 = torch.tensor(rng.uniform(-A_MAX, A_MAX, (4096, 2)), dtype=torch.float32)
    a2 = torch.tensor(rng.uniform(-A_MAX, A_MAX, (4096, 2)), dtype=torch.float32)
    with torch.no_grad():
        diff = (model.predict(zt_t, a1) - model.predict(zt_t, a2)).norm(dim=1)
    out["action_sensitivity"] = float((diff / zt_t.norm(dim=1)).mean())
    # latent stats
    out["z_std_mean"] = float(zt.std(0).mean())
    ev = np.linalg.eigvalsh(np.cov(zt.T))
    ev = np.clip(ev, 1e-12, None) / ev.sum()
    out["rankme"] = float(np.exp(-(ev * np.log(ev)).sum()))
    return out


def eval_rollout(model, data, H=20, n_windows=600):
    """Open-loop rollout error in the fixed linear-probe space for [p, o] (world units)."""
    s, a, s2, x, x2 = flat(data["S"], data["A"], data["X"])
    sub = np.random.default_rng(0).choice(len(x), 20000, replace=False)
    z = encode(model, x[sub])
    _, W, (mu, sd) = ridge_r2(z, s[sub][:, 0:4], z, s[sub][:, 0:4])

    def probe(zz):
        zz = (zz - mu) / sd
        return np.concatenate([zz, np.ones((len(zz), 1))], 1) @ W

    S, A, X = data["S_te"], data["A_te"], data["X_te"]
    rng = np.random.default_rng(2)
    ep = rng.integers(0, S.shape[0], n_windows)
    t0 = rng.integers(0, S.shape[1] - H - 1, n_windows)
    xs = np.stack([X[e, t] for e, t in zip(ep, t0)])
    zh = torch.tensor(encode(model, xs), dtype=torch.float32)
    errs = np.zeros(H + 1)
    errs[0] = np.sqrt(((probe(zh.numpy()) - np.stack([S[e, t, 0:4] for e, t in zip(ep, t0)])) ** 2).mean())
    with torch.no_grad():
        for k in range(1, H + 1):
            ak = torch.tensor(np.stack([A[e, t + k - 1] for e, t in zip(ep, t0)]), dtype=torch.float32)
            zh = model.predict(zh, ak)
            truth = np.stack([S[e, t + k, 0:4] for e, t in zip(ep, t0)])
            errs[k] = np.sqrt(((probe(zh.numpy()) - truth) ** 2).mean())
    # reference: error of "copy z_0" (no dynamics) in the same probe space
    z0 = probe(encode(model, xs))
    copy_err = np.array([np.sqrt(((z0 - np.stack([S[e, t + k, 0:4] for e, t in zip(ep, t0)])) ** 2).mean())
                         for k in range(H + 1)])
    return errs.tolist(), copy_err.tolist()


def cem_plan(cost_fn, H, n_samples=300, n_elites=30, n_iters=30, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.zeros((H, 2))
    std = np.full((H, 2), 0.1)
    for _ in range(n_iters):
        acts = np.clip(mean + std * rng.normal(size=(n_samples, H, 2)), -A_MAX, A_MAX)
        c = cost_fn(acts)
        el = acts[np.argsort(c)[:n_elites]]
        mean, std = el.mean(0), el.std(0) + 1e-3
    return mean


def make_planning_episodes(n=100, seed=7):
    rng = np.random.default_rng(seed)
    eps = []
    W = World(seed)
    while len(eps) < n:
        s0 = W.reset(1)[0]
        p0 = rng.uniform(-0.6, 0.6, 2)
        ang = rng.uniform(0, 2 * np.pi)
        o0 = p0 + rng.uniform(0.3, 0.5) * np.array([np.cos(ang), np.sin(ang)])
        ang2 = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(ang2), np.sin(ang2)])
        o_star = o0 + rng.uniform(0.2, 0.35) * u
        if np.abs(o0).max() > 0.9 or np.abs(o_star).max() > 0.9:
            continue
        s0[0:2], s0[2:4] = p0, o0
        sg = W.reset(1)[0]  # fresh distractor + noise in the goal frame
        sg[2:4] = o_star
        sg[0:2] = o_star - 0.12 * u
        eps.append((s0, sg, o_star))
    return eps


def eval_planning(model, data, H=8, n_ep=100):
    mixer = data["mixer"]
    eps = make_planning_episodes(n_ep)
    W = World(99)
    succ, final_dist = [], []
    for k, (s0, sg, o_star) in enumerate(eps):
        with torch.no_grad():
            z0 = model.E(torch.tensor(mixer(s0[None]), dtype=torch.float32))
            zg = model.E(torch.tensor(mixer(sg[None]), dtype=torch.float32))

        def cost(acts):
            with torch.no_grad():
                z = z0.repeat(len(acts), 1)
                at = torch.tensor(acts, dtype=torch.float32)
                for t in range(H):
                    z = model.predict(z, at[:, t])
                return ((z - zg) ** 2).sum(1).numpy()

        plan = cem_plan(cost, H, seed=k)
        s = s0[None].copy()
        for t in range(H):  # full commit, open loop
            s = W.step(s, plan[t][None])
        dist = np.linalg.norm(s[0, 2:4] - o_star)
        final_dist.append(dist)
        succ.append(dist < 0.15)
    return float(np.mean(succ)), float(np.mean(final_dist))


def eval_planning_reference(H=8, n_ep=100):
    """Oracle (true-state CEM) and random-action ceilings/floors for the same episodes."""
    eps = make_planning_episodes(n_ep)
    W = World(99)
    W2 = World(5)
    orc, rnd, init = [], [], []
    for k, (s0, sg, o_star) in enumerate(eps):
        def cost(acts):
            s = np.repeat(s0[None], len(acts), 0)
            for t in range(H):
                s = W2.step(s, acts[:, t])
            return ((s[:, 2:4] - o_star) ** 2).sum(1) + 0.1 * ((s[:, 0:2] - sg[0:2]) ** 2).sum(1)

        plan = cem_plan(cost, H, seed=k)
        s = s0[None].copy()
        for t in range(H):
            s = W.step(s, plan[t][None])
        orc.append(np.linalg.norm(s[0, 2:4] - o_star) < 0.15)
        s = s0[None].copy()
        for t in range(H):
            s = W.step(s, np.random.default_rng(k * 7 + t).uniform(-A_MAX, A_MAX, (1, 2)))
        rnd.append(np.linalg.norm(s[0, 2:4] - o_star) < 0.15)
        init.append(np.linalg.norm(s0[2:4] - o_star) < 0.15)
    return dict(oracle=float(np.mean(orc)), random=float(np.mean(rnd)), solved_at_reset=float(np.mean(init)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    data = make_data()
    model, hist = train(args.arm, args.seed, data, steps=args.steps)
    model.eval()
    t0 = time.time()
    res = dict(arm=args.arm, seed=args.seed, steps=args.steps, history=hist)
    res.update(eval_probes(model, data))
    res["rollout_rmse"], res["copy_rmse"] = eval_rollout(model, data)
    res["plan_success"], res["plan_final_dist"] = eval_planning(model, data)
    print(f"[{args.arm} s{args.seed}] eval time {time.time() - t0:.1f}s", flush=True)
    stem = os.path.join(RUNS, f"{args.arm}_s{args.seed}")
    torch.save({"E": model.E.state_dict(), "P": model.P.state_dict()}, stem + ".pt")
    with open(stem + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    keys = ["lin_r2_p", "lin_r2_o", "lin_r2_d", "lin_r2_n", "mlp_r2_action", "action_sensitivity",
            "rankme", "plan_success"]
    print(f"[{args.arm} s{args.seed}] " + " ".join(f"{k}={res[k]:.3f}" for k in keys), flush=True)


if __name__ == "__main__":
    main()
