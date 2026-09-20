"""Why does latent CEM fail for every arm at D=8? Three diagnostics on the saved models.
 (a) nuisance share: ||E(s_goal, fresh d) - E(s_goal, matched d)||^2 / ||z_goal - z_0||^2  (how much of the
     goal distance is distractor mismatch);
 (b) contact dynamics: on held-out transitions where the object actually moved (|delta o| > 0.01), R^2 of the
     probe-decoded predicted object displacement vs the true one (1-step) -- does the model know pushing at all?
 (c) decodability floor: probe RMSE of o (world units) vs the 0.15 success radius.
Also (d) the T3 duplicate-mask fraction: share of same-cluster action pairs with cosine > 0.98 (masked out of
the inverse InfoNCE negatives)."""
import glob, json, os
import numpy as np, torch
from common import ridge_r2
from t2_toy_world import DIMS, ToyWM, flat, make_data, make_planning_episodes
HERE = os.path.dirname(os.path.abspath(__file__))
data = make_data(); mixer = data["mixer"]
s, a, s2, x, x2 = flat(data["S"], data["A"], data["X"])
st, at, st2, xt, xt2 = flat(data["S_te"], data["A_te"], data["X_te"])
sub = np.random.default_rng(0).choice(len(x), 20000, replace=False)
T = lambda v: torch.tensor(v, dtype=torch.float32)
eps = make_planning_episodes(100)
out = {}
for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.pt"))):
    name = os.path.basename(path)[:-3]; arm, seed = name.rsplit("_s", 1)
    m = ToyWM(); sd = torch.load(path); m.E.load_state_dict(sd["E"]); m.P.load_state_dict(sd["P"]); m.eval()
    with torch.no_grad():
        z = m.E(T(x[sub])).numpy()
        _, W, (mu, sdv) = ridge_r2(z, s[sub][:, 0:4], z, s[sub][:, 0:4])
        probe = lambda zz: np.concatenate([(zz - mu) / sdv, np.ones((len(zz), 1))], 1) @ W
        # (a) nuisance share
        shares = []
        for s0, sg, o_star in eps:
            sgm = sg.copy(); sgm[4:10] = s0[4:10]
            z0, zg, zgm = [m.E(T(mixer(v[None]))).numpy()[0] for v in (s0, sg, sgm)]
            shares.append(((zg - zgm) ** 2).sum() / ((zg - z0) ** 2).sum())
        # (b) contact dynamics, 1 step, held-out
        zt, zt2 = m.E(T(xt)), m.E(T(xt2))
        pred = m.predict(zt, T(at))
        d_true = st2[:, 2:4] - st[:, 2:4]
        moved = np.linalg.norm(d_true, axis=1) > 0.01
        d_pred = probe(pred.numpy())[:, 2:4] - probe(zt.numpy())[:, 2:4]
        r2_contact = 1 - ((d_pred[moved] - d_true[moved]) ** 2).sum() / ((d_true[moved] - d_true[moved].mean(0)) ** 2).sum()
        # fraction of moved transitions where predicted displacement direction is right (cos > 0)
        cosv = (d_pred[moved] * d_true[moved]).sum(1) / (np.linalg.norm(d_pred[moved], axis=1) * np.linalg.norm(d_true[moved], axis=1) + 1e-9)
        # (c) decodability floor
        rmse_o = float(np.sqrt(((probe(zt.numpy())[:, 2:4] - st[:, 2:4]) ** 2).sum(1).mean()))
    out[name] = dict(arm=arm, seed=int(seed), nuisance_share=float(np.mean(shares)), contact_frac=float(moved.mean()),
                     contact_dispR2=float(r2_contact), contact_dir_acc=float((cosv > 0).mean()), probe_rmse_o=rmse_o,
                     mean_true_disp=float(np.linalg.norm(d_true[moved], axis=1).mean()),
                     mean_pred_disp=float(np.linalg.norm(d_pred[moved], axis=1).mean()))
    print(out[name], flush=True)
# (d) T3 duplicate-mask fraction
rng = np.random.default_rng(0); c = rng.choice([-1., 1.], 4000)
aa = np.stack([0.1 * c + rng.normal(0, 0.02, 4000), rng.normal(0, 0.02, 4000)], 1)
an = aa / np.linalg.norm(aa, axis=1, keepdims=True); cs = an @ an.T
same = c[:, None] == c[None, :]; np.fill_diagonal(same, False)
mask_frac_same = float((cs[same] > 0.98).mean()); mask_frac_all = float((cs[~np.eye(4000, dtype=bool)] > 0.98).mean())
out["t3_mask"] = dict(same_cluster_pairs_masked=mask_frac_same, all_pairs_masked=mask_frac_all)
print(out["t3_mask"])
json.dump(out, open(os.path.join(HERE, "t2_planning_diag.json"), "w"), indent=1)
arms = sorted(set(v["arm"] for k, v in out.items() if k != "t3_mask"))
print("\n| arm | nuisance share of goal dist | contact transitions | 1-step contact dR2 | dir. acc | pred/true |disp| | probe RMSE o |")
print("|---|---|---|---|---|---|---|")
for arm in arms:
    rs = [v for k, v in out.items() if k != "t3_mask" and v["arm"] == arm]
    f = lambda key: np.mean([r[key] for r in rs])
    print(f"| {arm} | {f('nuisance_share'):.2f} | {f('contact_frac'):.3f} | {f('contact_dispR2'):.2f} | {f('contact_dir_acc'):.2f} | {f('mean_pred_disp'):.3f}/{f('mean_true_disp'):.3f} | {f('probe_rmse_o'):.3f} |")
