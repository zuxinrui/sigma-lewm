"""eval.py + flow-IDM-as-amortized-actor (fm_idm.flow_actor.LeWMFlowActor).

Wraps a frozen LeWM so CEM seeds from the IDM's amortized plan. The point: sweep
solver.n_steps to trace the accuracy-vs-compute frontier and check whether IDM-seeded CEM
matches stock CEM at far fewer iterations (= beat CEM on compute).

Overrides:
    +flow.idm=<ckpt>            trained HC-FM-IDM / GC-ACID flow IDM (default fm_idm/hcfmidm.pt)
    +flow.h_mode={remaining,fixed,clamp}   how remaining-horizon h is set (default remaining)
    +flow.h_fixed=<int>         goal-distance in macro-steps (goal_offset/action_block=5)
    +flow.ode_steps=<int>       ODE steps for generation (default 1 = GC-IDM fast path)
    solver.n_steps=<k>          0 = pure react; k>0 = IDM-seeded CEM with k refine iters
"""

from trim.paths import REPO_ROOT
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from fm_idm.config import HCFMConfig
from fm_idm.model import HCFMIDM
from fm_idm.flow_actor import LeWMFlowActor


def _load_idm(ckpt, device):
    d = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = HCFMConfig(**{k: v for k, v in d["cfg"].items() if k in HCFMConfig.__dataclass_fields__})
    name = d.get("model_name", "hcfm")
    if name == "gcidm":
        from fm_idm.gcidm_model import GCIDM
        m = GCIDM(cfg).to(device)
    elif name == "acid":
        from fm_idm.acid_model import ACIDIDM
        m = ACIDIDM(cfg).to(device)
    else:
        m = HCFMIDM(cfg).to(device)
    m.load_state_dict(d["model"])
    return m.eval().requires_grad_(False)


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ei = dataset.get_col_data(col); si = dataset.get_col_data("step_idx")
    return np.array([np.max(si[ei == e]) + 1 for e in episodes])


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "config/eval"), config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dpath = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(cfg.eval.dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dpath)
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)

    process = {}
    for c in cfg.dataset.keys_to_cache:
        if c in ["pixels"]:
            continue
        pr = preprocessing.StandardScaler()
        cd = dataset.get_col_data(c); cd = cd[~np.isnan(cd).any(axis=1)]
        pr.fit(cd); process[c] = pr
        if c != "action":
            process[f"goal_{c}"] = pr

    try:
        model = swm.wm.utils.load_pretrained(cfg.policy)
    except RuntimeError as error:
        if "Unexpected key(s)" not in str(error):
            raise
        from trim.evaluation.planning import load_full_model

        model, _ = load_full_model(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    idm = _load_idm(cfg.flow.get("idm", "fm_idm/hcfmidm.pt"), "cuda")
    _cw = cfg.flow.get("cost_weight", None)
    model = LeWMFlowActor(model, idm, h_max=int(getattr(idm.cfg, "h_max", 12)),
                          ode_steps=int(cfg.flow.get("ode_steps", 1)),
                          h_mode=str(cfg.flow.get("h_mode", "remaining")),
                          h_fixed=int(cfg.flow.get("h_fixed", 5)),
                          cost_weight=(None if _cw in (None, "none") else str(_cw)),
                          arrival=int(cfg.flow.get("arrival", 5)), device="cuda").to("cuda").eval()
    print(f"[eval_flowactor] idm={cfg.flow.get('idm','fm_idm/hcfmidm.pt')} "
          f"h_mode={cfg.flow.get('h_mode','remaining')} ode_steps={cfg.flow.get('ode_steps',1)} "
          f"solver.n_steps={cfg.solver.n_steps}")

    config = swm.PlanConfig(**cfg.plan_config)
    if bool(cfg.flow.get("mix", False)):
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT / "knob_exp"))
        from solvers import MixedCEMSolver   # inject n_pi IDM samples, keep CEM mean unbiased
        solver = MixedCEMSolver(model, n_pi=int(cfg.flow.get("n_pi", 30)), pi_noise=0.3,
                                num_samples=cfg.solver.num_samples, var_scale=cfg.solver.var_scale,
                                n_steps=cfg.solver.n_steps, topk=cfg.solver.topk, device="cuda", seed=cfg.seed)
        print(f"[eval_flowactor] MIX mode: n_pi={cfg.flow.get('n_pi', 30)}")
    else:
        solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    elen = get_episodes_length(dataset, ep_indices)
    max_start = elen - cfg.eval.goal_offset_steps - 1
    msd = {e: max_start[i] for i, e in enumerate(ep_indices)}
    mspr = np.array([msd[e] for e in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= mspr)[0]
    g = np.random.default_rng(cfg.seed)
    idx = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_eps = dataset.get_row_data(idx)[col]; eval_start = dataset.get_row_data(idx)["step_idx"]

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    metrics = world.evaluate(dataset=dataset, start_steps=eval_start.tolist(), goal_offset=cfg.eval.goal_offset_steps,
                             eval_budget=cfg.eval.eval_budget, episodes_idx=eval_eps.tolist(),
                             callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True), video=results_path)
    dt = time.time() - t0
    print(metrics)
    rp = results_path / cfg.output.filename
    with rp.open("a") as f:
        f.write("\n==== CONFIG ====\n"); f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n"); f.write(f"metrics: {metrics}\n"); f.write(f"evaluation_time: {dt} seconds\n")


if __name__ == "__main__":
    run()
