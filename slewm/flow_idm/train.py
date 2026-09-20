"""HC-FM-IDM training loop (CLI). See docs/2026-07-14_hc_fm_idm_design.md §8.

Trains the flow-matching velocity field v_theta(a_s, s; z_t, z_g, h) on cached
LeWM latents. Pure latent-space: no world model / ViT in the loop. AdamW +
linear-warmup-then-cosine-to-0 schedule. Everything lives in the same normalized
action space as knob_exp/train_actor.py and eval.py's `process`.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from slewm.flow_idm.config import HCFMConfig
from slewm.flow_idm.model import HCFMIDM
from slewm.flow_idm.data import LatentPairSampler
from slewm.flow_idm.flow import fm_loss


def train(cfg: HCFMConfig, model_name: str = "hcfm") -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device

    if model_name == "acid":
        from slewm.flow_idm.acid_model import ACIDIDM     # lazy: only needed for --model acid
        model = ACIDIDM(cfg).to(dev)
    elif model_name == "gcidm":
        from slewm.flow_idm.gcidm_model import GCIDM       # faithful GC-IDM (MLP + MSE regression)
        model = GCIDM(cfg).to(dev)
    else:
        model = HCFMIDM(cfg).to(dev)
    is_mse = (model_name == "gcidm")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: {n_params/1e6:.2f}M params  W={cfg.width} L={cfg.depth} "
          f"heads={cfg.heads} n_sub={cfg.n_sub} h_max={cfg.h_max}", flush=True)

    sampler = LatentPairSampler(cfg.npz, cfg.h_max, dev,
                                val_frac=cfg.val_frac, seed=cfg.seed, scaler_h5=cfg.scaler_h5)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    warmup = max(1, cfg.warmup)
    total = max(1, cfg.steps)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup:
            return (step + 1) / warmup
        # cosine decay from 1 -> 0 over the post-warmup span
        progress = (step - cfg.warmup) / max(1, total - cfg.warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # fixed val batch (data fixed; flow noise/time resampled each eval)
    val_batch = sampler.sample(cfg.bs, "val")

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    s_beta = (cfg.s_beta_a, cfg.s_beta_b)          # ACID timestep schedule (<=0 -> uniform)

    def _loss(zt, zg, hh, aa):                     # MSE for gcidm, flow-matching otherwise
        if is_mse:
            return ((model(zt, zg, hh) - aa) ** 2).mean()
        return fm_loss(model, zt, zg, hh, aa, s_beta=s_beta)

    def save(bv: float, bs_: int):
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                    "best_val": bv, "best_step": bs_, "model_name": model_name}, cfg.out)

    best_val = float("inf")
    best_step = -1
    bad = 0                                         # val evals since last improvement
    t0 = time.time()
    for step in range(cfg.steps):
        model.train()
        z_t, z_g, h, a = sampler.sample(cfg.bs, "train")
        loss = _loss(z_t, z_g, h, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % cfg.val_every == 0 or step == cfg.steps - 1:
            model.eval()
            nd = max(1, cfg.val_draws)
            with torch.no_grad():
                zt, zg, hv, av = val_batch
                vloss = sum(_loss(zt, zg, hv, av).item()
                            for _ in range(nd)) / nd   # avg over noise -> stable best-val
            improved = vloss < best_val - 1e-4
            if improved:
                best_val, best_step, bad = vloss, step, 0
                save(best_val, best_step)              # BEST-val checkpoint -> cfg.out
            else:
                bad += 1
            its = (step + 1) / (time.time() - t0 + 1e-9)
            print(f"  step {step:7d}  train {loss.item():.4f}  val {vloss:.4f}  "
                  f"best {best_val:.4f}@{best_step}  bad {bad}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ({its:.0f} it/s)"
                  + ("  *saved" if improved else ""), flush=True)
            if cfg.patience > 0 and bad >= cfg.patience:
                print(f"EARLY STOP: no val improvement for {bad} evals "
                      f"(best {best_val:.4f} @ step {best_step})", flush=True)
                break

    dt = time.time() - t0
    print(f"saved BEST -> {cfg.out}  (best val {best_val:.4f} @ step {best_step}; "
          f"{dt:.0f}s, {dt/60:.1f} min)", flush=True)
    print("FM_IDM_TRAIN_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Train HC-FM-IDM")
    ap.add_argument("--model", type=str, default="hcfm", choices=["hcfm", "acid", "gcidm"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--scaler_h5", type=str, default=None)   # h5 defining the action StandardScaler
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--h_max", type=int, default=None)
    ap.add_argument("--d_action", type=int, default=None)   # cube macro-action = 5 frames x 5 = 25
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--val_every", type=int, default=None)
    ap.add_argument("--val_draws", type=int, default=None)
    ap.add_argument("--s_beta_a", type=float, default=None)
    ap.add_argument("--s_beta_b", type=float, default=None)
    args = ap.parse_args()

    cfg = HCFMConfig()
    for k in ("steps", "lr", "bs", "out", "npz", "scaler_h5", "device", "h_max", "d_action", "depth", "heads", "width",
              "dropout", "wd", "seed", "patience", "val_every", "val_draws", "s_beta_a", "s_beta_b"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    train(cfg, model_name=args.model)


if __name__ == "__main__":
    main()
