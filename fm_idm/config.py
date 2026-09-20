"""HC-FM-IDM configuration. See docs/2026-07-14_hc_fm_idm_design.md §8."""
from __future__ import annotations
from repro_paths import resolve_path as _resolve_artifact

from dataclasses import dataclass


@dataclass
class HCFMConfig:
    # dims
    d_latent: int = 192          # JEPA projector-space latent
    d_action: int = 10           # standardized macro-action (5 frames x 2-D PushT)
    n_sub: int = 5               # action tokenized into n_sub sub-tokens (each d_action//n_sub)
    # DiT backbone
    width: int = 192
    depth: int = 6
    heads: int = 4
    dim_head: int = 48
    mlp_dim: int = 384
    dropout: float = 0.0         # GC-IDM uses 10% dropout; 0 = none (our original)
    # horizon conditioning
    h_max: int = 12              # max step-gap (model-steps); matches knob_exp/train_actor.py max_goal
    # training
    bs: int = 512
    steps: int = 150_000
    lr: float = 1e-4             # ACID-aligned (was 3e-4; 3e-4 accelerated overfitting)
    wd: float = 1e-4
    warmup: int = 1_000
    ckpt_every: int = 10_000
    val_every: int = 2_000
    val_frac: float = 0.1
    # flow / inference
    ode_steps: int = 8
    # ACID-aligned flow-matching timestep schedule Beta(a,b) (set both <=0 -> Uniform)
    s_beta_a: float = 1.5
    s_beta_b: float = 1.0
    # early stopping / best-val checkpointing (val-eval units; <=0 disables early stop)
    patience: int = 8
    val_draws: int = 4          # noise draws averaged per val eval (stabilizes best-val selection)
    # io / env
    device: str = "cuda:0"
    seed: int = 0
    npz: str = "knob_exp/latents.npz"
    out: str = "fm_idm/hcfmidm.pt"
    # h5 whose raw `action` column defines the eval StandardScaler (cube: the ogbench cube h5)
    scaler_h5: str = _resolve_artifact('stable:datasets/pusht_expert_train.h5')
