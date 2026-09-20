"""LatentPairSampler for HC-FM-IDM. See docs/2026-07-14_hc_fm_idm_design.md §3, §8.

Samples goal-conditioned inverse-dynamics tuples (z_t, z_{t+h}, h, a_t) from the cached
LeWM latents (`knob_exp/latents.npz`). Mirrors `knob_exp/train_actor.py`:
  - hindsight goal relabel: gap h ~ Uniform{1..h_max}, z_g = z[t+h] in the SAME episode,
    a = act[t] (the next-step action, GC-IDM semantics);
  - GPU-gather batching (torch index_select on device, no DataLoader);
  - episode-level (not frame-level) train/val split.

Actions are STANDARDIZED into the eval/CEM StandardScaler space at load time (per
2-D action dim), matching eval.py's `process['action']` and knob_exp/train_actor.py.
This keeps the whole pipeline (train / generate / fm_residual / eval) in one space
where N(0, I) is the natural flow-noise and random-action baseline (design §2, §8).
The scaler is fit on the full expert-action distribution in the source .h5; if that
file is absent we fall back to the cached actions' own per-dim mean/std (with a warning).
`act_mean`/`act_std` (each (sub_dim,)) are kept so callers can inverse_transform.
"""
from __future__ import annotations

import os

import numpy as np
import torch

# expert dataset whose action column defines the eval StandardScaler (see eval.py,
# knob_exp/train_actor.py). Only the small `action` dataset is read, not the 46 GB of pixels.
_SCALER_H5 = os.path.join(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm")), "datasets", "pusht_expert_train.h5")


def _action_scaler(act: np.ndarray, sub_dim: int, scaler_h5: str):
    """Per-sub-dim (amean, astd), each shape (sub_dim,), matching eval.py's StandardScaler
    fit on the full h5 action column (NaN rows dropped). Falls back to the cached actions."""
    if scaler_h5 and os.path.exists(scaler_h5):
        try:
            import h5py
            with h5py.File(scaler_h5, "r") as fa:
                allact = fa["action"][:]                       # (M, sub_dim) raw expert actions
            allact = allact[~np.isnan(allact).any(axis=1)]
            amean = allact.mean(0).astype(np.float32)
            astd = (allact.std(0) + 1e-6).astype(np.float32)
            return amean, astd
        except Exception as e:  # h5py missing / corrupt -> fall back below
            print(f"[LatentPairSampler] scaler h5 unusable ({e}); "
                  f"falling back to cached-action stats", flush=True)
    else:
        print(f"[LatentPairSampler] scaler h5 not found at {scaler_h5}; "
              f"falling back to cached-action stats", flush=True)
    flat = act.reshape(-1, sub_dim).astype(np.float32)
    return flat.mean(0), flat.std(0) + 1e-6


class LatentPairSampler:
    def __init__(self, npz: str, h_max: int, device, val_frac: float = 0.1, seed: int = 0,
                 scaler_h5: str = _SCALER_H5):
        self.h_max = int(h_max)
        self.device = torch.device(device)
        self.g = torch.Generator(device=self.device).manual_seed(int(seed))

        d = np.load(npz)
        z = d["z"].astype(np.float32)          # (N,192) fp16 on disk -> fp32
        act = d["act"].astype(np.float32)      # (N,10) raw macro-action blocks
        ep = d["ep_id"].astype(np.int64)       # (N,)  episode id (episode-contiguous)
        pos = d["pos"].astype(np.int64)        # (N,)  in-episode frame index (0..len-1)
        N = len(z)

        # --- standardize actions into eval.py's StandardScaler space (per raw-frame sub-action) ---
        # macro-action is a block of FRAMESKIP(=5) frames x raw_action_dim; scaler is per raw dim.
        # PushT: d_action=10 -> sub_dim=2;  cube: d_action=25 -> sub_dim=5.
        d_action = act.shape[1]
        sub_dim = d_action // 5 if d_action % 5 == 0 else (2 if d_action % 2 == 0 else d_action)
        amean, astd = _action_scaler(act, sub_dim, scaler_h5)
        act = ((act.reshape(N, -1, sub_dim) - amean) / astd).reshape(N, d_action)
        self.act_mean = torch.from_numpy(np.asarray(amean, np.float32)).to(self.device)
        self.act_std = torch.from_numpy(np.asarray(astd, np.float32)).to(self.device)

        # per-anchor exclusive episode-end index (episodes are contiguous blocks; verified).
        uniq, first = np.unique(ep, return_index=True)
        order = np.argsort(first)
        uniq, first = uniq[order], first[order]
        starts = first.tolist() + [N]
        ep_end = np.zeros(N, np.int64)
        for k in range(len(uniq)):
            ep_end[starts[k]:starts[k + 1]] = starts[k + 1]
        # max in-episode gap for anchor i: number of future frames still in the episode.
        # (pos == arange within an episode, so ep_end[i]-1-i == pos_last - pos[i].)
        max_gap = ep_end - 1 - np.arange(N)

        # episode-level split: last `val_frac` of episodes -> val.
        n_val_ep = min(len(uniq) - 1, max(1, int(round(len(uniq) * val_frac))))
        val_ep = set(uniq[-n_val_ep:].tolist()) if n_val_ep > 0 else set()
        is_val = np.array([e in val_ep for e in ep])

        # valid anchors need >=1 future in-episode frame so t+h stays in-episode.
        valid = max_gap >= 1
        train_pool = np.nonzero(valid & ~is_val)[0]
        val_pool = np.nonzero(valid & is_val)[0]

        self.z = torch.from_numpy(z).to(self.device)                 # (N,192)
        self.act = torch.from_numpy(act).to(self.device)             # (N,10)
        self.max_gap = torch.from_numpy(max_gap).to(self.device)     # (N,) long
        self.pool = {
            "train": torch.from_numpy(train_pool).to(self.device),   # (n_train,) long
            "val": torch.from_numpy(val_pool).to(self.device),       # (n_val,) long
        }
        self.n_ep = len(uniq)
        self.n_val_ep = n_val_ep

    def sample(self, bs: int, split: str = "train"):
        """-> (z_t:(bs,192), z_g:(bs,192), h:(bs,) long, a:(bs,10)), all on device."""
        pool = self.pool[split]
        sel = torch.randint(len(pool), (bs,), device=self.device, generator=self.g)
        i = pool[sel]                                                # (bs,) anchor indices
        hi = self.max_gap[i].clamp(max=self.h_max)                   # (bs,) per-anchor max gap
        # h ~ Uniform{1..hi_i}:  floor(U[0,1) * hi) + 1  in [1, hi].
        r = torch.rand(bs, device=self.device, generator=self.g)
        h = (r * hi.to(r.dtype)).floor().long() + 1
        h = torch.clamp(h, min=1, max=self.h_max)                    # guard float edge cases
        j = i + h                                                    # stays in-episode (h <= hi)
        return self.z[i], self.z[j], h, self.act[i]
