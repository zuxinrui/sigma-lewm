"""LeWM Variant B: swap SIGReg -> whitening_barrier + FMCA((z_t, a_t), z_{t+1}).

One-file patch of train.py. Only the loss composition and the module wiring change;
everything else (config, dataset, optimizer, trainer, ckpt manager) is byte-identical
to train.py so any PushT sanity check that reproduces from train.py should reproduce
here up to the loss swap and the two projection heads' parameters.

Variant B loss (Hu Eq. 3, log-det form; anti-collapse via non-invariant barrier):

    total = pred_MSE
          + w_bar * whitening_barrier(emb.reshape(-1, D)) / D
          + w_f   * fmca_loss( head_f([z_t; a_t]),  head_g(z_{t+1}) )

where
  * whitening_barrier(C) = tr(C) - logdet(C + eps*I)   (min at C=I; NOT scale-invariant
    so it actually floors rank -- SIGReg's job).
  * fmca_loss(f, g)      = logdet R_XY - logdet R_X - logdet R_Y = sum_k log(1 - s_k^2)
    (Variant B slogdet form; centered; ridged; sums the log(1-lambda_k) spectrum via a
    scale-invariant log-det ratio -- dynamics-aware dependence, NOT a rank floor).

The two roles are complementary and both are needed: barrier floors the encoder rank
(SIGReg replacement), FMCA drives (z_t, a_t) <-> z_{t+1} predictability in the whitened
K-dim head space.

Config knobs (all optional; defaults hard-coded here, override via +key=value on the CLI):
  fmca_k            : int, projection dim K for the FMCA heads (default 64)
  fmca_weight       : float, w_f multiplier on fmca_loss (default 0.02)
  barrier_weight    : float, w_bar multiplier on whitening_barrier(emb)/D (default 0.03)
  fmca_eps          : float, ridge for RX/RY/RXY inside fmca_loss (default 1e-3)
  fmca_s_clamp      : float, canonical-corr clamp inside fmca_loss (default 1e-3)

The seed / lambd knob from lewm.yaml is preserved so `run_fmca_ab.sh` uses the same
seed as baseline train.py. `cfg.loss.sigreg.weight` is intentionally NOT read by this
file (SIGReg is gone); it is still allowed to exist in the YAML so the same config
loads unchanged.
"""

from slewm.paths import REPO_ROOT
import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from slewm.training.utilities import get_column_normalizer, get_img_preprocessor, SaveCkptCallback

# Reuse the vetted FMCA primitives shipped in fmca_exp/ (log-det slogdet loss,
# whitening barrier, small MLP projection head, RankMe diagnostic).

from fmca_exp.fmca import ProjHead, fmca_loss                       # slogdet Variant B loss + head
from fmca_exp.fmca_common import rankme, whitening_barrier          # anti-collapse barrier + diag


def _fmca_variant_b(f_feat, g_feat, eps, s_clamp):
    """Wrap fmca_loss with bf16-safe autocast disable and (B, T, K) flattening."""
    with torch.autocast(device_type=f_feat.device.type, enabled=False):
        r, _ = fmca_loss(
            f_feat.float(), g_feat.float(),
            eps=eps, s_clamp=s_clamp,
            center=True, loss_mode="slogdet", return_spectrum=False,
        )
    return r


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute Variant-B FMCA loss."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    # Variant-B mixing weights (config-overridable via +key=value)
    w_bar = float(cfg.get("barrier_weight", 0.03))
    w_f = float(cfg.get("fmca_weight", 0.02))
    fmca_eps = float(cfg.get("fmca_eps", 1e-3))
    fmca_s_clamp = float(cfg.get("fmca_s_clamp", 1e-3))

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]              # (B, T, D)
    act_emb = output["act_emb"]      # (B, T, A)

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]                       # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # ---- Prediction MSE (kept identical to train.py so CEM planning cost stays calibrated) ----
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    output["pred_loss"] = pred_loss

    # ---- Anti-collapse whitening barrier on the encoder embeddings (SIGReg replacement) ----
    D = emb.shape[-1]
    barrier_loss = whitening_barrier(emb.reshape(-1, D)) / D
    output["barrier_loss"] = barrier_loss

    # ---- FMCA((z_t, a_t), z_{t+1}) dependence in the whitened K-dim head space ----
    # X_t = [z_t; a_t] over t = 0 .. T-2 ; Y_t = z_{t+1} over the same t.
    X_t = torch.cat([emb[:, :-1], act_emb[:, :-1]], dim=-1)   # (B, T-1, D+A)
    Y_t = emb[:, 1:]                                          # (B, T-1, D)
    fmca_loss_val = _fmca_variant_b(
        self.model.head_f(X_t),
        self.model.head_g(Y_t),
        eps=fmca_eps, s_clamp=fmca_s_clamp,
    )
    output["fmca_loss"] = fmca_loss_val

    # ---- Total (Variant B, doc s.d3 recommendation) ----
    output["loss"] = pred_loss + w_bar * barrier_loss + w_f * fmca_loss_val

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    # RankMe on the encoder embeddings: decisive anti-collapse readout.
    try:
        self.log(
            f"{stage}/rankme",
            rankme(emb.reshape(-1, D)),
            on_step=True, sync_dist=True,
        )
    except Exception:
        pass

    return output


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "config/train"), config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # ---- Variant-B FMCA projection heads (K-dim), attached to `model` so the ----
    # ---- existing 'model_opt' optimizer trains them for free.                ----
    # The action_encoder emits emb_dim per-step (see config/train/model/lewm.yaml
    # -> action_encoder.emb_dim: ${embed_dim}), so head_f input dim is 2 * embed_dim.
    K_fmca = int(cfg.get("fmca_k", 64))
    D_emb = int(cfg.embed_dim)
    A_act = int(cfg.get("action_emb_dim", D_emb))
    world_model.head_f = ProjHead(D_emb + A_act, K_fmca, depth=2)
    world_model.head_g = ProjHead(D_emb,         K_fmca, depth=2)

    print(
        f"[VARIANT-B] K={K_fmca}  D_emb={D_emb}  A_act={A_act}  "
        f"w_bar={float(cfg.get('barrier_weight', 0.03))}  "
        f"w_fmca={float(cfg.get('fmca_weight', 0.02))}  "
        f"seed={cfg.seed}",
        flush=True,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    print(f"VARIANT_B_TRAIN_DONE seed={cfg.seed} subdir={run_id}", flush=True)
    return


if __name__ == "__main__":
    run()
