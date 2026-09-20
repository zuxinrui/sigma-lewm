#!/usr/bin/env python3
"""Post-hoc training-style metrics for PushT LeWM checkpoints.

This evaluates frozen checkpoints on a deterministic slice of the PushT
validation split. It does not run environment eval.
"""

from __future__ import annotations
from repro_paths import resolve_path as _resolve_artifact

import argparse
import csv
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fmca_exp"))

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from stable_pretraining.backbone.utils import vit_hf

from fmca_exp.fmca_common import rankme, whitening_barrier
from slewm.models.world_model import JEPA
from slewm.models.components import ARPredictor, Embedder, MLP, SIGReg
from slewm.training.utilities import get_img_preprocessor


STABLEWM = Path.home() / ".stable-wm"
DEFAULT_CKPTS = {
    "hf_official": STABLEWM / "hf_pusht" / "weights.pt",
    "paper_baseline_epoch4": Path(_resolve_artifact('legacy:lewm_baseline_compare/bundle/paper_baseline/ckpt.pth')),
    "C_S_long": Path(_resolve_artifact('legacy:lewm_baseline_compare/bundle/C_S_long/ckpt.pth')),
    "C_T_idmgrad_long": Path(_resolve_artifact('legacy:lewm_baseline_compare/bundle/C_T_idmgrad_long/ckpt.pth')),
    "C_W_mix": Path(_resolve_artifact('legacy:lewm_baseline_compare/bundle/C_W_mix/ckpt.pth')),
    "C_S_noidm": Path(_resolve_artifact('legacy:lewm_baseline_compare/bundle/C_S_noidm/ckpt.pth')),
    "C_V_bidir_simsiam": Path(_resolve_artifact('legacy:lewm_baseline_compare/bundle/C_V_bidir_simsiam/ckpt.pth')),
    "C_W_dynamo_triple": Path(_resolve_artifact('legacy:lewm_baseline_compare/bundle/C_W_dynamo_triple/ckpt.pth')),
}


def build_model(device: str) -> JEPA:
    model = JEPA(
        encoder=vit_hf(
            size="tiny",
            patch_size=14,
            image_size=224,
            pretrained=False,
            use_mask_token=False,
        ),
        predictor=ARPredictor(
            num_frames=3,
            input_dim=192,
            hidden_dim=192,
            output_dim=192,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.0,
        ),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        projector=MLP(
            input_dim=192,
            output_dim=192,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=192,
            output_dim=192,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
    )
    return model.to(device).eval()


def parse_ckpt_args(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return DEFAULT_CKPTS.copy()
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--ckpt expects name=/path/to/weights.pt, got: {value}")
        name, path = value.split("=", 1)
        out[name] = Path(path).expanduser()
    return out


def load_batches(args: argparse.Namespace) -> list[dict[str, torch.Tensor]]:
    h5_path = Path(args.h5_path).expanduser()
    ds = swm.data.load_dataset(
        str(h5_path),
        transform=None,
        num_steps=args.history_size + args.horizon,
        frameskip=5,
        keys_to_load=["pixels", "action"],
        keys_to_cache=[],
    )

    transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    )

    action_data = torch.from_numpy(np.asarray(ds.get_col_data("action")))
    action_data = action_data[~torch.isnan(action_data).any(dim=1)]
    action_mean = action_data.mean(0, keepdim=True).float()
    action_std = action_data.std(0, keepdim=True).float()
    action_dim = int(ds[0]["action"].shape[-1])
    if action_dim % action_mean.shape[-1] != 0:
        raise SystemExit(
            f"cannot broadcast action normalizer: item_dim={action_dim} "
            f"raw_dim={action_mean.shape[-1]}"
        )
    repeats = action_dim // action_mean.shape[-1]
    action_mean = action_mean.repeat(1, repeats)
    action_std = action_std.repeat(1, repeats)

    train_set, val_set = torch.utils.data.random_split(
        range(len(ds)),
        lengths=[args.train_split, 1.0 - args.train_split],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_indices = np.asarray(val_set.indices)
    total = args.num_batches * args.batch_size
    if total > len(val_indices):
        raise SystemExit(f"requested {total} samples, val split only has {len(val_indices)}")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(val_indices, size=total, replace=False)

    print(
        f"[data] len={len(ds)} val={len(val_indices)} samples={total} "
        f"num_steps={args.history_size + args.horizon}",
        flush=True,
    )

    batches: list[dict[str, torch.Tensor]] = []
    for b in range(args.num_batches):
        idx = picks[b * args.batch_size : (b + 1) * args.batch_size]
        pixels = torch.stack([torch.as_tensor(ds[int(i)]["pixels"]) for i in idx], dim=0)
        action = torch.stack([torch.as_tensor(ds[int(i)]["action"]) for i in idx], dim=0)
        item = transform({"pixels": pixels})
        action = ((action - action_mean) / action_std).float()
        batches.append({"pixels": item["pixels"], "action": action})
        if (b + 1) % max(1, args.num_batches // 4) == 0:
            print(f"[data] prepared batch {b + 1}/{args.num_batches}", flush=True)
    return batches


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def ar_rollout(model: JEPA, ctx_emb: torch.Tensor, act_emb: torch.Tensor, ctx_len: int, horizon: int) -> torch.Tensor:
    emb_full = ctx_emb
    preds = []
    for _ in range(horizon):
        length = emb_full.size(1)
        emb_trunc = emb_full[:, -ctx_len:]
        act_trunc = act_emb[:, length - ctx_len : length]
        pred_full = model.predict(emb_trunc, act_trunc)
        pred_step = pred_full[:, -1:]
        preds.append(pred_step)
        emb_full = torch.cat([emb_full, pred_step], dim=1)
    return torch.cat(preds, dim=1)


def evaluate_one(
    name: str,
    ckpt_path: Path,
    batches: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    device: str,
) -> dict[str, float | int | str]:
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    model = build_model(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    sigreg = SIGReg(knots=17, num_proj=args.sigreg_proj).to(device).eval()
    hs = args.history_size
    horizon = args.horizon

    tf_losses: list[float] = []
    ar_losses: list[float] = []
    step_losses: list[list[float]] = [[] for _ in range(horizon)]
    barrier_losses: list[float] = []
    sigreg_losses: list[float] = []
    emb_stds: list[float] = []
    emb_norms: list[float] = []
    z_chunks: list[torch.Tensor] = []

    t0 = time.time()
    with torch.inference_mode():
        for b_idx, batch in enumerate(batches):
            info = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model.encode(info)
            emb = out["emb"]
            act_emb = out["act_emb"]

            ctx_emb = emb[:, :hs]
            ctx_act = act_emb[:, :hs]
            tf_pred = model.predict(ctx_emb, ctx_act)
            tf_tgt = emb[:, 1 : 1 + hs]
            tf_losses.append(float((tf_pred - tf_tgt).pow(2).mean().item()))

            pred_seq = ar_rollout(model, ctx_emb, act_emb, hs, horizon)
            ar_tgt = emb[:, hs : hs + horizon]
            err = (pred_seq - ar_tgt).pow(2)
            ar_losses.append(float(err.mean().item()))
            for k in range(horizon):
                step_losses[k].append(float(err[:, k].mean().item()))

            z = emb.reshape(-1, emb.shape[-1]).float()
            z_chunks.append(z.cpu())
            barrier_losses.append(float((whitening_barrier(z) / z.shape[-1]).item()))
            emb_stds.append(float(z.std().item()))
            emb_norms.append(float(z.norm(dim=-1).mean().item()))

            torch.manual_seed(args.seed + b_idx)
            sigreg_losses.append(float(sigreg(emb.transpose(0, 1)).item()))

    Z = torch.cat(z_chunks, dim=0)
    feat_std = Z.std(dim=0)
    row: dict[str, float | int | str] = {
        "name": name,
        "ckpt": str(ckpt_path),
        "n_seq": args.num_batches * args.batch_size,
        "n_emb": int(Z.shape[0]),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "tf_loss": mean(tf_losses),
        "ar_loss": mean(ar_losses),
        "rankme": rankme(Z),
        "barrier_loss": mean(barrier_losses),
        "sigreg_loss": mean(sigreg_losses),
        "emb_std": mean(emb_stds),
        "emb_norm": mean(emb_norms),
        "feat_std_min": float(feat_std.min().item()),
        "feat_std_median": float(feat_std.median().item()),
        "feat_std_max": float(feat_std.max().item()),
        "wall_s": time.time() - t0,
    }
    for k, values in enumerate(step_losses, start=1):
        row[f"ar_step{k}"] = mean(values)

    del model
    torch.cuda.empty_cache()
    return row


def fmt(x: float | int | str) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        return "nan"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-path", default=str(STABLEWM / "datasets" / "pusht_expert_train.h5"))
    ap.add_argument("--output", type=Path, default=Path("/tmp/lewm_posthoc_wm_metrics.csv"))
    ap.add_argument("--ckpt", action="append", help="name=/abs/path/to/weights.pt; omit for defaults")
    ap.add_argument("--num-batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--history-size", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train-split", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--sigreg-proj", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = parse_ckpt_args(args.ckpt)
    batches = load_batches(args)

    rows = []
    for name, path in ckpts.items():
        print(f"\n[eval] {name}: {path}", flush=True)
        row = evaluate_one(name, path, batches, args, args.device)
        rows.append(row)
        print(
            f"[eval] {name} tf={row['tf_loss']:.6f} ar={row['ar_loss']:.6f} "
            f"rankme={row['rankme']:.2f} sigreg={row['sigreg_loss']:.4f} "
            f"missing={row['missing']} unexpected={row['unexpected']}",
            flush=True,
        )

    fieldnames = [
        "name",
        "ckpt",
        "n_seq",
        "n_emb",
        "missing",
        "unexpected",
        "tf_loss",
        "ar_loss",
        *[f"ar_step{k}" for k in range(1, args.horizon + 1)],
        "rankme",
        "barrier_loss",
        "sigreg_loss",
        "emb_std",
        "emb_norm",
        "feat_std_min",
        "feat_std_median",
        "feat_std_max",
        "wall_s",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[done] wrote {args.output}", flush=True)
    compact_cols = ["name", "tf_loss", "ar_loss", "ar_step5", "rankme", "barrier_loss", "sigreg_loss"]
    print(" | ".join(compact_cols))
    print(" | ".join("---" for _ in compact_cols))
    for row in rows:
        print(" | ".join(fmt(row[c]) for c in compact_cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
