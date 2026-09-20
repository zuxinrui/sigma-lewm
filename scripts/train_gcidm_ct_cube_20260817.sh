#!/usr/bin/env bash
# Rebuild the current-checkpoint Cube latent cache and the frozen GCIDM prior.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

GPU_ID="${1:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
CUBE_H5="$STABLEWM_HOME"/datasets/ogbench/cube_single_expert.h5
CUBE_CKPT="$STABLEWM_HOME"/checkpoints/tworoom_cube_best_20260716/cube_C_T_dcl_idm_sigreg_s3072/weights_epoch_8.pt
LATENT_CACHE="${LATENT_CACHE:-logs/gcidm_ct_cube_latents_100k.npz}"
GCIDM_OUT="${GCIDM_OUT:-logs/gcidm_ct_cube_s${TRAIN_SEED}.pt}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python knob_exp/make_latents.py \
    --h5 "$CUBE_H5" --out "$LATENT_CACHE" --episodes 2500 --bs 512 \
    --device cuda:0 --policy "$CUBE_CKPT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m fm_idm.train \
    --model gcidm --steps 150000 --lr 1e-4 --bs 512 \
    --out "$GCIDM_OUT" --npz "$LATENT_CACHE" --scaler_h5 "$CUBE_H5" \
    --device cuda:0 --h_max 12 --d_action 25 --depth 6 --heads 4 \
    --width 192 --dropout 0.1 --wd 1e-4 --seed "$TRAIN_SEED" \
    --patience 15 --val_every 1000 --val_draws 4

sha256sum "$CUBE_CKPT" "$LATENT_CACHE" "$GCIDM_OUT"
