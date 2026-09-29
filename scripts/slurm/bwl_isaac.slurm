#!/usr/bin/env bash
# bwlatent (docs/BWLATENT.md) on Gear-Assembly and Cube-Stacking: configs paper_{gear,franka}_ksbwl{,B,C}
# (Figure 6's ours with champion_backward_key: latent), 10 epochs, every epoch saved. Training only:
# evaluation needs Isaac Lab on an RTX GPU and runs off-cluster.
#   array index i: task = (gear, franka)[i / 3], training seed = (3072, 1, 2)[i % 3]
#   sbatch --array=0-5 --export=ALL,REPO=...,PY=...,STABLEWM_HOME=... bwl_isaac.slurm
#   # SMOKE=1: two epochs of three batches, own STABLEWM_HOME
#SBATCH --job-name=slewm-bwlisaac
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --constraint=scratch-node
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -uo pipefail
i=${SLURM_ARRAY_TASK_ID:?submit with --array}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$REPO/.venv/bin/python}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export STABLEWM_HOME=${STABLEWM_HOME:-/scratch-shared/$USER/stable-wm}
export SLEWM_DATA_ROOT=${SLEWM_DATA_ROOT:-$STABLEWM_HOME/isaac}
[[ "${SMOKE:-0}" == 1 ]] && export STABLEWM_HOME=${SMOKE_HOME:-/scratch-shared/$USER/stable-wm-smoke}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf} SPT_CACHE_DIR=${SPT_CACHE_DIR:-/scratch-shared/$USER/spt-cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"
TASKS=(gear franka); SEEDS=(3072 1 2)
task=${TASKS[$(( i / 3 ))]}; seed=${SEEDS[$(( i % 3 ))]}; epochs=${EPOCHS:-10}; X=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  epochs=2; X=(--override +trainer.limit_train_batches=3 --override +trainer.limit_val_batches=1 --override loader.batch_size=8)
fi
LOCAL=/scratch-node/$USER.$SLURM_JOB_ID
if [[ -d $LOCAL && -w $LOCAL ]]; then
  # Gear reads <root>/datasets/<name>, Cube-Stacking <root>/<name>: stage into both (as baselines.slurm).
  mkdir -p "$LOCAL/datasets"
  prefix=$([[ $task == gear ]] && echo gear_01036_effective4_persistent_start_h5 || echo franka_stack_aligned)
  for f in "$SLEWM_DATA_ROOT"/${prefix}_*$([[ $task == gear ]] && echo _v4).h5; do cp "$f" "$LOCAL/"; ln -sfn "../$(basename "$f")" "$LOCAL/datasets/$(basename "$f")"; done
  export SLEWM_DATA_ROOT=$LOCAL
fi
echo "==> code: $REPO ($(cat "$REPO/CODE_VERSION" 2>/dev/null)); $task / ksbwl / seed $seed / $epochs epochs  $(date -Is)"
"$PY" -m slewm.cli train --task "$task" --arm ksbwl --epochs "$epochs" --seed "$seed" --execute "${X[@]}"
echo "==> done rc=$? $(date -Is)"
