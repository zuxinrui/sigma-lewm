#!/usr/bin/env bash
# Trains every baseline arm the paper reports but does not take from a released checkpoint.
# One array task per (task, arm, seed).  See docs/BASELINES.md for what the arms are.
#
#   GROUP=standard : SMWM and AC-MTM on Push-T, Cube, Reacher, TwoRoom   --array=0-23
#   GROUP=isaac    : Sub-JEPA and RC-aux on Gear and Cube-Stacking       --array=0-11
#
# Submit from the repository root:
#   sbatch --array=0-23 --export=ALL,GROUP=standard scripts/slurm/baselines.slurm
#   sbatch --array=0-11 --export=ALL,GROUP=isaac    scripts/slurm/baselines.slurm
#   # dry run first: SMOKE=1 trains two epochs on three batches
#   sbatch --array=0-23 --export=ALL,GROUP=standard,SMOKE=1 scripts/slurm/baselines.slurm
#
# Paths come from the environment so that no account or cluster is baked in; override any
# of REPO, PY, SLEWM_EXTERNAL, STABLEWM_HOME or SLEWM_DATA_ROOT with --export.
#
#SBATCH --job-name=slewm-baselines
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

GROUP=${GROUP:?set GROUP=standard or GROUP=isaac}
i=${SLURM_ARRAY_TASK_ID:?submit with --array}

REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$REPO/.venv/bin/python}

# Deterministic training asserts this before Python starts, so it cannot be set later.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export STABLEWM_HOME=${STABLEWM_HOME:-/scratch-shared/$USER/stable-wm}
export SLEWM_DATA_ROOT=${SLEWM_DATA_ROOT:-$STABLEWM_HOME/isaac}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

SEEDS=(3072 1 2)
case $GROUP in
  standard)
    TASKS=(pusht cube reacher tworoom); ARMS=(smwm acmtm)
    task=${TASKS[$(( i / 6 ))]}; arm=${ARMS[$(( (i / 3) % 2 ))]}; epochs=15 ;;
  isaac)
    TASKS=(gear franka); ARMS=(subjepa rcaux)
    task=${TASKS[$(( i / 6 ))]}; arm=${ARMS[$(( (i / 3) % 2 ))]}
    epochs=30; [[ $task == franka ]] && epochs=20
    # Only these two arms read the upstream objectives.
    : "${SLEWM_EXTERNAL:?clone the pinned upstream repos and export SLEWM_EXTERNAL (docs/BASELINES.md)}"
    export SLEWM_EXTERNAL ;;
  *) echo "unknown GROUP=$GROUP" >&2; exit 2 ;;
esac
seed=${SEEDS[$(( i % 3 ))]}

EXTRA=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  epochs=2
  EXTRA=(--override +trainer.limit_train_batches=3 --override +trainer.limit_val_batches=1
         --override loader.batch_size=8)
fi

echo "==> array $i: $task / $arm / seed $seed / $epochs epochs  $(date -Is)"
# `slewm train` resolves the config and already pins trainer.devices=1: these arms zero the
# SIGReg weight, which leaves parameters out of the loss and makes DDP abort.
"$PY" -m slewm.cli train --task "$task" --arm "$arm" --epochs "$epochs" --seed "$seed" --execute "${EXTRA[@]}"
echo "==> done rc=$? $(date -Is)"
