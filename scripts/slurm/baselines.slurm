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
# Node-local NVMe: random reads of a 50-100 GB h5 over GPFS starve the GPU.
#SBATCH --constraint=scratch-node
# Ten minutes before the limit the batch shell gets USR1 and queues its own continuation.
#SBATCH --signal=B:USR1@600
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
if [[ "${SMOKE:-0}" == 1 ]]; then
  # Smoke runs write under their own root: the trainer resumes from any <run>_weights.ckpt it
  # finds, so a smoke checkpoint in the real run directory would become the real run's start.
  # The smoke root holds a `datasets` symlink to the real one.
  export STABLEWM_HOME=${SMOKE_HOME:-/scratch-shared/$USER/stable-wm-smoke}
fi
export SLEWM_DATA_ROOT=${SLEWM_DATA_ROOT:-$STABLEWM_HOME/isaac}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
# stable-pretraining keeps its own last.ckpt per run; keep it off the home quota.
export SPT_CACHE_DIR=${SPT_CACHE_DIR:-/scratch-shared/$USER/spt-cache}
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
    epochs=${EPOCHS:-$epochs}
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
elif [[ $GROUP == standard ]]; then
  # Write a full-state checkpoint at every epoch boundary so a continuation resumes from it.
  EXTRA=(--override ++save_resume_checkpoint=true --override ++resume_full_state=true)
fi
echo "==> code: $REPO ($(cat "$REPO/CODE_VERSION" 2>/dev/null || git -C "$REPO" rev-parse --short HEAD 2>/dev/null)); checkpoints under $STABLEWM_HOME"

# Stage the dataset onto node-local NVMe; `aligned.py` reads <LOCAL_DATASET_DIR>/datasets/.
if [[ $GROUP == standard ]]; then
  declare -A H5=([pusht]=pusht_expert_train.h5 [cube]=cube_single_expert.h5
                 [reacher]=reacher.h5 [tworoom]=tworoom.h5)
  LOCAL=/scratch-node/$USER.$SLURM_JOB_ID
  if [[ -d $LOCAL && -w $LOCAL ]]; then
    mkdir -p "$LOCAL/datasets"
    echo "==> staging ${H5[$task]} to $LOCAL  $(date -Is)"
    cp "$STABLEWM_HOME/datasets/${H5[$task]}" "$LOCAL/datasets/"
    export LOCAL_DATASET_DIR=$LOCAL
    echo "==> staged  $(date -Is)"
  else
    echo "==> no node-local scratch at $LOCAL; reading from $STABLEWM_HOME" >&2
  fi
elif [[ $GROUP == isaac ]]; then
  # Gear reads <SLEWM_DATA_ROOT>/datasets/<name>, Cube-Stacking <SLEWM_DATA_ROOT>/<name>:
  # stage this task's train and validation files into both places on node-local NVMe.
  LOCAL=/scratch-node/$USER.$SLURM_JOB_ID
  if [[ -d $LOCAL && -w $LOCAL ]]; then
    mkdir -p "$LOCAL/datasets"
    prefix=$([[ $task == gear ]] && echo gear_01036_effective4_persistent_start_h5 || echo franka_stack_aligned)
    for f in "$SLEWM_DATA_ROOT"/${prefix}_*.h5; do
      cp "$f" "$LOCAL/"; ln -sfn "../$(basename "$f")" "$LOCAL/datasets/$(basename "$f")"
    done
    export SLEWM_DATA_ROOT=$LOCAL
    echo "==> staged $(ls "$LOCAL"/*.h5 | xargs -n1 basename | tr '\n' ' ')to $LOCAL"
  fi
fi

# Continuation: on USR1 queue the same array index to start once this job has ended.
# A real run resumes from its last epoch checkpoint; RESUBMITS caps runaway chains.
n=${RESUBMITS:-0}
continue_run() {
  if [[ "${SMOKE:-0}" == 1 || $n -ge 5 ]]; then
    echo "==> time limit reached; not resubmitting (smoke=${SMOKE:-0}, resubmits=$n)" >&2; return
  fi
  local next
  next=$(env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') \
    sbatch --parsable --chdir="$REPO" --array="$i" --dependency="afterany:$SLURM_JOB_ID" \
      --export="ALL,GROUP=$GROUP,REPO=$REPO,PY=$PY,STABLEWM_HOME=$STABLEWM_HOME,SLEWM_DATA_ROOT=$SLEWM_DATA_ROOT,RESUBMITS=$((n + 1))" \
      "$REPO/scripts/slurm/baselines.slurm") \
    || { echo "==> continuation sbatch failed" >&2; return 0; }
  echo "==> time limit near; continuation queued as $next (resubmit $((n + 1)))" >&2
}
trap continue_run USR1

echo "==> array $i: $task / $arm / seed $seed / $epochs epochs / resubmit $n  $(date -Is)"
# `slewm train` resolves the config and already pins trainer.devices=1: these arms zero the
# SIGReg weight, which leaves parameters out of the loss and makes DDP abort.
"$PY" -m slewm.cli train --task "$task" --arm "$arm" --epochs "$epochs" --seed "$seed" --execute "${EXTRA[@]}" &
pid=$!
# `wait` returns early when USR1 arrives; keep waiting for the trainer itself.
rc=0
while kill -0 "$pid" 2>/dev/null; do wait "$pid" && rc=0 || rc=$?; done
echo "==> done rc=$rc $(date -Is)"
exit "$rc"
