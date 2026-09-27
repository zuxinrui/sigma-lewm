#!/usr/bin/env bash
# Phase-SIAM quick look (2026-09-27): the S1/S2/S3 arms on Gear-Assembly and OGBench-Scene, one
# training seed (1) each. One array task per line of $RUNS: "<task> <config name>", task in
# gear | scene; the config is a file under config/train (e.g. paper_gear_janus_osB).
#
#   gear   the config's own contract (paper_new_task_control: 50-100 epochs, loss-composite early
#          stopping, every epoch saved); evaluation needs Isaac Lab and runs off-cluster.
#   scene  10 epochs, then Table 1's evaluation of epoch 10: h5 CEM30, 5 planner seeds x 50.
#
#   sbatch --array=0-N --export=ALL,REPO=...,RUNS=...,PY=...,STABLEWM_HOME=... phase_siam.slurm
#   # SMOKE=1: two epochs of three batches, one 10-episode evaluation (scene), own STABLEWM_HOME
#
#SBATCH --job-name=slewm-psiam
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --constraint=scratch-node
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail
i=${SLURM_ARRAY_TASK_ID:?submit with --array}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$REPO/.venv/bin/python}
: "${RUNS:?point RUNS at the run list}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export STABLEWM_HOME=${STABLEWM_HOME:-/scratch-shared/$USER/stable-wm}
export SLEWM_DATA_ROOT=${SLEWM_DATA_ROOT:-$STABLEWM_HOME/isaac}
[[ "${SMOKE:-0}" == 1 ]] && export STABLEWM_HOME=${SMOKE_HOME:-/scratch-shared/$USER/stable-wm-smoke}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export SPT_CACHE_DIR=${SPT_CACHE_DIR:-/scratch-shared/$USER/spt-cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

read -r task config < <(sed -n "$(( i + 1 ))p" "$RUNS")
[[ -n ${config:-} && -f config/train/$config.yaml ]] || { echo "no run line $i in $RUNS or no config $config" >&2; exit 2; }
name=$(grep -m1 '^output_model_name:' "config/train/$config.yaml" | awk '{print $2}')
X=(++save_resume_checkpoint=true ++resume_full_state=true "output_model_name=$name")
evals="0 1 2 3 4"; episodes=50; epochs=10
if [[ "${SMOKE:-0}" == 1 ]]; then
  evals="0"; episodes=10; epochs=2
  X+=(trainer.max_epochs=2 trainer.min_epochs=0 +trainer.limit_train_batches=3 +trainer.limit_val_batches=1
      loader.batch_size=8)
  [[ $task == gear ]] && X+=(early_stopping.enabled=false)
fi

LOCAL=/scratch-node/$USER.$SLURM_JOB_ID
if [[ -d $LOCAL && -w $LOCAL ]]; then
  mkdir -p "$LOCAL/datasets"
  if [[ $task == gear ]]; then
    # Gear reads <SLEWM_DATA_ROOT>/datasets/<name>: stage train and validation onto NVMe.
    for f in "$SLEWM_DATA_ROOT"/gear_01036_effective4_persistent_start_h5_{train,validation}_v4.h5; do
      cp "$f" "$LOCAL/"; ln -sfn "../$(basename "$f")" "$LOCAL/datasets/$(basename "$f")"
    done
    export SLEWM_DATA_ROOT=$LOCAL
  else
    cp "$STABLEWM_HOME/datasets/visual_scene_play.h5" "$LOCAL/datasets/"
    export LOCAL_DATASET_DIR=$LOCAL
  fi
fi
echo "==> code: $REPO ($(cat "$REPO/CODE_VERSION" 2>/dev/null)); $task / $config -> $name  $(date -Is)"

n=${RESUBMITS:-0}
continue_run() {
  if [[ "${SMOKE:-0}" == 1 || $n -ge 5 ]]; then
    echo "==> time limit reached; not resubmitting (smoke=${SMOKE:-0}, resubmits=$n)" >&2; return 0
  fi
  env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') \
    sbatch --parsable --chdir="$REPO" --array="$i" --dependency="afterany:$SLURM_JOB_ID" \
      --export="ALL,REPO=$REPO,RUNS=$RUNS,PY=$PY,STABLEWM_HOME=$STABLEWM_HOME,RESUBMITS=$((n + 1))" \
      "$REPO/scripts/slurm/phase_siam.slurm" || echo "==> continuation sbatch failed" >&2
  return 0
}
trap continue_run USR1

[[ $task == gear ]] && export LOCAL_DATASET_DIR=$SLEWM_DATA_ROOT   # as slewm.cli does for Isaac tasks
"$PY" -m slewm.training.aligned --config-name="$config" "${X[@]}" &
pid=$!
rc=0; while kill -0 "$pid" 2>/dev/null; do wait "$pid" && rc=0 || rc=$?; done
[[ $rc -ne 0 ]] && { echo "==> training rc=$rc"; exit "$rc"; }
[[ $task == gear ]] && { echo "==> done (gear: evaluate off-cluster) $(date -Is)"; exit 0; }

CK=$STABLEWM_HOME/checkpoints/$name/weights_epoch_${epochs}.pt
[[ -f $CK ]] || { echo "no final checkpoint at $CK" >&2; exit 1; }
OUT=$STABLEWM_HOME/results/phase_siam; mkdir -p "$OUT"
# The evaluator reads $STABLEWM_HOME/datasets: point it at the NVMe copy (checkpoint paths are absolute).
EVAL_HOME=$STABLEWM_HOME
if [[ -n ${LOCAL_DATASET_DIR:-} ]]; then ln -sfn "$STABLEWM_HOME/checkpoints" "$LOCAL/checkpoints"; EVAL_HOME=$LOCAL; fi
for es in $evals; do
  log=$OUT/${name}_eval${es}.log
  STABLEWM_HOME=$EVAL_HOME "$PY" -m slewm.cli eval --task "$task" --checkpoint "$CK" --seed "$es" --episodes "$episodes" \
    --horizon 5 --cem 30 --execute > "$log" 2>&1
  sr=$(grep -ao "'success_rate': [0-9.]*" "$log" | tail -1 | awk '{print $2}')
  printf "%s\t%s\t1\t%s\t%s\t%s\n" "$task" "$config" "$es" "$episodes" "${sr:-FAIL}" | tee -a "$OUT/results.tsv"
done
echo "==> done $(date -Is)"
