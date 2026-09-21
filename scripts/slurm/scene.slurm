#!/usr/bin/env bash
# OGBench visual Scene: LeWM, Sigma-LeWM (A7), AC-MTM and SMWM, three training seeds each,
# 10 epochs (AC-MTM's Scene budget, no early stopping), then evaluated on five evaluation
# seeds x 50 episodes with the h=5 CEM30 protocol of config/eval/scene.yaml.
#
#   ARMS=single : lewm, acmtm, smwm  --array=0-8   one GPU each
#   ARMS=ours   : A7                 --array=0-2   two GPUs (the recipe trains with DDP)
#
# Submit from the repository root (scripts/slurm/run_scene.sh does all of this):
#   sbatch --array=0-8 --export=ALL,ARMS=single scripts/slurm/scene.slurm
#   sbatch --array=0-2 --gpus=2 --export=ALL,ARMS=ours scripts/slurm/scene.slurm
#   # SMOKE=1: two epochs on three batches, then a 10-episode evaluation
#
# Training resumes from the last finished epoch if a job is resubmitted after a time-out.
# Results: $STABLEWM_HOME/results/scene/results.tsv, one line per (arm, seed, eval seed).
#
#SBATCH --job-name=slewm-scene
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

ARMS=${ARMS:?set ARMS=single or ARMS=ours}
i=${SLURM_ARRAY_TASK_ID:?submit with --array}

REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$REPO/.venv/bin/python}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export STABLEWM_HOME=${STABLEWM_HOME:-/scratch-shared/$USER/stable-wm}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

SEEDS=(3072 1 2)
case $ARMS in
  single) LIST=(lewm acmtm smwm); arm=${LIST[$(( i / 3 ))]} ;;
  ours)   arm=A7 ;;
  *) echo "unknown ARMS=$ARMS" >&2; exit 2 ;;
esac
seed=${SEEDS[$(( i % 3 ))]}
suf=$(case $seed in 3072) echo "";; 1) echo B;; 2) echo C;; esac)
epochs=${EPOCHS:-10}; episodes=50; evals="0 1 2 3 4"

X=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  epochs=2; episodes=10; evals="0"
  X=(--override +trainer.limit_train_batches=3 --override +trainer.limit_val_batches=1
     --override loader.batch_size=8)
fi
case $arm in
  lewm)  name=paper_scene_lewm${suf}_seed3072     # the default name is shared by every LeWM run
         X+=(--override output_model_name=$name --override subdir=paper_std_baselines/scene/lewm${suf}_seed3072) ;;
  A7)    name=repro_A7_s${seed}_e${epochs} ;;
  *)     name=paper_scene_${arm}${suf}_seed3072 ;;
esac
[[ "${SMOKE:-0}" == 1 ]] && name_note=" (smoke)" || name_note=""

echo "==> array $i: scene / $arm / seed $seed / $epochs epochs$name_note  $(date -Is)"
"$PY" -m slewm.cli train --task scene --arm "$arm" --epochs "$epochs" --seed "$seed" --execute "${X[@]}"

CK=$STABLEWM_HOME/checkpoints/$name/weights_epoch_${epochs}.pt
[[ -f $CK ]] || { echo "no final checkpoint at $CK" >&2; exit 1; }
OUT=$STABLEWM_HOME/results/scene; mkdir -p "$OUT"
for es in $evals; do
  log=$OUT/${name}_eval${es}${name_note:+_smoke}.log
  "$PY" -m slewm.cli eval --task scene --checkpoint "$CK" --seed "$es" --episodes "$episodes" --execute > "$log" 2>&1
  sr=$(grep -ao "'success_rate': [0-9.]*" "$log" | tail -1 | awk '{print $2}')
  printf "scene\t%s\t%s\t%s\t%s\t%s%s\n" "$arm" "$seed" "$es" "$episodes" "${sr:-FAIL}" "$name_note" | tee -a "$OUT/results.tsv"
done
echo "==> done $(date -Is)"
