#!/usr/bin/env bash
# Σ-LeWM (A7) at epoch 5, the matched comparator for bwlatent at epoch 5 (docs/BWLATENT.md): Table 1's
# evaluation (h5 CEM30, goal +25, budget 50, 50 episodes, planner seeds 0-4). One array task per line of $LIST:
# "<task> <training seed> <checkpoint>". Rows: $STABLEWM_HOME/results/bwlatent/results.tsv (arm A7@5).
#SBATCH --job-name=slewm-a7ep5
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --constraint=scratch-node
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -uo pipefail
i=${SLURM_ARRAY_TASK_ID:?}; cd "$REPO"
export STABLEWM_HOME=${STABLEWM_HOME:-/scratch-shared/$USER/stable-wm} HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$REPO
read -r task seed ck < <(sed -n "$(( i + 1 ))p" "$LIST"); [[ -f $ck ]] || { echo "missing $ck" >&2; exit 2; }
declare -A H5=([pusht]=pusht_expert_train.h5 [cube]=cube_single_expert.h5 [reacher]=reacher.h5 [tworoom]=tworoom.h5 [scene]=visual_scene_play.h5)
OUT=$STABLEWM_HOME/results/bwlatent; mkdir -p "$OUT"; EVAL_HOME=$STABLEWM_HOME
LOCAL=/scratch-node/$USER.$SLURM_JOB_ID
if [[ -d $LOCAL && -w $LOCAL ]]; then mkdir -p $LOCAL/datasets; cp "$STABLEWM_HOME/datasets/${H5[$task]}" $LOCAL/datasets/; ln -sfn "$STABLEWM_HOME/checkpoints" $LOCAL/checkpoints; EVAL_HOME=$LOCAL; fi
echo "==> $task A7@5 seed $seed $ck ($(cat CODE_VERSION))"
for es in 0 1 2 3 4; do
  log=$OUT/A7ep5_${task}_s${seed}_eval${es}.log
  STABLEWM_HOME=$EVAL_HOME "$PY" -m slewm.cli eval --task "$task" --checkpoint "$ck" --seed "$es" --episodes 50 --horizon 5 --cem 30 --execute > "$log" 2>&1
  sr=$(grep -ao "'success_rate': [0-9.]*" "$log" | tail -1 | awk '{print $2}')
  printf "%s\tA7@5\t%s\t%s\t50\t%s\n" "$task" "$seed" "$es" "${sr:-FAIL}" | tee -a "$OUT/results.tsv"
done
