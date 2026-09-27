#!/usr/bin/env bash
# Figures 3, 4 and 5 at 3 training seeds x 5 planner seeds (docs/FIG345_PLAN.md), on the Table 1
# checkpoints listed in docs/results/fig345_checkpoints.tsv (one array task per checkpoint, 72).
#
#   FIG=4   h=5, goal +25, budget 50, CEM iterations 8/4/2           -> fig4_budget_raw.tsv
#   FIG=5   scripts/goal_tolerance_eval.py, h5 CEM30, per-step errors -> goal_tolerance/*.npz
#   FIG=3   CEM30, h in $HS (default "3 10"; then HS=15), goal 5h, budget 10h, receding h
#                                                                     -> fig3_horizon_raw.tsv
# Every evaluation is 50 episodes at planner seeds 0-4. Rerunning skips evaluations already done.
# The evaluators read $STABLEWM_HOME/datasets, so each job copies its task's dataset to node-local
# NVMe and points STABLEWM_HOME there (results and checkpoints stay on /scratch-shared). PAR
# evaluations (default 3) share the GPU: most of an evaluation's wall time is dataset caching and
# environment set-up, not planning. wall_seconds is therefore measured under that sharing.
#
#   sbatch --array=0-71 --export=ALL,REPO=...,PY=...,STABLEWM_HOME=...,FIG=4 fig345.slurm
#   # SMOKE=1: planner seed 0 only, 5 episodes, one setting per figure, output under fig345_smoke
#
#SBATCH --job-name=slewm-fig345
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
FIG=${FIG:?set FIG=3, 4 or 5}
LIST=${LIST:-$REPO/docs/results/fig345_checkpoints.tsv}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export STABLEWM_HOME=${STABLEWM_HOME:-/scratch-shared/$USER/stable-wm}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

read -r task method tseed ck < <(sed -n "$(( i + 2 ))p" "$LIST")
[[ -f ${ck:-} ]] || { echo "no checkpoint on line $i of $LIST: ${ck:-}" >&2; exit 2; }
sha=$(sha256sum "$ck" | cut -c1-64)

OUT=$STABLEWM_HOME/results/fig345
seeds="0 1 2 3 4"; episodes=50; cems="8 4 2"; HS=${HS:-3 10}
if [[ "${SMOKE:-0}" == 1 ]]; then
  OUT=$STABLEWM_HOME/results/fig345_smoke; seeds="0"; episodes=5; cems="2"; HS="3"
fi
mkdir -p "$OUT/logs" "$OUT/goal_tolerance"

declare -A H5=([pusht]=pusht_expert_train.h5 [cube]=cube_single_expert.h5
               [reacher]=reacher.h5 [tworoom]=tworoom.h5)
LOCAL=/scratch-node/$USER.$SLURM_JOB_ID
if [[ -d $LOCAL && -w $LOCAL ]]; then
  mkdir -p "$LOCAL/datasets"
  t0=$SECONDS; cp "$STABLEWM_HOME/datasets/${H5[$task]}" "$LOCAL/datasets/"
  ln -sfn "$STABLEWM_HOME/checkpoints" "$LOCAL/checkpoints"
  echo "==> staged ${H5[$task]} to $LOCAL in $((SECONDS - t0)) s"
  export STABLEWM_HOME=$LOCAL
fi
PAR=${PAR:-3}
slot() { while (( $(jobs -rp | wc -l) >= PAR )); do wait -n; done; }
echo "==> code: $REPO ($(cat "$REPO/CODE_VERSION" 2>/dev/null)); fig $FIG / $task / $method / training seed $tseed"
echo "==> checkpoint $ck sha256 $sha  $(date -Is)"

HEADER="task	method	training_seed	planner_seed	horizon	goal_offset	eval_budget	cem_iters	num_eval	success_rate	wall_seconds	checkpoint_sha256"
append() {  # file row: one writer at a time across the array
  ( flock 9; [[ -s $1 ]] || echo "$HEADER" >> "$1"; echo "$2" >> "$1" ) 9>"$1.lock"
}
done_already() {  # file task method tseed ps h cem episodes
  [[ -f $1 ]] && awk -F'\t' -v k="$2	$3	$4	$5	$6" -v c="$7" -v n="$8" \
    '$1"\t"$2"\t"$3"\t"$4"\t"$5==k && $8==c && $9==n && $10!="FAIL" {f=1} END {exit !f}' "$1"
}
evaluate() {  # file h cem ps
  local f=$1 h=$2 cem=$3 ps=$4
  done_already "$f" "$task" "$method" "$tseed" "$ps" "$h" "$cem" "$episodes" && return 0
  slot
  ( local log=$OUT/logs/fig${FIG}_${task}_${method}_ts${tseed}_h${h}_cem${cem}_s${ps}.log t0=$SECONDS
  "$PY" -m slewm.cli eval --task "$task" --checkpoint "$ck" --seed "$ps" --episodes "$episodes" \
    --horizon "$h" --cem "$cem" --execute > "$log" 2>&1
  local sr; sr=$(grep -ao "'success_rate': [0-9.]*" "$log" | tail -1 | awk '{print $2}')
  append "$f" "$task	$method	$tseed	$ps	$h	$((5 * h))	$((10 * h))	$cem	$episodes	${sr:-FAIL}	$((SECONDS - t0))	$sha"
  echo "fig$FIG $task $method ts$tseed h$h cem$cem s$ps -> ${sr:-FAIL} ($((SECONDS - t0)) s)" ) &
}

case $FIG in
  4) for cem in $cems; do for ps in $seeds; do evaluate "$OUT/fig4_budget_raw.tsv" 5 "$cem" "$ps"; done; done ;;
  3) for h in $HS; do for ps in $seeds; do evaluate "$OUT/fig3_horizon_raw.tsv" "$h" 30 "$ps"; done; done ;;
  5)
    X=(); [[ "${SMOKE:-0}" == 1 ]] && X=(--override eval.num_eval=$episodes)
    for ps in $seeds; do
      tag=${method}_ts${tseed}; npz=$OUT/goal_tolerance/${task}_${tag}_s${ps}.npz
      [[ -f $npz ]] && continue
      slot
      ( log=$OUT/logs/fig5_${task}_${tag}_s${ps}.log; t0=$SECONDS
      "$PY" scripts/goal_tolerance_eval.py --task "$task" --policy "$ck" --seed "$ps" --tag "$tag" \
        --out "$OUT/goal_tolerance" "${X[@]}" > "$log" 2>&1
      nat=$(grep -ao "native any-time success [0-9.]*" "$log" | tail -1 | awk '{print $4}')
      [[ -f $npz ]] || nat=FAIL
      append "$OUT/fig5_runs.tsv" "$task	$method	$tseed	$ps	5	25	50	30	$episodes	${nat:-FAIL}	$((SECONDS - t0))	$sha"
      echo "fig5 $task $method ts$tseed s$ps -> native T0 ${nat:-FAIL}% ($((SECONDS - t0)) s)" ) &
    done ;;
  *) echo "FIG must be 3, 4 or 5" >&2; exit 2 ;;
esac
wait
echo "==> done $(date -Is)"
