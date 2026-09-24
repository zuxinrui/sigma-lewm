#!/usr/bin/env bash
# Planning-latency table (the CEM ladder) on one GPU: 15 (task, method) runs, one after the
# other, then the summary. Each run times a single-environment solve at CEM 30/16/8/4/2/1
# (50 timed repetitions after 5 warm-ups) and exits; no episode is played to the end.
#
#   sbatch --export=ALL,STABLEWM_HOME=$HOME/lewm_latency scripts/slurm/latency.slurm
#
# STABLEWM_HOME must hold the unpacked lewm_latency.tar: checkpoints/ plus datasets/ cut down
# to their first episodes (scripts/make_latency_stub_dataset.py). Only the normalisation
# statistics come from the datasets; the planner's images come from the simulator, and
# timing depends on tensor shapes and iteration counts, not on the statistics' values.
#
#SBATCH --job-name=slewm-latency
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
REPO=${REPO:-$SLURM_SUBMIT_DIR}
PY=${PY:-$REPO/.venv/bin/python}
: "${STABLEWM_HOME:?point STABLEWM_HOME at the unpacked lewm_latency directory}"
OUT=${OUT:-$REPO/outputs/latency_$(date +%Y%m%d)_$SLURM_JOB_ID}
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"; mkdir -p "$OUT"

declare -A CKPT=(
  [pusht_lewm]=models--quentinll--lewm-pusht      [pusht_ours]=t3_ours_pusht_plan
  [cube_lewm]=models--quentinll--lewm-cube        [cube_ours]=t3_ours_cube_plan
  [tworoom_lewm]=models--quentinll--lewm-tworooms [tworoom_ours]=t3_ours_tworoom_plan
  [reacher_lewm]=models--quentinll--lewm-reacher  [reacher_ours]=t3_ours_reacher_plan
  [pusht_subjepa]=subjepa_pusht   [cube_subjepa]=subjepa_cube
  [tworoom_subjepa]=subjepa_tworoom [reacher_subjepa]=subjepa_reacher
  [cube_rcaux]=rcaux_cube [tworoom_rcaux]=rcaux_tworoom [reacher_rcaux]=rcaux_reacher)

for key in "${!CKPT[@]}"; do
  task=${key%%_*}; arm=${key#*_}; tag=micro_${task}_${arm}_c30
  [ -s "$OUT/$tag.json" ] && continue
  echo "==> $tag  $(date -Is)"
  BENCH_MODE=micro BENCH_TAG=$tag BENCH_OUT=$OUT/$tag.json \
    "$PY" -m slewm.evaluation.latency --config-name=$task \
      policy=${CKPT[$key]}/weights.pt +planner=cem \
      plan_config.horizon=5 plan_config.receding_horizon=5 \
      eval.num_eval=1 eval.eval_budget=50 world.max_episode_steps=100 \
      solver.n_steps=30 solver.num_samples=300 seed=0 > "$OUT/$tag.log" 2>&1 \
    || echo "    FAILED, see $OUT/$tag.log"
done

"$PY" -m slewm.evaluation.latency_summary "$OUT"
