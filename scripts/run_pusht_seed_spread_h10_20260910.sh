#!/bin/bash
# Seed spread on the Sec. 4.2 h=10/CEM8 cell (July protocol: h=10 full commit, goal+25, budget 100, eval seeds
# 1617884563 1641937927 522729137 x 50), three Σ-LeWM training seeds.
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
GPU="${1:-1}"; OUT="${OUT:-logs/pusht_seed_spread_20260910}"; mkdir -p "$OUT"
for tag in s1 s2 s3072; do for seed in 1617884563 1641937927 522729137; do
  ck=$HOME/.stable-wm/checkpoints/t3_ours_pusht_${tag}_plan/weights.pt
  log="$OUT/pusht_${tag}_cem_s${seed}_cem8_h10_off25_b100_n50.log"
  if grep -q "success_rate" "$log" 2>/dev/null; then echo "[SKIP] $log"; continue; fi
  echo "[START] $(date +%FT%T) $tag s$seed cem8 h10 off25 b100"
  CUDA_VISIBLE_DEVICES=$GPU python -m slewm.evaluation.planning --config-name=pusht policy="$ck" +planner=cem \
    plan_config.horizon=10 plan_config.receding_horizon=10 eval.goal_offset_steps=25 eval.eval_budget=100 \
    eval.num_eval=50 solver.n_steps=8 solver.num_samples=300 seed=$seed +idm_pool_size=2048 +idm_bank_seed=9001 >"$log" 2>&1
  echo "[END] rc=$? $(grep -oE "'success_rate': [0-9.]+" "$log" | tail -1)"
done; done
echo "[ALL_DONE] $(date +%FT%T)"
