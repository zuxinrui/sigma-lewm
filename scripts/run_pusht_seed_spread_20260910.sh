#!/bin/bash
# Training-seed spread for TrIM on Push-T: the three training seeds (s1, s2, s3072) evaluated with the
# stock CEM under the Table 1 protocol (h=5, CEM30, goal+25, budget 50, eval seeds 0-4 x 50) and the
# Fig. 1(d) cell (h=10 full commit, CEM8, goal+50, budget 100, eval seeds 0-2 x 50).
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
GPU="${1:-1}"; OUT="${OUT:-logs/pusht_seed_spread_20260910}"; mkdir -p "$OUT"
cell() {  # tag seed cem h
  local tag=$1 seed=$2 cem=$3 h=$4
  local off=$((5*h)) bud=$((10*h))
  local ck=$HOME/.stable-wm/checkpoints/t3_ours_pusht_${tag}_plan/weights.pt
  local log="$OUT/pusht_${tag}_cem_s${seed}_cem${cem}_h${h}_off${off}_b${bud}_n50.log"
  if grep -q "success_rate" "$log" 2>/dev/null; then echo "[SKIP] $log"; return; fi
  echo "[START] $(date +%FT%T) $tag s$seed cem$cem h$h"
  CUDA_VISIBLE_DEVICES=$GPU python eval_ext.py --config-name=pusht policy="$ck" +planner=cem \
    plan_config.horizon=$h plan_config.receding_horizon=$h eval.goal_offset_steps=$off eval.eval_budget=$bud \
    eval.num_eval=50 solver.n_steps=$cem solver.num_samples=300 seed=$seed +idm_pool_size=2048 +idm_bank_seed=9001 >"$log" 2>&1
  echo "[END] rc=$? $(grep -oE "'success_rate': [0-9.]+" "$log" | tail -1)"
}
for tag in s1 s2 s3072; do for s in 0 1 2 3 4; do cell $tag $s 30 5; done; done
for tag in s1 s2 s3072; do for s in 0 1 2; do cell $tag $s 8 10; done; done
echo "[ALL_DONE] $(date +%FT%T)"
