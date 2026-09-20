#!/bin/bash
# Compute-matched control for IDM ranking (App. H): the same TrIM checkpoints, stock CEM restarted
# N_RESTARTS times with cost-based selection (+planner=restart), on the identical episodes
# (seeds 0-1 x 50, h=5, goal+25, budget 50) as Table 9. IDM ranking costs 5.3-5.8x stock wall-clock
# per iteration, so six restarts is a conservative (>= 5.8x) match.
set -uo pipefail
cd "$(dirname "$0")/.."; source .venv/bin/activate
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
GPU="${1:-0}"; OUT="${OUT:-logs/restart_control_20260910}"; mkdir -p "$OUT"
N_RESTARTS="${N_RESTARTS:-6}"
declare -A CK=(
  [pusht]=$HOME/.stable-wm/checkpoints/ct_dcl_pusht/weights_epoch_6.pt
  [cube]=$HOME/.stable-wm/checkpoints/tworoom_cube_best_20260716/cube_C_T_dcl_idm_sigreg_s3072/weights_epoch_8.pt
  [tworoom]=$HOME/.stable-wm/checkpoints/tworoom_cube_best_20260716/tworoom_C_T_dcl_idm_sigreg_s3072/weights_epoch_10.pt
  [reacher]=$HOME/.stable-wm/checkpoints/lewm_reacher_v15_s3072/weights_epoch_10.pt
)
cell() {  # task seed cem
  local task=$1 seed=$2 cem=$3
  local log="$OUT/${task}_restart${N_RESTARTS}_s${seed}_cem${cem}_h5_off25_b50_n50.log"
  if grep -q "success_rate" "$log" 2>/dev/null; then echo "[SKIP] $log"; return; fi
  echo "[START] $(date +%FT%T) $task s$seed cem$cem"
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU python eval_ext.py --config-name="$task" policy="${CK[$task]}" +planner=restart \
    plan_config.horizon=5 plan_config.receding_horizon=5 eval.goal_offset_steps=25 eval.eval_budget=50 \
    eval.num_eval=50 solver.n_steps=$cem solver.num_samples=300 seed=$seed +idm_pool_size=2048 +idm_bank_seed=9001 \
    +n_restarts=$N_RESTARTS +restart_selection=cost +restart_trace=true >"$log" 2>&1
  echo "[END] rc=$? wall=$(( $(date +%s)-t0 ))s $(grep -oE "'success_rate': [0-9.]+" "$log" | tail -1)"
}
for s in 0 1; do for t in cube tworoom pusht reacher; do cell $t $s 30; done; cell pusht $s 2; done
echo "[ALL_DONE] $(date +%FT%T)"
