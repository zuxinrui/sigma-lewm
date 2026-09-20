#!/usr/bin/env bash
# RC-DB-DCL on the current Σ-LeWM champion checkpoints: h5/CEM30 main grid, CEM axis, horizon axis.
# Checkpoints = the paper's t3_ours_*_plan sources (verified tensor-identical, 2026-09-09).
# Frozen RC-DB-DCL flag set copied verbatim from scripts/run_rc_db_dcl_confirm_20260720.sh.
# Resumable: a cell is skipped if its log already contains success_rate.
# Usage: run_rc_db_dcl_axes.sh <gpu> <chain: main|cemaxis|haxis|smoke>
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}" MUJOCO_GL=egl SDL_VIDEODRIVER=dummy TOKENIZERS_PARALLELISM=false
GPU="$1"; CHAIN="$2"
OUT="${OUT:-logs/rcdb_axes_20260909}"; mkdir -p "$OUT"
SEEDS="${SEEDS:-0 1}"; NUM_EVAL="${NUM_EVAL:-50}"
declare -A CK=(
  [pusht]=$HOME/.stable-wm/checkpoints/ct_dcl_pusht/weights_epoch_6.pt
  [cube]=$HOME/.stable-wm/checkpoints/tworoom_cube_best_20260716/cube_C_T_dcl_idm_sigreg_s3072/weights_epoch_8.pt
  [tworoom]=$HOME/.stable-wm/checkpoints/tworoom_cube_best_20260716/tworoom_C_T_dcl_idm_sigreg_s3072/weights_epoch_10.pt
  [reacher]=$HOME/.stable-wm/checkpoints/lewm_reacher_v15_s3072/weights_epoch_10.pt
)
RC_FLAGS=(+dcl_beams=4 +dcl_branch_topk=8 +dcl_proposal_start=0 +dcl_proposal_every=1 +dcl_target_source=best
  +dcl_proposal_trust_rms=0 +dcl_min_goal_gain=0.10 +dcl_min_support_gain=0.0 +dcl_min_abs_support=0.0
  +dcl_source_chains=1 +dcl_source_pool=30 +dcl_retrieval_preselect=0 +dcl_retrieval_diversity=0.0 +dcl_refine_steps=0
  +dcl_include_diverse_challenger=true +dcl_diverse_pool_size=8192 +dcl_diverse_source_chains=2 +dcl_diverse_source_pool=30
  +dcl_diverse_retrieval_preselect=64 +dcl_diverse_retrieval_diversity=0.25 +dcl_diverse_min_goal_gain=0.02 +dcl_robust_samples=0 +dcl_trace=true)
# cell <task> <planner cem|safe_dcl_portfolio> <seed> <cem_steps> <horizon> <goal_offset> <budget>
cell() {
  local task=$1 planner=$2 seed=$3 cem=$4 h=$5 off=$6 bud=$7
  local log="$OUT/${task}_${planner}_s${seed}_cem${cem}_h${h}_off${off}_b${bud}_n${NUM_EVAL}.log"
  if grep -q "success_rate" "$log" 2>/dev/null; then echo "[SKIP] $log"; return; fi
  local extra=(+idm_pool_size=2048 +idm_bank_seed=9001); [[ $planner == safe_dcl_portfolio ]] && extra+=("${RC_FLAGS[@]}")
  echo "[START] $(date +%FT%T) $task $planner s$seed cem$cem h$h off$off b$bud"
  CUDA_VISIBLE_DEVICES=$GPU python eval_ext.py --config-name="$task" policy="${CK[$task]}" +planner="$planner" \
    plan_config.horizon=$h plan_config.receding_horizon=$h eval.goal_offset_steps=$off eval.eval_budget=$bud \
    eval.num_eval=$NUM_EVAL solver.n_steps=$cem solver.num_samples=300 seed=$seed "${extra[@]}" >"$log" 2>&1
  echo "[END] rc=$? $(grep -oE "'success_rate': [0-9.]+" "$log" | tail -1) $(date +%FT%T)"
}
case $CHAIN in
  smoke)   NUM_EVAL=5; cell pusht safe_dcl_portfolio 0 30 5 25 50 ;;
  main)    for s in $SEEDS; do for t in pusht tworoom reacher cube; do cell $t cem $s 30 5 25 50; cell $t safe_dcl_portfolio $s 30 5 25 50; done; done ;;
  cemaxis) for s in $SEEDS; do for t in pusht tworoom reacher cube; do for c in 2 4 8; do cell $t cem $s $c 5 25 50; cell $t safe_dcl_portfolio $s $c 5 25 50; done; done; done ;;
  haxis)   for s in $SEEDS; do for t in pusht tworoom reacher cube; do for h in 3 10 15; do cell $t cem $s 30 $h $((5*h)) $((10*h)); cell $t safe_dcl_portfolio $s 30 $h $((5*h)) $((10*h)); done; done; done ;;
esac
echo "[CHAIN_DONE] $CHAIN $(date +%FT%T)"
