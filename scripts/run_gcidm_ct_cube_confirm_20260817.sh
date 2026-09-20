#!/usr/bin/env bash
# Frozen Cube confirmation for the goal-conditioned IDM seeded CEM champion.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

GPU_ID="${1:-0}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-1617884563 1641937927}"
NUM_EVAL="${NUM_EVAL:-50}"
CEM_STEPS="${CEM_STEPS:-30}"
OUT_DIR="${OUT_DIR:-logs/gcidm_ct_cube_confirm_20260817}"
mkdir -p "$OUT_DIR"

CUBE_CKPT="$STABLEWM_HOME"/checkpoints/tworoom_cube_best_20260716/cube_C_T_dcl_idm_sigreg_s3072/weights_epoch_8.pt
GCIDM_CKPT=logs/gcidm_ct_cube_s0.pt

for eval_seed in $CONFIRM_SEEDS; do
    log_path="$OUT_DIR/cube_gcidm_s${eval_seed}_n${NUM_EVAL}_cem${CEM_STEPS}.log"
    echo "[START] $(date --iso-8601=seconds) seed=$eval_seed"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python eval_flowactor.py \
        --config-name=cube policy="$CUBE_CKPT" \
        +flow.idm="$GCIDM_CKPT" +flow.cost_weight=none \
        +flow.h_fixed=5 +flow.h_mode=remaining \
        plan_config.horizon=5 plan_config.receding_horizon=5 \
        eval.eval_budget=50 eval.num_eval="$NUM_EVAL" \
        solver.n_steps="$CEM_STEPS" solver.num_samples=300 \
        seed="$eval_seed" >"$log_path" 2>&1
    grep -E "success_rate|CEM solve time" "$log_path" | tail -4
    echo "[DONE] $(date --iso-8601=seconds) seed=$eval_seed"
done

echo "[ALL_DONE] $(date --iso-8601=seconds) out=$OUT_DIR"
