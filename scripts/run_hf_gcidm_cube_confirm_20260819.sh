#!/usr/bin/env bash
# Paired Cube confirmation: stock CEM versus HF-latent GCIDM-seeded CEM.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

STOCK_GPU="${1:-0}"
GCIDM_GPU="${2:-1}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-1617884563 1641937927}"
NUM_EVAL="${NUM_EVAL:-50}"
CEM_STEPS="${CEM_STEPS:-30}"
OUT_DIR="${OUT_DIR:-logs/hf_gcidm_cube_confirm_20260819}"
mkdir -p "$OUT_DIR"

HF_CKPT=quentinll/lewm-cube
GCIDM_CKPT=fm_idm/gcidm_cube.pt

run_stock_lane() {
    for eval_seed in $CONFIRM_SEEDS; do
        local log_path="$OUT_DIR/cube_hf_stock_s${eval_seed}_n${NUM_EVAL}_cem${CEM_STEPS}.log"
        echo "[START stock] $(date --iso-8601=seconds) seed=$eval_seed"
        CUDA_VISIBLE_DEVICES="$STOCK_GPU" python eval.py \
            --config-name=cube policy="$HF_CKPT" \
            plan_config.horizon=5 plan_config.receding_horizon=5 \
            eval.eval_budget=50 eval.num_eval="$NUM_EVAL" \
            solver.n_steps="$CEM_STEPS" solver.num_samples=300 \
            seed="$eval_seed" >"$log_path" 2>&1
        grep -E "success_rate|CEM solve time" "$log_path" | tail -4
        echo "[DONE stock] $(date --iso-8601=seconds) seed=$eval_seed"
    done
}

run_gcidm_lane() {
    for eval_seed in $CONFIRM_SEEDS; do
        local log_path="$OUT_DIR/cube_hf_gcidm_s${eval_seed}_n${NUM_EVAL}_cem${CEM_STEPS}.log"
        echo "[START gcidm] $(date --iso-8601=seconds) seed=$eval_seed"
        CUDA_VISIBLE_DEVICES="$GCIDM_GPU" python eval_flowactor.py \
            --config-name=cube policy="$HF_CKPT" \
            +flow.idm="$GCIDM_CKPT" +flow.cost_weight=none \
            +flow.h_fixed=5 +flow.h_mode=remaining \
            plan_config.horizon=5 plan_config.receding_horizon=5 \
            eval.eval_budget=50 eval.num_eval="$NUM_EVAL" \
            solver.n_steps="$CEM_STEPS" solver.num_samples=300 \
            seed="$eval_seed" >"$log_path" 2>&1
        grep -E "success_rate|CEM solve time" "$log_path" | tail -4
        echo "[DONE gcidm] $(date --iso-8601=seconds) seed=$eval_seed"
    done
}

run_stock_lane &
stock_pid=$!
run_gcidm_lane &
gcidm_pid=$!

status=0
wait "$stock_pid" || status=$?
wait "$gcidm_pid" || status=$?
if (( status != 0 )); then
    echo "[FAILED] $(date --iso-8601=seconds) status=$status" >&2
    exit "$status"
fi

python scripts/audit_hf_gcidm_cube_confirm.py --log-dir "$OUT_DIR"
echo "[ALL_DONE] $(date --iso-8601=seconds) out=$OUT_DIR"
