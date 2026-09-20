#!/usr/bin/env bash
# Run one cell of the complete 3x3 IDM x backward PushT ablation on both local GPUs.
#
# Each GPU receives 64 samples, while DCL-IDM, DCL-backward and SIGReg gather
# their loss inputs across both ranks.  SyncBatchNorm and a rank-zero broadcast
# for random-N preserve one logical global batch of 128.
#
# Usage:
#   bash scripts/run_ablation_9arm_local_pusht.sh A0
#   SEED=1 EPOCHS=15 bash scripts/run_ablation_9arm_local_pusht.sh A8
#   SMOKE=1 bash scripts/run_ablation_9arm_local_pusht.sh A7

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ARM="${1:?usage: $0 A0|A1|A2|A3|A4|A5|A6|A7|A8}"
SEED="${SEED:-3072}"
EPOCHS="${EPOCHS:-15}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
WORLD_SIZE=2
SAVE_EVERY="${SAVE_EVERY:-5}"
SMOKE="${SMOKE:-0}"

if (( GLOBAL_BATCH % WORLD_SIZE != 0 )); then
  echo "FATAL: GLOBAL_BATCH must be divisible by ${WORLD_SIZE} (got ${GLOBAL_BATCH})" >&2
  exit 2
fi
PER_DEVICE_BATCH=$((GLOBAL_BATCH / WORLD_SIZE))

# Four mutually exclusive loss switches.  A8 completes the full 3x3 grid.
MSE_IDM=0
DCL_IDM=0
MSE_BWD=0
DCL_BWD=0
case "$ARM" in
  A0) ;;
  A1) MSE_IDM=1 ;;
  A2) DCL_IDM=1 ;;
  A3) DCL_IDM=1; MSE_BWD=1 ;;
  A4) MSE_BWD=1 ;;
  A5) DCL_BWD=1 ;;
  A6) MSE_IDM=1; DCL_BWD=1 ;;
  A7) DCL_IDM=1; DCL_BWD=1 ;;
  A8) MSE_IDM=1; MSE_BWD=1 ;;
  *) echo "FATAL: unknown arm $ARM (want A0..A8)" >&2; exit 2 ;;
esac

IDM_WEIGHT=0.0
IDM_STOPGRAD=true
DCL_IDM_WEIGHT=0.0
PREDICTOR_INV_WEIGHT=0.0
FMCA_WEIGHT=0.0
FMCA_INVERSE_ON=false
MI_LOSS=""

if [[ "$MSE_IDM" == 1 ]]; then
  IDM_WEIGHT=0.03
  IDM_STOPGRAD=false
fi
if [[ "$DCL_IDM" == 1 ]]; then
  DCL_IDM_WEIGHT=0.05
fi
if [[ "$MSE_BWD" == 1 ]]; then
  PREDICTOR_INV_WEIGHT=1.0
fi
if [[ "$DCL_BWD" == 1 ]]; then
  FMCA_WEIGHT=0.1
  FMCA_INVERSE_ON=true
  MI_LOSS=dcl
fi

RUN_ID="pusht_abl9_${ARM}_s${SEED}_e${EPOCHS}_b${GLOBAL_BATCH}_ddp2"
if [[ "$SMOKE" == 1 ]]; then
  RUN_ID="${RUN_ID}_smoke"
fi
LOG_DIR="$REPO/logs/ablation9_pusht"
mkdir -p "$LOG_DIR"

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LEWM_STEP_INTERVAL="${LEWM_STEP_INTERVAL:-5000}"
if [[ "$SMOKE" == 1 ]]; then
  export LEWM_STEP_INTERVAL=0
fi

OVERRIDES=(
  "seed=${SEED}"
  "+full_seed_on=true"
  "num_preds=5"
  "data=pusht"
  "output_model_name=${RUN_ID}"
  "subdir=${RUN_ID}"
  "trainer.devices=2"
  "+trainer.strategy=ddp"
  "+trainer.sync_batchnorm=true"
  "trainer.max_epochs=${EPOCHS}"
  "+trainer.accumulate_grad_batches=1"
  "loader.batch_size=${PER_DEVICE_BATCH}"
  # Four workers total are enough to feed the two GPUs without refilling swap.
  "loader.num_workers=2"
  "loader.prefetch_factor=1"
  "save_epoch_interval=${SAVE_EVERY}"
  "+encoder_gradient_checkpointing=false"
  "+distributed_global_losses=true"
  "+sync_random_n=true"
  "+pred_weight=1.0"
  "+tf_weight=1.0"
  "+ar_weight=1.0"
  "+ar_warmup_epochs=2"
  "+random_n_on=true"
  "+unified_random_n_on=false"
  "+diffusion_forcing=true"
  "+df_sigma_max=0.3"
  "+df_tau_ctx=0.1"
  "+barrier_weight=0.0"
  "+standalone_sigreg_weight=0.09"
  "+idm_weight=${IDM_WEIGHT}"
  "+idm_stopgrad=${IDM_STOPGRAD}"
  "+idm_multistep_on=false"
  "+dcl_idm_weight=${DCL_IDM_WEIGHT}"
  "+dcl_idm_tau=0.1"
  "+dcl_idm_dup_cos=0.98"
  "+predictor_inv_weight=${PREDICTOR_INV_WEIGHT}"
  "+ar_inv_weight=0.0"
  "+fmca_weight=${FMCA_WEIGHT}"
  "+fmca_inverse_on=${FMCA_INVERSE_ON}"
  "+fmca_bidir_on=false"
  "+mi_multistep_on=false"
  "+mi_tau=0.1"
  "+joint_reg_on=false"
  "+fmca_idm_weight=0.0"
  "+idm_flow_weight=0.0"
  "+xidm_weight=0.0"
  "+gcidm_weight=0.0"
  "+flowgcidm_weight=0.0"
  "+lindcl_fwd_weight=0.0"
  "+lindcl_inv_weight=0.0"
  "+bidir_simsiam_weight=0.0"
  "+bidir_dynamo_weight=0.0"
  "+bidir_pure_weight=0.0"
  "+dcl_bidir_weight=0.0"
  "+dynamo_lewm_weight=0.0"
  "+bn_dyn_lewm_weight=0.0"
  "+inloop_eval_on=false"
  "wandb.enabled=false"
)
if [[ -n "$MI_LOSS" ]]; then
  OVERRIDES+=("+mi_loss=${MI_LOSS}")
fi
if [[ "$SMOKE" == 1 ]]; then
  OVERRIDES+=(
    "+trainer.limit_train_batches=5"
    "+trainer.limit_val_batches=2"
    "trainer.max_epochs=1"
  )
fi

{
  echo "[$(date --iso-8601=seconds)] START $RUN_ID GPUs=0,1"
  echo "MSE_IDM=$MSE_IDM DCL_IDM=$DCL_IDM MSE_BWD=$MSE_BWD DCL_BWD=$DCL_BWD"
  echo "global_batch=$GLOBAL_BATCH per_device_batch=$PER_DEVICE_BATCH epochs=$EPOCHS seed=$SEED"
  echo "distributed_global_losses=true sync_random_n=true sync_batchnorm=true"
  "${PY:-python}" -m slewm.training.multistep "${OVERRIDES[@]}"
  echo "[$(date --iso-8601=seconds)] DONE $RUN_ID GPUs=0,1"
} 2>&1 | tee "$LOG_DIR/${RUN_ID}.log"
