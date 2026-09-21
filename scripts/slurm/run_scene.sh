#!/usr/bin/env bash
# Submits the OGBench-Scene comparison: LeWM, AC-MTM, SMWM (one GPU each) and Sigma-LeWM
# (two GPUs), three training seeds each, training plus evaluation in one job.
#
#   ./scripts/slurm/run_scene.sh --smoke    # 2 epochs on 3 batches + short eval, proves the setup
#   ./scripts/slurm/run_scene.sh            # the real 12 runs
#   ./scripts/slurm/run_scene.sh --chain    # smoke first; the real runs start only if it passes
#
# Needs $STABLEWM_HOME/datasets/visual_scene_play.h5: run `python scripts/prep_ogb_scene.py`
# once on a login node (it downloads from the OGBench server, compute nodes are offline).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
JOB=scripts/slurm/scene.slurm

MODE=real
case "${1:-}" in
  --smoke) MODE=smoke ;;
  --chain) MODE=chain ;;
  "")      ;;
  *) echo "usage: $0 [--smoke|--chain]" >&2; exit 2 ;;
esac

PY=${PY:-$REPO/.venv/bin/python}
STABLEWM_HOME=${STABLEWM_HOME:-/scratch-shared/$USER/stable-wm}

fail() { echo "  FAIL  $*" >&2; bad=1; }
bad=0
echo "Preflight"
[[ -x $PY ]] && echo "  ok    interpreter $PY" || fail "no interpreter at PY=$PY"
[[ -x $PY ]] && { "$PY" -c 'import slewm, ogbench' 2>/dev/null \
  && echo "  ok    slewm and ogbench importable" || fail "run 'pip install -r requirements.txt && pip install -e .' with $PY"; }
f=$STABLEWM_HOME/datasets/visual_scene_play.h5
[[ -s $f ]] && echo "  ok    $f" || fail "missing $f: run 'python scripts/prep_ogb_scene.py' on a login node"
command -v sbatch >/dev/null && echo "  ok    sbatch" || fail "sbatch not on PATH: submit from a login node"
(( bad )) && { echo "Preflight failed; nothing submitted." >&2; exit 1; }

COMMON="ALL,PY=$PY,STABLEWM_HOME=$STABLEWM_HOME,REPO=$REPO"
submit() {  # submit <array> <arms> <gpus> <extra-export> [dependency]
  local dep=(); [[ -n ${5:-} ]] && dep=(--dependency=afterok:"$5")
  sbatch --parsable "${dep[@]}" --array="$1" --gpus="$3" --export="$COMMON,ARMS=$2${4:+,$4}" "$JOB"
}

echo
case $MODE in
  smoke)
    a=$(submit 0-8 single 1 SMOKE=1); b=$(submit 0-2 ours 2 SMOKE=1)
    echo "smoke: single=$a  ours=$b" ;;
  real)
    a=$(submit 0-8 single 1 ""); b=$(submit 0-2 ours 2 "")
    echo "real: single=$a  ours=$b" ;;
  chain)
    sa=$(submit 0-8 single 1 SMOKE=1); sb=$(submit 0-2 ours 2 SMOKE=1)
    ra=$(submit 0-8 single 1 "" "$sa"); rb=$(submit 0-2 ours 2 "" "$sb")
    echo "smoke: single=$sa  ours=$sb"
    echo "real : single=$ra (after $sa)  ours=$rb (after $sb)" ;;
esac
echo
echo "watch:    squeue -u \$USER"
echo "results:  $STABLEWM_HOME/results/scene/results.tsv"
echo "cancel:   scancel -u \$USER --name=slewm-scene"
