#!/usr/bin/env bash
# Submits every baseline run the paper needs: 24 standard-benchmark jobs (SMWM, AC-MTM on
# Push-T, Cube, Reacher, TwoRoom) and 12 manipulation jobs (Sub-JEPA, RC-aux on Gear and
# Cube-Stacking), three training seeds each.
#
#   ./scripts/slurm/run_all.sh --smoke    # 2 epochs on 3 batches, ~10 min, proves the setup
#   ./scripts/slurm/run_all.sh            # the real 36 runs
#   ./scripts/slurm/run_all.sh --chain    # smoke first; the real runs start only if it passes
#
# Everything is read from the environment, so no account or cluster is baked in:
#   PY               interpreter                    (default: ./.venv/bin/python)
#   STABLEWM_HOME    standard-benchmark datasets    (default: /scratch-shared/$USER/stable-wm)
#   SLEWM_DATA_ROOT  Gear / Cube-Stacking datasets  (default: $STABLEWM_HOME/isaac)
#   SLEWM_EXTERNAL   upstream Sub-JEPA and RC-aux   (no default; see docs/BASELINES.md)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
JOB=scripts/slurm/baselines.slurm

MODE=real
case "${1:-}" in
  --smoke) MODE=smoke ;;
  --chain) MODE=chain ;;
  "")      ;;
  *) echo "usage: $0 [--smoke|--chain]" >&2; exit 2 ;;
esac

PY=${PY:-$REPO/.venv/bin/python}
STABLEWM_HOME=${STABLEWM_HOME:-/scratch-shared/$USER/stable-wm}
SLEWM_DATA_ROOT=${SLEWM_DATA_ROOT:-$STABLEWM_HOME/isaac}

fail() { echo "  FAIL  $*" >&2; bad=1; }
bad=0
echo "Preflight"
[[ -x $PY ]] && echo "  ok    interpreter $PY ($("$PY" -V 2>&1))" || fail "no interpreter at PY=$PY"
[[ -x $PY ]] && { "$PY" -c 'import slewm' 2>/dev/null \
  && echo "  ok    slewm importable" || fail "slewm not installed: run 'pip install -e .' with $PY"; }
[[ -d $STABLEWM_HOME ]] && echo "  ok    STABLEWM_HOME $STABLEWM_HOME" || fail "STABLEWM_HOME missing: $STABLEWM_HOME"
for f in pusht_expert_train.h5 cube_single_expert.h5 reacher.h5 tworoom.h5; do
  [[ -e $STABLEWM_HOME/datasets/$f ]] || fail "dataset missing: $STABLEWM_HOME/datasets/$f"
done
[[ -d $SLEWM_DATA_ROOT ]] && echo "  ok    SLEWM_DATA_ROOT $SLEWM_DATA_ROOT" || fail "SLEWM_DATA_ROOT missing: $SLEWM_DATA_ROOT"
if [[ -n ${SLEWM_EXTERNAL:-} && -d ${SLEWM_EXTERNAL:-} ]]; then
  for d in sub-jepa/subjepa.py rc-aux/rcaux.py; do
    [[ -e $SLEWM_EXTERNAL/$d ]] || fail "upstream file missing: $SLEWM_EXTERNAL/$d (docs/BASELINES.md)"
  done
  echo "  ok    SLEWM_EXTERNAL $SLEWM_EXTERNAL"
else
  fail "SLEWM_EXTERNAL unset or missing; Sub-JEPA and RC-aux cannot run (docs/BASELINES.md)"
fi
command -v sbatch >/dev/null && echo "  ok    sbatch" || fail "sbatch not on PATH: submit from a login node"
(( bad )) && { echo "Preflight failed; nothing submitted." >&2; exit 1; }

COMMON="ALL,PY=$PY,STABLEWM_HOME=$STABLEWM_HOME,SLEWM_DATA_ROOT=$SLEWM_DATA_ROOT,SLEWM_EXTERNAL=$SLEWM_EXTERNAL,REPO=$REPO"
submit() {  # submit <array> <group> <extra-export> [dependency]
  local dep=(); [[ -n ${4:-} ]] && dep=(--dependency=afterok:"$4")
  sbatch --parsable "${dep[@]}" --array="$1" --export="$COMMON,GROUP=$2${3:+,$3}" "$JOB"
}

echo
case $MODE in
  smoke)
    a=$(submit 0-23 standard SMOKE=1); b=$(submit 0-11 isaac SMOKE=1)
    echo "smoke: standard=$a  isaac=$b" ;;
  real)
    a=$(submit 0-23 standard ""); b=$(submit 0-11 isaac "")
    echo "real: standard=$a  isaac=$b" ;;
  chain)
    sa=$(submit 0-23 standard SMOKE=1); sb=$(submit 0-11 isaac SMOKE=1)
    ra=$(submit 0-23 standard "" "$sa"); rb=$(submit 0-11 isaac "" "$sb")
    echo "smoke: standard=$sa  isaac=$sb"
    echo "real : standard=$ra (after $sa)  isaac=$rb (after $sb)" ;;
esac
echo
echo "watch:   squeue -u \$USER"
echo "cancel:  scancel -u \$USER --name=slewm-baselines"
