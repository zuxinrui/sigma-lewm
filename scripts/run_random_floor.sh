#!/usr/bin/env bash
# Random-policy floor ("free successes") on the exact evaluation seeds of the T6 horizon axis / T4 CEM ladder.
# policy=random -> swm.policy.RandomPolicy; the episode sampler depends only on seed+goal_offset, so the
# start/goal pairs are identical to the world-model runs. CEM budget is irrelevant for a random policy.
set -u
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
OUT=logs/random_floor_20260910; mkdir -p "$OUT"
cell() {  # task h seed
  local t=$1 h=$2 s=$3 off=$((5*$2)) bud=$((10*$2))
  local log="$OUT/${t}_random_h${h}_s${s}.log"
  grep -q "success_rate" "$log" 2>/dev/null && return 0
  CUDA_VISIBLE_DEVICES="" python -m slewm.evaluation.benchmark --config-name="$t" policy=random seed="$s" eval.num_eval=50 \
    eval.goal_offset_steps="$off" eval.eval_budget="$bud" > "$log" 2>&1
}
export -f cell; export OUT
jobs=()
for t in pusht cube tworoom reacher; do
  for s in 0 1 2 3 4; do jobs+=("$t 3 $s" "$t 5 $s"); done
  for s in 0 1 2; do jobs+=("$t 10 $s" "$t 15 $s"); done
done
# Each cell loads the full dataset + eval.num_eval parallel envs (~7-11 GB RSS).
# At -P 6 that peaked ~55 GB on a 62 GB box and the OOM killer took the desktop down.
# Override with PARALLEL=n if you have headroom to spare.
printf '%s\n' "${jobs[@]}" | xargs -P "${PARALLEL:-3}" -I{} bash -c 'cell {}'
# summary
echo "task,label,seed,horizon,goal_offset,eval_budget,num_eval,success_rate,exit_code" > "$OUT/summary.csv"
for f in "$OUT"/*_random_h*_s*.log; do
  b=$(basename "$f" .log); t=${b%%_random_*}; h=$(echo "$b" | sed -E 's/.*_h([0-9]+)_s.*/\1/'); s=${b##*_s}
  sr=$(grep -aoE "success_rate': [0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+$")
  echo "$t,random,$s,$h,$((5*h)),$((10*h)),50,${sr:-NA},$([ -n "$sr" ] && echo 0 || echo 1)" >> "$OUT/summary.csv"
done
echo DONE
