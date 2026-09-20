#!/bin/bash
# Launch the 18 T2 runs: 6 arms in parallel, 3 seeds sequential, 2 torch threads each. CPU only.
export CUDA_VISIBLE_DEVICES="" TOY_THREADS=2
cd "$(dirname "$0")"
PY="${PY:-python}"
for arm in "F+SIG" "F+B+SIG" "F+I+SIG" "F+B+I+SIG" "F+Bmse+Imse+SIG" "F+Ishuf+SIG"; do
  ( for s in 0 1 2; do $PY t2_toy_world.py --arm "$arm" --seed $s --steps 4000 2>&1 | grep -v Warning; done ) > "runs/logs/${arm}.log" 2>&1 &
done
wait
echo ALL_DONE
