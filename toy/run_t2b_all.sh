#!/bin/bash
# T2b bottleneck: Z_DIM in {4,3}, 6 arms in parallel per dim, 2 seeds sequential, 2 threads each. CPU only.
export CUDA_VISIBLE_DEVICES="" TOY_THREADS=2
cd "$(dirname "$0")"
PY="${PY:-python}"
mkdir -p runs_d4/logs runs_d3/logs
for zd in 4 3; do
  for arm in "F+SIG" "F+B+SIG" "F+I+SIG" "F+B+I+SIG" "F+Bmse+Imse+SIG" "F+Ishuf+SIG"; do
    ( for s in 0 1; do T2_ZDIM=$zd $PY t2_toy_world.py --arm "$arm" --seed $s --steps 4000 2>&1 | grep -v Warning; done ) > "runs_d${zd}/logs/${arm}.log" 2>&1 &
  done
done
wait
echo ALL_DONE
