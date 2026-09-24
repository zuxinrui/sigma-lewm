# Training the baselines

Every baseline in the paper can be trained from this repository. Four of them are trained
here rather than replayed from a released checkpoint, because their authors publish
checkpoints only for the four standard benchmarks:

| Arm | Objective | Where it is used |
|---|---|---|
| `smwm` | forward MSE + inverse-action **regression**, no SIGReg | all six tasks |
| `acmtm` | forward MSE + in-batch action **discrimination**, no SIGReg | all six tasks |
| `subjepa` | LeWM's loss with SIGReg computed inside 32 random orthogonal subspaces | Gear, Cube-Stacking |
| `rcaux` | LeWM's loss plus a training-only reachability head (the `λ=0` arm) | Gear, Cube-Stacking |

`subjepa` and `rcaux` call the authors' own implementations. This repository does not
redistribute them: clone the two upstream repositories at the commits pinned in
[`external_repos.lock.json`](external_repos.lock.json) and point `SLEWM_EXTERNAL` at the
directory that holds them.

```bash
mkdir -p ~/slewm-external && cd ~/slewm-external
git clone https://github.com/intcomp/sub-jepa.git   && git -C sub-jepa checkout ef945ed
git clone https://github.com/Guang000/RC-aux.git    && git -C RC-aux  checkout ecb4496
mv RC-aux rc-aux
export SLEWM_EXTERNAL=~/slewm-external
```

`smwm` and `acmtm` need no external checkout.

## Run

```bash
slewm train --task pusht --arm smwm  --epochs 15            # standard benchmark
slewm train --task gear  --arm rcaux --epochs 30 --seed 1   # manipulation task
```

Add `--execute` to launch instead of printing the command. `--seed` takes 3072, 1 or 2;
those are the three training seeds the paper reports per arm. `--override KEY=VALUE` passes
an extra Hydra override through and may be repeated.

## On a Slurm cluster

`scripts/slurm/baselines.slurm` trains one arm per array task. It bakes in no account and
no cluster: the paths come from `REPO`, `PY`, `STABLEWM_HOME`, `SLEWM_DATA_ROOT` and
`SLEWM_EXTERNAL`, each overridable with `--export`.

`scripts/slurm/run_all.sh` submits all 36 in one go. It checks the interpreter, the
installation, both dataset roots and the upstream checkout first, and submits nothing if
any of them is missing.

```bash
export STABLEWM_HOME=/scratch-shared/$USER/stable-wm
export SLEWM_DATA_ROOT=$STABLEWM_HOME/isaac
export SLEWM_EXTERNAL=~/slewm-external
export PY=/path/to/python            # optional; defaults to ./.venv/bin/python

./scripts/slurm/run_all.sh --smoke   # two epochs on three batches, about ten minutes
./scripts/slurm/run_all.sh           # the real runs
./scripts/slurm/run_all.sh --chain   # smoke first, real runs held until it passes
```

To submit one group by hand instead, the array index maps to `(task, arm, seed)` in that
order, three seeds per arm:

```bash
sbatch --array=0-23 --export=ALL,GROUP=standard scripts/slurm/baselines.slurm
sbatch --array=0-11 --export=ALL,GROUP=isaac    scripts/slurm/baselines.slurm
```

## Inverse-dynamics coefficients

SMWM publishes a per-environment inverse coefficient, and the configs use the published
value for each task rather than a single global one:

| TwoRoom | Reacher | Push-T | OGBench-Cube | Gear / Cube-Stacking |
|---|---|---|---|---|
| 0.1 | 5 | 30 | 1 | 1 |

AC-MTM publishes no per-environment value; every AC-MTM arm here uses the same
`inverse_weight: 0.30` with `action_nce_temperature: 0.10`.

## One constraint

These arms set the SIGReg weight to zero, which leaves part of the network out of the
loss. DistributedDataParallel rejects that, so the configs train on a single device; the
CLI passes `trainer.devices=1` for you.

## Results

Success rates, protocol and the fixes that invalidated earlier runs: [RESULTS.md](RESULTS.md).
