# Evaluating every task

All six tasks are scored by closed-loop planning with CEM. The four standard tasks run in
this repository's environment; Gear-Assembly and Cube-Stacking run inside Isaac Lab.

## Standard tasks: Push-T, OGBench-Cube, Two-Room, Reacher

Put the datasets under `$STABLEWM_HOME/datasets/` with the names in `config/eval/<task>.yaml`,
and the checkpoints under `$STABLEWM_HOME/checkpoints/` (see the README for the Hugging Face
download). Then, for each task and evaluation seed:

```bash
slewm eval --task pusht --checkpoint sigma-lewm/pusht/weights.pt --horizon 5 --cem 30 --seed 0 --execute
```

`--task` is one of `pusht`, `cube`, `tworoom`, `reacher`. The paper's protocol for all four:

| setting | value |
|---|---|
| planning horizon | 5, all steps executed (`receding_horizon = horizon`) |
| CEM | 30 iterations, 300 samples, top 30 |
| goal | the dataset state 25 steps ahead of the start |
| episode budget | 50 environment steps |
| episodes | 50 per evaluation seed; seeds 0-4 |

The CEM solve is not bit-reproducible on GPU: the same seed can differ by a few episodes
between runs. Report the mean over seeds, never a single run.

## Gear-Assembly and Cube-Stacking (Isaac Lab)

These need the Isaac environment below, the task data, and three variables:

```bash
export SLEWM_DATA_ROOT=/path/to/gear-franka-data
export SLEWM_CHECKPOINT_ROOT=/path/to/gear-franka-checkpoints
export ISAAC_PYTHON=/path/to/isaac-env/bin/python
```

Evaluate a checkpoint on the sealed test split, once per planner seed (42 and 10042):

```bash
# LeWM (L0) or Sigma-LeWM (C0)
python -m scripts.run_paper_h5_checkpoint --task gear --arm C0 \
  --checkpoint $SLEWM_CHECKPOINT_ROOT/<run>/weights_epoch_<N>.pt --epoch <N> \
  --split test --allow-sealed-test --planner-seed-base 42 --output-dir out/gear/c0/primary

# AC-MTM or SMWM
python -m scripts.run_external_h5_checkpoint --task gear --arm SMWM ...   # same arguments
```

`--task franka` selects Cube-Stacking. The case lists live in
`config/eval/pools/paper_new_tasks_h5_pools_v2.json`; `--split dev` evaluates the
development cases used for epoch selection. Pick the epoch on `dev` (or by validation loss),
then touch `test` once.

## Installing Isaac (minimal)

Tested with Isaac Sim 5.1, Isaac Lab 2.3.0 and Python 3.11 on an RTX 3090.

```bash
conda create -n isaac python=3.11 -y && conda activate isaac
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

git clone -b v2.3.0 https://github.com/isaac-sim/IsaacLab.git
(cd IsaacLab && ./isaaclab.sh --install)

# Gear-Assembly comes from NVlabs SRSA (clone with Git LFS)
git lfs install && git clone https://github.com/NVlabs/SRSA.git
(cd SRSA/rl_games_sil && pip install .)
pip install -e SRSA/source/SRSA
(cd SRSA && tar xzf data.tar.gz && tar xzf checkpoints.tar.gz)

export ISAAC_PYTHON=$(which python)
export OMNI_KIT_ACCEPT_EULA=YES      # first launch otherwise stops at the EULA prompt
```

Cube-Stacking is the stock Isaac Lab task `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0`
and needs nothing beyond Isaac Lab. The first run downloads robot and object assets from
NVIDIA's asset server, so it needs internet access.
