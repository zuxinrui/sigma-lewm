# Figures 3, 4 and 5 at 3 training seeds x 5 planner seeds (Snellius, from 2026-09-27)

The paper's Figures 3–5 still come from single-seed local runs, and they cover only four arms. Redo
all three with the Table 1 checkpoints and protocol: every arm, 3 training seeds x 5 planner seeds.

## Arms and checkpoints

There are six arms: LeWM, Sub-JEPA, RC-aux, SMWM, AC-MTM and Σ-LeWM. SMWM and AC-MTM are new to
these figures.

Use exactly the files behind Table 1: `docs/results/main_table_checkpoints.tsv`, i.e.
`checkpoints/<task>/<method>/seed{3072,1,2}/weights.pt` in the main-table bundle.
- **RC-aux:** use `seed3072` (the authors' train.py on the official LeWM weights), not
  `seed3072_released`.
- **Σ-LeWM TwoRoom and Reacher, seed 3072:** these were trained off-cluster.
  - They come from le-wm in `sigma_seed3072_tworoom_reacher.tar` (sha256 `8fa39050…2827d598`),
    which holds `t3_ours_{tworoom,reacher}_plan/{config.json,weights.pt}`.
  - The config uses `stable_worldmodel.wm.lewm.LeWM`.
  - Table 1 uses these two files for seed 3072.

## Tasks

Push-T, TwoRoom, Reacher and OGBench-Cube, the four panels of each figure. Scene is not in these
figures.

## Shared protocol

The shared protocol is the one of Table 1 unless stated:
- CEM with 300 samples and top-30 elites;
- 50 episodes per evaluation;
- planner seeds 0–4;
- action block 5;
- one real frame.

## Figure 3: planning horizon

h ∈ {3, 10, 15}, with CEM30. For each h:
- goal offset = 5h env steps;
- eval budget = 10h;
- `receding_horizon = h` (full commit).

The h = 5 cell is Table 1 as it stands (goal +25, budget 50), so do not rerun it.

- Evaluations: 4 tasks × 6 arms × 3 horizons × 3 training seeds × 5 planner seeds = **1,080**.
- Cost from our 3090 timings (median wall seconds per evaluation):

  | | h = 3 | h = 10 | h = 15 |
  |---|---|---|---|
  | Push-T | 59 | 180 | 272 |
  | Cube | 121 | 240 | 292 |
  | Reacher | 74 | 186 | 256 |
  | TwoRoom | 46 | 160 | 248 |

  That is about 53 GPU-hours on a 3090, or roughly 35 on an H100. h = 15 is half of that.

## Figure 4: search budget

h = 5, goal +25 and budget 50, as Table 1. CEM iterations ∈ {8, 4, 2}; CEM30 is Table 1.

- Evaluations: 4 × 6 × 3 × 3 × 5 = **1,080**.
- Cost: about 21 GPU-hours on a 3090.
  - At CEM 8/4/2, Push-T takes 60/54/50 s per evaluation, Cube 128/120/120 s, Reacher 68/62/57 s
    and TwoRoom 42/35/34 s.

## Figure 5: goal tolerance

This figure needs every step's goal error, not only success.

- **Run:** le-wm `scripts/goal_tolerance_eval.py`, once per checkpoint and planner seed:
  ```
  python scripts/goal_tolerance_eval.py --task <task> --policy <checkpoint>/weights.pt \
      --seed <planner seed 0-4> --tag <method>_ts<training seed>
  ```
  It is eval.py with the same episodes and CEM seed, termination disabled and the per-step errors
  saved. It uses h = 5, CEM30, goal +25, budget 50.
- **Output:** `outputs/goal_tolerance/<task>_<method>_ts<seed>_s<planner seed>.npz`.
- **Re-scoring:** `scripts/goal_tolerance_rescore.py` applies the thresholds offline.
  - The levels are Push-T 20/15/10/7 (px and deg), Cube 0.04/0.02/0.01/0.005 m, TwoRoom 16/8/4/2 px
    and Reacher 0.05/0.03/0.02/0.01 rad.
  - Its `--arms` and tag parsing must accept the six arms and the training-seed suffix.
- **Evaluations:** 4 × 6 × 15 = **360**, about 10 GPU-hours on a 3090.
- **Cross-check:** the T0 level is the native criterion, so T0 must reproduce Table 1 within
  planner noise. A mismatch means the episodes differ.

## Order

1. **Figure 4 first:** it is the cheapest, and CEM30 already exists for every cell.
2. **Figure 5 next.**
3. **Figure 3 last:** do h = 3 and h = 10 before h = 15.

If time is short, do not drop any training seed; drop h = 15 instead.

## What to push

- `docs/results/fig3_horizon_raw.tsv` and `docs/results/fig4_budget_raw.tsv`, one row per evaluation:
  `task, method, training_seed, planner_seed, horizon, goal_offset, eval_budget, cem_iters, num_eval,
  success_rate, wall_seconds, checkpoint_sha256`.
- For Figure 5:
  - a tar of `outputs/goal_tolerance/*.npz` (le-wm pulls it);
  - the rescored table `docs/results/fig5_tolerance.tsv`, with one row per task × arm × level
    holding the pooled success % and a Wilson interval.
- A short section in `docs/RESULTS.md` per figure, as for the main table. Note there anything that
  deviates from the protocol above.

The le-wm figure scripts (`build_fig3_long_horizon.py`, `build_fig5_frontier.py`,
`build_fig_tolerance.py`) are switched to these files locally once they land.
