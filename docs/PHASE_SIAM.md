# Phase-SIAM quick look: S1, S2, S3 (from 2026-09-27)

Does a two-frame state make backward prediction, and through it the representation, better? Three
arms share one recipe and differ only in what the encoder sees. One training seed (1) each, on
Gear-Assembly and OGBench-Scene.

## Shared recipe (Janus-OS)

le-wm's `paper_gear_janus_os` (19e4f60), which local runs found best:

- **Architecture:** the released LeWM one: a ViT-tiny encoder, a 6-layer AR transformer predictor
  and `pred_proj`.
- **Backward predictor:** built exactly like the forward one (same transformer config, same
  `pred_proj`), with its own initialisation stream. It runs on the time-reversed window: position
  j holds a state and the action that led into it, and outputs that state's predecessor.
- **Objective:** one-step teacher forcing in both directions, with no rollout, noise or context
  drop.
  - `L_fwd = MSE(P(z_≤t, a_≤t), sg(z_{t+1}))`
  - `L_bwd = MSE(B(z_≥t+1, a_≥t), sg(z_t))`
  - `L = 0.5 L_fwd + 0.5 L_bwd + 1.0 L_DCL-IDM`
  - The stop-gradient is on the target in both directions.
  - The inverse term is the champion's DCL-IDM on the raw action (temperature 0.1).
- **No SIGReg and no sphere normalisation.** Planning uses the released latent MSE cost.

## The three arms

| Arm | Encoder input | Training data | Planning input |
|---|---|---|---|
| **S1** | `o_t` (3 channels) | unchanged | `o_t`, goal `o_g` (Table 1) |
| **S2** `obs_velocity` | `[o_t, o_t − o_{t−1}]` (6 channels) | + the frame one block before each clip | `[o_{t−1}, o_t]`, goal `[o_{g−1}, o_g]` |
| **S3** `frame_pair` | `[o_t, o_{t−1}]` (6 channels) | same as S2 | same as S2 |

- **Initialisation:** in S2 and S3 the patch embedding is widened from 3 to 6 input channels. The
  first half keeps the RGB weights and the second half starts at zero, so at initialisation both
  compute S1 exactly.
- **Information content:** S2 and S3 carry the same information, since the inputs are an invertible
  linear map of each other. They differ only in the input the zero-initialised half learns from.
- **Why these are not trivial:** backward prediction from `z_{t+1} = E(o_{t+1}, o_t)` to
  `z_t = E(o_t, o_{t−1})` needs `o_{t−1}`, which the input does not contain. This is unlike a velocity
  built from latents: with `v_{t+1} = z_{t+1} − z_t`, the state `z_t = z_{t+1} − v_{t+1}` would follow
  by subtraction.

## Where the previous frame comes from

"Previous" means one action block back (5 env steps), which is one frame at the rate the models
are trained and plan at.

- **Gear training:** `AlignedControlHDF5Dataset(previous_frame=true)` adds `pixels_prev`, the row
  `frameskip` before the clip's first state. At an episode's first rows it repeats the first state
  (zero velocity); the persistent-start episodes begin at rest.
- **Scene training:** each window loads one extra leading frame (`num_steps = history + num_preds + 1`).
  It is used only as the predecessor of the first state, and its action is dropped.
- **Planning** (`slewm/planning/two_frame_policy.py`, `TwoFramePolicy`):
  - The current state's previous frame is the rendered frame 5 env steps ago.
  - Before 5 steps have passed, it is the dataset frame 5 rows before the start row. That frame is
    the true predecessor of the state the environment was reset to, or the start frame itself at an
    episode start.
  - The goal's previous frame is the dataset frame 5 rows before the goal row.
  - These frames are read with `dataset.load_chunk`, as `World` reads the start and goal frames.
- **Gear evaluation runs on Isaac off-cluster.** For S2 and S3 its policy server needs the same
  two-frame input, which is not implemented yet.

## Code (branch `baseline-fixes`, 0d341d8)

| File | What |
|---|---|
| `slewm/models/siam.py` | `SiamJEPA`, le-wm's `siam_jepa_model.py` with the JEPA import changed; `siam_input_mode` (`single`, `obs_velocity`, `frame_pair`), two-frame `encode` and `rollout` |
| `slewm/objectives/siam.py` | le-wm's `siam_jepa_losses.py`. The six-state check is relaxed to any window, for the standard tasks |
| `slewm/training/aligned.py` | the siam branch for aligned clips (Gear) and standard windows (Scene); `pixels_prev` preprocessing |
| `slewm/data/aligned_hdf5.py` | `previous_frame` |
| `slewm/evaluation/planning.py`, `slewm/planning/two_frame_policy.py` | two-frame planning |
| `config/train/paper_{gear,scene}_janus_os{B,_v2B,_p2B}.yaml` | S1, S2, S3 at seed 1 (`paper_gear_janus_os*` and `paper_gear_siam_*` copied from le-wm) |
| `scripts/slurm/phase_siam.slurm` | training; Scene is evaluated after training |

**Tests:**
- `tests/test_siam_port.py`: with the same weights and inputs, the port gives le-wm's losses and
  gradients to 1e-6. It also checks the stop-gradient in both directions and any window length.
- `tests/test_two_frame_siam.py`:
  - at initialisation S2 and S3 equal S1, both for encoding and for a rollout from two planning
    frames;
  - `pixels_prev` and a leading frame give the same result;
  - the policy passes the frame from one block back;
  - on the Gear validation file, `pixels_prev` is the frame five rows back, or the first frame at an
    episode start.

## Runs (Snellius, A100, seed 1)

| Task | Budget and checkpoint | Evaluation |
|---|---|---|
| OGBench-Scene | 10 epochs, epoch 10 (the Table 1 budget) | h5 CEM30, 5 planner seeds × 50 on Snellius |
| Gear-Assembly | le-wm's contract (`paper_new_task_control`): 50–100 epochs; early stopping and epoch selection on the validation loss-composite EMA; every epoch saved | Isaac test30 locally. The selected, final and two neighbouring epochs are packaged, because the selector has picked poorly for MSE siam variants before (epoch 1 for `siam_mse_symidm_inv005`); dev30 first |

**References (3 training × 5 planner seeds):**
- Scene: LeWM 57.7 ± 8.0, A0 (multi-step only) 53.6 ± 5.9, Σ-LeWM 80.7 ± 4.2.
- Gear: LeWM and Σ-LeWM (`c0_symidm_nosig`) from Figure 6.
- Gear S1 at seed 3072 is the local `janus_os`.

**Collapse monitor:** with no SIGReg, the thing to watch is `siam_unit_std_ratio`, the per-channel
std of z/|z| times √d: about 1 is healthy and it falls towards 0 on collapse. It is logged on Gear
only, because the standard-window path logs only loss keys.

**Status:**
- S1: job 27267506. S2 and S3: job 27268724.
- The smoke tests passed, and they ran two-frame planning.
- 2026-09-27 21:30: after 9–10 epochs, Gear `siam_unit_std_ratio` is 0.68–0.71 in all three arms.

Results will follow in `docs/RESULTS.md`.
