# Results: baselines, Scene and planning latency

Status of 2026-09-24. Every success rate below is closed-loop planning with the h=5 CEM30
planner (300 candidates, goal offset 25, budget 50). One evaluation is 50 episodes at one
planner seed; each cell pools 5 planner seeds (0-4) and, where the run was trained here,
3 training seeds (3072, 1, 2). Cells are mean ± sd over those evaluations.

## Standard benchmarks

| Task | LeWM | SMWM | AC-MTM | Sub-JEPA | RC-aux | Σ-LeWM |
|---|---|---|---|---|---|---|
| Push-T | 85.2 ± 4.1 | 88.5 ± 5.2 | 89.2 ± 3.6 | 93.2 ± 2.7 | — | pending |
| Cube | 64.8 ± 4.1 | 81.9 ± 4.7 | 77.3 ± 5.7 | 69.2 ± 5.3 | 68.4 ± 5.3 | pending |
| Reacher | 87.6 ± 4.5 | 84.5 ± 5.6 | 84.9 ± 2.3 | 84.0 ± 4.9 | 84.8 ± 5.2 | pending |
| TwoRoom | 90.4 ± 4.3 | 73.1 ± 37.6 † | 92.4 ± 2.7 | 94.0 ± 2.2 | 94.8 ± 2.7 | pending |

The columns do not share one source of variance, and the text should say so:

- **SMWM, AC-MTM**: trained here, 15 epochs, 3 training seeds × 5 planner seeds (n = 15).
  Their sd includes training-seed variance.
- **LeWM, Sub-JEPA, RC-aux**: the authors' released checkpoints, one training run each × 5
  planner seeds (n = 5). Their sd covers the planner only and understates the spread.
  RC-aux publishes no Push-T checkpoint (its fourth pixel-control checkpoint is `wall`).
- † SMWM collapses on TwoRoom in one training seed out of three: seeds 1 and 2 reach 99.6,
  seed 3072 reaches 20.0, which is the success rate of a collapsed encoder. SMWM's published
  TwoRoom inverse coefficient is 0.1, the smallest of the four tasks, and it does not always
  hold the representation up once SIGReg is off. Report the per-seed numbers, not only the mean.

## OGBench-Scene

All six methods trained here for 10 epochs (AC-MTM's Scene budget, no early stopping),
3 training seeds × 5 planner seeds (n = 15 per cell).

| Σ-LeWM (A7) | AC-MTM | SMWM | RC-aux | LeWM | Sub-JEPA |
|---|---|---|---|---|---|
| **80.7 ± 4.0** | 60.9 ± 4.9 | 60.0 ± 3.0 | 58.8 ± 6.7 | 57.7 ± 7.7 | 55.5 ± 7.8 |

SMWM publishes no Scene coefficient; its Scene config uses the OGBench-Cube value (1.0).
A7 trains with DDP on two GPUs and needs more than 40 GB per GPU at batch 64; these runs used
H100s.

## Planning latency (H100)

`slewm.evaluation.latency`, single-environment MPC solve, h=5, 300 candidates; mean ± sd over
15 (task, method) runs: the four standard tasks × LeWM, Sub-JEPA, RC-aux and Σ-LeWM, less
RC-aux on Push-T. One H100 quarter node.

| | CEM30 | CEM8 | CEM4 | CEM2 |
|---|---|---|---|---|
| Nominal budget reduction | ×1 | ×3.75 | ×7.5 | ×15 |
| Solve latency (ms), H100 | 450.9 ± 3.6 | 127.0 ± 1.2 | 68.2 ± 0.7 | 38.6 ± 0.6 |
| Solve latency (ms), RTX 3090 | 692.2 ± 7.9 | 191.0 ± 1.9 | 99.5 ± 0.9 | 53.8 ± 0.7 |
| Replanning rate (Hz), H100 | 2.21 ± 0.02 | 7.82 ± 0.07 | 14.48 ± 0.14 | 25.30 ± 0.32 |
| Measured speed-up, H100 | ×1.00 | ×3.53 ± 0.01 | ×6.54 ± 0.03 | ×11.43 ± 0.08 |
| Measured speed-up, RTX 3090 | ×1.00 | ×3.61 ± 0.02 | ×6.91 ± 0.05 | ×12.69 ± 0.10 |

Linear fit on the H100: 9.2 ms fixed + 14.7 ms per CEM iteration. The H100 is only about
1.5× faster than the 3090: a ViT-tiny scoring 300 candidates is latency-bound, not
compute-bound. The fixed share grows as iterations get cheaper, so the measured speed-ups sit
further below the nominal ratios than on the 3090.

`latency_summary.py` adds observation preprocessing (about 1 ms) to the replanning-rate and
speed-up rows but not to the latency row. At CEM2 that is the difference between
1000 / 38.6 = 25.9 Hz and the reported 25.30 Hz; the caption should say which rows include it.

## Fixes behind these numbers

Four defects made earlier runs of this repository train something other than the named
baseline. Numbers produced before these fixes should not be used.

1. **SMWM and AC-MTM trained no inverse term on the standard benchmarks.** `external_inverse`
   was implemented only in `aligned_h5_forward` (Gear, Cube-Stacking); `lejepa_forward`
   ignored it. With SIGReg at 0 both arms reduced to LeWM without SIGReg, which collapses
   (validation prediction loss ~1e-6, SIGReg statistic frozen at 51.5). The inverse term now
   runs on both paths, and a model with an inverse head refuses to train without it.
2. **RC-aux trained no reachability term on the standard benchmarks**, for the same reason.
   Wired and guarded the same way.
3. **Sub-JEPA's subspace SIGReg saw the time axis instead of the batch axis.** The trainer
   passes LeWM's `(T, B, D)`; the authors' `MultiSubspaceSIGReg` takes `(B, T, D)`. The
   Epps-Pulley test then averaged over four time steps and could not see the embeddings
   shrink: at 1/100 scale it returned 1.6 instead of 51.4. This affected every path,
   **including Gear and Cube-Stacking**; any Sub-JEPA number trained with this repository
   before the fix must be retrained. `tests/test_subjepa_sigreg_axes.py` pins the axis order.
4. **Smoke runs shared the real run directory.** The trainer resumes from any
   `<run>_weights.ckpt` it finds, so an earlier Scene run started its "10 epochs" from the
   smoke run's epoch-2 state. Smoke runs now write under their own `STABLEWM_HOME`.

RC-aux is this repository's implementation of the reachability head and its
time-contrastive loss (`slewm/baselines/auxiliary.py`); Sub-JEPA imports the authors'
`subjepa.py`. Only the latter is an adapter over upstream code.

## A7 without SIGReg (Push-T, Cube)

The counterpart of Gear's `c0_nosig` / `c0_symidm_nosig`: Σ-LeWM (A7) as published ("base"),
with `standalone_sigreg_weight=0` ("nosig"), and additionally with the symmetric DCL inverse
("symidm_nosig"). One training seed (3072), 10 epochs, 5 planner seeds x 50 episodes, h5 CEM30.

| | base | nosig | symidm_nosig |
|---|---|---|---|
| Push-T | **95.2 ± 3.0** | 90.4 ± 2.3 | 84.8 ± 1.6 |
| Cube | 82.4 ± 3.4 | **84.4 ± 5.0** | 83.2 ± 6.0 |

For reference, the Gear arms (3 training seeds, sealed test30): c0 39.4, c0_nosig 72.2 ± 1.9,
c0_symidm_nosig 73.9 ± 5.4. Removing SIGReg helps Gear a lot, costs 5 points on Push-T and is
within noise on Cube. One training seed per cell: read the Cube differences as no effect.

## RC-aux with the authors' objective

`scripts/slurm/rcaux_authors.slurm` runs the authors' `train.py` (rc-aux ecb4496, `rcaux_default`)
as they produced the released checkpoints: a fine-tune of a LeWM checkpoint, 125 batches per
epoch, 4 epochs (TwoRoom 1), cosine to step 500 (125), read from the scheduler and loop state in
the released files. Evaluation uses this repository's planner, which ignores the reachability
head (the lambda=0 arm).

Reproduction of the released seed-3072 checkpoints from the official LeWM weights:

| | Cube | Reacher | TwoRoom |
|---|---|---|---|
| released | 68.4 | 84.8 | 94.8 |
| reproduced here | 68.8 | 86.0 | 92.0 |

New cells: Push-T seed 3072 84.4 ± 3.7 (the authors publish no Push-T checkpoint); Scene,
3 training seeds, 58.0 ± 5.3, which replaces the simplified-RC-aux Scene number above
(58.8 ± 6.7). The simplified head in `RCAuxJEPA` is not the authors' objective and should not
be reported as RC-aux.

## LeWM and Sub-JEPA over three training seeds

Seeds 1 and 2 retrained here with each released checkpoint's recipe (10 epochs; Sub-JEPA: the
authors' config, 16 subspaces of dim 12), next to the released seed-3072 checkpoints. Each cell
pools 3 training seeds x 5 planner seeds (n = 15), 50 episodes each, h5 CEM30; sd is the sample
sd over the 15 evaluations. Per-seed columns are means over the 5 planner seeds.

| Task | Method | 3 seeds | seed 3072 (released) | seed 1 | seed 2 |
|---|---|---|---|---|---|
| Push-T | LeWM | 90.0 ± 5.7 | 85.2 | 91.2 | 93.6 |
| Push-T | Sub-JEPA | 84.0 ± 8.7 | 93.2 | 77.6 | 81.2 |
| Cube | LeWM | 66.5 ± 6.3 | 64.8 | 67.6 | 67.2 |
| Cube | Sub-JEPA | 70.3 ± 6.6 | 69.2 | 69.6 | 72.0 |
| Reacher | LeWM | 86.0 ± 6.3 | 87.6 | 86.0 | 84.4 |
| Reacher | Sub-JEPA | 80.7 ± 5.9 | 84.0 | 80.0 | 78.0 |
| TwoRoom | LeWM | 90.9 ± 4.1 | 90.4 | 93.2 | 89.2 |
| TwoRoom | Sub-JEPA | 93.6 ± 2.8 | 94.0 | 95.6 | 91.2 |

The released Sub-JEPA Push-T checkpoint (93.2) sits 12-16 points above both seeds trained here;
the other cells agree with their released checkpoints within the planner noise. The three-seed
means are the ones to put in the main table; the released-only numbers in the first table
understate the spread.

## Gear v5 checkpoints (for Isaac evaluation)

Longer Gear dataset, one training seed (3072), 10 epochs, under
`$STABLEWM_HOME/checkpoints/`: `gear_v5_l0_s3072_e10` (LeWM) and `gear_v5_c0_bw1inv1_s3072_e10`
(Σ-LeWM champion), `weights_epoch_{1..10}.pt` each. Evaluation runs on an RTX machine.

## RC-aux over three training seeds

Seeds 1 and 2 are the authors' fine-tune (`rcaux_default`, 4 epochs, TwoRoom 1) of the LeWM
seed-1 and seed-2 checkpoints trained here; seed 3072 is the fine-tune of the official LeWM
weights reproduced above (Push-T and Scene: of the seed-3072 LeWM, as there is no released
RC-aux checkpoint). 3 training seeds x 5 planner seeds (n = 15), h5 CEM30, sample sd.

| Task | 3 seeds | seed 3072 | seed 1 | seed 2 | with released 3072 instead |
|---|---|---|---|---|---|
| Push-T | 87.9 ± 5.2 | 84.4 | 89.2 | 90.0 | — |
| Cube | 70.3 ± 5.4 | 68.8 | 73.2 | 68.8 | 70.1 ± 6.1 |
| Reacher | 84.8 ± 5.4 | 86.0 | 84.0 | 84.4 | 84.4 ± 5.0 |
| TwoRoom | 91.6 ± 3.4 | 92.0 | 92.0 | 90.8 | 92.5 ± 3.1 |
| Scene | 58.0 ± 5.5 | 56.8 | 58.8 | 58.4 | — |

Swapping the reproduced seed-3072 run for the released checkpoint moves no cell by more than
one point, so the pipeline and the released recipe agree.

## Main table over three training seeds

Every cell pools training seeds x 5 planner seeds (50 episodes each, h5 CEM30), sample sd.
All cells recomputed from the raw result rows with the sample sd (SMWM and AC-MTM therefore
differ from the first table in the last digit). Scene RC-aux is the authors' objective.
Σ-LeWM is A7 at epoch 10 of a 10-epoch run. On TwoRoom and Reacher its seed-3072 checkpoints
were trained off-cluster, so those two cells pool seeds 1 and 2 only (n = 10).

| Task | LeWM | SMWM | AC-MTM | Sub-JEPA | RC-aux | Σ-LeWM |
|---|---|---|---|---|---|---|
| Push-T | 90.0 ± 5.7 | 88.5 ± 5.4 | 89.2 ± 3.8 | 84.0 ± 8.7 | 87.9 ± 5.2 | **94.5 ± 3.0** |
| Cube | 66.5 ± 6.3 | **81.9 ± 4.9** | 77.3 ± 5.9 | 70.3 ± 6.6 | 70.3 ± 5.4 | 81.6 ± 5.0 |
| Reacher | **86.0 ± 6.3** | 84.5 ± 5.8 | 84.9 ± 2.4 | 80.7 ± 5.9 | 84.8 ± 5.4 | 84.0 ± 4.5 ‡ |
| TwoRoom | 90.9 ± 4.1 | 73.1 ± 38.9 † | 92.4 ± 2.7 | 93.6 ± 2.8 | 91.6 ± 3.4 | **94.0 ± 3.8** ‡ |
| Scene | 57.7 ± 8.0 | 60.0 ± 3.1 | 60.9 ± 5.1 | 55.5 ± 8.1 | 58.0 ± 5.5 | **80.7 ± 4.2** |

Raw rows (one per evaluation, 440): `docs/results/main_table_raw.tsv`; the checkpoint behind
each training seed: `docs/results/main_table_checkpoints.tsv`.

Σ-LeWM per training seed (3072 / 1 / 2): Push-T 95.2 / 93.2 / 95.2, Cube 82.4 / 80.0 / 82.4,
Reacher — / 82.8 / 85.2, TwoRoom — / 94.0 / 94.0. ‡ seeds 1 and 2 only. Σ-LeWM leads on
Push-T and TwoRoom, ties SMWM on Cube (0.3 apart, sd 5) and is within noise of the pack on Reacher.

## Loss-weight ablation (Table 2) over three training seeds

Push-T, the A7 recipe with one weight scaled 0.5x or 2x (backward = FMCA 0.05 / 0.1 / 0.2,
inverse = DCL-IDM 0.025 / 0.05 / 0.1, SIGReg 0.045 / 0.09 / 0.18), 10 epochs, epoch 10.
3 training seeds x 5 planner seeds (n = 15). Paper values in brackets.

fc-h10 (h=10, full commit, goal +25, budget 100: the protocol of the paper's numbers):

| | 0.5x | 1x | 2x |
|---|---|---|---|
| inverse | 85.6 ± 4.7 (82) | 86.3 ± 4.5 (90) | 88.9 ± 4.4 (87) |
| backward | 84.8 ± 4.5 (86) | 86.3 ± 4.5 (90) | 84.4 ± 6.9 (82) |
| SIGReg | 86.8 ± 4.1 (88) | 86.3 ± 4.5 (90) | 80.8 ± 5.7 (76) |

h5 (the main-table protocol):

| | 0.5x | 1x | 2x |
|---|---|---|---|
| inverse | 93.6 ± 4.2 | 94.5 ± 3.0 | 94.3 ± 2.7 |
| backward | 93.6 ± 3.0 | 94.5 ± 3.0 | 94.0 ± 3.4 |
| SIGReg | 91.1 ± 4.7 | 94.5 ± 3.0 | 94.1 ± 4.0 |

With three seeds the recipe is flat to 0.5x-2x on every weight except SIGReg 2x at fc-h10
(-5.5, the paper's -14 direction). The paper's 1x (90) is above all three seeds here (84.0 /
87.2 / 87.6); its 0.5x/2x dips for inverse and backward are within one sd. Inverse 2x reaches
88.9, so the source record's 38 does not reproduce. Under h5 every cell is 91-95.

## Phase-JEPA on Reacher

le-wm's DynamicsJEPA (ported as `slewm.models.dynamics`, zero-initialised velocity pathways and
`second_order_gain`), the LeWM recipe with the model swapped, seed 1, 10 epochs, paired with
LeWM seed 1 (same data, schedule and shared initial weights). 5 planner seeds x 50, h5 CEM30.

| | success | final val pred / SIGReg |
|---|---|---|
| LeWM seed 1 | **86.0 ± 4.0** | 0.024 / 1.7 |
| latent_velocity | 84.0 ± 3.2 | 0.025 / 1.6 |
| second_order | 84.0 ± 3.7 | 0.020 / 1.9 |
| obs_velocity | 66.0 ± 7.1 | 0.001 / 1.7 |

No variant beats LeWM on its own seed. second_order started with a high SIGReg (76.5 after epoch
1, 28 after epoch 2) and settled at LeWM's level by epoch 5; no persistence collapse.
obs_velocity drives the validation prediction loss to 0.001, 20x below LeWM, and loses 20
points: with the frame difference as an input channel the latent becomes easy to predict
without carrying what the planner needs. One training seed; read latent_velocity and
second_order as ties.

## Pending

- **Sub-JEPA and RC-aux on Gear and Cube-Stacking**, 3 training seeds, 20 epochs: training.
  Evaluation needs Isaac Lab on an RTX GPU (the sealed `test30` pools, 30 cases per planner
  seed), so it runs outside the cluster.
