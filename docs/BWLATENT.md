# bwlatent on TwoRoom, Cube, Scene and Cube-Stacking (from 2026-09-29)

**bwlatent** is one arm of le-wm's 2026-09-20 key-space ablation. It changes one thing relative to Σ-LeWM:

- Σ-LeWM's backward term contrasts `head_f([z_(t+1); e(a_t)])` with `head_g(z_t)`, where both sides have a learned
  projection head.
- bwlatent removes the key-side head (`head_g` = identity) and sizes `head_f` to land in the 192-d latent space,
  so the backward query is aligned with z_t itself.

Everything else is unchanged.

## Code

Branch `baseline-fixes`:

- `slewm/training/multistep.py`: `+backward_key=latent`. The default is `head`, the champion. It is only
  defined for the one-step inverse-state DCL (`fmca_inverse_on=true`, `mi_multistep_on=false`), which is what A7 uses.
- `slewm/models/triadic.py`: `champion_backward_key: latent` on `AlignedChampionJEPA` and
  `AlignedChampionResidualTransitionJEPA`. The default is `head`.
- `slewm/registry/training_recipes.json`: `A7bwl` is A7 plus `+backward_key=latent`.
  - Output name: `repro_A7bwl_s{SEED}_e{EPOCHS}`.
- `config/train/paper_franka_ksbwl{,B,C}.yaml`: Cube-Stacking, seeds 3072 / 1 / 2.
  - This is Figure 6's ours (le-wm `paper_franka_c0_symidm_nosig`: `paper_franka_c0` + SIGReg 0 + symmetric
    DCL-IDM) with `champion_backward_key: latent`.
- `tests/test_bwlatent.py`: default heads unchanged, latent heads (key = z_t), the recipe equals A7 plus the key,
  and the Stack configs differ from c0 only in the key, SIGReg and the symmetric IDM.

## Runs

Three training seeds (3072, 1, 2) per task, like Table 1.

```bash
# TwoRoom, Cube, Scene: 5 epochs (A7 pipeline: 2 GPUs, DDP)
for task in tworoom cube scene; do for seed in 3072 1 2; do
  python -m slewm.cli train --task $task --arm A7bwl --epochs 5 --seed $seed --execute
done; done

# Cube-Stacking: 10 epochs (aligned pipeline; SLEWM_DATA_ROOT as for the other franka runs)
for seed in 3072 1 2; do
  python -m slewm.cli train --task franka --arm ksbwl --epochs 10 --seed $seed --execute
done
```

## Evaluation

**TwoRoom, Cube, Scene:** Table 1 protocol (h5, CEM30, goal +25, budget 50), planner seeds 0–4, at **epoch 5**:

```bash
python -m slewm.cli eval --task $task --checkpoint <run>/weights_epoch_5.pt --seed $s --execute
```

- **Comparator:** Σ-LeWM (A7) evaluated at the **same epoch 5** of the main-table runs (they save every epoch),
  same planner seeds. Table 1 is epoch 10, so it is not the matched comparison.
- Save evaluation outputs outside the source tree, or under the ignored `outputs/` directory.

**Cube-Stacking** is evaluated locally on Isaac (sealed v4eval, projected CEM).

- **What to send:** the three `ksbwl` run directories (epoch-10 weights + config.json).
- **Local comparator:** Figure 6's ours (`c0_symidm_nosig{,B,C}`) at epoch 10.

## Planning export

The training checkpoints carry the auxiliary heads. With `backward_key=latent`, `head_g` is an identity and has no
weights. Strip the heads for planning as for the other A7 runs.
