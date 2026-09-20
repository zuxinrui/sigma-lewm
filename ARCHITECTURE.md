# Code map

Start with `trim/objectives/triadic.py`. It composes the actual aligned training
objective, not a separate illustrative implementation.

| Idea | Implementation |
| --- | --- |
| Encode observations, predict latents, roll out and score goals | [world_model.py](trim/models/world_model.py) |
| Action-conditioned transformer | [components.py](trim/models/components.py) |
| Forward: teacher-forced and autoregressive prediction | [forward.py](trim/objectives/forward.py) |
| Backward: discriminate the predecessor from successor + action | `symmetric_dcl_loss` in [contrastive.py](trim/objectives/contrastive.py) |
| Inverse: discriminate the action from adjacent states | `dcl_idm_loss` in [contrastive.py](trim/objectives/contrastive.py) |
| Global-batch inverse and backward losses | [inverse.py](trim/objectives/inverse.py), [distributed_contrastive.py](trim/objectives/distributed_contrastive.py) |
| Compose forward + backward + inverse | [triadic.py](trim/objectives/triadic.py) |
| Prevent latent collapse | [regularization.py](trim/objectives/regularization.py) |
| Residual transition and detached base | [residual.py](trim/models/residual.py) |
| Optional action/persistence task constraints | [task_constraints.py](trim/objectives/task_constraints.py) |
| AC-MTM / SMWM objectives | [inverse.py](trim/baselines/inverse.py) |

## Data flow

```text
observations ── encoder ── latents ── forward predictor ── predicted latents
                              ├── backward head: (next latent, action) → predecessor
                              └── inverse head:  (latent, next latent) → action
```

The arrows for backward/inverse describe positive pairings, not regression
targets. Backward uses symmetric DCL. The default inverse objective uses one-way
InfoNCE with near-duplicate actions masked; it is not symmetric DCL by default.
Neither auxiliary head changes the stock latent-MSE CEM criterion.

Aligned clips have `states[B,T+1,D]`, `actions[B,T,A]`. An action at index `t`
connects state `t` to `t+1`; no terminal action is invented. The aligned champion
recipe fixes `T=5`. The benchmark adapter retains its original stored action
layout and slices the terminal slot explicitly.

## Execution and records

- `trim/training/`: training adapters, sampler, RNG state, stopping and telemetry.
- `trim/training/multistep.py`: benchmark construction and launch.
- `trim/training/multistep_step.py`: historical ablation dispatch; individual
  contrastive primitives are imported from `trim/objectives/`.
- `trim/training/audit_callbacks.py`, `trim/protocols/`: execution/protocol checks,
  deliberately separated from the method.
- `trim/evaluation/`, `trim/planning/`: environment evaluation and planner variants.
- `config/`, `repro/`: training recipes and the experiment index.
- `scripts/`, `toy/`: study orchestration and synthetic experiments, not the starting
  point for understanding the model.

The benchmark ablation adapter still supports historical variants. It is not
replaced with the aligned Gear/Franka recipe: their indexing, weighting and
distributed semantics are not interchangeable.

## Compatibility

Root `train*.py` / `eval*.py` scripts delegate to the package. The small model
aliases at root preserve common old Hydra and pickle paths. Other historical
module aliases are in `compat/`; add that directory to `PYTHONPATH` only when
loading an artifact that explicitly requires an old module name. New code should
import `trim.*`. See `repro/module_migration.json` for exact mappings.

Model parameter names and objective formulas are preserved. Some historical
configuration names (`inverse_dcl_weight`, `head_f`, `head_g`) refer to the
**backward** term; they remain unchanged to preserve recipe/checkpoint contracts.
Global-batch DDP execution and simulator parity require separate validation;
passing CPU tests does not certify full experimental reproducibility.
