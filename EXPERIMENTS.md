# Experiment index

Each experiment family and the code that runs it. Results, logs and checkpoints are not
part of this repository; every command starts a new run.

| Experiment | Entry points |
|---|---|
| standard: Four-task h5 CEM30 main table | [eval_ext.py](eval_ext.py) |
| dinowm_pusht: DINO-WM measured CEM1 | [eval.py](eval.py) |
| cem_axis: CEM2/4/8/30, four tasks | [eval_ext.py](eval_ext.py) |
| horizon_axis: h3/5/10/15, goal offset 5h | [eval_ext.py](eval_ext.py) |
| factorial_pusht: A0-A8 PushT h10 offset25, CEM8/30 | [train_multistep.py](train_multistep.py), [run_ablation_9arm_local_pusht.sh](scripts/run_ablation_9arm_local_pusht.sh), [eval_ext.py](eval_ext.py) |
| factorial_cube: Cube near/far common-epoch factorial | [train_multistep.py](train_multistep.py), [eval_ext.py](eval_ext.py) |
| dose: Half/base/double B/I/SIGReg weights, collapse telemetry | [train_multistep.py](train_multistep.py) |
| linear_backward_crossstep: Linear backward and cross-step inverse regression | [train_multistep.py](train_multistep.py) |
| training_seed_spread: PushT three training seeds, h5/h10 | [run_pusht_seed_spread_20260910.sh](scripts/run_pusht_seed_spread_20260910.sh), [train_multistep.py](train_multistep.py) |
| training_budget: A100 timings and parameter counts | [train_multistep.py](train_multistep.py) |
| paired_effects: Eight preregistered contrasts, flow variant, Holm | [train_multistep.py](train_multistep.py) |
| random_floor: Random-policy floors at four horizons | [run_random_floor.sh](scripts/run_random_floor.sh) |
| nonfree: Cube free/nonfree stratification and parking analysis | [compute_nonfree_success.py](scripts/compute_nonfree_success.py) |
| gear_training: gear L0/C0/AC-MTM/SMWM training and loss selection | [train.py](train.py), [select_external_new_task_checkpoint.py](scripts/select_external_new_task_checkpoint.py), [paper_gear_l0.yaml](config/train/paper_gear_l0.yaml), [paper_gear_c0.yaml](config/train/paper_gear_c0.yaml), [paper_gear_acmtm.yaml](config/train/paper_gear_acmtm.yaml), [paper_gear_smwm.yaml](config/train/paper_gear_smwm.yaml) |
| gear_h5: gear sealed test30 x 2 planner seeds | [run_paper_h5_checkpoint.py](scripts/run_paper_h5_checkpoint.py), [run_external_h5_checkpoint.py](scripts/run_external_h5_checkpoint.py) |
| gear_controls: gear proposal-only, zero, random and goal-agnostic controls | [run_paper_t7a_controls.py](scripts/run_paper_t7a_controls.py), [run_paper_t7a_random_baseline.py](scripts/run_paper_t7a_random_baseline.py) |
| franka_training: franka L0/C0/AC-MTM/SMWM training and loss selection | [train.py](train.py), [select_external_new_task_checkpoint.py](scripts/select_external_new_task_checkpoint.py), [paper_franka_l0.yaml](config/train/paper_franka_l0.yaml), [paper_franka_c0.yaml](config/train/paper_franka_c0.yaml), [paper_franka_acmtm.yaml](config/train/paper_franka_acmtm.yaml), [paper_franka_smwm.yaml](config/train/paper_franka_smwm.yaml) |
| franka_h5: franka sealed test30 x 2 planner seeds | [run_paper_h5_checkpoint.py](scripts/run_paper_h5_checkpoint.py), [run_external_h5_checkpoint.py](scripts/run_external_h5_checkpoint.py) |
| franka_controls: franka proposal-only, zero, random and goal-agnostic controls | [run_paper_t7a_controls.py](scripts/run_paper_t7a_controls.py), [run_paper_t7a_random_baseline.py](scripts/run_paper_t7a_random_baseline.py) |
| gear_final_epoch: Gear final-epoch selection reversal | [run_paper_h5_checkpoint.py](scripts/run_paper_h5_checkpoint.py) |
| excluded_constraint_arms: L1/C1 trained but explicitly excluded | [train.py](train.py) |
| toy_pushing_d8: Synthetic pushing 6 objectives, D8, 3 seeds | [t2_toy_world.py](toy/t2_toy_world.py), [t2_analyze.py](toy/t2_analyze.py) |
| toy_pushing_d4: Capacity bottleneck D4, 2 seeds | [t2_toy_world.py](toy/t2_toy_world.py), [t2_analyze.py](toy/t2_analyze.py) |
| toy_static_d4: Static phase and removal of wall, D4 | [t3_alignment.py](toy/t3_alignment.py) |
| toy_static_d3: Static phase tight capacity, D3 | [t3_alignment.py](toy/t3_alignment.py) |
| toy_analytic: Finite-state counterexamples and MI constructions | [t1_exact_mi.py](toy/t1_exact_mi.py), [t4_mi_space.py](toy/t4_mi_space.py) |
| pusht_atlas: pusht six-variable positions/goals/motion/focused atlas | [pusht_physical_atlas.py](scripts/pusht_physical_atlas.py), [rank_physical_probes.py](scripts/rank_physical_probes.py), [build_focused_physical_atlas.py](scripts/build_focused_physical_atlas.py) |
| cube_atlas: cube six-variable positions/goals/motion/focused atlas | [cube_gear_physical_atlas.py](scripts/cube_gear_physical_atlas.py), [rank_physical_probes.py](scripts/rank_physical_probes.py), [build_focused_physical_atlas.py](scripts/build_focused_physical_atlas.py) |
| gear_atlas: gear six-variable positions/goals/motion/focused atlas | [cube_gear_physical_atlas.py](scripts/cube_gear_physical_atlas.py), [rank_physical_probes.py](scripts/rank_physical_probes.py), [build_focused_physical_atlas.py](scripts/build_focused_physical_atlas.py) |
| idm_ranking: Stock vs IDM ranking CEM/horizon, seeds0/1 | [run_rc_db_dcl_axes.sh](scripts/run_rc_db_dcl_axes.sh), [solver_safe_dcl_portfolio.py](trim/planning/solver_safe_dcl_portfolio.py) |
| idm_ranking_replication: IDM ranking extra two seeds | [eval_ext.py](eval_ext.py), [solver_safe_dcl_portfolio.py](trim/planning/solver_safe_dcl_portfolio.py) |
| restarts: Compute-matched six-restart CEM | [run_restart_control_20260910.sh](scripts/run_restart_control_20260910.sh), [multirestart_cem.py](trim/planning/multirestart_cem.py) |
| gcidm: Frozen-latent GC-IDM training, PushT frontier and Cube paired | [gcidm.py](trim/planning/gcidm.py), [train_gcidm_ct_cube_20260817.sh](scripts/train_gcidm_ct_cube_20260817.sh), [run_gcidm_ct_cube_confirm_20260817.sh](scripts/run_gcidm_ct_cube_confirm_20260817.sh) |
