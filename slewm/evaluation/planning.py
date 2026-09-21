"""Extended PushT eval: baseline CEM / DTVH / IPI-MPPI planner selection on FULL checkpoints.

Identical protocol to eval.py (same fixed-seed episodes, callables, success metric); differences:
  1. loads the base JEPA and builds IDMDecoder from either the same state_dict or an optional
     separate raw auxiliary checkpoint (``+aux_state_dict_path=...``);
  2. planner selection via +planner={cem,dtvh,ipi_mppi,cover_cem,portfolio_cem};
  3. optional variance-head attach via +var_head_ckpt=path.

Usage (mirrors eval.py):
  python eval_ext.py --config-name=pusht policy=<ckpt-rel-path> +planner=dtvh \
      plan_config.horizon=5 eval.eval_budget=50 eval.num_eval=50
"""

from slewm.paths import REPO_ROOT
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from slewm.evaluation.benchmark import img_transform, get_episodes_length, get_dataset  # reuse verbatim helpers
from slewm.planning.idm_decode import IDMDecoder, select_kcenter_indices


def load_full_model(policy: str):
    """Load the deployed JEPA state dict; any in-file aux keys remain available
    to ``IDMDecoder`` and are ignored by the base module under ``strict=False``."""
    from hydra.utils import instantiate
    cache = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    ckpt_path = cache / policy
    cfg_path = ckpt_path.parent / "config.json"
    import json
    with open(cfg_path) as f:
        model_cfg = json.load(f)
    model = instantiate(model_cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    base_missing = [k for k in missing if k.split(".")[0] in
                    ("encoder", "predictor", "action_encoder", "projector", "pred_proj")]
    assert not base_missing, f"base JEPA keys missing from ckpt: {base_missing[:5]}"
    print(f"[eval_ext] loaded {policy}: {len(unexpected)} aux keys ignored by JEPA "
          f"(consumed by IDMDecoder)")
    return model, sd


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "config/eval"), config_name="pusht")
def run(cfg: DictConfig):
    assert (cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget)
    planner = str(cfg.get("planner", "cem"))

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    # Frames must come out at the dataset's resolution (Scene is 64x64); the policy transform
    # resizes to img_size. Tasks without world.width/height keep the old 224x224.
    world = swm.World(**cfg.world, image_shape=(int(cfg.world.get("height", 224)), int(cfg.world.get("width", 224))))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        pr = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        pr.fit(col_data)
        process[col] = pr
        if col != "action":
            process[f"goal_{col}"] = pr

    # ---- model + decoder ----
    model, policy_sd = load_full_model(cfg.policy)
    raw_sd = policy_sd
    aux_path = cfg.get("aux_state_dict_path", None)
    if aux_path:
        raw_sd = torch.load(str(aux_path), map_location="cpu")
        if isinstance(raw_sd, dict) and "state_dict" in raw_sd:
            raw_sd = raw_sd["state_dict"]
        # The split load is valid only when the raw checkpoint is the exact
        # unstripped parent of the deployed core checkpoint.
        missing_core = [k for k in policy_sd if k not in raw_sd]
        mismatched_core = [
            k for k in policy_sd
            if k in raw_sd and not torch.equal(policy_sd[k], raw_sd[k])
        ]
        assert not missing_core and not mismatched_core, (
            "aux checkpoint does not match deployed base: "
            f"missing={missing_core[:3]}, mismatched={mismatched_core[:3]}"
        )
        print(
            f"[eval_ext] auxiliary heads loaded from {aux_path}; "
            f"verified {len(policy_sd)} core tensors exactly"
        )
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # Optional planning-cost override (+cost_mode=minovertime|softmin|lastk|discounted).
    # Aligns the CEM objective with the LATCHED eval success metric (reached at ANY step).
    cost_mode = cfg.get("cost_mode", None)
    if cost_mode:
        import types
        from slewm.planning.cost_modes import BUILDERS
        if cost_mode not in BUILDERS:
            raise ValueError(f"unknown cost_mode {cost_mode!r}; expected {list(BUILDERS)}")
        model.criterion = types.MethodType(BUILDERS[cost_mode](cfg), model)
        print(f"[eval_ext] planning cost_mode = {cost_mode} (criterion patched)")

    # candidate-action pool for the contrastive decoder = the empirical action marginal
    # (matches the InfoNCE negative distribution — in-batch actions were dataset actions)
    col_e = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_a = np.asarray(dataset.get_col_data(col_e))
    st_a = np.asarray(dataset.get_col_data("step_idx"))
    ab = int(cfg.plan_config.action_block)
    ia = np.arange(len(ep_a) - ab)
    ok_a = (ep_a[ia + ab] == ep_a[ia]) & (st_a[ia + ab] == st_a[ia] + ab)
    # Hold the finite action/negative bank fixed across evaluation seeds.
    # Evaluation seed should change episodes and CEM noise, not the IDM itself.
    bank_seed = int(cfg.get("idm_bank_seed", 0))
    rng_pool = np.random.default_rng(bank_seed)
    pool_size = min(int(cfg.get("idm_pool_size", 512)), int(ok_a.sum()))
    if pool_size < 2:
        raise RuntimeError(f"IDM action pool needs at least two transitions, got {pool_size}")
    pk = np.sort(rng_pool.choice(ia[ok_a], size=pool_size, replace=False))

    def make_action_blocks(start_rows: np.ndarray) -> np.ndarray:
        """Batch-load exact contiguous action blocks for an IDM bank.

        The previous per-block dataset calls dominated Cube startup (thousands
        of Python/HDF5 round trips).  Flattening all requested row indices once
        preserves the order and values exactly while reducing it to one read.
        """
        rows = (
            start_rows[:, None] + np.arange(ab, dtype=start_rows.dtype)[None]
        ).reshape(-1)
        # `get_col_data` is already memory-cached by the evaluation setup.
        # Index it directly: `get_row_data` also reads pixels/proprio/state and
        # would turn an action-only bank build into a large HDF5 image read.
        raw = np.asarray(dataset.get_col_data("action"))[rows]
        standardized = process["action"].transform(raw)
        return standardized.reshape(len(start_rows), -1).astype(np.float32)

    acts = make_action_blocks(pk)                                      # (pool,A) std blocks
    action_pool = torch.from_numpy(acts)

    decoder = None
    try:
        decoder = IDMDecoder(
            raw_sd,
            A=int(action_pool.shape[-1]),
            tau=float(cfg.get("idm_tau", 0.1)),
            device="cuda",
            pool=action_pool,
        )
        print(f"[eval_ext] IDMDecoder mode = {decoder.mode}, pool = {getattr(decoder, 'pool_src', '-')}")
    except ValueError as e:
        print(f"[eval_ext] no IDM heads ({e}); d_t trigger / IDM proposal unavailable")

    # Optional second inverse-retrieval arm.  It intentionally owns a separate,
    # larger empirical action bank so the original 2,048-bank arm remains a
    # literal reproduction rather than silently changing under the portfolio.
    diverse_decoder = None
    if bool(cfg.get("dcl_include_diverse_challenger", False)):
        diverse_pool_size = min(
            int(cfg.get("dcl_diverse_pool_size", 8192)), int(ok_a.sum())
        )
        diverse_rng = np.random.default_rng(bank_seed)
        diverse_pk = np.sort(
            diverse_rng.choice(ia[ok_a], size=diverse_pool_size, replace=False)
        )
        diverse_acts = make_action_blocks(diverse_pk)
        diverse_decoder = IDMDecoder(
            raw_sd,
            A=int(diverse_acts.shape[-1]),
            tau=float(cfg.get("idm_tau", 0.1)),
            device="cuda",
            pool=torch.from_numpy(diverse_acts),
        )
        print(
            f"[eval_ext] separate diverse IDM bank = {diverse_pool_size} "
            "(base bank unchanged)"
        )

    coreset_decoder = None
    if bool(cfg.get("dcl_include_coreset_challenger", False)):
        superpool_size = min(
            int(cfg.get("dcl_coreset_superpool_size", 16384)), int(ok_a.sum())
        )
        coreset_size = min(
            int(cfg.get("dcl_coreset_pool_size", 2048)), superpool_size
        )
        coreset_rng = np.random.default_rng(bank_seed + 104_729)
        coreset_pk = np.sort(
            coreset_rng.choice(ia[ok_a], size=superpool_size, replace=False)
        )
        coreset_acts = make_action_blocks(coreset_pk)
        coreset_decoder = IDMDecoder(
            raw_sd,
            A=int(coreset_acts.shape[-1]),
            tau=float(cfg.get("idm_tau", 0.1)),
            device="cuda",
            pool=torch.from_numpy(coreset_acts),
        )
        coreset_indices = select_kcenter_indices(
            coreset_decoder.pool_g,
            coreset_size,
            seed=bank_seed + 104_729,
        )
        coreset_decoder.pool = coreset_decoder.pool[coreset_indices]
        coreset_decoder.pool_g = coreset_decoder.pool_g[coreset_indices]
        coreset_decoder.pool_src = "dataset-marginal-dcl-kcenter"
        print(
            f"[eval_ext] DCL k-center action coreset = {coreset_size} "
            f"from fixed superpool {superpool_size}"
        )

    var_head = None
    vh_path = cfg.get("var_head_ckpt", None)
    if vh_path:
        from slewm.planning.varhead import VarianceHead
        var_head = VarianceHead().to("cuda").eval()
        var_head.load_state_dict(torch.load(vh_path, map_location="cuda"))
        print(f"[eval_ext] variance head loaded from {vh_path}")

    config = swm.PlanConfig(**cfg.plan_config)

    if planner == "portfolio_cem":
        from slewm.planning.solver_portfolio_cem import PortfolioCEMSolver

        evaluator_cfg = cfg.get("portfolio_evaluators", "")
        if isinstance(evaluator_cfg, str):
            evaluator_specs = evaluator_cfg.split(",")
        else:
            evaluator_specs = list(evaluator_cfg)
        evaluator_specs = [
            str(spec).strip() for spec in evaluator_specs if str(spec).strip()
        ]
        evaluator_models = []
        evaluator_names = [str(cfg.policy)]
        for spec in evaluator_specs:
            if spec == str(cfg.policy):
                continue
            evaluator, _ = load_full_model(spec)
            evaluator = evaluator.to("cuda").eval()
            evaluator.requires_grad_(False)
            evaluator.interpolate_pos_encoding = True
            evaluator_models.append(evaluator)
            evaluator_names.append(spec)
        print(
            f"[eval_ext] portfolio evaluators ({len(evaluator_names)}): "
            + ", ".join(evaluator_names)
        )
        solver = PortfolioCEMSolver(
            model=model,
            batch_size=int(cfg.solver.batch_size),
            num_samples=int(cfg.solver.num_samples),
            n_steps=int(cfg.solver.n_steps),
            topk=int(cfg.solver.topk),
            var_scale=float(cfg.solver.var_scale),
            device="cuda",
            seed=int(cfg.seed),
            evaluators=evaluator_models,
            evaluator_names=evaluator_names,
            restarts=int(cfg.get("portfolio_restarts", 2)),
            restart_seed_stride=int(cfg.get("portfolio_seed_stride", 100_003)),
            mode=str(cfg.get("portfolio_mode", "consensus")),
            disagreement_weight=float(cfg.get("portfolio_disagreement_weight", 0.25)),
            min_consensus_gain=float(cfg.get("portfolio_min_gain", 0.0)),
            primary_cost_margin=float(
                cfg.get("portfolio_primary_cost_margin", float("inf"))
            ),
            trace=bool(cfg.get("portfolio_trace", False)),
        )
        policy_obj = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    elif planner == "ipi_mppi":
        from slewm.planning.solver_ipi_mppi import IPIMPPISolver
        assert decoder is not None, "ipi_mppi needs IDM heads in the ckpt"
        solver = IPIMPPISolver(
            model=model, batch_size=1,
            num_samples=int(cfg.get("ipi_num_samples", 300)),
            n_steps=int(cfg.get("ipi_n_steps", 30)),
            temperature=float(cfg.get("ipi_temperature", 0.5)),
            var_scale=1.0, device="cuda", seed=cfg.seed,
            idm_frac=float(cfg.get("ipi_idm_frac", 0.1)),
            idm_sigma=float(cfg.get("ipi_idm_sigma", 1.0)),
            shell_radius=float(cfg.get("ipi_shell_radius", 3.0)),
            ratio_clip=float(cfg.get("ipi_ratio_clip", float("inf"))),
            ess_floor=float(cfg.get("ipi_ess_floor", 1.0)),
            ess_fraction=float(cfg.get("ipi_ess_fraction", 0.0)),
            ess_ratio_floor=float(cfg.get("ipi_ess_ratio_floor", 0.5)),
            proposal_kl=float(cfg.get("ipi_proposal_kl", 2.0)),
            accept_margin=float(cfg.get("ipi_accept_margin", 0.0)),
            forward_verify_topk=int(cfg.get("ipi_forward_verify_topk", 8)),
            cache_costs=bool(cfg.get("ipi_cache_costs", True)),
            verify_cost_cache=bool(cfg.get("ipi_verify_cost_cache", False)),
            idm_tau=float(cfg.get("idm_tau", 0.1)),
        )
        solver._decoder = decoder
        policy_obj = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    elif planner in {"dcl_beam_cem", "safe_dcl_portfolio"}:
        # Use the inverse contrastive IDM only as a multi-modal proposal
        # generator.  All vanilla CEM candidates are preserved and every
        # retrieved plan is forward-closed + ranked by the unchanged WM goal
        # cost; no uncalibrated IDM energy enters the planning objective.
        from slewm.planning.solver_dcl_beam_cem import DCLBeamCEMSolver

        assert decoder is not None and decoder.mode == "contrastive", (
            "dcl_beam_cem needs head_f_idm/head_g_idm from a "
            "C_T_dcl_idm checkpoint"
        )
        common_solver = dict(
            model=model,
            batch_size=int(cfg.solver.batch_size),
            num_samples=int(cfg.solver.num_samples),
            n_steps=int(cfg.solver.n_steps),
            topk=int(cfg.solver.topk),
            var_scale=float(cfg.solver.var_scale),
            device="cuda",
            seed=int(cfg.seed),
        )
        dcl_kwargs = dict(
            decoder=decoder,
            proposal_beams=int(cfg.get("dcl_beams", 4)),
            branch_topk=int(cfg.get("dcl_branch_topk", 8)),
            proposal_start=int(cfg.get("dcl_proposal_start", 0)),
            proposal_every=int(cfg.get("dcl_proposal_every", 1)),
            target_source=str(cfg.get("dcl_target_source", "best")),
            proposal_trust_rms=float(cfg.get("dcl_proposal_trust_rms", 0.0)),
            source_chains=int(cfg.get("dcl_source_chains", 1)),
            source_pool=int(cfg.get("dcl_source_pool", 30)),
            retrieval_preselect=int(cfg.get("dcl_retrieval_preselect", 0)),
            retrieval_diversity=float(cfg.get("dcl_retrieval_diversity", 0.0)),
            proposal_mode=str(cfg.get("dcl_proposal_mode", "reconstruct")),
            output_mode=str(cfg.get("dcl_output_mode", "mean")),
            search_mode=str(cfg.get("dcl_search_mode", "single")),
            mixture_components=int(cfg.get("dcl_mixture_components", 4)),
            mixture_initial_scale=float(
                cfg.get("dcl_mixture_initial_scale", 0.5)
            ),
            mixture_scale_floor=float(
                cfg.get("dcl_mixture_scale_floor", 0.05)
            ),
            mixture_elites=int(cfg.get("dcl_mixture_elites", 0)),
            history_size=int(cfg.get("dcl_history_size", 3)),
        )
        if planner == "dcl_beam_cem":
            solver = DCLBeamCEMSolver(
                **common_solver,
                **dcl_kwargs,
                trace=bool(cfg.get("dcl_trace", False)),
            )
        else:
            from slewm.planning.solver_safe_dcl_portfolio import SafeDCLPortfolioSolver

            secondary_model = None
            secondary_policy = str(cfg.get("dcl_secondary_policy", "")).strip()
            if secondary_policy:
                secondary_model, _ = load_full_model(secondary_policy)
                secondary_model = secondary_model.to("cuda").eval()
                secondary_model.requires_grad_(False)
                secondary_model.interpolate_pos_encoding = True
                print(
                    "[eval_ext] Safe-DCL secondary world-model veto = "
                    f"{secondary_policy}"
                )

            challenger_variants = []
            challenger_variant_gains = []
            hybrid_enabled = False
            mixture_enabled = False
            seeded_models = []
            gcidm_enabled = False
            if diverse_decoder is not None:
                challenger_variants.append({
                    "decoder": diverse_decoder,
                    "source_chains": int(cfg.get("dcl_diverse_source_chains", 2)),
                    "source_pool": int(cfg.get("dcl_diverse_source_pool", 30)),
                    "retrieval_preselect": int(
                        cfg.get("dcl_diverse_retrieval_preselect", 64)
                    ),
                    "retrieval_diversity": float(
                        cfg.get("dcl_diverse_retrieval_diversity", 0.25)
                    ),
                })
                challenger_variant_gains.append(
                    float(
                        cfg.get(
                            "dcl_diverse_min_goal_gain",
                            cfg.get("dcl_min_goal_gain", 0.02),
                        )
                    )
                )
            base_arm_count = 2 + int(diverse_decoder is not None)
            if coreset_decoder is not None:
                challenger_variants.append({
                    "decoder": coreset_decoder,
                    "output_mode": str(
                        cfg.get("dcl_coreset_output_mode", "mean")
                    ),
                    "source_chains": int(cfg.get("dcl_coreset_source_chains", 2)),
                    "source_pool": int(cfg.get("dcl_coreset_source_pool", 30)),
                    "retrieval_preselect": int(
                        cfg.get("dcl_coreset_retrieval_preselect", 64)
                    ),
                    "retrieval_diversity": float(
                        cfg.get("dcl_coreset_retrieval_diversity", 0.25)
                    ),
                })
                challenger_variant_gains.append(
                    float(cfg.get("dcl_coreset_min_goal_gain", 0.05))
                )
            if diverse_decoder is not None:
                if bool(cfg.get("dcl_include_hybrid_repair_challenger", False)):
                    from slewm.planning.idm_decode import BalancedIDMDecoder

                    hybrid_decoder = BalancedIDMDecoder(
                        [decoder, diverse_decoder],
                        retrieval_kwargs=[
                            {},
                            {
                                "preselect": int(
                                    cfg.get("dcl_hybrid_large_preselect", 64)
                                ),
                                "diversity_weight": float(
                                    cfg.get("dcl_hybrid_large_diversity", 0.25)
                                ),
                            },
                        ],
                    )
                    challenger_variants.append({
                        "decoder": hybrid_decoder,
                        "proposal_mode": str(
                            cfg.get("dcl_hybrid_proposal_mode", "repair")
                        ),
                        "output_mode": str(
                            cfg.get("dcl_hybrid_output_mode", "mean")
                        ),
                        "source_chains": int(
                            cfg.get("dcl_hybrid_source_chains", 2)
                        ),
                        "source_pool": int(cfg.get("dcl_hybrid_source_pool", 30)),
                        "retrieval_preselect": 0,
                        "retrieval_diversity": 0.0,
                    })
                    hybrid_enabled = True
                    challenger_variant_gains.append(
                        float(cfg.get("dcl_hybrid_min_goal_gain", 0.10))
                    )
                if bool(cfg.get("dcl_include_mixture_challenger", False)):
                    from slewm.planning.idm_decode import BalancedIDMDecoder

                    mixture_decoder = BalancedIDMDecoder(
                        [decoder, diverse_decoder],
                        retrieval_kwargs=[
                            {},
                            {
                                "preselect": int(
                                    cfg.get("dcl_mixture_large_preselect", 64)
                                ),
                                "diversity_weight": float(
                                    cfg.get("dcl_mixture_large_diversity", 0.25)
                                ),
                            },
                        ],
                    )
                    challenger_variants.append({
                        "decoder": mixture_decoder,
                        "proposal_mode": "reconstruct",
                        "output_mode": str(
                            cfg.get("dcl_mixture_output_mode", "best_seen")
                        ),
                        "search_mode": "mixture",
                        "proposal_beams": int(
                            cfg.get("dcl_mixture_proposal_beams", 4)
                        ),
                        "branch_topk": int(
                            cfg.get("dcl_mixture_branch_topk", 8)
                        ),
                        "source_chains": int(
                            cfg.get("dcl_mixture_source_chains", 2)
                        ),
                        "source_pool": int(
                            cfg.get("dcl_mixture_source_pool", 30)
                        ),
                        "retrieval_preselect": 0,
                        "retrieval_diversity": 0.0,
                        "mixture_components": int(
                            cfg.get("dcl_mixture_components", 4)
                        ),
                        "mixture_initial_scale": float(
                            cfg.get("dcl_mixture_initial_scale", 0.5)
                        ),
                        "mixture_scale_floor": float(
                            cfg.get("dcl_mixture_scale_floor", 0.05)
                        ),
                        "mixture_elites": int(
                            cfg.get("dcl_mixture_elites", 0)
                        ),
                    })
                    mixture_enabled = True
                    challenger_variant_gains.append(
                        float(cfg.get("dcl_mixture_min_goal_gain", 0.10))
                    )
            gcidm_path = str(cfg.get("dcl_gcidm_path", "")).strip()
            if gcidm_path:
                from slewm.evaluation.flow import _load_idm
                from slewm.flow_idm.flow_actor import LeWMFlowActor

                gcidm = _load_idm(gcidm_path, "cuda")
                gcidm_actor = LeWMFlowActor(
                    model,
                    gcidm,
                    h_max=int(getattr(gcidm.cfg, "h_max", 12)),
                    ode_steps=1,
                    h_mode=str(cfg.get("dcl_gcidm_h_mode", "remaining")),
                    h_fixed=int(cfg.get("dcl_gcidm_h_fixed", 5)),
                    cost_weight=None,
                    arrival=5,
                    device="cuda",
                ).to("cuda").eval()
                seeded_models.append(gcidm_actor)
                gcidm_enabled = True
                challenger_variant_gains.append(
                    float(cfg.get("dcl_gcidm_min_goal_gain", 0.02))
                )
            challenger_goal_gains = [
                float(cfg.get("dcl_min_goal_gain", 0.02)),
                *challenger_variant_gains,
            ]
            experimental_enabled = (
                coreset_decoder is not None
                or hybrid_enabled
                or mixture_enabled
                or gcidm_enabled
            )

            solver = SafeDCLPortfolioSolver(
                **common_solver,
                decoder=decoder,
                min_relative_goal_gain=float(cfg.get("dcl_min_goal_gain", 0.02)),
                challenger_goal_gains=challenger_goal_gains,
                min_support_gain=float(cfg.get("dcl_min_support_gain", 0.0)),
                min_dcl_support=float(cfg.get("dcl_min_abs_support", 0.0)),
                refine_steps=int(cfg.get("dcl_refine_steps", 0)),
                refine_step_size=float(cfg.get("dcl_refine_step_size", 0.05)),
                refine_radius=float(cfg.get("dcl_refine_radius", 0.15)),
                evaluator=secondary_model,
                min_secondary_relative_gain=float(
                    cfg.get("dcl_min_secondary_gain", float("-inf"))
                ),
                robust_samples=int(cfg.get("dcl_robust_samples", 0)),
                robust_noise_scale=float(cfg.get("dcl_robust_noise_scale", 0.25)),
                robust_scale_floor=float(cfg.get("dcl_robust_scale_floor", 0.05)),
                min_robust_median_gain=float(
                    cfg.get("dcl_min_robust_median_gain", 0.02)
                ),
                min_robust_win_fraction=float(
                    cfg.get("dcl_min_robust_win_fraction", 0.75)
                ),
                challenger_variants=challenger_variants,
                seeded_models=seeded_models,
                residual_start_index=(base_arm_count if experimental_enabled else None),
                min_residual_relative_gain=float(
                    cfg.get(
                        "dcl_experimental_min_residual_gain",
                        cfg.get("dcl_hybrid_min_residual_gain", 0.02),
                    )
                ),
                capture_plans=bool(cfg.get("dcl_capture_plans", False)),
                force_arm=(
                    int(cfg.get("dcl_force_arm"))
                    if cfg.get("dcl_force_arm", None) is not None
                    else None
                ),
                trace=bool(cfg.get("dcl_trace", False)),
                dcl_kwargs={k: v for k, v in dcl_kwargs.items() if k != "decoder"},
            )
        print(
            f"[eval_ext] {planner}: "
            f"pool={pool_size}, beams={int(cfg.get('dcl_beams', 4))}, "
            f"branch_topk={int(cfg.get('dcl_branch_topk', 8))}, "
            f"source={str(cfg.get('dcl_target_source', 'best'))}, "
            f"trust_rms={float(cfg.get('dcl_proposal_trust_rms', 0.0))}, "
            f"source_chains={int(cfg.get('dcl_source_chains', 1))}, "
            f"retrieval={int(cfg.get('dcl_retrieval_preselect', 0))}/"
            f"{float(cfg.get('dcl_retrieval_diversity', 0.0))}"
        )
        policy_obj = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    elif planner == "cover_cem":
        from slewm.planning.solver_cover_cem import SafeCoVerCEMSolver
        cover_source = str(cfg.get("cover_source", "idm"))
        verifier = decoder
        if cover_source == "dcl":
            from slewm.planning.dcl_transition import ForwardDCLVerifier

            # C_S DCL was trained against the empirical next-state marginal.
            # Encode the same fixed dataset picks used for the action bank and
            # use their real successor latents to estimate the NCE partition.
            successor_rows = dataset.get_row_data(pk + ab)["pixels"]
            pixel_transform = img_transform(cfg)
            successor_latents = []
            with torch.inference_mode():
                for start in range(0, len(successor_rows), 128):
                    pixels = torch.stack([
                        pixel_transform(pixel)
                        for pixel in successor_rows[start : start + 128]
                    ]).to("cuda")
                    hidden = model.encoder(
                        pixels, interpolate_pos_encoding=True
                    ).last_hidden_state[:, 0]
                    successor_latents.append(model.projector(hidden).float().cpu())
            successor_bank = torch.cat(successor_latents, dim=0)
            verifier = ForwardDCLVerifier(
                raw_sd,
                action_encoder=model.action_encoder,
                successor_bank=successor_bank,
                tau=float(cfg.get("idm_tau", 0.1)),
                device="cuda",
                A=int(action_pool.shape[-1]),
            )
            print(
                "[eval_ext] CoVer source = forward DCL; "
                f"successor bank = {successor_bank.shape[0]}"
            )
        elif cover_source != "idm":
            raise ValueError(
                f"unknown cover_source {cover_source!r}; expected 'idm' or 'dcl'"
            )
        assert verifier is not None, "cover_cem needs the requested verifier head"
        solver = SafeCoVerCEMSolver(
            model=model,
            batch_size=int(cfg.solver.batch_size),
            num_samples=int(cfg.solver.num_samples),
            n_steps=int(cfg.solver.n_steps),
            topk=int(cfg.solver.topk),
            var_scale=float(cfg.solver.var_scale),
            device="cuda",
            seed=int(cfg.seed),
            verifier=verifier,
            tau=float(cfg.get("idm_tau", 0.1)),
            cost_margin=float(cfg.get("cover_cost_margin", 0.0)),
            # Disabled by default: neither transition-IDM flavor is a
            # calibrated task-success verifier (see the CoVer audit report).
            verifier_margin_sigma=float(cfg.get("cover_margin_sigma", float("inf"))),
            worst_step_weight=float(cfg.get("cover_worst_weight", 0.5)),
            trace=bool(cfg.get("cover_trace", False)),
        )
        policy_obj = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    elif planner == "gbw":
        # GBW-Collocation seam rerank (2026-07-15 design workflow top pick).
        from slewm.planning.dcl_transition import ForwardDCLVerifier
        from slewm.planning.gbw_seam import SafeGBWCEMSolver

        # Backward witness uses head_g on the PREDECESSOR argument, so the NCE
        # partition bank is encoded from the PREDECESSOR frames pk (not pk+ab).
        pred_rows = dataset.get_row_data(pk)["pixels"]
        pixel_transform = img_transform(cfg)
        bank_latents = []
        with torch.inference_mode():
            for start in range(0, len(pred_rows), 128):
                pixels = torch.stack([
                    pixel_transform(pixel) for pixel in pred_rows[start:start + 128]
                ]).to("cuda")
                hidden = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                bank_latents.append(model.projector(hidden).float().cpu())
        successor_bank = torch.cat(bank_latents, dim=0)
        verifier = ForwardDCLVerifier(
            raw_sd, action_encoder=model.action_encoder, successor_bank=successor_bank,
            tau=float(cfg.get("idm_tau", 0.1)), device="cuda",
            A=int(action_pool.shape[-1]))
        beta = float(cfg.get("gbw_beta", 0.0))
        anchor = str(cfg.get("gbw_anchor", "goal"))
        print(f"[eval_ext] GBW: inverse-DCL backward witness, predecessor bank = "
              f"{successor_bank.shape[0]}, anchor={anchor}, beta={beta}")
        solver = SafeGBWCEMSolver(
            model=model,
            batch_size=int(cfg.solver.batch_size),
            num_samples=int(cfg.solver.num_samples),
            n_steps=int(cfg.solver.n_steps),
            topk=int(cfg.solver.topk),
            var_scale=float(cfg.solver.var_scale),
            device="cuda", seed=int(cfg.seed),
            verifier=verifier,
            idm_verifier=(decoder if (beta > 0 and decoder is not None
                                      and decoder.mode == "contrastive") else None),
            anchor=anchor, beta=beta,
            tau=float(cfg.get("idm_tau", 0.1)),
            cost_margin=float(cfg.get("gbw_cost_margin", 0.0)),
            verifier_margin_sigma=float(cfg.get("gbw_margin_sigma", float("inf"))),
            trace=bool(cfg.get("gbw_trace", False)),
        )
        policy_obj = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    elif planner == "restart":
        # Best-of-K multi-restart CEM (targets h10 CEM-optimization variance).
        from slewm.planning.multirestart_cem import MultiRestartCEMSolver
        selection = str(cfg.get("restart_selection", "cost"))
        rverifier = None
        if selection == "witness":
            from slewm.planning.dcl_transition import ForwardDCLVerifier
            pred_rows = dataset.get_row_data(pk)["pixels"]
            pixel_transform = img_transform(cfg)
            bl = []
            with torch.inference_mode():
                for start in range(0, len(pred_rows), 128):
                    pixels = torch.stack([pixel_transform(p) for p in pred_rows[start:start+128]]).to("cuda")
                    hidden = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                    bl.append(model.projector(hidden).float().cpu())
            rverifier = ForwardDCLVerifier(raw_sd, action_encoder=model.action_encoder,
                                           successor_bank=torch.cat(bl, dim=0),
                                           tau=float(cfg.get("idm_tau", 0.1)), device="cuda",
                                           A=int(action_pool.shape[-1]))
        print(f"[eval_ext] MultiRestart-CEM: K={int(cfg.get('n_restarts',4))}, selection={selection}")
        solver = MultiRestartCEMSolver(
            model=model,
            batch_size=int(cfg.solver.batch_size),
            num_samples=int(cfg.solver.num_samples),
            n_steps=int(cfg.solver.n_steps),
            topk=int(cfg.solver.topk),
            var_scale=float(cfg.solver.var_scale),
            device="cuda", seed=int(cfg.seed),
            n_restarts=int(cfg.get("n_restarts", 4)),
            selection=selection, verifier=rverifier,
            tau=float(cfg.get("idm_tau", 0.1)),
            trace=bool(cfg.get("restart_trace", False)),
        )
        policy_obj = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform)
    else:
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        if planner == "dtvh":
            from slewm.planning.dtvh_policy import DTVHPolicy, calibrate_thresholds
            tau_g, tau_d = calibrate_thresholds(
                model, dataset, img_transform(cfg), process["action"],
                action_block=int(cfg.plan_config.action_block),
                n_pairs=int(cfg.get("dtvh_calib_pairs", 256)),
                q=float(cfg.get("dtvh_quantile", 0.9)),
                margin=float(cfg.get("dtvh_margin", 1.5)),
                seed=int(cfg.seed), idm_decoder=decoder, device="cuda")
            print(f"[DTVH] calibrated tau_g={tau_g:.6f} tau_d={tau_d:.6f}")
            policy_obj = DTVHPolicy(
                solver=solver, config=config, process=process, transform=transform,
                tau_g=tau_g, tau_d=tau_d,
                hysteresis_ratio=float(cfg.get("dtvh_hysteresis", 0.5)),
                max_extra_replans=int(cfg.get("dtvh_max_extra", 4)),
                idm_decoder=decoder, var_head=var_head)
        else:
            policy_obj = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform)

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent

    # ---- episode selection (verbatim eval.py logic) ----
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(policy_obj)
    results_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    dt = time.time() - t0
    print(metrics)
    if hasattr(policy_obj, "stats"):
        print(f"[DTVH] stats: {policy_obj.stats}")
    if hasattr(solver, "diagnostics"):
        print(f"[solver] diagnostics: {solver.diagnostics()}")

    capture_path = str(cfg.get("dcl_capture_path", "")).strip()
    if capture_path:
        if not getattr(solver, "capture_records", None):
            raise RuntimeError(
                "dcl_capture_path requested but solver captured no planning records"
            )
        capture_file = Path(capture_path)
        capture_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "planner": planner,
            "task": str(cfg.eval.dataset_name),
            "eval_seed": int(cfg.seed),
            "episode_indices": torch.as_tensor(eval_episodes),
            "start_steps": torch.as_tensor(eval_start_idx),
            "episode_successes": torch.as_tensor(
                metrics["episode_successes"], dtype=torch.bool
            ),
            "planning_records": solver.capture_records,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(payload, capture_file)
        print(
            f"[eval_ext] saved {len(solver.capture_records)} planning capture(s) "
            f"to {capture_file}"
        )

    out = results_path / f"pusht_{planner}_results.txt"
    with out.open("a") as f:
        f.write(f"\n==== PLANNER {planner} ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\nmetrics: {metrics}\nevaluation_time: {dt} s\n")
        if hasattr(policy_obj, "stats"):
            f.write(f"dtvh_stats: {policy_obj.stats}\n")
        if hasattr(solver, "diagnostics"):
            f.write(f"solver_diagnostics: {solver.diagnostics()}\n")


if __name__ == "__main__":
    run()
