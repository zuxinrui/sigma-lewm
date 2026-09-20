"""Frozen, outcome-independent specs for the Demo-Goal successor study.

This module is intentionally separate from ``formal_method_specs.py``.  The
Delta/Concat Simple Triangle study is already frozen and must not acquire new
arms, method names, or attempt slots merely because a successor was added.
"""

from trim.protocols.formal_method_specs import COMMON_AUXILIARY_SHAPES, DCL_IDM_AUXILIARY_SHAPES


DEMO_GOAL_STUDY_IDS = {
    "pusht": "demo-goal-successor-pusht-v1",
    "tworoom": "demo-goal-successor-tworoom-v1",
}

DEMO_GOAL_METHOD_LOCK_ID = "demo-goal-successor-method-lock-v1"
DEMO_GOAL_FINAL_CONFIRMATION_ID = "demo-goal-successor-two-task-confirmation-v1"

DEMO_GOAL_TRAINING_SEEDS = (3072, 1, 2)
DEMO_GOAL_FORMAL_ARMS = ("champion", "dg_champion")

DEMO_GOAL_BANK_SEED_NAMESPACES = {
    "discovery": "lewm-demo-goal-successor-discovery-v1",
    "confirmation": "lewm-demo-goal-successor-confirmation-v1",
}
DEMO_GOAL_BANKS_PER_SET = {"discovery": 3, "confirmation": 5}
DEMO_GOAL_EPISODES_PER_BANK = 50
DEMO_GOAL_RESERVED_EPISODES_PER_TASK = 400
DEMO_GOAL_TASK_PROTOCOLS = {
    "pusht": (
        {
            "name": "primary_fc_h10_o25",
            "horizon": 10,
            "goal_offset": 25,
            "receding_horizon": 10,
            "action_block": 5,
            "eval_budget": 100,
        },
        {
            "name": "secondary_stock_h5_o25",
            "horizon": 5,
            "goal_offset": 25,
            "receding_horizon": 5,
            "action_block": 5,
            "eval_budget": 50,
        },
    ),
    "tworoom": (
        {
            "name": "primary_stock_h5_o25",
            "horizon": 5,
            "goal_offset": 25,
            "receding_horizon": 5,
            "action_block": 5,
            "eval_budget": 50,
        },
        {
            "name": "secondary_fc_h10_o25",
            "horizon": 10,
            "goal_offset": 25,
            "receding_horizon": 10,
            "action_block": 5,
            "eval_budget": 100,
        },
    ),
}

# Count-domain gates avoid ambiguous rounding at 3% and 4% thresholds.
DEMO_GOAL_DISCOVERY_GATES = {
    "pusht": {
        "primary_pairs": 150,
        "primary_min_delta_count": 6,
        "primary_min_positive_banks": 2,
        "secondary_pairs": 150,
        "secondary_min_delta_count": -3,
    },
    "tworoom": {
        "primary_pairs": 150,
        "primary_min_delta_count": 5,
        "primary_min_positive_banks": 2,
        "secondary_pairs": 150,
        "secondary_min_delta_count": -3,
    },
}
DEMO_GOAL_CONFIRMATION_GATES = {
    "primary_pairs": 750,
    "primary_min_delta_count": 23,
    "every_training_seed_positive": True,
    "two_way_cluster_bootstrap_95ci_lower_strictly_positive": True,
    "secondary_pairs": 750,
    "secondary_min_delta_count": -15,
}

DEMO_GOAL_FORMAL_TRAINING_JOB_SCHEDULE = (
    # Discovery: one paired seed per task.
    ("pusht", "champion", 3072),
    ("pusht", "dg_champion", 3072),
    ("tworoom", "champion", 3072),
    ("tworoom", "dg_champion", 3072),
    # Confirmation adds two new paired seeds per task.  Seed 3072 is reused.
    ("pusht", "champion", 1),
    ("pusht", "dg_champion", 1),
    ("pusht", "champion", 2),
    ("pusht", "dg_champion", 2),
    ("tworoom", "champion", 1),
    ("tworoom", "dg_champion", 1),
    ("tworoom", "champion", 2),
    ("tworoom", "dg_champion", 2),
)

# Every key here must match exactly for a formal run.  Method-varying axes live
# only in DEMO_GOAL_METHOD_SIGNATURES below.
DEMO_GOAL_FROZEN_TRAINING_CONFIG = {
    "full_seed_on": True,
    "episode_split_on": True,
    "split_seed": 20260822,
    "train_split": 0.9,
    "component_seeded_init": False,
    "img_size": 224,
    "embed_dim": 192,
    "history_size": 3,
    "data.dataset.frameskip": 5,
    "num_preds": 5,
    "trainer.max_epochs": 10,
    "trainer.devices": 2,
    "trainer.strategy": "ddp_find_unused_parameters_true",
    "trainer.sync_batchnorm": True,
    "trainer.accumulate_grad_batches": 1,
    "trainer.precision": "bf16",
    "trainer.accelerator": "gpu",
    "trainer.gradient_clip_val": 1.0,
    "loader.batch_size": 64,
    "loader.num_workers": 2,
    "loader.prefetch_factor": 1,
    "loader.persistent_workers": True,
    "loader.pin_memory": True,
    "optimizer.type": "AdamW",
    "optimizer.lr": 5e-5,
    "optimizer.weight_decay": 1e-3,
    "encoder_gradient_checkpointing": False,
    "fmca_k": 64,
    "distributed_global_losses": True,
    "sync_random_n": True,
    "pred_weight": 1.0,
    "tf_weight": 1.0,
    "ar_weight": 1.0,
    "ar_warmup_epochs": 2,
    "random_n_on": True,
    "unified_random_n_on": False,
    "diffusion_forcing": True,
    "df_sigma_max": 0.3,
    "df_tau_ctx": 0.1,
    "barrier_weight": 0.0,
    "standalone_sigreg_weight": 0.09,
    "loss.sigreg.weight": 0.09,
    "loss.sigreg.kwargs.knots": 17,
    "loss.sigreg.kwargs.num_proj": 1024,
    "idm_multistep_on": False,
    "arc_weight": 0.0,
    "dcl_idm_tau": 0.1,
    "dcl_idm_k": 64,
    "dcl_idm_dup_cos": 0.98,
    "shared_action_rank_weight": 0.0,
    "ibr_switch_epoch": 1,
    "fmca_weight": 0.10,
    "predictor_inv_weight": 0.0,
    "ar_inv_weight": 0.0,
    "fmca_inverse_on": True,
    "fmca_direct_on": False,
    "fmca_bidir_on": False,
    "mi_loss": "dcl",
    "mi_multistep_on": False,
    "mi_tau": 0.1,
    "joint_reg_on": False,
    "pred_target_sg": False,
    "fmca_idm_weight": 0.0,
    "idm_flow_weight": 0.0,
    "xidm_weight": 0.0,
    "gcidm_weight": 0.0,
    "flowgcidm_weight": 0.0,
    "flowpolicy_weight": 0.0,
    "bridge_weight": 0.0,
    "lindcl_fwd_weight": 0.0,
    "lindcl_inv_weight": 0.0,
    "bidir_simsiam_weight": 0.0,
    "bidir_dynamo_weight": 0.0,
    "bidir_pure_weight": 0.0,
    "dcl_bidir_weight": 0.0,
    "dynamo_lewm_weight": 0.0,
    "bn_dyn_lewm_weight": 0.0,
    "inloop_eval_on": False,
    "transition_triangle_execution_audit": True,
    "transition_triangle_expected_world_size": 2,
    "transition_triangle_require_clean_source": True,
    "transition_triangle_formal_execution": True,
    "transition_triangle_study_family": "demo_goal",
    "wandb.enabled": False,
    "save_epoch_interval": 999,
    "save_epoch_milestones": [3, 6, 8, 10],
}

DEMO_GOAL_METHOD_SIGNATURE_KEYS = (
    "idm_weight",
    "idm_input_mode",
    "idm_head_arch",
    "idm_stopgrad",
    "dcl_idm_weight",
    "demo_goal_action_weight",
    "demo_goal_index",
    "demo_goal_target_indices",
    "ibr_schedule_v1",
)

DEMO_GOAL_METHOD_SIGNATURES = {
    "champion": {
        "idm_weight": 0.0,
        "idm_input_mode": "concat",
        "idm_head_arch": "simple",
        "idm_stopgrad": False,
        "dcl_idm_weight": 0.05,
        "demo_goal_action_weight": 0.0,
        "demo_goal_index": 3,
        "demo_goal_target_indices": [0, 1, 2],
        "ibr_schedule_v1": False,
    },
    "dg_champion": {
        "idm_weight": 0.0,
        "idm_input_mode": "concat",
        "idm_head_arch": "simple",
        "idm_stopgrad": False,
        "dcl_idm_weight": 0.05,
        "demo_goal_action_weight": 0.10,
        "demo_goal_index": 3,
        "demo_goal_target_indices": [0, 1, 2],
        "ibr_schedule_v1": False,
    },
}

# This is intentionally not a member of DEMO_GOAL_METHOD_SIGNATURES.  It is a
# later simplicity ablation, not a second candidate that may rescue the locked
# two-arm successor study after outcomes are observed.
DEMO_GOAL_EXPLORATORY_SIGNATURES = {
    "dg_mse": {
        "idm_weight": 0.10,
        "idm_input_mode": "concat",
        "idm_head_arch": "dalewm",
        "idm_stopgrad": False,
        "dcl_idm_weight": 0.0,
        "demo_goal_action_weight": 0.10,
        "demo_goal_index": 3,
        "demo_goal_target_indices": [0, 1, 2],
        "ibr_schedule_v1": False,
    },
}


DEMO_GOAL_ACTION_AUXILIARY_SHAPES = {
    "demo_goal_action_mlp.0.weight": (256, 384),
    "demo_goal_action_mlp.0.bias": (256,),
    "demo_goal_action_mlp.1.weight": (256,),
    "demo_goal_action_mlp.1.bias": (256,),
    "demo_goal_action_mlp.3.weight": (256, 256),
    "demo_goal_action_mlp.3.bias": (256,),
    "demo_goal_action_mlp.4.weight": (256,),
    "demo_goal_action_mlp.4.bias": (256,),
    "demo_goal_action_mlp.6.weight": (10, 256),
    "demo_goal_action_mlp.6.bias": (10,),
}

DALEWM_IDM_AUXILIARY_SHAPES = {
    "idm_mlp.0.weight": (256, 384),
    "idm_mlp.0.bias": (256,),
    "idm_mlp.1.weight": (256,),
    "idm_mlp.1.bias": (256,),
    "idm_mlp.3.weight": (256, 256),
    "idm_mlp.3.bias": (256,),
    "idm_mlp.4.weight": (256,),
    "idm_mlp.4.bias": (256,),
    "idm_mlp.6.weight": (10, 256),
    "idm_mlp.6.bias": (10,),
}

DEMO_GOAL_TRAINING_AUXILIARY_PREFIXES = (
    "head_f.",
    "head_g.",
    "head_f_idm.",
    "head_g_idm.",
    "idm_mlp.",
    "demo_goal_action_mlp.",
    "sigreg_module.",
)


def demo_goal_study_id(task):
    """Return the immutable successor study identity for one task."""
    try:
        return DEMO_GOAL_STUDY_IDS[str(task)]
    except KeyError as error:
        raise ValueError(f"unsupported Demo-Goal task: {task!r}") from error


def demo_goal_method_signature(config):
    """Extract exactly the axes that distinguish successor checkpoints."""
    signature = {
        key: config.get(key, None)
        for key in DEMO_GOAL_METHOD_SIGNATURE_KEYS
    }
    target_indices = signature.get("demo_goal_target_indices")
    if target_indices is not None:
        signature["demo_goal_target_indices"] = [
            int(index) for index in target_indices
        ]
    return signature


def demo_goal_method_name(signature):
    """Resolve a complete signature; partial or hybrid recipes fail closed."""
    for name, expected in DEMO_GOAL_METHOD_SIGNATURES.items():
        if signature == expected:
            return name
    raise ValueError(f"unknown Demo-Goal method signature: {signature}")


def demo_goal_auxiliary_shapes(method_name):
    """Return exact float32 training-only state required by a successor arm."""
    if method_name not in DEMO_GOAL_METHOD_SIGNATURES:
        raise KeyError(f"unknown Demo-Goal method: {method_name}")
    expected = dict(COMMON_AUXILIARY_SHAPES)
    expected.update(DCL_IDM_AUXILIARY_SHAPES)
    if method_name != "champion":
        expected.update(DEMO_GOAL_ACTION_AUXILIARY_SHAPES)
    return expected


def demo_goal_exploratory_auxiliary_shapes(method_name):
    """Return head shapes for a separately registered, non-formal ablation."""
    if method_name not in DEMO_GOAL_EXPLORATORY_SIGNATURES:
        raise KeyError(f"unknown exploratory Demo-Goal method: {method_name}")
    expected = dict(COMMON_AUXILIARY_SHAPES)
    expected.update(DALEWM_IDM_AUXILIARY_SHAPES)
    expected.update(DEMO_GOAL_ACTION_AUXILIARY_SHAPES)
    return expected
