"""Evaluate one frozen checkpoint on immutable SimpleTT protocol banks."""

from __future__ import annotations

from trim.paths import REPO_ROOT

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from trim.evaluation.protocol import (
    FORMAL_EVAL_MANIFEST_SCHEMA_VERSION,
    FORMAL_EVAL_RESULT_SCHEMA_VERSION,
    _canonical_sha256,
    _dataset_fingerprint,
    _file_sha256,
    _validate_formal_bank_structure,
    _validate_binary_episode_successes,
    _validate_task_protocols,
    verify_demo_goal_manifest_external_bindings,
)
from trim.protocols.formal_method_specs import (
    COMMON_AUXILIARY_SHAPES as _COMMON_AUXILIARY_SHAPES,
    DCL_IDM_AUXILIARY_SHAPES as _DCL_IDM_AUXILIARY_SHAPES,
    FORMAL_METHOD_SIGNATURES,
    MSE_IDM_AUXILIARY_SHAPES as _MSE_IDM_AUXILIARY_SHAPES,
    TRAINING_AUXILIARY_PREFIXES as _INFERENCE_AUX_PREFIXES,
)
from trim.protocols.demo_goal_formal_specs import (
    DEMO_GOAL_FROZEN_TRAINING_CONFIG,
    DEMO_GOAL_METHOD_SIGNATURE_KEYS,
    DEMO_GOAL_METHOD_SIGNATURES,
    DEMO_GOAL_STUDY_IDS,
    DEMO_GOAL_TRAINING_AUXILIARY_PREFIXES,
    demo_goal_auxiliary_shapes,
)
from trim.evaluation.inloop import InLoopEvaluator
from trim.protocols.formal_training_completion import (
    EPISODE_SPLIT_SCHEMA_VERSION,
    verify_formal_training_completion,
)
from trim.training.utilities import (
    git_tracked_tree_sha256,
    runtime_environment_compatibility_fingerprint,
    runtime_environment_fingerprint,
)


_METHOD_KEYS = (
    "idm_weight",
    "idm_input_mode",
    "dcl_idm_weight",
    "ibr_schedule_v1",
)
_DELTA_INFERENCE_AUX_PREFIXES = _INFERENCE_AUX_PREFIXES
_INFERENCE_AUX_PREFIXES = tuple(dict.fromkeys(
    _DELTA_INFERENCE_AUX_PREFIXES + DEMO_GOAL_TRAINING_AUXILIARY_PREFIXES
))
_RUN_SPECIFIC_CONFIG_KEYS = (
    "output_model_name",
    "subdir",
    "initial_full_model_sha256",
    "formal_stage_authorization_path",
    "formal_stage_authorization_sha256",
    "formal_smoke_attestation_path",
    "formal_slurm_job_id",
    "formal_spt_cache_dir_sha256",
)
_CROSS_SEED_CONFIG_KEYS = (
    "seed",
    "initial_shared_core_sha256",
)
_FROZEN_FORMAL_CONFIG = {
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
    "idm_stopgrad": False,
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
    "wandb.enabled": False,
    "save_epoch_interval": 999,
    "save_epoch_milestones": [3, 6, 8, 10],
}
_FORBIDDEN_FORMAL_TRAINER_KEYS = (
    "limit_train_batches",
    "limit_val_batches",
    "max_steps",
    "fast_dev_run",
    "overfit_batches",
)
_EXACT_FORMAL_MAPPING_KEYS = {
    "trainer": {
        "max_epochs",
        "devices",
        "accelerator",
        "precision",
        "gradient_clip_val",
        "strategy",
        "sync_batchnorm",
        "accumulate_grad_batches",
    },
    "loader": {
        "batch_size",
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
        "pin_memory",
    },
    "optimizer": {"type", "lr", "weight_decay"},
    "loss.sigreg": {"weight", "kwargs"},
    "loss.sigreg.kwargs": {"knots", "num_proj"},
}

_CANONICAL_SIMPLETT_INFERENCE_CONFIG = {
    "_target_": "trim.models.world_model.JEPA",
    "encoder": {
        "_target_": "stable_pretraining.backbone.utils.vit_hf",
        "size": "tiny",
        "patch_size": 14,
        "image_size": 224,
        "pretrained": False,
        "use_mask_token": False,
    },
    "predictor": {
        "_target_": "trim.models.components.ARPredictor",
        "num_frames": 3,
        "input_dim": 192,
        "hidden_dim": 192,
        "output_dim": 192,
        "depth": 6,
        "heads": 16,
        "mlp_dim": 2048,
        "dim_head": 64,
        "dropout": 0.1,
        "emb_dropout": 0.0,
    },
    "action_encoder": {
        "_target_": "trim.models.components.Embedder",
        "input_dim": 10,
        "emb_dim": 192,
    },
    "projector": {
        "_target_": "trim.models.components.MLP",
        "input_dim": 192,
        "output_dim": 192,
        "hidden_dim": 2048,
        "norm_fn": {
            "_target_": "torch.nn.BatchNorm1d",
            "_partial_": True,
        },
    },
    "pred_proj": {
        "_target_": "trim.models.components.MLP",
        "input_dim": 192,
        "output_dim": 192,
        "hidden_dim": 2048,
        "norm_fn": {
            "_target_": "torch.nn.BatchNorm1d",
            "_partial_": True,
        },
    },
}


def _nested_config_value(config, dotted_key):
    value = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def _validate_canonical_inference_config(config):
    if config != _CANONICAL_SIMPLETT_INFERENCE_CONFIG:
        raise RuntimeError(
            "formal transition-triangle study requires the canonical "
            "PushT/TwoRoom LeWM architecture"
        )


def _same_frozen_value(actual, expected):
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _study_family_from_payload(payload):
    study_id = payload.get("study_id")
    if study_id is None:
        return "delta_triangle"
    if study_id in set(DEMO_GOAL_STUDY_IDS.values()):
        task = payload.get("task")
        if DEMO_GOAL_STUDY_IDS.get(task) != study_id:
            raise RuntimeError(
                "Demo-Goal task/study_id mapping mismatch: "
                f"task={task!r} study_id={study_id!r}"
            )
        return "demo_goal"
    raise RuntimeError(f"unknown formal study_id: {study_id!r}")


def _method_signature(config, *, study_family="delta_triangle"):
    if study_family == "demo_goal":
        signature = {
            key: config.get(key, None)
            for key in DEMO_GOAL_METHOD_SIGNATURE_KEYS
        }
        if signature.get("demo_goal_target_indices") is not None:
            signature["demo_goal_target_indices"] = [
                int(value)
                for value in signature["demo_goal_target_indices"]
            ]
        return signature
    if study_family != "delta_triangle":
        raise ValueError(f"unknown formal study family: {study_family!r}")
    return {key: config.get(key, None) for key in _METHOD_KEYS}


def _formal_method_name(method_signature, *, study_family=None):
    if study_family is None:
        study_family = (
            "demo_goal"
            if set(method_signature) == set(DEMO_GOAL_METHOD_SIGNATURE_KEYS)
            else "delta_triangle"
        )
    signatures = (
        DEMO_GOAL_METHOD_SIGNATURES
        if study_family == "demo_goal"
        else FORMAL_METHOD_SIGNATURES
    )
    for name, expected in signatures.items():
        if method_signature == expected:
            return name
    raise RuntimeError(f"unknown formal method signature: {method_signature}")


def _validate_frozen_training_config(
    config, training, run_dir, *, study_family="delta_triangle"
):
    mismatches = {}
    frozen_config = (
        DEMO_GOAL_FROZEN_TRAINING_CONFIG
        if study_family == "demo_goal"
        else _FROZEN_FORMAL_CONFIG
    )
    for key, expected in frozen_config.items():
        try:
            actual = _nested_config_value(config, key)
        except KeyError:
            mismatches[key] = ("<missing>", expected)
            continue
        if not _same_frozen_value(actual, expected):
            mismatches[key] = (actual, expected)

    expected_manifest_values = {
        "seed": int(training["training_seed"]),
        "split_seed": int(training["split_seed"]),
        "initial_shared_core_sha256": training["initial_shared_core_sha256"],
        "initial_full_model_sha256": training["initial_full_model_sha256"],
        "episode_split_train_sha256": training["train_sha256"],
        "episode_split_val_sha256": training["validation_sha256"],
        "subdir": Path(run_dir).name,
        "output_model_name": Path(run_dir).name,
        "formal_stage_authorization_sha256": training[
            "formal_stage_authorization_sha256"
        ],
        "formal_smoke_attestation_sha256": training[
            "formal_smoke_attestation_sha256"
        ],
        "formal_slurm_job_id": training["formal_slurm_job_id"],
        "formal_slurm_restart_count": 0,
        "formal_spt_cache_dir_sha256": training[
            "formal_spt_cache_dir_sha256"
        ],
        "formal_dataset_file_sha256": training["dataset_file_sha256"],
    }
    if study_family == "demo_goal":
        expected_manifest_values.update({
            "initial_champion_predecessor_sha256": training[
                "initial_champion_predecessor_sha256"
            ],
            "transition_triangle_study_family": "demo_goal",
            "formal_method_lock_sha256": training[
                "formal_method_lock_sha256"
            ],
        })
    for key, expected in expected_manifest_values.items():
        actual = config.get(key, "<missing>")
        if actual != expected:
            mismatches[key] = (actual, expected)

    trainer_config = config.get("trainer", {})
    forbidden = {
        key: trainer_config[key]
        for key in _FORBIDDEN_FORMAL_TRAINER_KEYS
        if key in trainer_config
    }
    if forbidden:
        mismatches["forbidden_trainer_limits"] = (forbidden, {})
    for dotted_key, expected_keys in _EXACT_FORMAL_MAPPING_KEYS.items():
        try:
            mapping = _nested_config_value(config, dotted_key)
        except KeyError:
            continue
        actual_keys = set(mapping) if isinstance(mapping, dict) else set()
        if actual_keys != expected_keys:
            mismatches[f"{dotted_key}.__keys__"] = (
                sorted(actual_keys), sorted(expected_keys)
            )
    if mismatches:
        raise RuntimeError(f"formal training config violates frozen recipe: {mismatches}")
    _formal_method_name(
        _method_signature(config, study_family=study_family),
        study_family=study_family,
    )


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"checkpoint contains no files: {path}")
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        try:
            digest.update(value.numpy().tobytes())
        except TypeError:
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_inference_core(
    checkpoint: Path,
    method_signature=None,
    *,
    enforce_canonical_architecture=True,
    study_family=None,
):
    """Strictly load the JEPA inference core and audit discarded training heads."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "formal evaluation requires one explicit weights_epoch_10.pt file"
        )
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {config_path}")
    config = json.loads(config_path.read_text())
    if enforce_canonical_architecture:
        _validate_canonical_inference_config(config)
    model = instantiate(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state_dict.items()
    ):
        raise TypeError("formal checkpoint must be a raw tensor state_dict")

    expected = set(model.state_dict())
    present = set(state_dict)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    forbidden = [
        key for key in unexpected
        if not any(key.startswith(prefix) for prefix in _INFERENCE_AUX_PREFIXES)
    ]
    if missing or forbidden:
        raise RuntimeError(
            "checkpoint cannot be reduced to the inference core: "
            f"missing={missing[:10]} forbidden={forbidden[:10]}"
        )
    core_state = {key: state_dict[key] for key in model.state_dict()}
    auxiliary_state = {key: state_dict[key] for key in unexpected}
    model.load_state_dict(core_state, strict=True)
    dropped_prefix_counts = {
        prefix.rstrip("."): sum(key.startswith(prefix) for key in unexpected)
        for prefix in _INFERENCE_AUX_PREFIXES
        if any(key.startswith(prefix) for key in unexpected)
    }
    if method_signature is not None:
        if study_family is None:
            study_family = (
                "demo_goal"
                if set(method_signature) == set(DEMO_GOAL_METHOD_SIGNATURE_KEYS)
                else "delta_triangle"
            )
        method_name = _formal_method_name(
            method_signature, study_family=study_family
        )
        if study_family == "demo_goal":
            expected_shapes = demo_goal_auxiliary_shapes(method_name)
            required_prefixes = {
                "head_f", "head_g", "head_f_idm", "head_g_idm", "sigreg_module"
            }
            if method_name == "dg_champion":
                required_prefixes.add("demo_goal_action_mlp")
        else:
            expected_shapes = dict(_COMMON_AUXILIARY_SHAPES)
            if method_name in ("champion", "ibr"):
                expected_shapes.update(_DCL_IDM_AUXILIARY_SHAPES)
                required_prefixes = {
                    "head_f", "head_g", "head_f_idm", "head_g_idm", "sigreg_module"
                }
            else:
                input_mode = method_signature["idm_input_mode"]
                expected_shapes.update(_MSE_IDM_AUXILIARY_SHAPES[input_mode])
                required_prefixes = {"head_f", "head_g", "idm_mlp", "sigreg_module"}
        absent = sorted(required_prefixes - set(dropped_prefix_counts))
        if absent:
            raise RuntimeError(f"checkpoint lacks required training auxiliary heads: {absent}")
        actual_auxiliary_keys = set(auxiliary_state)
        expected_auxiliary_keys = set(expected_shapes)
        missing_auxiliary = sorted(expected_auxiliary_keys - actual_auxiliary_keys)
        extra_auxiliary = sorted(actual_auxiliary_keys - expected_auxiliary_keys)
        malformed_auxiliary = {
            key: {
                "actual_shape": tuple(auxiliary_state[key].shape),
                "expected_shape": expected_shape,
                "actual_dtype": str(auxiliary_state[key].dtype),
            }
            for key, expected_shape in expected_shapes.items()
            if key in auxiliary_state
            and (
                tuple(auxiliary_state[key].shape) != expected_shape
                or auxiliary_state[key].dtype != torch.float32
            )
        }
        if missing_auxiliary or extra_auxiliary or malformed_auxiliary:
            raise RuntimeError(
                "checkpoint auxiliary signature mismatch: "
                f"missing={missing_auxiliary[:10]} extra={extra_auxiliary[:10]} "
                f"malformed={dict(list(malformed_auxiliary.items())[:5])}"
            )
    audit = {
        "inference_config_path": str(config_path),
        "inference_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "inference_core_state_sha256": _state_dict_sha256(core_state),
        "discarded_auxiliary_state_sha256": _state_dict_sha256(auxiliary_state),
        "discarded_auxiliary_key_count": len(unexpected),
        "discarded_auxiliary_prefix_counts": dropped_prefix_counts,
    }
    return model, audit


def _verify_manifest(payload):
    if int(payload.get("schema_version", -1)) != FORMAL_EVAL_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "formal transition-triangle study requires manifest schema_version="
            f"{FORMAL_EVAL_MANIFEST_SCHEMA_VERSION}"
        )
    claimed = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    actual = _canonical_sha256(unsigned)
    if claimed != actual:
        raise RuntimeError(f"formal manifest hash mismatch: claimed={claimed}, actual={actual}")
    _validate_task_protocols(payload.get("task"), payload.get("protocols"))
    _validate_formal_bank_structure(
        payload,
        expected_discovery_banks=3,
        expected_confirmation_banks=5,
        expected_num_eval=50,
        expected_goal_offset=25,
    )
    verify_demo_goal_manifest_external_bindings(payload)
    runtime = payload.get("runtime_environment")
    if _canonical_sha256(runtime) != payload.get("runtime_environment_sha256"):
        raise RuntimeError("formal manifest training runtime self-hash is invalid")
    compatibility = runtime_environment_compatibility_fingerprint(runtime)
    if compatibility != payload.get("runtime_compatibility"):
        raise RuntimeError("formal manifest runtime compatibility payload is invalid")
    if _canonical_sha256(compatibility) != payload.get(
        "runtime_compatibility_sha256"
    ):
        raise RuntimeError("formal manifest runtime compatibility hash is invalid")


def _verify_dataset(payload, dataset_path: Path):
    stat_before = dataset_path.stat()
    with h5py.File(dataset_path, "r", swmr=True) as handle:
        lengths = handle["ep_len"][:]
        offsets = handle["ep_offset"][:]
    actual = _dataset_fingerprint(
        payload["dataset_name"],
        # Window count is frozen indirectly by the training dataset fingerprint.
        # Recover it from the split manifest's fingerprint contract via the
        # explicit value copied into newer formal manifests.
        payload["dataset_windows"],
        lengths,
        offsets,
        dataset_path.stat().st_size,
    )
    if actual != payload["dataset_fingerprint_sha256"]:
        raise RuntimeError("evaluation dataset fingerprint differs from training")
    if int(stat_before.st_size) != int(payload["dataset_size_bytes"]):
        raise RuntimeError("evaluation dataset size differs from the frozen manifest")
    if int(stat_before.st_mtime_ns) != int(payload["dataset_mtime_ns"]):
        raise RuntimeError("evaluation dataset mtime differs from the frozen manifest")
    actual_file_sha256 = _file_sha256(dataset_path)
    if actual_file_sha256 != payload["dataset_file_sha256"]:
        raise RuntimeError("evaluation dataset full-file SHA256 differs from training")
    stat_after = dataset_path.stat()
    if (
        int(stat_after.st_size) != int(stat_before.st_size)
        or int(stat_after.st_mtime_ns) != int(stat_before.st_mtime_ns)
    ):
        raise RuntimeError("evaluation dataset changed while it was being verified")


def _training_contract(checkpoint: Path, formal_payload, training_manifest_path: Path):
    """Verify and summarize the training provenance of one evaluated checkpoint."""
    training_manifest_path = Path(training_manifest_path).resolve()
    if not training_manifest_path.is_file():
        raise FileNotFoundError(training_manifest_path)
    run_dir = training_manifest_path.parent
    if checkpoint.name != "weights_epoch_10.pt" or not checkpoint.is_file():
        raise RuntimeError(
            "formal comparison is frozen to the explicit weights_epoch_10.pt checkpoint"
        )
    try:
        checkpoint.relative_to(run_dir)
    except ValueError as error:
        raise RuntimeError(
            f"checkpoint {checkpoint} is not inside training run {run_dir}"
        ) from error

    training = json.loads(training_manifest_path.read_text())
    study_family = _study_family_from_payload(formal_payload)
    expected = {
        "schema_version": EPISODE_SPLIT_SCHEMA_VERSION,
        "split_seed": formal_payload["split_seed"],
        "train_sha256": formal_payload["train_split_sha256"],
        "validation_sha256": formal_payload["validation_split_sha256"],
        "dataset_fingerprint_sha256": formal_payload["dataset_fingerprint_sha256"],
        "dataset_file_sha256": formal_payload["dataset_file_sha256"],
        "dataset_size_bytes": formal_payload["training_dataset_size_bytes"],
        "normalizer_fit": "train_episodes_only",
        "source_git_commit": formal_payload["created_by_git_commit"],
        "source_tracked_tree_sha256": formal_payload["source_tracked_tree_sha256"],
        "runtime_environment_sha256": formal_payload["runtime_environment_sha256"],
    }
    # Delta's legacy protocol froze one physical training file path.  Demo-Goal
    # stages each paired job into a fresh job-owned /tmp directory, so identity
    # is the full file SHA256 + fingerprint + size, never a transient pathname
    # or mtime that necessarily differs between champion and candidate jobs.
    if study_family != "demo_goal":
        expected.update({
            "dataset_path": formal_payload["training_dataset_path"],
            "dataset_mtime_ns": formal_payload["training_dataset_mtime_ns"],
        })
    mismatches = {
        key: (training.get(key), value)
        for key, value in expected.items()
        if training.get(key) != value
    }
    training_normalizer_sha256 = _canonical_sha256(training.get("normalizer_stats"))
    formal_normalizer_sha256 = _canonical_sha256(formal_payload["normalizer_stats"])
    if training_normalizer_sha256 != formal_normalizer_sha256:
        mismatches["normalizer_stats_sha256"] = (
            training_normalizer_sha256,
            formal_normalizer_sha256,
        )
    if mismatches:
        raise RuntimeError(f"training manifest is incompatible with formal protocol: {mismatches}")

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"training config is missing: {config_path}")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    _validate_frozen_training_config(
        config, training, run_dir, study_family=study_family
    )
    method_signature = _method_signature(config, study_family=study_family)
    method_name = _formal_method_name(
        method_signature, study_family=study_family
    )
    if study_family == "demo_goal":
        from trim.protocols.demo_goal_training_artifacts import (
            verify_demo_goal_training_completion,
        )

        completion_path = run_dir / "demo_goal_training_complete.json"
        completion = verify_demo_goal_training_completion(
            completion_path,
            run_dir=run_dir,
            checkpoint=checkpoint,
            expected_method_name=method_name,
            expected_task=formal_payload["task"],
            expected_training_seed=int(training["training_seed"]),
            expected_method_lock_sha256=formal_payload["method_lock_sha256"],
            require_registry=True,
        )
    else:
        completion_path = run_dir / "formal_training_complete.json"
        completion = verify_formal_training_completion(
            completion_path,
            run_dir=run_dir,
            checkpoint=checkpoint,
            expected_method_name=method_name,
            expected_task=formal_payload["task"],
            expected_training_seed=int(training["training_seed"]),
            require_registry=True,
        )
    completion_expected = {
        "source_git_commit": training["source_git_commit"],
        "source_tracked_tree_sha256": training[
            "source_tracked_tree_sha256"
        ],
        "runtime_environment_sha256": training["runtime_environment_sha256"],
        "dataset_file_sha256": training["dataset_file_sha256"],
        "stage_authorization_sha256": training[
            "formal_stage_authorization_sha256"
        ],
        "smoke_attestation_sha256": training[
            "formal_smoke_attestation_sha256"
        ],
        "formal_slurm_job_id": training["formal_slurm_job_id"],
    }
    completion_mismatches = {
        key: (completion.get(key), value)
        for key, value in completion_expected.items()
        if completion.get(key) != value
    }
    if completion_mismatches:
        raise RuntimeError(
            "formal completion provenance differs from training manifest: "
            f"{completion_mismatches}"
        )
    comparison_config = json.loads(json.dumps(config))
    method_keys = (
        DEMO_GOAL_METHOD_SIGNATURE_KEYS
        if study_family == "demo_goal"
        else _METHOD_KEYS
    )
    for key in method_keys + _RUN_SPECIFIC_CONFIG_KEYS:
        comparison_config.pop(key, None)
    recipe_config = json.loads(json.dumps(comparison_config))
    cross_seed_keys = list(_CROSS_SEED_CONFIG_KEYS)
    if study_family == "demo_goal":
        cross_seed_keys.append("initial_champion_predecessor_sha256")
    for key in cross_seed_keys:
        recipe_config.pop(key, None)

    return {
        "training_manifest_path": str(training_manifest_path),
        "training_manifest_sha256": hashlib.sha256(
            training_manifest_path.read_bytes()
        ).hexdigest(),
        "training_config_path": str(config_path),
        "training_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "training_config_comparison_sha256": _canonical_sha256(comparison_config),
        "training_recipe_sha256": _canonical_sha256(recipe_config),
        "method_signature": method_signature,
        "method_name": method_name,
        "study_family": study_family,
        "training_seed": int(training["training_seed"]),
        "split_seed": int(training["split_seed"]),
        "train_split_sha256": training["train_sha256"],
        "validation_split_sha256": training["validation_sha256"],
        "training_normalizer_sha256": training_normalizer_sha256,
        "training_dataset_file_sha256": training["dataset_file_sha256"],
        "initial_shared_core_sha256": training["initial_shared_core_sha256"],
        "initial_full_model_sha256": training["initial_full_model_sha256"],
        "initial_champion_predecessor_sha256": training.get(
            "initial_champion_predecessor_sha256"
        ),
        "training_source_git_commit": training["source_git_commit"],
        "training_source_tracked_tree_sha256": training[
            "source_tracked_tree_sha256"
        ],
        "training_runtime_environment_sha256": training[
            "runtime_environment_sha256"
        ],
        "formal_training_completion_path": str(completion_path),
        "formal_training_completion_sha256": completion["completion_sha256"],
        "formal_training_smoke_attestation_sha256": completion[
            "smoke_attestation_sha256"
        ],
        "formal_training_stage_authorization_sha256": completion[
            "stage_authorization_sha256"
        ],
    }


def run_evaluation(
    *,
    checkpoint: Path,
    manifest_path: Path,
    training_manifest_path: Path,
    output_path: Path,
    bank_set: str,
    protocol_names,
    dataset_path: Path | None,
):
    checkpoint = Path(checkpoint).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise RuntimeError(
            f"refusing to overwrite formal evaluation result: {output_path}"
        )
    payload = json.loads(manifest_path.read_text())
    _verify_manifest(payload)
    training_contract = _training_contract(
        checkpoint, payload, training_manifest_path
    )
    if bank_set not in payload["banks"]:
        raise KeyError(f"unknown bank set {bank_set!r}")
    if dataset_path is None:
        dataset_path = Path(payload["dataset_path"])
    dataset_path = Path(dataset_path).resolve()
    _verify_dataset(payload, dataset_path)

    repo = REPO_ROOT
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if current_commit != payload["created_by_git_commit"]:
        raise RuntimeError("formal evaluation source commit differs from its manifest")
    if git_tracked_tree_sha256(repo) != payload["source_tracked_tree_sha256"]:
        raise RuntimeError("formal evaluation tracked source content differs from its manifest")
    runtime_environment = runtime_environment_fingerprint()
    runtime_environment_sha256 = _canonical_sha256(runtime_environment)
    runtime_compatibility_sha256 = _canonical_sha256(
        runtime_environment_compatibility_fingerprint(runtime_environment)
    )
    if runtime_compatibility_sha256 != payload["runtime_compatibility_sha256"]:
        raise RuntimeError(
            "formal evaluation runtime is not inference-compatible with training"
        )
    runtime_files = {
        repo / "config" / "eval" / f"{payload['task']}.yaml": payload["eval_config_sha256"],
        repo / "config" / "eval" / "solver" / "cem.yaml": payload["solver_config_sha256"],
    }
    for path, expected_sha256 in runtime_files.items():
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"formal runtime config drifted after manifest creation: {path}")

    selected_protocols = [
        protocol for protocol in payload["protocols"]
        if not protocol_names or protocol["name"] in protocol_names
    ]
    if not selected_protocols:
        raise RuntimeError("no requested protocol exists in the manifest")

    model, inference_core_audit = _load_inference_core(
        checkpoint,
        method_signature=training_contract["method_signature"],
        study_family=training_contract["study_family"],
    )
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    records = []
    for protocol in selected_protocols:
        if int(protocol["goal_offset"]) != int(payload["goal_offset"]):
            raise RuntimeError("mixed goal offsets require a separate frozen bank manifest")
        for bank in payload["banks"][bank_set]:
            evaluator = InLoopEvaluator(
                num_eval=int(payload["num_eval_per_bank"]),
                seed=int(bank["seed"]),
                task=str(payload["task"]),
                unique_episodes=True,
                normalizer_stats=payload["normalizer_stats"],
                fixed_rows=bank["row_indices"],
                dataset_path=dataset_path,
            )
            metrics = evaluator.evaluate(
                model,
                horizon=int(protocol["horizon"]),
                receding_horizon=int(protocol["receding_horizon"]),
                action_block=int(protocol["action_block"]),
                eval_budget=int(protocol["eval_budget"]),
                return_metrics=True,
            )
            actual_rows = [int(value) for value in evaluator.eval_row_indices]
            actual_episodes = [int(value) for value in evaluator.eval_episodes]
            actual_starts = [int(value) for value in evaluator.eval_start_idx]
            actual_goals = [
                start + int(evaluator.goal_offset) for start in actual_starts
            ]
            actual_contract = {
                "row_indices": actual_rows,
                "episode_ids": actual_episodes,
                "start_steps": actual_starts,
                "goal_steps": actual_goals,
            }
            expected_contract = {
                field: [int(value) for value in bank[field]]
                for field in actual_contract
            }
            if actual_contract != expected_contract:
                raise RuntimeError(
                    "evaluator execution rows differ from the frozen bank: "
                    f"actual={actual_contract} expected={expected_contract}"
                )
            raw_successes = metrics["episode_successes"]
            if torch.is_tensor(raw_successes):
                raw_successes = raw_successes.detach().cpu().numpy()
            successes = _validate_binary_episode_successes(
                raw_successes,
                expected_count=int(payload["num_eval_per_bank"]),
                context=(
                    f"evaluator episode_successes protocol={protocol['name']} "
                    f"bank={bank['bank_index']}"
                ),
            )
            records.append({
                "protocol": protocol["name"],
                "bank_set": bank_set,
                "bank_index": int(bank["bank_index"]),
                "bank_seed": int(bank["seed"]),
                "horizon": int(protocol["horizon"]),
                "receding_horizon": int(protocol["receding_horizon"]),
                "action_block": int(protocol["action_block"]),
                "eval_budget": int(protocol["eval_budget"]),
                "row_indices": actual_rows,
                "episode_ids": actual_episodes,
                "start_steps": actual_starts,
                "goal_steps": actual_goals,
                "episode_successes": successes.tolist(),
                "success_rate": float(successes.mean() * 100.0),
            })

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    result = {
        "schema_version": FORMAL_EVAL_RESULT_SCHEMA_VERSION,
        "task": payload["task"],
        "source_git_commit": source_commit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "formal_manifest_path": str(manifest_path),
        "formal_manifest_sha256": payload["manifest_sha256"],
        "dataset_path": str(dataset_path),
        "dataset_fingerprint_sha256": payload["dataset_fingerprint_sha256"],
        "dataset_file_sha256": payload["dataset_file_sha256"],
        "bank_set": bank_set,
        "records": records,
    }
    if payload.get("study_id") is not None:
        result.update({
            "study_id": payload["study_id"],
            "method_lock_sha256": payload["method_lock_sha256"],
        })
    result.update(training_contract)
    result.update(inference_core_audit)
    result["evaluation_runtime_environment"] = runtime_environment
    result["evaluation_runtime_environment_sha256"] = runtime_environment_sha256
    result["evaluation_runtime_compatibility_sha256"] = (
        runtime_compatibility_sha256
    )
    result["result_sha256"] = _canonical_sha256(result)
    if output_path.exists():
        raise RuntimeError(
            f"formal evaluation output appeared during execution: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output_path, flags, 0o600)
    try:
        encoded = (
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        ).encode()
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--bank-set", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--protocol", action="append", default=[])
    args = parser.parse_args()
    result = run_evaluation(
        checkpoint=args.checkpoint,
        manifest_path=args.manifest,
        training_manifest_path=args.training_manifest,
        output_path=args.output,
        bank_set=args.bank_set,
        protocol_names=set(args.protocol),
        dataset_path=args.dataset_path,
    )
    print(
        f"FORMAL_EVAL_DONE task={result['task']} banks={args.bank_set} "
        f"sha256={result['result_sha256']} output={args.output}"
    )


if __name__ == "__main__":
    main()
