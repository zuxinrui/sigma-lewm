#!/usr/bin/env python3
"""Strictly audit completed DINO-WM/PLDM paired compute frontiers.

The raw JSONL records remain the authority.  This script rejects missing cells,
wrong planning settings, incomplete episode arrays, outcome/count mismatches,
and any episode-selection drift relative to the preregistered HF baseline.
"""

from __future__ import annotations
from slewm.paths import resolve_path as _resolve_artifact

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401: registers the compression filters used by pixels
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs/iclr_ct_idm_holdout_v1"
OUT = ROOT / "outputs/tables/external_baseline_audit.json"
PARTIAL_OUT = ROOT / "outputs/tables/external_baseline_audit_partial.json"
SEEDS = (2102905372, 1521362602, 4021952115, 586350286, 2724438844)
STEPS = (1, 2, 4, 8, 30)
MODELS = {
    "dinowm_community": {
        "policy": Path(
            _resolve_artifact('stable:checkpoints/dinowm_patch_prop_pusht')
        ),
        "weight": "weights.pt",
        "sha256": "e569c30a6d570d37b2a61a4dd8930ee65ae30ece66c9590d358e330e970ebe51",
        "config": "config.json",
        "config_sha256": "3e7d7b17dbb10dba6d1bbc11ebe2c4045eed2cc62249974204b378ba8992158c",
        "aux_files": {
            "norm_stats.json": "cf8288f3666805aebb06f7fcae52c87253061639b7b686498eb7dee9652493d0",
            "README.md": "7e30fd713e6da1de4a0bc3e3e8fd1c20228e01005cf5a56b7b30166c56317ea9",
        },
    },
    "pldm_official": {
        "policy": Path(
            _resolve_artifact('stable:checkpoints/pldm_pusht_lewm_paper10_s3072_20260716')
        ),
        "record_policy": Path(
            _resolve_artifact('stable:checkpoints/pldm_pusht_lewm_paper10_s3072_20260716/weights_epoch_10.pt')
        ),
        "weight": "weights_epoch_10.pt",
        "sha256": None,
        "config": "config.yaml",
        "config_sha256": "04d4ec5e5101ce2b6ce154f7ca664f80dd29adf2d3f49279fff1396aaf65fca3",
        "aux_files": {},
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"AUDIT FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_one(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing record: {path}")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"expected exactly one record in {path}, found {len(lines)}")
    try:
        return json.loads(lines[0])
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def nested(config: dict, path: str):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            fail(f"config is missing {path}")
        value = value[key]
    return value


def check_config(model: str, spec: dict) -> dict:
    path = spec["policy"] / spec["config"]
    if not path.is_file():
        fail(f"missing {model} config: {path}")
    digest = sha256(path)
    if digest != spec["config_sha256"]:
        fail(f"{model} config SHA-256 mismatch: {digest}")
    if path.suffix == ".json":
        config = json.loads(path.read_text())
    else:
        config = yaml.safe_load(path.read_text())

    if model == "dinowm_community":
        expected = {
            "_target_": "stable_worldmodel.wm.PreJEPA",
            "encoder.name": "dinov2_small",
            "history_size": 3,
            "num_pred": 1,
            "predictor.num_patches": 256,
            "predictor.dim": 404,
            "predictor.depth": 6,
            "predictor.heads": 16,
            "extra_encoders.modules.action.in_chans": 10,
            "extra_encoders.modules.proprio.in_chans": 4,
        }
    else:
        expected = {
            "trainer.max_epochs": 10,
            "loader.batch_size": 128,
            "optimizer.type": "AdamW",
            "optimizer.lr": 5e-5,
            "optimizer.weight_decay": 1e-3,
            "seed": 3072,
            "wm.history_size": 3,
            "wm.num_preds": 1,
            "wm.embed_dim": 192,
            "wm.use_proprio": False,
            "model.encoder.size": "tiny",
            "model.encoder.patch_size": 14,
            "model.encoder.pretrained": False,
            "model.predictor.depth": 6,
            "model.predictor.heads": 16,
            "loss.std.weight": 18,
            "loss.cov.weight": 12.0,
            "loss.temp_align.weight": 0.2,
            "loss.std_t.weight": 0.7,
            "loss.cov_t.weight": 0.0,
            "loss.idm.weight": 0.0,
            "data.dataset.frameskip": 5,
            "data.dataset.name": "pusht_expert_train.h5",
        }
    for key, expected_value in expected.items():
        actual = nested(config, key)
        if actual != expected_value:
            fail(f"{model} config {key}={actual!r}, expected {expected_value!r}")
    auxiliary = {}
    for name, expected_sha in spec["aux_files"].items():
        aux_path = spec["policy"] / name
        if not aux_path.is_file():
            fail(f"missing {model} auxiliary file: {aux_path}")
        actual_sha = sha256(aux_path)
        if actual_sha != expected_sha:
            fail(f"{model} {name} SHA-256 mismatch: {actual_sha}")
        auxiliary[name] = actual_sha
    if model == "dinowm_community":
        norm = json.loads((spec["policy"] / "norm_stats.json").read_text())
        if norm.get("action", {}).get("method") != "zscore":
            fail("DINO-WM action normalization is not zscore")
        if norm.get("proprio", {}).get("method") != "zscore":
            fail("DINO-WM proprio normalization is not zscore")
        if len(norm["action"]["mean"][0]) != 2 or len(norm["proprio"]["mean"][0]) != 4:
            fail("DINO-WM normalization dimensions do not match action/proprio")
        normalization_compatibility = {}
        public_data = Path(
            _resolve_artifact('stable:datasets/pusht_expert_train.h5')
        )
        with h5py.File(public_data, "r") as handle:
            for column in ("action", "proprio"):
                values = handle[column][:]
                values = values[~np.isnan(values).any(axis=1)]
                evaluator_scaler = StandardScaler().fit(values)
                checkpoint_mean = np.asarray(
                    norm[column]["mean"], dtype=np.float64
                ).reshape(-1)
                checkpoint_scale = np.asarray(
                    norm[column]["std"], dtype=np.float64
                ).reshape(-1)
                standardized_mean_offset = float(
                    np.max(
                        np.abs(checkpoint_mean - evaluator_scaler.mean_)
                        / evaluator_scaler.scale_
                    )
                )
                scale_relative_diff = float(
                    np.max(
                        np.abs(checkpoint_scale - evaluator_scaler.scale_)
                        / evaluator_scaler.scale_
                    )
                )
                if standardized_mean_offset > 2e-3 or scale_relative_diff > 2e-3:
                    fail(
                        f"DINO-WM {column} checkpoint/evaluator normalization "
                        "mismatch is too large: "
                        f"mean={standardized_mean_offset}, scale={scale_relative_diff}"
                    )
                normalization_compatibility[column] = {
                    "checkpoint_mean": checkpoint_mean.tolist(),
                    "checkpoint_scale": checkpoint_scale.tolist(),
                    "unified_evaluator_mean": evaluator_scaler.mean_.tolist(),
                    "unified_evaluator_scale": evaluator_scaler.scale_.tolist(),
                    "max_standardized_mean_offset": standardized_mean_offset,
                    "max_scale_relative_diff": scale_relative_diff,
                }
    result = {
        "path": str(path),
        "sha256": digest,
        "checked_fields": expected,
        "auxiliary_sha256": auxiliary,
    }
    if model == "dinowm_community":
        result["normalization_compatibility"] = normalization_compatibility
    if model == "pldm_official":
        from hydra.utils import instantiate

        runtime_path = spec["policy"] / "config.json"
        if not runtime_path.is_file():
            fail(f"missing PLDM evaluator config: {runtime_path}")
        runtime_config = json.loads(runtime_path.read_text())
        resolved_training_config = OmegaConf.to_container(
            OmegaConf.load(path).model, resolve=True
        )
        if runtime_config != resolved_training_config:
            fail("PLDM evaluator config.json differs from resolved config.yaml.model")
        runtime_model = instantiate(runtime_config)
        model_parameter_count = sum(
            parameter.numel() for parameter in runtime_model.parameters()
        )
        if model_parameter_count != 18_034_478:
            fail(f"PLDM evaluator model has {model_parameter_count} parameters")
        result["evaluator_config"] = {
            "path": str(runtime_path),
            "sha256": sha256(runtime_path),
            "matches_resolved_training_model": True,
            "model_parameter_count": model_parameter_count,
        }
    return result


def check_weight_structure(model: str, path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not torch.is_tensor(v) for v in state.values()):
        fail(f"{model} weight is not a tensor-only state dict")
    tensor_count = len(state)
    tensor_elements = sum(value.numel() for value in state.values())
    expected = (
        (302, 45_791_000)
        if model == "dinowm_community"
        else (303, 18_042_672)
    )
    if (tensor_count, tensor_elements) != expected:
        fail(
            f"{model} weight structure mismatch: "
            f"{(tensor_count, tensor_elements)} != {expected}"
        )
    result = {
        "tensor_count": tensor_count,
        "tensor_elements_including_buffers": tensor_elements,
    }
    if model == "dinowm_community":
        import stable_worldmodel as swm

        runtime_model = swm.wm.utils.load_pretrained(str(path.parent))
        if type(runtime_model).__name__ != "PreJEPA":
            fail(
                "DINO-WM runtime type is "
                f"{type(runtime_model).__module__}.{type(runtime_model).__name__}"
            )
        parameter_count = sum(
            parameter.numel() for parameter in runtime_model.parameters()
        )
        if parameter_count != 42_252_056:
            fail(f"DINO-WM runtime model has {parameter_count} parameters")
        result.update(
            {
                "runtime_type": (
                    f"{type(runtime_model).__module__}."
                    f"{type(runtime_model).__name__}"
                ),
                "runtime_parameter_count": parameter_count,
                "strict_runtime_load": True,
            }
        )
    return result


def check_pldm_source_and_data() -> dict:
    source = Path(_resolve_artifact('legacy:stable-worldmodel-src'))
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_commit = "0a63b2bb8040224a4c3ee0fe480e6db181ee061d"
    if commit != expected_commit:
        fail(f"PLDM source commit={commit}, expected {expected_commit}")
    changed = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--name-only"], text=True
    ).splitlines()
    if changed != ["scripts/train/pldm.py"]:
        fail(f"unexpected PLDM source modifications: {changed}")
    source_text = (source / "scripts/train/pldm.py").read_text()
    if "weights_only=False" not in source_text:
        fail("PLDM source is missing the documented full-state resume fix")

    # The pinned upstream training entry point uses cfg.seed=3072 for the
    # random_split/DataLoader generator but does not pass it to Manager.  Since
    # the model is instantiated before Manager.__call__, Manager's eventual
    # seed_everything(None)->0 cannot retroactively seed parameter
    # initialization.  Preserve and disclose this official-code behavior
    # rather than silently changing the training recipe after the run began.
    seed_source_invariants = {
        "split_and_loader_use_config_seed": (
            "torch.Generator().manual_seed(cfg.seed)" in source_text
            and "DataLoader(train_set, **cfg.loader, generator=rnd_gen)"
            in source_text
        ),
        "model_is_instantiated_before_manager": (
            source_text.index("world_model = hydra.utils.instantiate(cfg.model)")
            < source_text.index("manager = spt.Manager(")
        ),
        "manager_is_not_passed_config_seed": (
            "seed=cfg.seed" not in source_text[
                source_text.index("manager = spt.Manager(") :
                source_text.index("manager()")
            ]
        ),
    }
    if not all(seed_source_invariants.values()):
        fail(f"PLDM upstream seed-plumbing behavior changed: {seed_source_invariants}")
    train_log = ROOT / "logs/pldm_pusht_lewm_paper10_s3072_20260716.log"
    if not train_log.is_file():
        fail(f"missing PLDM training log: {train_log}")
    train_log_text = train_log.read_text(errors="replace")
    seed_log_invariants = {
        "manager_warned_seed_missing": (
            "User didn't specify seed, runs won't be exactly reproducible!"
            in train_log_text
        ),
        "manager_recorded_seed_none": "seed: None" in train_log_text,
        "manager_fell_back_to_zero": "Seed set to 0" in train_log_text,
    }
    if not all(seed_log_invariants.values()):
        fail(f"PLDM training seed log evidence changed: {seed_log_invariants}")

    dataset = Path(
        _resolve_artifact('stable:datasets/pusht_expert_train.h5')
    )
    sampled_digest = hashlib.sha256()
    with h5py.File(dataset, "r") as handle:
        ep_len = handle["ep_len"][:]
        episodes = int(len(ep_len))
        frames = int(ep_len.sum())
        windows = int((ep_len - 19).clip(min=0).sum())
        if handle["pixels"].shape[0] != frames:
            fail("PLDM dataset pixel count differs from sum(ep_len)")

        expected_layout = {
            "action": ((2_336_736, 2), "float32"),
            "ep_len": ((18_685,), "int32"),
            "ep_offset": ((18_685,), "int64"),
            "episode_idx": ((2_336_736,), "int64"),
            "pixels": ((2_336_736, 224, 224, 3), "uint8"),
            "proprio": ((2_336_736, 4), "float32"),
            "state": ((2_336_736, 7), "float32"),
            "step_idx": ((2_336_736,), "int64"),
        }
        actual_layout = {
            key: (tuple(handle[key].shape), str(handle[key].dtype))
            for key in sorted(handle.keys())
        }
        if actual_layout != expected_layout:
            fail(f"PLDM public dataset layout mismatch: {actual_layout}")

        # Training normalizes each raw 2-D action before Dataset.__getitem__
        # reshapes five consecutive actions into one 10-D macro action.  The
        # official eval_wm.py (and our paired evaluator) deliberately rebuilds
        # sklearn's StandardScaler from the same public HDF5.  NumPy float32
        # reductions and sklearn float64 reductions are not bit-identical, so
        # audit both the small numerical gap and, more importantly, the exact
        # row-major macro-action ordering used by training, CEM and execution.
        actions = handle["action"][:]
        episode_idx = handle["episode_idx"][:]
        step_idx = handle["step_idx"][:]

        from stable_worldmodel.data.normalization import ZScoreScaler

        train_scaler = ZScoreScaler().fit(actions)
        eval_scaler = StandardScaler().fit(actions)
        mean_abs_diff = float(
            np.max(np.abs(train_scaler.mean.reshape(-1) - eval_scaler.mean_))
        )
        scale_abs_diff = float(
            np.max(np.abs(train_scaler.std.reshape(-1) - eval_scaler.scale_))
        )
        scale_rel_diff = float(
            np.max(
                np.abs(train_scaler.std.reshape(-1) - eval_scaler.scale_)
                / eval_scaler.scale_
            )
        )
        if mean_abs_diff > 1e-5 or scale_rel_diff > 2e-3:
            fail(
                "PLDM train/eval action normalization drift is larger than "
                f"the pinned official behavior: mean={mean_abs_diff}, "
                f"relative scale={scale_rel_diff}"
            )

        continuous = np.flatnonzero(
            (episode_idx[:-14] == episode_idx[14:])
            & (step_idx[14:] == step_idx[:-14] + 14)
        )
        if len(continuous) == 0:
            fail("PLDM dataset has no continuous 15-action training window")
        start = int(continuous[0])
        raw_window = actions[start : start + 15]
        train_blocks = train_scaler.transform(raw_window).reshape(3, 10)
        eval_blocks = np.stack(
            [
                eval_scaler.transform(raw_window[offset : offset + 5]).reshape(-1)
                for offset in (0, 5, 10)
            ]
        )
        eval_direct = eval_scaler.transform(raw_window).reshape(3, 10)
        if not np.array_equal(eval_blocks, eval_direct):
            fail("PLDM evaluator macro-action construction changes row-major order")
        recovered = eval_scaler.inverse_transform(eval_blocks.reshape(-1, 2))
        recovery_max_abs_error = float(np.max(np.abs(recovered - raw_window)))
        if recovery_max_abs_error > 1e-6:
            fail(
                "PLDM CEM macro-action flatten/execution inverse is not lossless: "
                f"{recovery_max_abs_error}"
            )
        standardized_block_max_abs_diff = float(
            np.max(np.abs(train_blocks - eval_blocks))
        )
        if standardized_block_max_abs_diff > 5e-3:
            fail(
                "PLDM official train/eval standardized macro-action gap is too large: "
                f"{standardized_block_max_abs_diff}"
            )

        dataset_source = (source / "stable_worldmodel/data/dataset.py").read_text()
        hdf5_source = (
            source / "stable_worldmodel/data/formats/hdf5.py"
        ).read_text()
        cem_source = (
            source / "stable_worldmodel/planning/solver/cem.py"
        ).read_text()
        policy_source = (source / "stable_worldmodel/policy.py").read_text()
        eval_source = (source / "scripts/plan/eval_wm.py").read_text()
        source_invariants = {
            "training_transform_precedes_macro_reshape": (
                "steps = self._load_slice" in dataset_source
                and "steps['action'] = steps['action'].reshape(self.num_steps, -1)"
                in dataset_source
                and "return self.transform(steps) if self.transform else steps"
                in hdf5_source
            ),
            "raw_actions_are_not_frameskipped_before_reshape": (
                "if col != 'action':" in hdf5_source
                and "data = data[:: self.frameskip]" in hdf5_source
            ),
            "cem_macro_dim_is_env_dim_times_action_block": (
                "return self._action_dim * self._config.action_block" in cem_source
            ),
            "policy_executes_row_major_macro_actions": (
                "plan = plan.reshape(" in policy_source
                and "self.flatten_receding_horizon" in policy_source
                and "inverse_transform(action)" in policy_source
            ),
            "official_eval_uses_sklearn_standard_scaler": (
                "processor = preprocessing.StandardScaler()" in eval_source
                and "processor.fit(col_data)" in eval_source
            ),
        }
        if not all(source_invariants.values()):
            fail(f"PLDM action-interface source invariant failed: {source_invariants}")

        # Hash complete trajectory indexing plus deterministic, evenly spaced
        # samples from every model input.  A full pixel-file hash would reread
        # hundreds of GB and contend with training; this compact fingerprint
        # catches same-size substitutions while keeping the final gate cheap.
        for key in ("ep_len", "ep_offset"):
            value = handle[key][:]
            sampled_digest.update(key.encode())
            sampled_digest.update(str(value.shape).encode())
            sampled_digest.update(str(value.dtype).encode())
            sampled_digest.update(value.tobytes())
        indices = np.unique(np.linspace(0, frames - 1, 1024, dtype=np.int64))
        for key in ("action", "proprio", "state"):
            value = handle[key][indices]
            sampled_digest.update(key.encode())
            sampled_digest.update(indices.tobytes())
            sampled_digest.update(str(value.shape).encode())
            sampled_digest.update(str(value.dtype).encode())
            sampled_digest.update(value.tobytes())
        pixel_indices = np.unique(
            np.linspace(0, frames - 1, 32, dtype=np.int64)
        )
        value = handle["pixels"][pixel_indices]
        sampled_digest.update(b"pixels")
        sampled_digest.update(pixel_indices.tobytes())
        sampled_digest.update(str(value.shape).encode())
        sampled_digest.update(str(value.dtype).encode())
        sampled_digest.update(value.tobytes())
    expected = (18_685, 2_336_736, 1_981_721)
    if (episodes, frames, windows) != expected:
        fail(
            "PLDM public dataset identity mismatch: "
            f"{(episodes, frames, windows)} != {expected}"
        )
    sampled_sha256 = sampled_digest.hexdigest()
    expected_sampled_sha256 = (
        "6988f8bc9f314cb02a128ece46f18cbacc30aa0ea0b351ae6538bcce3a11ea9f"
    )
    if sampled_sha256 != expected_sampled_sha256:
        fail(f"PLDM public dataset sampled-content SHA-256={sampled_sha256}")
    return {
        "source_commit": commit,
        "source_modification": "resume-only: scripts/train/pldm.py weights_only=False",
        "training_randomness": {
            "config_seed": 3072,
            "config_seed_scope": "train/validation split and training DataLoader generator",
            "manager_seed_argument": None,
            "manager_runtime_fallback_seed": 0,
            "parameter_initialization": (
                "instantiated before Manager seeding; not bit-exact reproducible "
                "from config alone in the pinned upstream entry point"
            ),
            "fixed_artifact_identity": "final checkpoint SHA-256",
            "source_invariants": seed_source_invariants,
            "log_invariants": seed_log_invariants,
            "training_log": str(train_log),
        },
        "dataset": str(dataset),
        "stored_episodes": episodes,
        "stored_frames": frames,
        "valid_sequence_windows": windows,
        "sampled_content_sha256": sampled_sha256,
        "action_interface": {
            "macro_action_order": "[a_t_x,a_t_y,...,a_t+4_x,a_t+4_y]",
            "train_scaler": "stable_worldmodel.data.ZScoreScaler (NumPy float32 reduction)",
            "eval_scaler": "sklearn.preprocessing.StandardScaler (official eval_wm.py behavior)",
            "train_mean": train_scaler.mean.reshape(-1).tolist(),
            "eval_mean": eval_scaler.mean_.tolist(),
            "train_scale": train_scaler.std.reshape(-1).tolist(),
            "eval_scale": eval_scaler.scale_.tolist(),
            "mean_max_abs_diff": mean_abs_diff,
            "scale_max_abs_diff": scale_abs_diff,
            "scale_max_relative_diff": scale_rel_diff,
            "standardized_block_max_abs_diff": standardized_block_max_abs_diff,
            "roundtrip_max_abs_error": recovery_max_abs_error,
            "source_invariants": source_invariants,
        },
    }


def check_protocol(record: dict, model: str, steps: int, seed: int, path: Path) -> None:
    expected = {
        "label": f"{model}_h5_cem{steps}_s{seed}",
        "seed": seed,
        "horizon": 5,
        "receding_horizon": 5,
        "goal_cost": "terminal",
        "n_steps": steps,
        "num_samples": 300,
        "num_eval": 50,
        "eval_budget": 50,
        "goal_offset_steps": 25,
        "arrival": 5,
        "proposal": False,
        "acid_lam": 0.0,
        "energy_acid_lam": 0.0,
        "gcidm_acid_lam": 0.0,
        "native_flow_proposal": False,
        "native_flow_lam": 0.0,
        "ensemble_beta": 0.0,
        "ensemble_ckpts": [],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            fail(f"{path.name}: {key}={record.get(key)!r}, expected {value!r}")

    # New records carry this field directly.  Earlier records predate the
    # metadata addition, so their top-k=30 is grounded by the pinned runner and
    # config/eval/solver/cem.yaml; absence is recorded in the audit artifact.
    if "num_elites" in record and record["num_elites"] != 30:
        fail(f"{path.name}: num_elites={record['num_elites']}, expected 30")

    metrics = record.get("metrics", {})
    outcomes = metrics.get("episode_successes")
    if not isinstance(outcomes, list) or len(outcomes) != 50:
        fail(f"{path.name}: episode_successes must contain 50 entries")
    if any(type(item) is not bool for item in outcomes):
        fail(f"{path.name}: episode_successes contains a non-boolean value")
    expected_rate = 100.0 * sum(outcomes) / len(outcomes)
    if not math.isclose(float(metrics.get("success_rate", -1)), expected_rate):
        fail(f"{path.name}: success_rate does not match binary outcomes")
    for key in ("chosen_rows", "eval_episodes", "eval_starts"):
        if not isinstance(record.get(key), list) or len(record[key]) != 50:
            fail(f"{path.name}: {key} must contain 50 entries")


def audit_model(model: str, selected_steps: tuple[int, ...]) -> dict:
    spec = MODELS[model]
    policy = spec["policy"]
    record_policy = spec.get("record_policy", policy)
    weight_path = policy / spec["weight"]
    if not weight_path.is_file():
        fail(f"missing {model} weight: {weight_path}")
    weight_sha = sha256(weight_path)
    if spec["sha256"] is not None and weight_sha != spec["sha256"]:
        fail(f"{model} SHA-256 mismatch: {weight_sha}")

    config_audit = check_config(model, spec)
    weight_structure = check_weight_structure(model, weight_path)
    reproduction_audit = check_pldm_source_and_data() if model == "pldm_official" else None

    records = []
    missing_elite_metadata = []
    missing_policy_metadata = []
    by_step = {}
    for steps in selected_steps:
        rates = []
        for seed in SEEDS:
            path = LOG_DIR / f"{model}_h5_cem{steps}_s{seed}.jsonl"
            record = load_one(path)
            check_protocol(record, model, steps, seed, path)

            policy_meta = record.get("policy")
            if policy_meta is None:
                log_path = path.with_suffix(".log")
                needle = (
                    f"[eval_ext] loaded {record_policy}:"
                    if model == "pldm_official"
                    else f"Loading checkpoint from folder {record_policy}"
                )
                if not log_path.is_file() or needle not in log_path.read_text(errors="ignore"):
                    fail(f"{path.name}: missing policy metadata and log provenance")
                missing_policy_metadata.append(path.name)
            elif Path(policy_meta) != record_policy:
                fail(
                    f"{path.name}: policy={policy_meta!r}, "
                    f"expected {str(record_policy)!r}"
                )
            if "num_elites" not in record:
                missing_elite_metadata.append(path.name)

            hf_path = LOG_DIR / f"hf_h5_cem{steps}_s{seed}.jsonl"
            hf = load_one(hf_path)
            for key in ("chosen_rows", "eval_episodes", "eval_starts"):
                if record[key] != hf.get(key):
                    fail(f"{path.name}: paired {key} differs from {hf_path.name}")

            rates.append(float(record["metrics"]["success_rate"]))
            records.append(path.name)
        by_step[str(steps)] = {
            "seed_rates_pct": rates,
            "mean_pct": statistics.mean(rates),
            "sample_std_pct": statistics.stdev(rates),
        }

    return {
        "status": "PASS",
        "policy": str(record_policy),
        "checkpoint_dir": str(policy),
        "weight": str(weight_path),
        "weight_sha256": weight_sha,
        "weight_structure": weight_structure,
        "config": config_audit,
        "reproduction": reproduction_audit,
        "records": len(records),
        "episodes": len(records) * 50,
        "missing_policy_metadata_grounded_by_log": missing_policy_metadata,
        "missing_num_elites_metadata_grounded_by_pinned_cem_config": missing_elite_metadata,
        "frontier": by_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", action="append", choices=tuple(MODELS), required=True
    )
    parser.add_argument(
        "--cem-step", action="append", type=int, choices=STEPS,
        help="audit only selected completed CEM columns (default: full frontier)",
    )
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.model))
    selected_steps = (
        tuple(dict.fromkeys(args.cem_step)) if args.cem_step else STEPS
    )
    result = {
        "status": "PASS",
        "protocol": {
            "horizon": 5,
            "receding_horizon": 5,
            "cem_iterations": list(selected_steps),
            "seeds": list(SEEDS),
            "episodes_per_seed": 50,
            "num_samples": 300,
            "num_elites": 30,
            "goal_cost": "terminal",
        },
        "models": {
            model: audit_model(model, selected_steps) for model in selected
        },
    }
    out = OUT if selected_steps == STEPS else PARTIAL_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"AUDIT PASS: {', '.join(selected)} -> {out}")


if __name__ == "__main__":
    main()
