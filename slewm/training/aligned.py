
from slewm.paths import REPO_ROOT
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from hydra.utils import get_class

from slewm.models.components import SIGReg
from slewm.training.utilities import (
    SaveCkptCallback,
    get_column_normalizer,
    get_column_stats,
    get_img_preprocessor,
)
from slewm.objectives.forward import aligned_horizon_loss
from slewm.objectives.task_constraints import action_contrastive_horizon_loss, franka_terminal_listwise_loss
from slewm.objectives.triadic import aligned_champion_horizon_loss
from slewm.baselines.inverse import external_inverse_horizon_loss
from slewm.training.stopping import LossCompositeEarlyStopping
from slewm.training.sampler import EpochSeededRandomSampler
from slewm.training.rng import PaperRNGState
from slewm.training.telemetry import PaperTrainingTelemetry


def embedding_geometry_metrics(emb):
    """Return scale, effective rank and a robust covariance condition number."""

    with torch.no_grad():
        # AMP may autocast even explicitly-float inputs back to bf16 for the
        # covariance matmul. CUDA eigvalsh has no bf16 kernel, and the paper
        # diagnostic itself should be evaluated in stable float32 arithmetic.
        with torch.autocast(device_type=emb.device.type, enabled=False):
            flattened = emb.detach().float().reshape(-1, emb.shape[-1])
            centered = flattened - flattened.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / max(flattened.shape[0] - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
            trace = eigenvalues.sum().clamp_min(
                torch.finfo(eigenvalues.dtype).eps
            )
            probability = eigenvalues / trace
            entropy = -(
                probability * probability.clamp_min(1.0e-12).log()
            ).sum()
            effective_rank = entropy.exp()
            # The extreme minimum eigenvalue is dominated by finite-batch
            # noise. q90/q10 reacts to broad dimensional collapse without
            # mistaking one null direction for it.
            low = torch.quantile(eigenvalues, 0.10)
            high = torch.quantile(eigenvalues, 0.90)
            condition = high / torch.maximum(low, high * 1.0e-6)
            condition = condition.clamp_min(1.0)
            feature_std = flattened.std(dim=0, unbiased=False)
    return {
        "embedding_std_mean": feature_std.mean(),
        "embedding_std_min": feature_std.min(),
        "embedding_effective_rank": effective_rank,
        "embedding_covariance_condition": condition,
    }


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    if cfg.loss.get("prediction_mode", "legacy") == "aligned_h5":
        return aligned_h5_forward(self, batch, stage, cfg)

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def aligned_h5_forward(self, batch, stage, cfg):
    """Train on six real states and five exactly aligned transition actions."""

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    if batch["pixels"].shape[1] != 6 or batch["action"].shape[1] != 5:
        raise ValueError(
            "aligned_h5 requires exactly six physical states and five actions"
        )
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    champion_cfg = cfg.loss.get("aligned_champion", {})
    external_cfg = cfg.loss.get("external_inverse", {})
    if bool(champion_cfg.get("enabled", False)) and bool(
        external_cfg.get("enabled", False)
    ):
        raise ValueError("champion and external inverse objectives are mutually exclusive")
    if bool(external_cfg.get("enabled", False)):
        prediction = external_inverse_horizon_loss(
            self.model,
            emb,
            act_emb,
            batch["action"],
            max_history=int(cfg.history_size),
            method=str(external_cfg.get("method")),
            inverse_weight=float(external_cfg.get("inverse_weight", 1.0)),
            action_nce_temperature=float(
                external_cfg.get("action_nce_temperature", 0.1)
            ),
        )
    elif bool(champion_cfg.get("enabled", False)):
        prediction = aligned_champion_horizon_loss(
            self.model,
            emb,
            act_emb,
            batch["action"],
            max_history=int(cfg.history_size),
            current_epoch=int(self.current_epoch),
            training=stage == "train",
            random_horizon=bool(champion_cfg.get("random_horizon", True)),
            context_drop_probability=float(
                champion_cfg.get("context_drop_probability", 0.1)
            ),
            diffusion_sigma_max=float(
                champion_cfg.get("diffusion_sigma_max", 0.3)
            ),
            teacher_weight=float(champion_cfg.get("teacher_weight", 1.0)),
            autoregressive_weight=float(
                champion_cfg.get("autoregressive_weight", 1.0)
            ),
            autoregressive_warmup_epochs=int(
                champion_cfg.get("autoregressive_warmup_epochs", 2)
            ),
            inverse_dcl_weight=float(
                champion_cfg.get("inverse_dcl_weight", 0.1)
            ),
            inverse_dcl_temperature=float(
                champion_cfg.get("inverse_dcl_temperature", 0.1)
            ),
            dcl_idm_weight_value=float(
                champion_cfg.get("dcl_idm_weight", 0.05)
            ),
            dcl_idm_temperature=float(
                champion_cfg.get("dcl_idm_temperature", 0.1)
            ),
            dcl_idm_duplicate_cosine=float(
                champion_cfg.get("dcl_idm_duplicate_cosine", 0.98)
            ),
            dcl_idm_symmetric=bool(
                champion_cfg.get("dcl_idm_symmetric", False)
            ),
        )
    else:
        prediction = aligned_horizon_loss(
            self.model,
            emb,
            act_emb,
            max_history=int(cfg.history_size),
            teacher_forced_weight=float(
                cfg.loss.get("teacher_forced_weight", 1.0)
            ),
            free_run_weight=float(cfg.loss.get("free_run_weight", 1.0)),
            terminal_weight=float(cfg.loss.get("terminal_weight", 1.0)),
        )
    for key, value in prediction.items():
        output[key] = value
    action_rank_weight = float(cfg.loss.get("action_rank_weight", 0.0))
    action_rank_probe = bool(cfg.loss.get("action_rank_probe", False))
    compute_action_rank = action_rank_weight > 0 or (
        stage == "validate" and action_rank_probe
    )
    if compute_action_rank:
        if "_episode_index" not in batch:
            raise ValueError("action ranking requires dataset episode identities")
        ranking = action_contrastive_horizon_loss(
            self.model,
            emb,
            batch["action"],
            prediction["rollout_pred_emb"],
            batch["_episode_index"],
            max_history=int(cfg.history_size),
            margin_fraction=float(cfg.loss.get("action_rank_margin_fraction", 0.1)),
            margin_floor=float(cfg.loss.get("action_rank_margin_floor", 1.0e-4)),
            persistence_weight=float(
                cfg.loss.get("action_rank_persistence_weight", 0.0)
            ),
            action_schema=cfg.loss.get("action_rank_schema"),
        )
        prediction_objective = output["pred_loss"]
        if "base_pred_loss" not in output:
            output["base_pred_loss"] = prediction_objective
        output.update(ranking)
        output["pred_loss"] = prediction_objective + (
            action_rank_weight * output["action_rank_loss"]
        )
    terminal_listwise_weight = float(
        cfg.loss.get("franka_terminal_listwise_weight", 0.0)
    )
    terminal_listwise_probe = bool(
        cfg.loss.get("franka_terminal_listwise_probe", False)
    )
    compute_terminal_listwise = terminal_listwise_weight > 0 or (
        stage == "validate" and terminal_listwise_probe
    )
    if compute_terminal_listwise:
        if cfg.loss.get("action_rank_schema") != "franka_stack":
            raise ValueError("terminal listwise loss is Franka-only")
        if "_episode_index" not in batch:
            raise ValueError("terminal listwise loss requires episode identities")
        listwise = franka_terminal_listwise_loss(
            self.model,
            emb,
            batch["action"],
            prediction["rollout_pred_emb"],
            batch["_episode_index"],
            max_history=int(cfg.history_size),
            temperature=float(
                cfg.loss.get("franka_terminal_listwise_temperature", 0.1)
            ),
            scale_floor=float(
                cfg.loss.get("franka_terminal_listwise_scale_floor", 1.0e-4)
            ),
            standardized_zero_action=cfg.loss.get(
                "franka_standardized_zero_action"
            ),
        )
        output.update(listwise)
        output["pred_loss"] = output["pred_loss"] + (
            terminal_listwise_weight
            * output["franka_terminal_listwise_loss"]
        )
    if "action_rank_loss" in output:
        if "franka_terminal_listwise_loss" in output:
            output["task_action_probe_loss"] = 0.5 * (
                output["action_rank_loss"]
                + output["franka_terminal_listwise_loss"]
            )
        else:
            output["task_action_probe_loss"] = output["action_rank_loss"]
    output.update(embedding_geometry_metrics(emb))
    sigreg_weight = float(cfg.loss.sigreg.weight)
    # AC-MTM and SMWM remove SIGReg by definition.  Do not merely multiply its
    # stochastic sketch by zero: sampling the sketch would perturb the global
    # RNG stream (and therefore later dropout masks) while wasting substantial
    # compute, so that would not be a faithful no-SIGReg arm.
    output["sigreg_loss"] = (
        self.sigreg(emb.transpose(0, 1))
        if sigreg_weight > 0
        else emb.new_zeros(())
    )
    output["loss"] = (
        output["pred_loss"]
        + sigreg_weight * output["sigreg_loss"]
    )
    metrics_dict = {
        f"{stage}/{key}": value.detach()
        for key, value in output.items()
        if torch.is_tensor(value) and value.numel() == 1
    }
    # Source-validation checkpoint selection requires an epoch aggregate, not
    # the last validation batch. Keep per-step logging for optimization while
    # making every aligned loss auditable at epoch granularity.
    self.log_dict(
        metrics_dict,
        on_step=stage == "train",
        on_epoch=True,
        sync_dist=True,
    )
    return output


def _load_configured_dataset(dataset_cfg, dataset_factory_cfg, cache_dir):
    """Load a legacy SWM dataset or an explicit opt-in dataset factory."""

    resolved = OmegaConf.to_container(dataset_cfg, resolve=True)
    dataset_name = resolved.pop("name")
    if dataset_factory_cfg is None:
        return swm.data.load_dataset(
            dataset_name,
            transform=None,
            cache_dir=cache_dir,
            **resolved,
        )
    factory = hydra.utils.instantiate(dataset_factory_cfg)
    return factory(
        name=dataset_name,
        cache_dir=cache_dir,
        transform=None,
        **resolved,
    )


def partition_episode_indices(dataset, train_fraction, generator):
    """Return deterministic, disjoint train/validation episode indices."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    episode_count = len(dataset.lengths)
    if episode_count < 2:
        raise ValueError("episode split requires at least two episodes")
    permutation = torch.randperm(episode_count, generator=generator)
    train_count = min(
        max(int(round(train_fraction * episode_count)), 1),
        episode_count - 1,
    )
    train_episodes = permutation[:train_count].tolist()
    val_episodes = permutation[train_count:].tolist()
    return train_episodes, val_episodes


def clip_subset_for_episodes(dataset, episode_indices):
    """Select every clip belonging to the requested dataset episodes."""

    episodes = {int(episode) for episode in episode_indices}
    indices = [
        index
        for index, (episode, _) in enumerate(dataset.clip_indices)
        if episode in episodes
    ]
    if not indices:
        raise RuntimeError("episode selection produced an empty clip subset")
    return torch.utils.data.Subset(dataset, indices)


def split_by_episode(dataset, train_fraction, generator):
    """Create disjoint train/validation clip subsets at episode granularity."""

    train_episodes, val_episodes = partition_episode_indices(
        dataset, train_fraction, generator
    )
    train_set = clip_subset_for_episodes(dataset, train_episodes)
    val_set = clip_subset_for_episodes(dataset, val_episodes)
    return train_set, val_set


def instantiate_component_seeded_model(model_cfg, seed):
    """Instantiate shared LeWM components from method-independent seeds.

    Action-interface variants have different parameter counts. A single
    global RNG stream would therefore give later shared modules (projector and
    prediction projector) different initial weights, confounding a one-seed
    comparison. Each named component receives its own deterministic stream;
    shared component configs then initialize bitwise identically across raw,
    graph, and effect interfaces.
    """

    component_names = (
        "encoder",
        "predictor",
        "action_encoder",
        "projector",
        "pred_proj",
    )
    components = {}
    for index, name in enumerate(component_names, start=1):
        if name not in model_cfg:
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 1009 * index)
            components[name] = hydra.utils.instantiate(model_cfg[name])
    model_class = get_class(str(model_cfg["_target_"]))
    extras = {
        key: value
        for key, value in OmegaConf.to_container(model_cfg, resolve=True).items()
        if key not in {*component_names, "_target_"}
    }
    return model_class(**components, **extras)


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "config/train"), config_name="lewm")
def run(cfg):
    required_cublas = cfg.get("deterministic_cublas_workspace_config")
    if required_cublas is not None and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != str(required_cublas):
        raise RuntimeError(
            "formal deterministic training requires "
            f"CUBLAS_WORKSPACE_CONFIG={required_cublas} before Python starts"
        )
    #########################
    ##       dataset       ##
    #########################

    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset_factory_cfg = cfg.data.get("dataset_factory")
    dataset = _load_configured_dataset(
        cfg.data.dataset,
        dataset_factory_cfg,
        cache_dir,
    )
    validation_dataset = None
    if cfg.data.get("validation_dataset") is not None:
        validation_dataset = _load_configured_dataset(
            cfg.data.validation_dataset,
            dataset_factory_cfg,
            cache_dir,
        )
        if hasattr(dataset, "morphology_names") and hasattr(
            validation_dataset, "morphology_names"
        ):
            from slewm.data.morphopush_v2_dataset import assert_disjoint_morphologies

            assert_disjoint_morphologies(dataset, validation_dataset)

    # Split before fitting any data-dependent transform.  In particular, an
    # episode-disjoint split is not actually leakage-free if the action z-score
    # still sees validation rows.  External validation files use the complete
    # (already isolated) training file for statistics.
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    normalizer_episode_indices = None
    if validation_dataset is not None:
        train_set, val_set = dataset, validation_dataset
    elif cfg.data.get("split_mode", "clip") == "episode":
        train_episodes, val_episodes = partition_episode_indices(
            dataset, cfg.train_split, rnd_gen
        )
        train_set = clip_subset_for_episodes(dataset, train_episodes)
        val_set = clip_subset_for_episodes(dataset, val_episodes)
        normalizer_episode_indices = train_episodes
    else:
        train_set, val_set = spt.data.random_split(
            dataset,
            lengths=[cfg.train_split, 1 - cfg.train_split],
            generator=rnd_gen,
        )

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    context_keys = set(cfg.data.get("context_keys", []))
    no_normalize_keys = context_keys | set(
        cfg.data.get("no_normalize_keys", [])
    )
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            if col in no_normalize_keys:
                continue
            normalizer, column_mean, column_std = get_column_normalizer(
                dataset,
                col,
                col,
                episode_indices=normalizer_episode_indices,
                return_stats=True,
            )
            transforms.append(normalizer)
            if (
                col == "action"
                and "franka_terminal_listwise_probe" in cfg.loss
            ):
                column_mean_tensor = torch.as_tensor(column_mean)
                column_std_tensor = torch.as_tensor(column_std)
                standardized_zero = (
                    torch.zeros_like(column_mean_tensor) - column_mean_tensor
                ) / column_std_tensor
                with open_dict(cfg.loss):
                    cfg.loss.franka_standardized_zero_action = (
                        standardized_zero.reshape(-1).tolist()
                    )

        action_target = str(cfg.model.action_encoder.get("_target_", ""))
        if action_target.endswith(
            ("KinematicEffectEmbedder", "KinematicRobotEffectEmbedder")
        ):
            max_joints = int(dataset.get_dim("action"))
            frameskip = int(cfg.data.dataset.frameskip)
            if "action" in no_normalize_keys:
                action_mean = torch.zeros(max_joints).numpy()
                action_std = torch.ones(max_joints).numpy()
            else:
                action_mean, action_std = get_column_stats(
                    dataset,
                    "action",
                    episode_indices=normalizer_episode_indices,
                )
            cfg.model.action_encoder.max_joints = max_joints
            cfg.model.action_encoder.frameskip = frameskip
            cfg.model.action_encoder.action_mean = (
                action_mean.flatten().repeat(frameskip).tolist()
            )
            cfg.model.action_encoder.action_std = (
                action_std.flatten().repeat(frameskip).tolist()
            )
        else:
            cfg.model.action_encoder.input_dim = (
                cfg.data.dataset.frameskip * dataset.get_dim("action")
            )
        if "champion_action_input_dim" in cfg.model:
            cfg.model.champion_action_input_dim = int(
                cfg.data.dataset.frameskip * dataset.get_dim("action")
            )
        if "inverse_action_dim" in cfg.model:
            cfg.model.inverse_action_dim = int(
                cfg.data.dataset.frameskip * dataset.get_dim("action")
            )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    if validation_dataset is not None:
        # Validation receives transforms fit from training data only.
        validation_dataset.transform = transform

    loader_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    if int(loader_cfg.get("num_workers", 0)) == 0:
        # PyTorch rejects multiprocessing-only knobs when workers are disabled.
        # Keeping this normalization here makes one-batch CPU-loader smoke runs
        # exercise the exact same model/loss path as the formal configuration.
        loader_cfg.pop("prefetch_factor", None)
        loader_cfg["persistent_workers"] = False

    if bool(cfg.data.get("paired_batching", False)):
        if train_set is not dataset or validation_dataset is None:
            raise ValueError(
                "paired batching requires the external v2.2 train dataset"
            )
        from slewm.data.morphopush_v22_counterfactual_dataset import (
            PairedAnchorBatchSampler,
        )

        train_loader_cfg = dict(loader_cfg)
        batch_size = int(train_loader_cfg.pop("batch_size"))
        paired_sampler = PairedAnchorBatchSampler(
            dataset,
            batch_size=batch_size,
            generator=rnd_gen,
            drop_last=True,
        )
        train = torch.utils.data.DataLoader(
            train_set,
            batch_sampler=paired_sampler,
            **train_loader_cfg,
        )
    elif bool(cfg.get("epoch_seeded_sampler", False)):
        train = torch.utils.data.DataLoader(
            train_set,
            **loader_cfg,
            sampler=EpochSeededRandomSampler(
                train_set,
                seed=int(cfg.seed),
                initial_epoch=int(cfg.get("epoch_seeded_sampler_start_epoch", 0)),
            ),
            shuffle=False,
            drop_last=True,
        )
    else:
        train = torch.utils.data.DataLoader(
            train_set,
            **loader_cfg,
            shuffle=True,
            drop_last=True,
            generator=rnd_gen,
        )
    val = torch.utils.data.DataLoader(
        val_set, **loader_cfg, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    if cfg.get("component_seeded_init", False):
        pl.seed_everything(cfg.seed, workers=True)
        world_model = instantiate_component_seeded_model(
            cfg.model, cfg.seed
        )
    else:
        world_model = hydra.utils.instantiate(cfg.model)

    scheduler_config = {"type": "LinearWarmupCosineAnnealingLR"}
    scheduler_horizon_epochs = cfg.get("scheduler_horizon_epochs")
    if scheduler_horizon_epochs is not None:
        scheduler_max_steps = int(scheduler_horizon_epochs) * len(train)
        scheduler_warmup_steps = max(
            1,
            int(
                float(cfg.get("scheduler_warmup_fraction", 0.01))
                * scheduler_max_steps
            ),
        )
        if scheduler_max_steps <= scheduler_warmup_steps:
            raise ValueError("scheduler horizon must exceed warmup")
        scheduler_config.update(
            warmup_steps=scheduler_warmup_steps,
            max_steps=scheduler_max_steps,
            warmup_start_lr=0.0,
            eta_min=0.0,
        )
        with open_dict(cfg):
            cfg.scheduler_steps_per_epoch = len(train)
            cfg.scheduler_max_steps = scheduler_max_steps
            cfg.scheduler_warmup_steps = scheduler_warmup_steps

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": scheduler_config,
            # Historical configs used epoch stepping even though the scheduler
            # derives warmup/max_steps from optimizer-step counts. Formal runs
            # explicitly use optimizer-step scheduling and freeze the horizon
            # at 100 complete epochs, independent of early stopping.
            "interval": str(cfg.get("scheduler_interval", "epoch")),
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=int(cfg.get("save_epoch_interval", 1)),
        resume_subdir=str(cfg.get("subdir") or cfg.output_model_name),
        save_resume_checkpoint=bool(
            cfg.get("save_resume_checkpoint", False)
        ),
        audit_source_validation_metrics=bool(
            cfg.get("audit_source_validation_metrics", False)
        ),
    )

    callbacks = []
    if bool(cfg.get("paper_rng_state", False)):
        callbacks.append(PaperRNGState())
    if bool(cfg.get("paper_training_telemetry", False)):
        # Telemetry is observational only. Put it before stateful control/save
        # callbacks so its epoch record is captured in boundary checkpoints.
        callbacks.append(PaperTrainingTelemetry(run_dir=run_dir))
    early_stopping_cfg = cfg.get("early_stopping", {})
    if bool(early_stopping_cfg.get("enabled", False)):
        weights = OmegaConf.to_container(
            early_stopping_cfg.get("weights", {}), resolve=True
        )
        loss_control_kwargs = {
            "min_epochs": int(early_stopping_cfg.get("min_epochs", 50)),
            "max_epochs": int(early_stopping_cfg.get("max_epochs", 100)),
            "ema_window": int(early_stopping_cfg.get("ema_window", 3)),
            "patience": int(early_stopping_cfg.get("patience", 10)),
            "slope_window": int(early_stopping_cfg.get("slope_window", 5)),
            "min_improvement_fraction": float(
                early_stopping_cfg.get("min_improvement_fraction", 0.005)
            ),
            "degradation_fraction": float(
                early_stopping_cfg.get("degradation_fraction", 0.05)
            ),
            "degradation_patience": int(
                early_stopping_cfg.get("degradation_patience", 5)
            ),
            "collapse_std_ratio": float(
                early_stopping_cfg.get("collapse_std_ratio", 0.10)
            ),
            "collapse_effective_rank_ratio": float(
                early_stopping_cfg.get(
                    "collapse_effective_rank_ratio", 0.10
                )
            ),
            "collapse_condition_growth": float(
                early_stopping_cfg.get("collapse_condition_growth", 10.0)
            ),
            "collapse_patience": int(
                early_stopping_cfg.get("collapse_patience", 3)
            ),
            "epsilon": float(early_stopping_cfg.get("epsilon", 1.0e-12)),
        }
        if weights:
            loss_control_kwargs["weights"] = weights
        component_keys = OmegaConf.to_container(
            early_stopping_cfg.get("component_keys", {}), resolve=True
        )
        callbacks.append(
            LossCompositeEarlyStopping(
                run_dir=run_dir,
                component_keys=(component_keys or None),
                **loss_control_kwargs,
            )
        )
    # The loss controller must update its serializable state before the
    # checkpoint callback captures the epoch-boundary resume checkpoint.
    callbacks.append(object_dump_callback)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.setdefault("num_sanity_val_steps", 1)
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        seed=(
            int(cfg.seed)
            if cfg.get("component_seeded_init", False)
            else None
        ),
        # Historical runs keep the prior weights-only behavior. V2 explicitly
        # opts into full optimizer/scheduler/RNG restoration.
        weights_only=not bool(cfg.get("resume_full_state", False)),
    )

    manager()
    return


if __name__ == "__main__":
    run()
