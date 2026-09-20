import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def git_tracked_tree_sha256(repo: Path) -> str:
    """Hash the path and content of every tracked file in the worktree."""
    repo = Path(repo).resolve()
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo)
    digest = hashlib.sha256()
    for encoded_path in raw.split(b"\0"):
        if not encoded_path:
            continue
        relative = encoded_path.decode()
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"tracked source file is missing: {relative}")
        digest.update(encoded_path)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_tree_sha256(module_name: str) -> str:
    module = importlib.import_module(module_name)
    root = Path(module.__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    if not files:
        raise RuntimeError(f"package source tree is empty: {module_name} at {root}")
    for path in files:
        relative = path.relative_to(root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def runtime_environment_fingerprint() -> dict:
    """Fingerprint the software and accelerator stack defining training semantics.

    Formal paired runs compare the hash of this payload.  Recording only the
    Python packages is insufficient: changing the GPU family, driver, cuDNN, or
    NCCL can alter mixed-precision/DDP numerics while leaving the Python
    environment untouched.
    """
    distributions = {}
    for distribution, module_name in (
        ("stable-worldmodel", "stable_worldmodel"),
        ("stable-pretraining", "stable_pretraining"),
    ):
        distributions[distribution] = {
            "version": importlib.metadata.version(distribution),
            "source_tree_sha256": _package_tree_sha256(module_name),
        }
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append({
                "index": int(index),
                "name": str(properties.name),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "total_memory_bytes": int(properties.total_memory),
            })

    driver_versions = []
    try:
        raw_versions = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        driver_versions = sorted({
            version.strip() for version in raw_versions.splitlines()
            if version.strip()
        })
    except (FileNotFoundError, subprocess.CalledProcessError):
        # CPU-only tests and environments without nvidia-smi remain auditable.
        driver_versions = []

    num_devices = len(cuda_devices)
    visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_spec:
        visible_tokens = [token.strip() for token in visible_spec.split(",")]
    else:
        visible_tokens = [str(index) for index in range(num_devices)]
    physical_indices = (
        [int(token) for token in visible_tokens]
        if len(visible_tokens) == num_devices
        and all(token.isdigit() for token in visible_tokens)
        else None
    )
    if physical_indices is None and visible_spec and len(visible_tokens) == num_devices:
        try:
            raw_inventory = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            uuid_to_index = {}
            for line in raw_inventory.splitlines():
                fields = [field.strip() for field in line.split(",", 1)]
                if len(fields) == 2 and fields[0].isdigit():
                    uuid_to_index[fields[1]] = int(fields[0])
            mapped = [uuid_to_index.get(token) for token in visible_tokens]
            if all(index is not None for index in mapped):
                physical_indices = mapped
        except (FileNotFoundError, subprocess.CalledProcessError):
            physical_indices = None

    gpu_topology = None
    try:
        raw_topology = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        # Keep only the GPU-to-GPU matrix.  CPU affinity can vary with a job's
        # scheduler allocation even when the accelerator topology is identical.
        topology_by_physical_row = {}
        for line in raw_topology.splitlines():
            fields = line.split()
            label = fields[0] if fields else ""
            if (
                label.startswith("GPU")
                and label[3:].isdigit()
            ):
                topology_by_physical_row[int(label[3:])] = fields[1:]
        topology_rows = []
        if physical_indices is not None:
            for logical_row, physical_row in enumerate(physical_indices):
                row = topology_by_physical_row.get(physical_row)
                if row is None or any(index >= len(row) for index in physical_indices):
                    topology_rows = []
                    break
                topology_rows.append([
                    f"GPU{logical_row}",
                    *[row[index] for index in physical_indices],
                ])
        gpu_topology = topology_rows or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu_topology = None

    nccl_version = None
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        try:
            raw_nccl_version = torch.cuda.nccl.version()
            if isinstance(raw_nccl_version, tuple):
                nccl_version = list(raw_nccl_version)
            else:
                nccl_version = str(raw_nccl_version)
        except (AttributeError, RuntimeError):
            nccl_version = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": cuda_devices,
        "cuda_visible_devices": physical_indices,
        "nvidia_driver_versions": driver_versions,
        "gpu_topology": gpu_topology,
        "cudnn_version": (
            None if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version())
        ),
        "nccl_version": nccl_version,
        "lightning": importlib.metadata.version("lightning"),
        "packages": distributions,
    }


def runtime_environment_compatibility_fingerprint(environment: dict) -> dict:
    """Project a runtime fingerprint onto inference-semantic compatibility.

    Formal training still compares the complete fingerprint, including the
    two-rank visible-device list and GPU topology.  Evaluation is deliberately
    single-process and may expose one of the same accelerator type.  For that
    boundary, retain the complete software/driver stack and accelerator
    properties while ignoring only device multiplicity, logical/physical
    indices, and inter-GPU topology.
    """
    if not isinstance(environment, dict):
        raise TypeError("runtime environment fingerprint must be a dictionary")
    device_types = []
    for device in environment.get("cuda_devices", []):
        if not isinstance(device, dict):
            raise TypeError("runtime cuda_devices entries must be dictionaries")
        signature = {
            "name": device.get("name"),
            "compute_capability": device.get("compute_capability"),
            "total_memory_bytes": device.get("total_memory_bytes"),
        }
        if signature not in device_types:
            device_types.append(signature)
    device_types.sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    ignored = {
        "cuda_devices",
        "cuda_visible_devices",
        "gpu_topology",
    }
    compatible = {
        key: value for key, value in environment.items() if key not in ignored
    }
    compatible["cuda_device_types"] = device_types
    return compatible

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(
    dataset,
    source: str,
    target: str,
    *,
    episode_indices=None,
    return_stats: bool = False,
):
    """Get a column normalizer, optionally fitted on selected episodes only."""
    mean, std = get_column_stats(
        dataset, source, episode_indices=episode_indices
    )
    transform = dt.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std), source=source, target=target
    )
    if return_stats:
        return transform, mean, std
    return transform


def get_column_stats(dataset, source: str, *, episode_indices=None):
    """Return finite source-only statistics with safe constant dimensions.

    ``episode_indices`` refers to the dataset's internal episode indices.  When
    supplied, statistics are accumulated only from the corresponding raw row
    ranges, avoiding validation-episode leakage without materialising a second
    copy of a potentially large column.
    """

    # Exact-alignment readers may carry a rectangular placeholder on the final
    # observation row of each episode.  Let them define the statistically valid
    # row set so z-scoring does not see fake actions.
    dataset_stats = getattr(dataset, "get_column_stats", None)
    if callable(dataset_stats):
        return dataset_stats(source, episode_indices=episode_indices)

    col_data = dataset.get_col_data(source)
    if episode_indices is not None:
        if not hasattr(dataset, "offsets") or not hasattr(dataset, "lengths"):
            raise TypeError(
                "episode-restricted normalization requires dataset offsets/lengths"
            )
        episode_indices = [int(ep) for ep in episode_indices]
        if not episode_indices:
            raise ValueError("episode-restricted normalization received no episodes")

        feature_shape = tuple(col_data.shape[1:])
        feature_dim = int(np.prod(feature_shape)) if feature_shape else 1
        total = np.zeros(feature_dim, dtype=np.float64)
        total_sq = np.zeros(feature_dim, dtype=np.float64)
        count = 0
        for ep in episode_indices:
            start = int(dataset.offsets[ep])
            end = start + int(dataset.lengths[ep])
            chunk = np.asarray(col_data[start:end]).reshape(-1, feature_dim)
            finite = np.isfinite(chunk).all(axis=1)
            chunk = chunk[finite]
            if chunk.size == 0:
                continue
            chunk64 = chunk.astype(np.float64, copy=False)
            total += chunk64.sum(axis=0)
            total_sq += np.square(chunk64).sum(axis=0)
            count += int(chunk64.shape[0])
        if count == 0:
            raise ValueError(f"no finite rows found for column {source!r}")

        mean_np = total / count
        if count > 1:
            # Match torch.std's default unbiased sample variance.
            var_np = (total_sq - count * np.square(mean_np)) / (count - 1)
        else:
            var_np = np.zeros_like(mean_np)
        std_np = np.sqrt(np.maximum(var_np, 0.0))
        mean = torch.from_numpy(mean_np.reshape((1,) + feature_shape)).float()
        std = torch.from_numpy(std_np.reshape((1,) + feature_shape)).float()
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return mean, std

    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained.

    Also supports STEP-based saves (2026-07-04) so short-wall Snellius runs (1h/2h)
    can still write a resumable ckpt before wall-kill. Set env LEWM_STEP_INTERVAL=N
    to save every N training steps (0 = disabled)."""

    _SOURCE_VALIDATION_KEYS = (
        "validate/loss",
        "validate/pred_loss",
        "validate/teacher_pred_loss",
        "validate/rollout_pred_loss",
        "validate/terminal_pred_loss",
        "validate/sigreg_loss",
    )
    _OPTIONAL_SOURCE_VALIDATION_KEYS = (
        "validate/base_pred_loss",
        "validate/action_rank_loss",
        "validate/action_rank_cross_episode_loss",
        "validate/action_rank_temporal_loss",
    )

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        *,
        epoch_milestones=None,
        resume_subdir: str | None = None,
        save_resume_checkpoint: bool = False,
        audit_source_validation_metrics: bool = False,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.epoch_milestones = (
            None if epoch_milestones is None
            else {int(epoch) for epoch in epoch_milestones}
        )
        if self.epoch_milestones is not None and (
            not self.epoch_milestones or min(self.epoch_milestones) <= 0
        ):
            raise ValueError("epoch_milestones must contain positive epochs")
        self.resume_subdir = resume_subdir
        self.save_resume_checkpoint = save_resume_checkpoint
        self.audit_source_validation_metrics = (
            audit_source_validation_metrics
        )
        import os as _os
        self.step_interval = int(_os.environ.get("LEWM_STEP_INTERVAL", "0"))
        self.keep_step_weights = bool(int(_os.environ.get("LEWM_KEEP_STEP_WEIGHTS", "1")))
        self._last_saved_step = -1

    def _should_save_epoch(self, epoch: int, max_epochs: int) -> bool:
        if self.epoch_milestones is None:
            scheduled = int(epoch) % self.epoch_interval == 0
        else:
            scheduled = int(epoch) in self.epoch_milestones
        return bool(scheduled or int(epoch) == int(max_epochs))

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1
            should_save = self._should_save_epoch(epoch, trainer.max_epochs)
            if should_save:
                checkpoint = self._save(
                    pl_module.model, f"epoch_{epoch}"
                )
                if self.audit_source_validation_metrics:
                    self._record_source_validation_metrics(
                        trainer, checkpoint, epoch
                    )
                if self.save_resume_checkpoint:
                    trainer.save_checkpoint(
                        str(self._resume_checkpoint_path())
                    )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Step-based saves for short-wall runs where epoch boundary is unreachable.
        if self.step_interval <= 0 or not trainer.is_global_zero:
            return
        gs = trainer.global_step
        if gs > 0 and gs % self.step_interval == 0 and gs != self._last_saved_step:
            step_tag = f"step_{gs}" if self.keep_step_weights else "step_latest"
            self._save(pl_module.model, step_tag)
            # Also write Lightning-format ckpt so train_multistep.py's resume path picks it up
            # (train_multistep.py:245-247: `ckpt_path if ckpt_path.exists() else None`)
            try:
                from pathlib import Path
                import stable_worldmodel as _swm
                run_id = getattr(getattr(pl_module, "hparams", object()), "cfg", None)
                # Recover ckpt_path exactly like train_multistep.py's run() does
                run_dir = Path(_swm.data.utils.get_cache_dir(sub_folder='checkpoints'), self._resume_subdir())
                run_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = run_dir / f"{self.run_name}_weights.ckpt"
                trainer.save_checkpoint(str(ckpt_path))
            except Exception as _e:
                print(f"[SaveCkptCallback] Lightning resume-ckpt save failed: {_e}", flush=True)
            self._last_saved_step = gs

    def _resume_subdir(self):
        """Return the subdir under checkpoints/ that train_multistep.py uses at load-time.

        train_multistep.py sets `subdir` via CLI/env; SLURM uses SUBDIR_TAG.
        Falls back to run_name if SUBDIR is unset.
        """
        if self.resume_subdir:
            return self.resume_subdir
        import os as _os
        # SLURM launcher sets SUBDIR_TAG=multistep_${ARM}_s${SEED}_e${EPOCHS} which then becomes cfg.subdir
        return _os.environ.get("LEWM_RESUME_SUBDIR", _os.environ.get("SUBDIR_TAG", self.run_name))

    def _resume_checkpoint_path(self) -> Path:
        import stable_worldmodel as _swm

        run_dir = Path(
            _swm.data.utils.get_cache_dir(sub_folder="checkpoints"),
            self._resume_subdir(),
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / f"{self.run_name}_weights.ckpt"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _record_source_validation_metrics(
        self,
        trainer,
        checkpoint: Path,
        epoch: int,
    ) -> None:
        """Atomically persist source-validation metrics beside each state dict."""

        missing = [
            key
            for key in self._SOURCE_VALIDATION_KEYS
            if key not in trainer.callback_metrics
        ]
        if missing:
            raise RuntimeError(
                "source-validation metric audit is missing epoch aggregates: "
                + ", ".join(missing)
            )
        metrics = {
            key.removeprefix("validate/"): float(
                trainer.callback_metrics[key].detach().cpu()
            )
            for key in self._SOURCE_VALIDATION_KEYS
        }
        metrics.update(
            {
                key.removeprefix("validate/"): float(
                    trainer.callback_metrics[key].detach().cpu()
                )
                for key in self._OPTIONAL_SOURCE_VALIDATION_KEYS
                if key in trainer.callback_metrics
            }
        )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError("source-validation metrics must all be finite")
        config_path = checkpoint.parent / "config.json"
        if not config_path.is_file():
            raise RuntimeError("saved checkpoint is missing config.json")
        ledger_path = checkpoint.parent / "source_validation_metrics.json"
        if ledger_path.is_file():
            document = json.loads(ledger_path.read_text(encoding="utf-8"))
        else:
            document = {
                "schema_version": (
                    "lewm-source-validation-checkpoint-metrics-v1"
                ),
                "selection_scope": (
                    "held-out source-validation trajectories only; never a closed-loop task metric"
                ),
                "primary_endpoint": (
                    "held-out closed-loop task success after action-causality gating; "
                    "not represented by this ledger"
                ),
                "records": [],
            }
        record = {
            "epoch": int(epoch),
            "global_step": int(trainer.global_step),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": self._sha256(checkpoint),
            "config_sha256": self._sha256(config_path),
            "metrics": metrics,
        }
        records = {
            int(item["epoch"]): item for item in document.get("records", [])
        }
        records[int(epoch)] = record
        document["records"] = [
            records[key] for key in sorted(records)
        ]
        temporary = ledger_path.with_name(
            f"{ledger_path.name}.tmp.{os.getpid()}"
        )
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, ledger_path)

    def _save(self, model, tag):
        from stable_worldmodel.wm.utils import save_pretrained
        import stable_worldmodel as _swm

        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_{tag}.pt',
        )
        return Path(
            _swm.data.utils.get_cache_dir(sub_folder="checkpoints"),
            self.run_name,
            f"weights_{tag}.pt",
        )
