"""Auditable optimizer/runtime telemetry for the new-task paper matrix."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import time

import lightning as pl
import torch


class PaperTrainingTelemetry(pl.Callback):
    """Persist per-epoch LR, gradient norm, throughput, time and peak memory.

    The callback has no control authority: none of these values can select a
    checkpoint or stop training.  They exist to expose compute/fairness bugs
    and to support the preregistered training visualizations.
    """

    def __init__(self, *, run_dir: str | Path) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.records: list[dict] = []
        self._epoch_started = 0.0
        self._step_started = 0
        self._examples = 0
        self._gradient_norms: list[float] = []
        self._clip_identity_digest = hashlib.sha256()
        self._first_batch_clip_identities: list[list[int]] | None = None

    @property
    def state_key(self) -> str:
        return "paper_training_telemetry_v1"

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_started = time.perf_counter()
        self._step_started = int(trainer.global_step)
        self._examples = 0
        self._gradient_norms = []
        self._clip_identity_digest = hashlib.sha256()
        self._first_batch_clip_identities = None
        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if isinstance(batch, dict) and "pixels" in batch:
            self._examples += int(batch["pixels"].shape[0])
        if isinstance(batch, dict) and {
            "_episode_index",
            "_clip_start",
        } <= set(batch):
            identities = torch.stack(
                (
                    batch["_episode_index"].detach().to("cpu", torch.int64),
                    batch["_clip_start"].detach().to("cpu", torch.int64),
                ),
                dim=1,
            ).contiguous()
            rows = identities.tolist()
            if self._first_batch_clip_identities is None:
                self._first_batch_clip_identities = rows
            self._clip_identity_digest.update(memoryview(identities.numpy()).cast("B"))

    def on_after_backward(self, trainer, pl_module) -> None:
        """Measure the raw norm before Stable Pretraining clips gradients."""

        device = pl_module.device
        squared = torch.zeros((), dtype=torch.float32, device=device)
        found = False
        with torch.no_grad():
            for parameter in pl_module.parameters():
                if parameter.grad is None:
                    continue
                found = True
                gradient = parameter.grad.detach().float()
                squared.add_(gradient.square().sum())
        if found:
            value = float(squared.sqrt().cpu())
            if math.isfinite(value):
                self._gradient_norms.append(value)

    def _write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "training_telemetry.json"
        payload = {
            "schema": "paper_training_telemetry_v1",
            "selection_scope": (
                "audit/visualization only; forbidden for early stopping and checkpoint selection"
            ),
            "gradient_norm_stage": "after_backward_before_gradient_clipping",
            "records": self.records,
        }
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        duration = max(time.perf_counter() - self._epoch_started, 1.0e-9)
        optimizer_steps = int(trainer.global_step) - self._step_started
        learning_rates = [
            float(group["lr"])
            for optimizer in trainer.optimizers
            for group in optimizer.param_groups
        ]
        peak_allocated = None
        peak_reserved = None
        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            peak_allocated = int(
                torch.cuda.max_memory_allocated(pl_module.device)
            )
            peak_reserved = int(
                torch.cuda.max_memory_reserved(pl_module.device)
            )
        gradients = self._gradient_norms
        record = {
            "epoch": int(trainer.current_epoch) + 1,
            "global_step": int(trainer.global_step),
            "optimizer_steps": optimizer_steps,
            "learning_rates": learning_rates,
            "gradient_norm_mean": (
                sum(gradients) / len(gradients) if gradients else None
            ),
            "gradient_norm_max": max(gradients) if gradients else None,
            "clip_identity_sha256": self._clip_identity_digest.hexdigest(),
            "first_batch_clip_identities": self._first_batch_clip_identities,
            "examples": int(self._examples),
            "epoch_wall_seconds": duration,
            "examples_per_second": float(self._examples) / duration,
            "peak_cuda_memory_allocated_bytes": peak_allocated,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
        }
        by_epoch = {int(item["epoch"]): item for item in self.records}
        by_epoch[record["epoch"]] = record
        self.records = [by_epoch[key] for key in sorted(by_epoch)]
        if trainer.is_global_zero:
            self._write()

    def state_dict(self) -> dict:
        return {"records": self.records}

    def load_state_dict(self, state_dict: dict) -> None:
        records = state_dict.get("records")
        if not isinstance(records, list):
            raise ValueError("resume checkpoint has no training telemetry records")
        self.records = list(records)


__all__ = ["PaperTrainingTelemetry"]
