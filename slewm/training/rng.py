"""Explicit RNG checkpointing for bit-exact epoch-boundary restoration."""

from __future__ import annotations

import random

import lightning as pl
import numpy as np
import torch


class PaperRNGState(pl.Callback):
    """Restore Python/NumPy/torch RNGs immediately before resumed training."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: dict | None = None

    @property
    def state_key(self) -> str:
        return "paper_rng_state_v1"

    @staticmethod
    def capture() -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }

    @staticmethod
    def restore(state: dict) -> None:
        required = {"python", "numpy", "torch_cpu", "torch_cuda"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("resume checkpoint has an invalid RNG state")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_states = state["torch_cuda"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
            if len(cuda_states) != torch.cuda.device_count():
                raise RuntimeError("CUDA device count differs from RNG checkpoint")
            torch.cuda.set_rng_state_all(cuda_states)

    def state_dict(self) -> dict:
        return {"rng": self.capture()}

    def load_state_dict(self, state_dict: dict) -> None:
        state = state_dict.get("rng")
        if not isinstance(state, dict):
            raise ValueError("resume checkpoint has no explicit RNG state")
        self._pending = state

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self._pending is not None:
            self.restore(self._pending)
            self._pending = None


__all__ = ["PaperRNGState"]
