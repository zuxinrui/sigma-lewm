"""Task-aware projected CEM for LeWM planning.

The stock StableWorldModel CEM samples unconstrained standardized actions.  That
is unsuitable for hybrid robot action spaces: clipping only the action sent to
the environment means the world model scores a different action from the one
that is executed, while patching ``model.get_cost`` leaves the elite update and
returned mean unprojected.

This module makes projection part of CEM itself.  Every sampled candidate is
projected *before* scoring, and the scored tensor is also the tensor used for
elite selection, distribution updates, callbacks, and the returned plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.utils import prepare_init_action


TASK_ACTION_SCHEMAS = (
    "generic",
    "cube",
    "gear_effective4",
    "gear_legacy6",
    "gear",  # Backward-compatible alias for gear_legacy6.
    "franka_stack",
)
ACTION_BOUND_MODES = ("none", "physical", "data_minmax", "data_quantile")
TASK_ACTION_DIMS = {
    "cube": 5,
    "gear_effective4": 4,
    "gear_legacy6": 6,
    "gear": 6,
    "franka_stack": 7,
}


def _as_per_dim(
    value: float | Sequence[float], action_dim: int, name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(action_dim, float(array), dtype=np.float64)
    if array.shape != (action_dim,):
        raise ValueError(f"{name} must be scalar or shape ({action_dim},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


class TrainingZScoreScaler:
    """NumPy-compatible scaler matching ``train.py`` action normalization.

    Training computes statistics with ``torch.mean`` and the default unbiased
    ``torch.std`` (sample standard deviation, ddof=1), then replaces standard
    deviations below ``1e-6`` with one.  ``WorldModelPolicy`` only requires the
    sklearn-like ``transform`` and ``inverse_transform`` methods provided here.
    """

    ddof = 1

    def __init__(self, mean: np.ndarray, scale: np.ndarray, sample_count: int) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.n_features_in_ = int(self.mean_.shape[0])
        self.n_samples_seen_ = int(sample_count)
        if self.mean_.ndim != 1 or self.scale_.shape != self.mean_.shape:
            raise ValueError("mean and scale must be same-length vectors")
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.scale_).all():
            raise ValueError("scaler statistics contain non-finite values")
        if np.any(self.scale_ <= 0):
            raise ValueError("scaler standard deviations must be positive")

    @classmethod
    def fit(cls, action_data: np.ndarray) -> "TrainingZScoreScaler":
        data = np.asarray(action_data)
        if data.ndim != 2:
            raise ValueError(f"action_data must be rank 2, got {data.shape}")
        data = data[np.isfinite(data).all(axis=1)]
        if len(data) == 0:
            raise ValueError("action_data has no finite rows")
        # Keep source float precision and operations aligned with
        # utils.get_column_stats rather than silently switching to ddof=0.
        tensor = torch.from_numpy(np.array(data))
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0) if len(data) > 1 else torch.zeros_like(mean)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return cls(
            mean.detach().cpu().numpy(),
            std.detach().cpu().numpy(),
            len(data),
        )

    def _validate(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"expected final dimension {self.n_features_in_}, got {array.shape[-1]}"
            )
        return array

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = self._validate(values)
        return (array - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = self._validate(values)
        return array * self.scale_ + self.mean_

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "ddof": self.ddof,
            "sample_count": self.n_samples_seen_,
        }


@dataclass(frozen=True)
class StandardizedActionProjection:
    """Projection contract for one flattened LeWM macro action.

    ``low``/``high``, ``fixed`` and ``categorical`` are all expressed in the
    standardized space consumed by the world model.  Base-dimension rules are
    repeated independently for each raw controller step in ``action_block``;
    the flattened layout is frame-major, ``[step0_dims, step1_dims, ...]``.
    """

    name: str
    action_dim: int
    action_block: int
    low: tuple[float, ...]
    high: tuple[float, ...]
    raw_low: tuple[float, ...]
    raw_high: tuple[float, ...]
    fixed: tuple[tuple[int, float], ...] = ()
    categorical: tuple[tuple[int, tuple[float, ...]], ...] = ()
    raw_fixed: tuple[tuple[int, float], ...] = ()
    raw_categorical: tuple[tuple[int, tuple[float, ...]], ...] = ()
    single_switch_categorical: tuple[int, ...] = ()
    terminal_open_steps: int = 0
    terminal_switch_steps: tuple[int, ...] = ()
    terminal_switch_counts: tuple[tuple[int, int], ...] = ()
    terminal_support_sample_count: int = 0
    continuous_action_indices: tuple[int, ...] = ()
    continuous_plan_tv_max: float | None = None
    continuous_plan_tv_quantile: float | None = None

    def __post_init__(self) -> None:
        if self.action_dim < 1 or self.action_block < 1:
            raise ValueError("action_dim and action_block must be positive")
        if len(self.low) != self.action_dim or len(self.high) != self.action_dim:
            raise ValueError("standardized bounds must have one entry per action dimension")
        if len(self.raw_low) != self.action_dim or len(self.raw_high) != self.action_dim:
            raise ValueError("raw bounds must have one entry per action dimension")
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        raw_low = np.asarray(self.raw_low, dtype=np.float64)
        raw_high = np.asarray(self.raw_high, dtype=np.float64)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("standardized bounds contain non-finite values")
        if not np.isfinite(raw_low).all() or not np.isfinite(raw_high).all():
            raise ValueError("raw bounds contain non-finite values")
        if np.any(low > high):
            raise ValueError("standardized lower bounds exceed upper bounds")
        if np.any(raw_low > raw_high):
            raise ValueError("raw lower bounds exceed upper bounds")

        fixed_indices: set[int] = set()
        for index, value in self.fixed:
            if not 0 <= index < self.action_dim:
                raise ValueError(f"fixed action index {index} is out of range")
            if index in fixed_indices:
                raise ValueError(f"fixed action index {index} is duplicated")
            if not np.isfinite(value) or not low[index] <= value <= high[index]:
                raise ValueError(f"fixed value for action index {index} is outside bounds")
            fixed_indices.add(index)

        categorical_indices: set[int] = set()
        for index, values in self.categorical:
            if not 0 <= index < self.action_dim:
                raise ValueError(f"categorical action index {index} is out of range")
            if index in fixed_indices or index in categorical_indices:
                raise ValueError(f"action index {index} has conflicting projection rules")
            if len(values) < 2:
                raise ValueError("categorical action dimensions need at least two values")
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all() or len(np.unique(array)) != len(array):
                raise ValueError(f"categorical values for action index {index} are invalid")
            if np.any(array < low[index]) or np.any(array > high[index]):
                raise ValueError(f"categorical values for action index {index} are outside bounds")
            categorical_indices.add(index)

        if tuple(index for index, _ in self.raw_fixed) != tuple(
            index for index, _ in self.fixed
        ):
            raise ValueError("raw and standardized fixed rules do not match")
        if tuple(index for index, _ in self.raw_categorical) != tuple(
            index for index, _ in self.categorical
        ):
            raise ValueError("raw and standardized categorical rules do not match")
        if len(set(self.single_switch_categorical)) != len(
            self.single_switch_categorical
        ):
            raise ValueError("single-switch categorical indices are duplicated")
        for index in self.single_switch_categorical:
            if index not in categorical_indices:
                raise ValueError(
                    "single-switch dimensions must also be categorical"
                )
            values = dict(self.categorical)[index]
            if len(values) != 2:
                raise ValueError(
                    "single-switch categorical dimensions must be binary"
                )
        if self.terminal_open_steps < 0:
            raise ValueError("terminal_open_steps must be non-negative")
        if self.terminal_open_steps and self.name != "franka_stack":
            raise ValueError("terminal open projection is only defined for Franka Stack")
        if self.terminal_switch_steps:
            if self.name != "franka_stack" or not self.terminal_open_steps:
                raise ValueError(
                    "terminal switch steps require Franka terminal-open projection"
                )
            if tuple(sorted(set(self.terminal_switch_steps))) != self.terminal_switch_steps:
                raise ValueError("terminal switch steps must be sorted and unique")
            if self.terminal_switch_steps[0] < 0:
                raise ValueError("terminal switch steps must be non-negative")
        elif self.terminal_open_steps:
            raise ValueError(
                "terminal open projection requires train-fitted switch steps"
            )
        switch_count_map = dict(self.terminal_switch_counts)
        if len(switch_count_map) != len(self.terminal_switch_counts):
            raise ValueError("terminal switch counts contain duplicate steps")
        if switch_count_map:
            if tuple(sorted(switch_count_map)) != self.terminal_switch_steps:
                raise ValueError(
                    "terminal switch count keys must match terminal switch steps"
                )
            if any(count <= 0 for count in switch_count_map.values()):
                raise ValueError("terminal switch counts must be positive")
            if sum(switch_count_map.values()) != self.terminal_support_sample_count:
                raise ValueError(
                    "terminal switch counts do not sum to the support sample count"
                )
        if self.terminal_support_sample_count < 0:
            raise ValueError("terminal support sample count must be non-negative")
        if (self.continuous_plan_tv_max is None) != (
            self.continuous_plan_tv_quantile is None
        ):
            raise ValueError(
                "continuous plan-TV maximum and quantile must be specified together"
            )
        if self.continuous_plan_tv_max is not None:
            if self.name not in {"franka_stack", "gear_effective4"}:
                raise ValueError(
                    "continuous plan-TV projection is only defined for Franka Stack "
                    "and effective-4D Gear"
                )
            if not self.continuous_action_indices:
                raise ValueError(
                    "continuous plan-TV projection needs at least one action index"
                )
            if len(set(self.continuous_action_indices)) != len(
                self.continuous_action_indices
            ):
                raise ValueError("continuous plan-TV action indices are duplicated")
            if any(
                index < 0 or index >= self.action_dim
                for index in self.continuous_action_indices
            ):
                raise ValueError("continuous plan-TV action index is out of range")
            if not np.isfinite(self.continuous_plan_tv_max) or self.continuous_plan_tv_max <= 0:
                raise ValueError("continuous plan-TV maximum must be positive and finite")
            if not 0.0 < self.continuous_plan_tv_quantile <= 1.0:
                raise ValueError("continuous plan-TV quantile must lie in (0,1]")
        elif self.continuous_action_indices:
            raise ValueError(
                "continuous plan-TV action indices require a maximum and quantile"
            )

    @property
    def flat_action_dim(self) -> int:
        return self.action_dim * self.action_block

    def _expanded_bounds(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.as_tensor(self.low, device=device, dtype=dtype).repeat(self.action_block)
        high = torch.as_tensor(self.high, device=device, dtype=dtype).repeat(self.action_block)
        return low, high

    def project(self, actions: torch.Tensor) -> torch.Tensor:
        """Return projected actions without mutating the caller's tensor."""

        if not torch.is_tensor(actions) or not actions.is_floating_point():
            raise TypeError("actions must be a floating-point torch.Tensor")
        if actions.shape[-1] != self.flat_action_dim:
            raise ValueError(
                f"expected flattened action dim {self.flat_action_dim}, got {actions.shape[-1]}"
            )
        low, high = self._expanded_bounds(device=actions.device, dtype=actions.dtype)
        projected = torch.maximum(torch.minimum(actions, high), low)

        single_switch = set(self.single_switch_categorical)
        for block in range(self.action_block):
            offset = block * self.action_dim
            for index, value in self.fixed:
                projected[..., offset + index] = value
            for index, values in self.categorical:
                if index in single_switch:
                    continue
                categories = torch.as_tensor(values, device=actions.device, dtype=actions.dtype)
                current = projected[..., offset + index]
                nearest = (current.unsqueeze(-1) - categories).abs().argmin(dim=-1)
                projected[..., offset + index] = categories[nearest]

        # Franka demonstrations contain only constant or one-transition gripper
        # commands inside each five-step macro (all 38,508 train macros; no
        # reopen/reclose flicker). Project each macro jointly here; the solver's
        # plan-level projection below additionally enforces this over all macros.
        for index in self.single_switch_categorical:
            categories = torch.as_tensor(
                dict(self.categorical)[index],
                device=actions.device,
                dtype=actions.dtype,
            )
            lo, hi = categories[0], categories[1]
            patterns = []
            for split in range(self.action_block + 1):
                patterns.append(
                    torch.cat(
                        (
                            lo.expand(split),
                            hi.expand(self.action_block - split),
                        )
                    )
                )
                patterns.append(
                    torch.cat(
                        (
                            hi.expand(split),
                            lo.expand(self.action_block - split),
                        )
                    )
                )
            patterns = torch.stack(patterns)
            current = torch.stack(
                [
                    projected[..., block * self.action_dim + index]
                    for block in range(self.action_block)
                ],
                dim=-1,
            )
            costs = (current.unsqueeze(-2) - patterns).square().sum(dim=-1)
            chosen = patterns[costs.argmin(dim=-1)]
            for block in range(self.action_block):
                projected[..., block * self.action_dim + index] = chosen[..., block]
        return projected

    def zero_fixed_variance(self, variance: torch.Tensor) -> torch.Tensor:
        """Set variance of dimensions fixed by the schema to exactly zero."""

        if variance.shape[-1] != self.flat_action_dim:
            raise ValueError("variance has the wrong flattened action dimension")
        variance = variance.clone()
        for block in range(self.action_block):
            offset = block * self.action_dim
            for index, _ in self.fixed:
                variance[..., offset + index] = 0.0
        return variance

    def project_raw(self, actions: np.ndarray) -> np.ndarray:
        """Enforce the same schema after inverse scaling in raw action space.

        A float32 standardized category does not necessarily inverse-transform
        to bit-exact raw ``-1``, ``0`` or ``+1``. This final canonicalization
        guarantees that the controller sees exact task-valid values.
        """

        array = np.asarray(actions)
        if array.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected raw action dim {self.action_dim}, got {array.shape[-1]}"
            )
        projected = np.clip(
            array,
            np.asarray(self.raw_low, dtype=array.dtype),
            np.asarray(self.raw_high, dtype=array.dtype),
        )
        projected = np.array(projected, copy=True)
        for index, value in self.raw_fixed:
            projected[..., index] = value
        for index, values in self.raw_categorical:
            categories = np.asarray(values, dtype=projected.dtype)
            nearest = np.abs(projected[..., index, None] - categories).argmin(axis=-1)
            projected[..., index] = categories[nearest]
        return projected

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action_dim": self.action_dim,
            "action_block": self.action_block,
            "standardized_low": list(self.low),
            "standardized_high": list(self.high),
            "raw_low": list(self.raw_low),
            "raw_high": list(self.raw_high),
            "standardized_fixed": {str(index): value for index, value in self.fixed},
            "standardized_categorical": {
                str(index): list(values) for index, values in self.categorical
            },
            "raw_fixed": {str(index): value for index, value in self.raw_fixed},
            "raw_categorical": {
                str(index): list(values) for index, values in self.raw_categorical
            },
            "single_switch_categorical": list(self.single_switch_categorical),
            "global_single_switch_categorical": (
                list(self.single_switch_categorical)
                if self.terminal_open_steps
                else []
            ),
            "terminal_open_steps": int(self.terminal_open_steps),
            "terminal_switch_steps": list(self.terminal_switch_steps),
            "terminal_switch_counts": {
                str(step): count for step, count in self.terminal_switch_counts
            },
            "terminal_support_sample_count": int(self.terminal_support_sample_count),
            "continuous_action_indices": list(self.continuous_action_indices),
            "continuous_plan_tv_max": self.continuous_plan_tv_max,
            "continuous_plan_tv_quantile": self.continuous_plan_tv_quantile,
        }


class ProjectedActionScaler:
    """Training scaler plus exact raw-space enforcement at policy output."""

    def __init__(
        self,
        scaler: TrainingZScoreScaler,
        projection: StandardizedActionProjection,
    ) -> None:
        self.scaler = scaler
        self.projection = projection
        self.mean_ = scaler.mean_
        self.scale_ = scaler.scale_
        self.n_features_in_ = scaler.n_features_in_
        self.n_samples_seen_ = scaler.n_samples_seen_
        self.ddof = scaler.ddof
        if self.n_features_in_ != projection.action_dim:
            raise ValueError("scaler and projection action dimensions do not match")

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.transform(values)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.projection.project_raw(self.scaler.inverse_transform(values))

    def to_dict(self) -> dict[str, Any]:
        return self.scaler.to_dict()


def build_task_action_projection(
    schema: str,
    *,
    action_dim: int,
    action_block: int,
    scaler: Any | None,
    raw_low: float | Sequence[float] = -1.0,
    raw_high: float | Sequence[float] = 1.0,
    terminal_open_steps: int = 0,
    terminal_switch_steps: Sequence[int] = (),
    terminal_switch_counts: Sequence[tuple[int, int]] = (),
    terminal_support_sample_count: int = 0,
    continuous_plan_tv_max: float | None = None,
    continuous_plan_tv_quantile: float | None = None,
) -> StandardizedActionProjection | None:
    """Build a validated task schema in the model's standardized action space.

    ``generic`` deliberately returns ``None`` and therefore retains literal
    stock-CEM behavior.  All task-aware schemas require the exact training
    scaler so raw physical constraints are converted without approximation.
    """

    if schema not in TASK_ACTION_SCHEMAS:
        raise ValueError(f"unknown task action schema {schema!r}")
    if action_dim < 1 or action_block < 1:
        raise ValueError("action_dim and action_block must be positive")
    if schema == "generic":
        return None

    expected = TASK_ACTION_DIMS[schema]
    if action_dim != expected:
        raise ValueError(
            f"task action schema {schema!r} requires action_dim={expected}, got {action_dim}"
        )
    if scaler is None or not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError(f"task action schema {schema!r} requires a fitted action scaler")

    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if mean.shape != (action_dim,) or scale.shape != (action_dim,):
        raise ValueError(
            f"action scaler must have {action_dim} features, got mean={mean.shape}, scale={scale.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("action scaler contains invalid mean/scale values")

    raw_lo = _as_per_dim(raw_low, action_dim, "raw_low")
    raw_hi = _as_per_dim(raw_high, action_dim, "raw_high")
    if np.any(raw_lo > raw_hi):
        raise ValueError("raw lower bounds exceed upper bounds")
    low = (raw_lo - mean) / scale
    high = (raw_hi - mean) / scale

    fixed_raw: Mapping[int, float] = (
        {3: 0.0, 4: 0.0} if schema in {"gear", "gear_legacy6"} else {}
    )
    categorical_raw: Mapping[int, tuple[float, ...]] = (
        {6: (-1.0, 1.0)} if schema == "franka_stack" else {}
    )
    fixed = tuple((index, float((value - mean[index]) / scale[index])) for index, value in fixed_raw.items())
    categorical = tuple(
        (
            index,
            tuple(float((value - mean[index]) / scale[index]) for value in values),
        )
        for index, values in categorical_raw.items()
    )
    return StandardizedActionProjection(
        name=schema,
        action_dim=action_dim,
        action_block=action_block,
        low=tuple(float(value) for value in low),
        high=tuple(float(value) for value in high),
        raw_low=tuple(float(value) for value in raw_lo),
        raw_high=tuple(float(value) for value in raw_hi),
        fixed=fixed,
        categorical=categorical,
        raw_fixed=tuple(fixed_raw.items()),
        raw_categorical=tuple(categorical_raw.items()),
        single_switch_categorical=(6,) if schema == "franka_stack" else (),
        terminal_open_steps=terminal_open_steps,
        terminal_switch_steps=tuple(int(step) for step in terminal_switch_steps),
        terminal_switch_counts=tuple(
            (int(step), int(count)) for step, count in terminal_switch_counts
        ),
        terminal_support_sample_count=int(terminal_support_sample_count),
        continuous_action_indices=(
            tuple(range(6))
            if schema == "franka_stack" and continuous_plan_tv_max is not None
            else tuple(range(4))
            if schema == "gear_effective4" and continuous_plan_tv_max is not None
            else ()
        ),
        continuous_plan_tv_max=continuous_plan_tv_max,
        continuous_plan_tv_quantile=continuous_plan_tv_quantile,
    )


def resolve_raw_action_bounds(
    mode: str,
    *,
    action_dim: int,
    physical_low: float | Sequence[float],
    physical_high: float | Sequence[float],
    action_data: np.ndarray | None = None,
    quantile: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve explicit raw-space support bounds for task-aware planning."""

    if mode not in ACTION_BOUND_MODES:
        raise ValueError(f"unknown action bound mode {mode!r}")
    if mode == "none":
        raise ValueError("bound mode 'none' has no raw action bounds")
    if mode == "physical":
        low = _as_per_dim(physical_low, action_dim, "physical_low")
        high = _as_per_dim(physical_high, action_dim, "physical_high")
    else:
        if action_data is None:
            raise ValueError(f"bound mode {mode!r} requires training action data")
        data = np.asarray(action_data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != action_dim:
            raise ValueError(
                f"training action data must have shape (N,{action_dim}), got {data.shape}"
            )
        data = data[np.isfinite(data).all(axis=1)]
        if len(data) == 0:
            raise ValueError("training action data has no finite rows")
        if mode == "data_minmax":
            low, high = data.min(axis=0), data.max(axis=0)
        else:
            if not 0.0 <= quantile < 0.5:
                raise ValueError("action support quantile must lie in [0,0.5)")
            low = np.quantile(data, quantile, axis=0)
            high = np.quantile(data, 1.0 - quantile, axis=0)
    if np.any(low > high):
        raise ValueError("resolved raw lower bounds exceed upper bounds")
    return low.astype(np.float64), high.astype(np.float64)


def infer_model_flat_action_dim(model: Any) -> int | None:
    """Infer the flattened action width of released LeWM-style encoders."""

    encoder = getattr(model, "action_encoder", None)
    if encoder is None:
        return None
    patch_embed = getattr(encoder, "patch_embed", None)
    if isinstance(patch_embed, torch.nn.Conv1d):
        return int(patch_embed.in_channels)
    explicit = getattr(encoder, "input_dim", None)
    if isinstance(explicit, int):
        return explicit
    for module in encoder.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    return None


def validate_action_scaler_contract(scaler: Any | None, *, action_dim: int) -> None:
    """Fail fast if an inverse scaler cannot serve the configured raw action."""

    if scaler is None:
        return
    mean = np.asarray(getattr(scaler, "mean_", None))
    scale = np.asarray(getattr(scaler, "scale_", None))
    if mean.shape != (action_dim,) or scale.shape != (action_dim,):
        raise ValueError(
            f"action scaler must have {action_dim} features, "
            f"got mean={mean.shape}, scale={scale.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("action scaler contains invalid mean/scale values")


def validate_model_action_contract(model: Any, *, action_dim: int, action_block: int) -> int:
    expected = int(action_dim) * int(action_block)
    actual = infer_model_flat_action_dim(model)
    if actual is None:
        raise ValueError("cannot infer checkpoint action-encoder input dimension")
    if actual != expected:
        raise ValueError(
            f"checkpoint action encoder expects {actual} values per macro step, "
            f"but action_dim*action_block is {expected}"
        )
    return actual


class ProjectedCEMSolver(CEMSolver):
    """Stock-compatible CEM with an optional first-class action projection."""

    def __init__(
        self,
        *args: Any,
        action_projection: StandardizedActionProjection | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_projection = action_projection
        self._latched_categorical: dict[int, float] = {}
        if self.n_steps < 1:
            raise ValueError("n_steps must be positive")
        if not 1 <= self.topk <= self.num_samples:
            raise ValueError("topk must lie in [1, num_samples]")

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        super().configure(action_space=action_space, n_envs=n_envs, config=config)
        if self.action_projection is not None:
            if self.action_projection.action_dim != self._action_dim:
                raise ValueError(
                    "projection action_dim does not match configured action space: "
                    f"{self.action_projection.action_dim} != {self._action_dim}"
                )
            if self.action_projection.action_block != int(config.action_block):
                raise ValueError(
                    "projection action_block does not match PlanConfig: "
                    f"{self.action_projection.action_block} != {config.action_block}"
                )
            projection = self.action_projection
            if projection.terminal_switch_steps:
                total_steps = int(config.horizon) * projection.action_block
                if projection.terminal_switch_steps[-1] >= total_steps:
                    raise ValueError(
                        "terminal switch step lies outside the configured plan"
                    )
                fitted_open_steps = total_steps - projection.terminal_switch_steps[-1]
                if projection.terminal_open_steps != fitted_open_steps:
                    raise ValueError(
                        "terminal_open_steps does not match the fitted latest switch"
                    )

    def reset_action_latches(self) -> None:
        """Reset episode-scoped categorical latches before a new environment reset."""

        self._latched_categorical.clear()

    def observe_returned_raw_action(self, action: np.ndarray) -> None:
        """Latch Franka open after that exact raw command is returned for execution."""

        projection = self.action_projection
        if projection is None or projection.name != "franka_stack":
            return
        raw = np.asarray(action)
        if raw.shape[-1:] != (projection.action_dim,):
            raise ValueError(
                f"returned raw action must end in dimension {projection.action_dim}"
            )
        for index in projection.single_switch_categorical:
            raw_categories = dict(projection.raw_categorical)[index]
            if 1.0 not in raw_categories:
                continue
            if bool(np.any(raw[..., index] == 1.0)):
                standardized_categories = dict(projection.categorical)[index]
                self._latched_categorical[index] = standardized_categories[
                    raw_categories.index(1.0)
                ]

    def latch_raw_categorical(self, index: int, raw_value: float) -> bool:
        """Latch one categorical dimension and report whether it changed.

        This is used by feedback-aware clients.  In particular, Franka must
        not interpret an *attempted* open command as evidence that placement
        succeeded; the client latches open only after geometry says the stack
        and goal pose are ready for release.
        """

        projection = self.action_projection
        if projection is None:
            raise ValueError("cannot latch a categorical action without a projection")
        raw_categories = dict(projection.raw_categorical).get(int(index))
        categories = dict(projection.categorical).get(int(index))
        if raw_categories is None or categories is None:
            raise ValueError(f"action dimension {index} is not categorical")
        try:
            category_index = raw_categories.index(float(raw_value))
        except ValueError as error:
            raise ValueError(
                f"raw value {raw_value} is outside categorical support {raw_categories}"
            ) from error
        standardized_value = categories[category_index]
        changed = self._latched_categorical.get(int(index)) != standardized_value
        self._latched_categorical[int(index)] = standardized_value
        return changed

    def _project(self, actions: torch.Tensor) -> torch.Tensor:
        projected = (
            actions
            if self.action_projection is None
            else self.action_projection.project(actions)
        )
        projection = self.action_projection
        if (
            projection is not None
            and projection.continuous_plan_tv_max is not None
            and projected.ndim >= 3
        ):
            # Per-coordinate marginal bounds are not a joint action-sequence
            # support constraint. Diagonal CEM can otherwise assemble legal
            # controller commands into high-frequency Franka or Gear plans
            # never observed in a demonstration. Project the schema's continuous
            # dimensions onto the training plan's total-variation envelope before
            # both model scoring and elite updates.
            leading = projected.shape[:-2]
            total_steps = projected.shape[-2] * projection.action_block
            plan = projected.reshape(*leading, total_steps, projection.action_dim)
            continuous_indices = torch.as_tensor(
                projection.continuous_action_indices,
                device=projected.device,
                dtype=torch.long,
            )
            continuous = plan.index_select(-1, continuous_indices)
            deltas = continuous[..., 1:, :] - continuous[..., :-1, :]
            total_variation = deltas.abs().sum(dim=(-2, -1))
            limit = torch.as_tensor(
                projection.continuous_plan_tv_max,
                device=projected.device,
                dtype=projected.dtype,
            )
            scale = torch.clamp(limit / total_variation.clamp_min(1e-12), max=1.0)
            scaled_deltas = deltas * scale[..., None, None]
            reconstructed = torch.cat(
                (
                    continuous[..., :1, :],
                    continuous[..., :1, :] + scaled_deltas.cumsum(dim=-2),
                ),
                dim=-2,
            )
            plan = plan.clone()
            plan[..., continuous_indices] = reconstructed
            projected = plan.reshape_as(projected)
        if (
            projection is not None
            and (projection.terminal_switch_steps or self._latched_categorical)
            and projected.ndim >= 3
        ):
            # Before release, project the entire 25-step Franka gripper plan to
            # one of the exact train-observed close-to-open phases. Once an open
            # command has actually been returned, the episode latch instead
            # makes every later candidate all-open before model scoring.
            total_steps = projected.shape[-2] * projection.action_block
            source_plan = actions.reshape(
                *actions.shape[:-2], total_steps, projection.action_dim
            )
            plan = projected.reshape(
                *projected.shape[:-2], total_steps, projection.action_dim
            ).clone()
            for gripper_index in projection.single_switch_categorical:
                if gripper_index in self._latched_categorical:
                    plan[..., gripper_index] = self._latched_categorical[
                        gripper_index
                    ]
                    continue
                categories = dict(projection.categorical)[gripper_index]
                raw_categories = dict(projection.raw_categorical)[gripper_index]
                close_category = categories[raw_categories.index(-1.0)]
                open_category = categories[raw_categories.index(1.0)]
                # Use the exact, possibly non-contiguous split set observed in
                # train-only first-success windows. Never infer support from a
                # modal split plus a slack interval.
                splits = torch.as_tensor(
                    projection.terminal_switch_steps,
                    device=projected.device,
                )
                raw_steps = torch.arange(total_steps, device=projected.device)
                patterns = torch.where(
                    raw_steps.unsqueeze(0) >= splits.unsqueeze(1),
                    torch.as_tensor(
                        open_category,
                        device=projected.device,
                        dtype=projected.dtype,
                    ),
                    torch.as_tensor(
                        close_category,
                        device=projected.device,
                        dtype=projected.dtype,
                    ),
                )
                current = source_plan[..., gripper_index]
                pattern_costs = (
                    current.unsqueeze(-2) - patterns
                ).square().sum(dim=-1)
                plan[..., gripper_index] = patterns[pattern_costs.argmin(dim=-1)]

            projected = plan.reshape_as(projected)
        return projected

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        # Preserve literal stock behavior (including its RNG stream) for generic
        # tasks and Cube reproductions that explicitly request no projection.
        if self.action_projection is None:
            return super().solve(info_dict, init_action)

        start_time = time.time()
        outputs: dict[str, Any] = {"costs": [], "mean": [], "var": []}
        total_envs = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(
            self.model,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        )
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = self._project(mean.to(self.device))
        var = self.action_projection.zero_fixed_variance(var.to(self.device))

        for callback in self.callbacks:
            callback.reset()

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded_infos: dict[str, Any] = {}
            for key, value in info_dict.items():
                if torch.is_tensor(value):
                    batch = value[start_idx:end_idx]
                    target_dtype = self.dtype if batch.is_floating_point() else None
                    batch = batch.to(device=self.device, dtype=target_dtype)
                    expanded_infos[key] = batch.unsqueeze(1).expand(
                        current_bs, self.num_samples, *batch.shape[1:]
                    )
                elif isinstance(value, np.ndarray):
                    batch = value[start_idx:end_idx]
                    expanded_infos[key] = np.repeat(batch[:, None, ...], self.num_samples, axis=1)
                elif isinstance(value, list):
                    expanded_infos[key] = value[start_idx:end_idx]
                else:
                    expanded_infos[key] = value

            for callback in self.callbacks:
                callback.start_batch()

            final_candidates: torch.Tensor | None = None
            final_costs: torch.Tensor | None = None
            final_elite_cost: torch.Tensor | None = None
            for step in range(self.n_steps):
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean
                # This tensor is now the single source of truth for scoring,
                # elite selection, updates, callbacks, and the returned plan.
                candidates = self._project(candidates)
                costs = self.model.get_cost(expanded_infos, candidates)
                if not isinstance(costs, torch.Tensor):
                    raise TypeError(f"model cost must be a tensor, got {type(costs)}")
                if costs.shape != (current_bs, self.num_samples):
                    raise ValueError(
                        f"expected cost shape {(current_bs, self.num_samples)}, got {tuple(costs.shape)}"
                    )
                if not bool(torch.isfinite(costs).all()):
                    raise ValueError("model returned non-finite CEM costs")

                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                prev_mean, prev_var = batch_mean, batch_var
                batch_mean = self._project(topk_candidates.mean(dim=1))
                # Match stock StableWorldModel CEM's sample standard deviation.
                batch_var = topk_candidates.std(dim=1)
                batch_var = self.action_projection.zero_fixed_variance(batch_var)

                for callback in self.callbacks:
                    callback(
                        step=step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=topk_vals,
                        topk_inds=topk_inds,
                        topk_candidates=topk_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=prev_mean,
                        prev_var=prev_var,
                    )
                final_candidates = candidates
                final_costs = costs
                final_elite_cost = topk_vals.mean(dim=1)

            if (
                final_candidates is None
                or final_costs is None
                or final_elite_cost is None
            ):  # guarded in __init__, retained for clarity
                raise RuntimeError("projected CEM did not run an optimization step")
            projected_mean = self._project(batch_mean)
            # The projected elite mean is a new feasible plan, not one of the
            # candidates whose cost was evaluated above. Explicitly score both
            # it and the best feasible final sample. Keeping the original sample
            # axis preserves model caches whose embeddings were expanded to the
            # configured CEM population size.
            _, best_sample_index = final_costs.min(dim=1)
            rows = torch.arange(current_bs, device=self.device)
            best_sample = final_candidates[rows, best_sample_index]
            selection_candidates = final_candidates.clone()
            selection_candidates[:, 0] = projected_mean
            if self.num_samples > 1:
                selection_candidates[:, 1] = best_sample
            selection_costs = self.model.get_cost(
                expanded_infos, selection_candidates
            )
            if not isinstance(selection_costs, torch.Tensor):
                raise TypeError(
                    f"model cost must be a tensor, got {type(selection_costs)}"
                )
            if selection_costs.shape != (current_bs, self.num_samples):
                raise ValueError(
                    "expected final-selection cost shape "
                    f"{(current_bs, self.num_samples)}, got {tuple(selection_costs.shape)}"
                )
            if not bool(torch.isfinite(selection_costs).all()):
                raise ValueError("model returned non-finite final-selection costs")
            projected_mean_cost = selection_costs[:, 0]
            best_sample_cost = (
                selection_costs[:, 1]
                if self.num_samples > 1
                else final_costs[:, 0]
            )
            choose_mean = projected_mean_cost <= best_sample_cost
            returned_plan = torch.where(
                choose_mean[:, None, None], projected_mean, best_sample
            )
            returned_cost = torch.where(
                choose_mean, projected_mean_cost, best_sample_cost
            )

            mean[start_idx:end_idx] = returned_plan
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(returned_cost.detach().cpu().tolist())
            outputs.setdefault("projected_mean_costs", []).extend(
                projected_mean_cost.detach().cpu().tolist()
            )
            outputs.setdefault("best_sample_costs", []).extend(
                best_sample_cost.detach().cpu().tolist()
            )
            outputs.setdefault("final_elite_mean_costs", []).extend(
                final_elite_cost.detach().cpu().tolist()
            )
            outputs.setdefault("returned_plan_source", []).extend(
                [
                    "projected_mean" if selected else "best_feasible_sample"
                    for selected in choose_mean.detach().cpu().tolist()
                ]
            )

        mean = self._project(mean)
        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        outputs["action_projection"] = self.action_projection.to_dict()
        if self.callbacks:
            outputs["callbacks"] = {}
            for callback in self.callbacks:
                callback.end_solve()
                outputs["callbacks"][callback.output_key] = callback.history

        print(f"Projected CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs


__all__ = [
    "TASK_ACTION_SCHEMAS",
    "TASK_ACTION_DIMS",
    "ACTION_BOUND_MODES",
    "TrainingZScoreScaler",
    "ProjectedActionScaler",
    "StandardizedActionProjection",
    "ProjectedCEMSolver",
    "build_task_action_projection",
    "resolve_raw_action_bounds",
    "infer_model_flat_action_dim",
    "validate_action_scaler_contract",
    "validate_model_action_contract",
]
