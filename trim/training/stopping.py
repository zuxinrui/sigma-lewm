"""Loss-only checkpoint selection and early stopping for new-task experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Mapping

import lightning as pl


DEFAULT_COMPONENT_KEYS = {
    "teacher": "teacher_pred_loss",
    "free_run": "rollout_pred_loss",
    "terminal": "terminal_pred_loss",
    "action": "action_rank_loss",
    "sigreg": "sigreg_loss",
}

DEFAULT_WEIGHTS = {
    "teacher": 0.30,
    "free_run": 0.35,
    "terminal": 0.20,
    "action": 0.10,
    "sigreg": 0.05,
}


class LossControlError(ValueError):
    """Raised when the preregistered loss-control contract is violated."""


@dataclass
class LossControlDecision:
    epoch: int
    validation_components: dict[str, float]
    validation_factor: float
    validation_ema: float
    train_components: dict[str, float]
    train_factor: float | None
    generalization_gap: float | None
    embedding_std_mean: float
    embedding_std_ratio: float
    embedding_effective_rank: float
    embedding_effective_rank_ratio: float
    embedding_covariance_condition: float
    embedding_covariance_condition_ratio: float
    best_epoch: int
    best_validation_ema: float
    epochs_without_improvement: int
    degradation_epochs: int
    collapse_epochs: int
    stop: bool
    stop_reason: str | None


@dataclass
class LossCompositeController:
    """Pure state machine implementing the frozen loss-only stop rule."""

    min_epochs: int = 50
    max_epochs: int = 100
    ema_window: int = 3
    patience: int = 10
    slope_window: int = 5
    min_improvement_fraction: float = 0.005
    degradation_fraction: float = 0.05
    degradation_patience: int = 5
    collapse_std_ratio: float = 0.10
    collapse_effective_rank_ratio: float = 0.10
    collapse_condition_growth: float = 10.0
    collapse_patience: int = 3
    epsilon: float = 1e-12
    weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    validation_baseline: dict[str, float] | None = None
    train_baseline: dict[str, float] | None = None
    embedding_std_baseline: float | None = None
    embedding_effective_rank_baseline: float | None = None
    embedding_covariance_condition_baseline: float | None = None
    validation_factors: list[float] = field(default_factory=list)
    validation_emas: list[float] = field(default_factory=list)
    train_factors: list[float] = field(default_factory=list)
    best_validation_ema: float = math.inf
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    degradation_epochs: int = 0
    collapse_epochs: int = 0
    decisions: list[LossControlDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise LossControlError("require 1 <= min_epochs <= max_epochs")
        if min(
            self.ema_window,
            self.patience,
            self.slope_window,
            self.degradation_patience,
            self.collapse_patience,
        ) < 1:
            raise LossControlError("all loss-control windows must be positive")
        if self.epsilon <= 0:
            raise LossControlError("epsilon must be positive")
        if not 0 < self.collapse_std_ratio < 1:
            raise LossControlError("collapse std ratio must lie in (0,1)")
        if not 0 < self.collapse_effective_rank_ratio < 1:
            raise LossControlError(
                "collapse effective-rank ratio must lie in (0,1)"
            )
        if self.collapse_condition_growth <= 1:
            raise LossControlError(
                "collapse covariance-condition growth must exceed one"
            )
        if set(self.weights) != set(DEFAULT_COMPONENT_KEYS):
            raise LossControlError("loss-control weights have unexpected components")
        if any(value < 0 for value in self.weights.values()):
            raise LossControlError("loss-control weights must be non-negative")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-12):
            raise LossControlError("loss-control weights must sum to one")

    def _validate_components(
        self,
        values: Mapping[str, float],
        *,
        where: str,
        require_all: bool = True,
    ) -> dict[str, float]:
        missing = set(self.weights) - set(values)
        if require_all and missing:
            raise LossControlError(
                f"{where} components missing: {sorted(missing)}"
            )
        result = {
            key: float(values[key]) for key in self.weights if key in values
        }
        if not result:
            raise LossControlError(f"{where} has no recognized components")
        if not all(math.isfinite(value) and value >= 0 for value in result.values()):
            raise LossControlError(f"{where} components must be finite and non-negative")
        return result

    def _factor(
        self, values: Mapping[str, float], baseline: Mapping[str, float]
    ) -> float:
        keys = [
            key for key in self.weights if key in values and key in baseline
        ]
        weight_total = sum(self.weights[key] for key in keys)
        if weight_total <= 0:
            raise LossControlError("composite has no positive-weight components")
        log_factor = sum(
            (self.weights[key] / weight_total)
            * math.log(
                max(float(values[key]), self.epsilon)
                / max(float(baseline[key]), self.epsilon)
            )
            for key in keys
        )
        return math.exp(log_factor)

    def _ema(self) -> float:
        window = self.validation_factors[-self.ema_window :]
        return sum(window) / len(window)

    def update(
        self,
        *,
        epoch: int,
        validation_components: Mapping[str, float],
        train_components: Mapping[str, float] | None,
        embedding_std_mean: float,
        embedding_effective_rank: float | None = None,
        embedding_covariance_condition: float | None = None,
    ) -> LossControlDecision:
        if epoch != len(self.decisions) + 1:
            raise LossControlError(
                f"epochs must be consecutive from one; got {epoch}"
            )
        validation = self._validate_components(
            validation_components, where="validation"
        )
        train = (
            self._validate_components(
                train_components, where="train", require_all=False
            )
            if train_components is not None
            else {}
        )
        embedding_std_mean = float(embedding_std_mean)
        if not math.isfinite(embedding_std_mean) or embedding_std_mean <= 0:
            raise LossControlError("embedding std mean must be finite and positive")
        embedding_effective_rank = float(
            1.0 if embedding_effective_rank is None else embedding_effective_rank
        )
        embedding_covariance_condition = float(
            1.0
            if embedding_covariance_condition is None
            else embedding_covariance_condition
        )
        if (
            not math.isfinite(embedding_effective_rank)
            or embedding_effective_rank <= 0
        ):
            raise LossControlError(
                "embedding effective rank must be finite and positive"
            )
        if (
            not math.isfinite(embedding_covariance_condition)
            or embedding_covariance_condition < 1
        ):
            raise LossControlError(
                "embedding covariance condition must be finite and >= 1"
            )

        if self.validation_baseline is None:
            self.validation_baseline = dict(validation)
            self.train_baseline = dict(train) if train else None
            self.embedding_std_baseline = embedding_std_mean
            self.embedding_effective_rank_baseline = embedding_effective_rank
            self.embedding_covariance_condition_baseline = (
                embedding_covariance_condition
            )
        validation_factor = self._factor(validation, self.validation_baseline)
        self.validation_factors.append(validation_factor)
        validation_ema = self._ema()
        self.validation_emas.append(validation_ema)

        train_factor = None
        generalization_gap = None
        if train and self.train_baseline is not None:
            train_factor = self._factor(train, self.train_baseline)
            self.train_factors.append(train_factor)
            generalization_gap = validation_factor / max(
                train_factor, self.epsilon
            )

        improved = (
            not math.isfinite(self.best_validation_ema)
            or validation_ema
            <= self.best_validation_ema
            * (1.0 - self.min_improvement_fraction)
        )
        if improved:
            self.best_validation_ema = validation_ema
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        degrading = (
            epoch >= self.min_epochs
            and validation_ema
            >= self.best_validation_ema * (1.0 + self.degradation_fraction)
            and train_factor is not None
            and len(self.train_factors) >= self.degradation_patience
            and train_factor < self.train_factors[-self.degradation_patience]
        )
        self.degradation_epochs = self.degradation_epochs + 1 if degrading else 0

        embedding_std_ratio = embedding_std_mean / max(
            float(self.embedding_std_baseline), self.epsilon
        )
        embedding_effective_rank_ratio = embedding_effective_rank / max(
            float(self.embedding_effective_rank_baseline), self.epsilon
        )
        embedding_covariance_condition_ratio = (
            embedding_covariance_condition
            / max(
                float(self.embedding_covariance_condition_baseline),
                self.epsilon,
            )
        )
        collapsed = epoch >= self.min_epochs and (
            embedding_std_ratio < self.collapse_std_ratio
            or embedding_effective_rank_ratio
            < self.collapse_effective_rank_ratio
            or embedding_covariance_condition_ratio
            > self.collapse_condition_growth
        )
        self.collapse_epochs = self.collapse_epochs + 1 if collapsed else 0

        slope_non_decreasing = False
        if len(self.validation_emas) >= self.slope_window:
            recent = self.validation_emas[-self.slope_window :]
            x_mean = (len(recent) - 1) / 2.0
            y_mean = sum(recent) / len(recent)
            numerator = sum(
                (index - x_mean) * (value - y_mean)
                for index, value in enumerate(recent)
            )
            denominator = sum(
                (index - x_mean) ** 2 for index in range(len(recent))
            )
            slope_non_decreasing = numerator / denominator >= 0

        stop_reason = None
        if epoch >= self.max_epochs:
            stop_reason = "max_epochs"
        elif epoch >= self.min_epochs and self.collapse_epochs >= self.collapse_patience:
            stop_reason = "embedding_collapse"
        elif epoch >= self.min_epochs and self.degradation_epochs >= self.degradation_patience:
            stop_reason = "validation_degradation_with_train_improvement"
        elif (
            epoch >= self.min_epochs
            and self.epochs_without_improvement >= self.patience
            and slope_non_decreasing
        ):
            stop_reason = "validation_plateau"

        decision = LossControlDecision(
            epoch=epoch,
            validation_components=validation,
            validation_factor=validation_factor,
            validation_ema=validation_ema,
            train_components=train,
            train_factor=train_factor,
            generalization_gap=generalization_gap,
            embedding_std_mean=embedding_std_mean,
            embedding_std_ratio=embedding_std_ratio,
            embedding_effective_rank=embedding_effective_rank,
            embedding_effective_rank_ratio=embedding_effective_rank_ratio,
            embedding_covariance_condition=embedding_covariance_condition,
            embedding_covariance_condition_ratio=(
                embedding_covariance_condition_ratio
            ),
            best_epoch=self.best_epoch,
            best_validation_ema=self.best_validation_ema,
            epochs_without_improvement=self.epochs_without_improvement,
            degradation_epochs=self.degradation_epochs,
            collapse_epochs=self.collapse_epochs,
            stop=stop_reason is not None,
            stop_reason=stop_reason,
        )
        self.decisions.append(decision)
        return decision


class LossCompositeEarlyStopping(pl.Callback):
    """Lightning adapter that consumes only train/validation loss metrics."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        component_keys: Mapping[str, str] | None = None,
        **controller_kwargs,
    ) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.component_keys = dict(component_keys or DEFAULT_COMPONENT_KEYS)
        if set(self.component_keys) != set(DEFAULT_COMPONENT_KEYS):
            raise LossControlError("component key mapping is incomplete")
        self.controller = LossCompositeController(**controller_kwargs)

    @property
    def state_key(self) -> str:
        return "paper_loss_control_v1"

    @staticmethod
    def _metric(metrics, key: str) -> float:
        if key not in metrics:
            raise LossControlError(f"Lightning callback metric is missing: {key}")
        value = metrics[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def _components(self, metrics, prefix: str) -> dict[str, float] | None:
        keys = {
            name: f"{prefix}/{metric}"
            for name, metric in self.component_keys.items()
        }
        if prefix == "validate" and any(
            key not in metrics for key in keys.values()
        ):
            missing = [key for key in keys.values() if key not in metrics]
            raise LossControlError(
                "validation metrics are incomplete: " + ", ".join(missing)
            )
        result = {
            name: self._metric(metrics, key) for name, key in keys.items()
            if key in metrics
        }
        return result or None

    def _write_ledger(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "paper_loss_control_v1",
            "selection_scope": (
                "training/held-out-validation losses only; simulator H5 metrics are forbidden"
            ),
            "component_keys": self.component_keys,
            "controller": {
                key: value
                for key, value in asdict(self.controller).items()
                if key != "decisions"
            },
            "decisions": [asdict(item) for item in self.controller.decisions],
        }
        path = self.run_dir / "loss_control.json"
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Decide only after Lightning finalizes both fit and validation means."""

        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        validation = self._components(metrics, "validate")
        # Stable Pretraining names the optimization stage ``fit``.  This is the
        # authoritative train-side epoch aggregate used for overfit detection.
        train = self._components(metrics, "fit")
        embedding_std = self._metric(metrics, "validate/embedding_std_mean")
        embedding_effective_rank = self._metric(
            metrics, "validate/embedding_effective_rank"
        )
        embedding_covariance_condition = self._metric(
            metrics, "validate/embedding_covariance_condition"
        )
        decision = self.controller.update(
            epoch=int(trainer.current_epoch) + 1,
            validation_components=validation,
            train_components=train,
            embedding_std_mean=embedding_std,
            embedding_effective_rank=embedding_effective_rank,
            embedding_covariance_condition=embedding_covariance_condition,
        )
        if trainer.is_global_zero:
            self._write_ledger()
            print(
                "[LOSS-CONTROL] "
                f"epoch={decision.epoch} factor={decision.validation_factor:.6g} "
                f"ema={decision.validation_ema:.6g} best_epoch={decision.best_epoch} "
                f"patience={decision.epochs_without_improvement}/"
                f"{self.controller.patience} stop={decision.stop} "
                f"reason={decision.stop_reason}",
                flush=True,
            )
        if decision.stop:
            trainer.should_stop = True

    def state_dict(self) -> dict:
        return {"controller": asdict(self.controller)}

    def load_state_dict(self, state_dict: dict) -> None:
        raw_state = state_dict.get("controller")
        state = dict(raw_state) if isinstance(raw_state, dict) else None
        if not isinstance(state, dict):
            raise LossControlError("resume checkpoint has no loss controller state")
        decisions = [LossControlDecision(**item) for item in state.pop("decisions", [])]
        self.controller = LossCompositeController(**state)
        self.controller.decisions = decisions


__all__ = [
    "DEFAULT_COMPONENT_KEYS",
    "DEFAULT_WEIGHTS",
    "LossCompositeController",
    "LossCompositeEarlyStopping",
    "LossControlDecision",
    "LossControlError",
]
