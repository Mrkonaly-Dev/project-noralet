"""Focused configuration and values for online predictive plasticity."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _positive_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


@dataclass(frozen=True, slots=True)
class NoraletLearningConfig:
    """The three controls required for one-transition predictive learning."""

    learning_rate: float
    max_gradient_norm: float
    predictor_hidden_size: int

    def __post_init__(self) -> None:
        learning_rate = _positive_finite_real(
            "learning_rate",
            self.learning_rate,
        )
        max_gradient_norm = _positive_finite_real(
            "max_gradient_norm",
            self.max_gradient_norm,
        )
        if type(self.predictor_hidden_size) is not int:
            raise TypeError("predictor_hidden_size must be an integer")
        if self.predictor_hidden_size <= 0:
            raise ValueError("predictor_hidden_size must be positive")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "max_gradient_norm", max_gradient_norm)


@dataclass(frozen=True, slots=True)
class PredictiveLearningResult:
    """One successful individual predictive update without routing identity."""

    prediction_loss: float
    gradient_norm: float

    def __post_init__(self) -> None:
        for name in ("prediction_loss", "gradient_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            if converted < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, converted)
