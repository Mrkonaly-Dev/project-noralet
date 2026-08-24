"""Focused configuration and values for independent lifetime plasticity."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from noralet.noralets.experience import Interoception


def _positive_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _non_negative_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0:
        raise ValueError(f"{name} cannot be negative")
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


@dataclass(frozen=True, slots=True)
class NoraletHomeostaticPlasticityConfig:
    """Controls for distress modulation and action-head eligibility traces."""

    energy_distress_weight: float
    condition_distress_weight: float
    homeostatic_modulation_scale: float
    eligibility_decay: float
    action_learning_rate: float
    max_homeostatic_update_norm: float

    def __post_init__(self) -> None:
        energy_weight = _non_negative_finite_real(
            "energy_distress_weight",
            self.energy_distress_weight,
        )
        condition_weight = _non_negative_finite_real(
            "condition_distress_weight",
            self.condition_distress_weight,
        )
        if energy_weight == 0.0 and condition_weight == 0.0:
            raise ValueError("at least one distress weight must be positive")
        modulation_scale = _positive_finite_real(
            "homeostatic_modulation_scale",
            self.homeostatic_modulation_scale,
        )
        decay = _non_negative_finite_real(
            "eligibility_decay",
            self.eligibility_decay,
        )
        if decay >= 1.0:
            raise ValueError("eligibility_decay must be less than one")
        action_learning_rate = _positive_finite_real(
            "action_learning_rate",
            self.action_learning_rate,
        )
        maximum_update_norm = _positive_finite_real(
            "max_homeostatic_update_norm",
            self.max_homeostatic_update_norm,
        )
        object.__setattr__(self, "energy_distress_weight", energy_weight)
        object.__setattr__(self, "condition_distress_weight", condition_weight)
        object.__setattr__(self, "homeostatic_modulation_scale", modulation_scale)
        object.__setattr__(self, "eligibility_decay", decay)
        object.__setattr__(self, "action_learning_rate", action_learning_rate)
        object.__setattr__(
            self,
            "max_homeostatic_update_norm",
            maximum_update_norm,
        )


def homeostatic_drive(
    interoception: Interoception,
    config: NoraletHomeostaticPlasticityConfig,
) -> float:
    """Return weighted negative drive from brain-facing distress sensations."""

    if not isinstance(interoception, Interoception):
        raise TypeError("interoception must be an Interoception")
    if not isinstance(config, NoraletHomeostaticPlasticityConfig):
        raise TypeError("config must be a NoraletHomeostaticPlasticityConfig")
    numerator = (
        config.energy_distress_weight * interoception.energy_distress
        + config.condition_distress_weight * interoception.condition_distress
    )
    denominator = (
        config.energy_distress_weight + config.condition_distress_weight
    )
    drive = numerator / denominator
    if not math.isfinite(drive) or not 0.0 <= drive <= 1.0:
        raise FloatingPointError("homeostatic drive must be finite and in [0, 1]")
    return drive


def homeostatic_modulation(
    drive_before: float,
    drive_after: float,
    config: NoraletHomeostaticPlasticityConfig,
) -> float:
    """Bound actual lived distress improvement to a finite global modulation."""

    if not isinstance(config, NoraletHomeostaticPlasticityConfig):
        raise TypeError("config must be a NoraletHomeostaticPlasticityConfig")
    drives: list[float] = []
    for name, value in (
        ("drive_before", drive_before),
        ("drive_after", drive_after),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        converted = float(value)
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        drives.append(converted)
    modulation = math.tanh(
        (drives[0] - drives[1]) / config.homeostatic_modulation_scale
    )
    modulation = min(
        math.nextafter(1.0, 0.0),
        max(math.nextafter(-1.0, 0.0), modulation),
    )
    if not math.isfinite(modulation):
        raise FloatingPointError("homeostatic modulation is non-finite")
    return modulation


@dataclass(slots=True)
class ActionEligibilityTraces:
    """Detached traces corresponding to each action head's parameter tensors."""

    acceleration: tuple[Tensor, ...]
    consume: tuple[Tensor, ...]
    signal: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        devices: set[torch.device] = set()
        for name in ("acceleration", "consume", "signal"):
            traces = getattr(self, name)
            if not isinstance(traces, tuple) or not traces:
                raise TypeError(f"{name} traces must be a non-empty tensor tuple")
            if not all(isinstance(trace, Tensor) for trace in traces):
                raise TypeError(f"every {name} trace must be a Tensor")
            if any(trace.requires_grad or trace.grad_fn is not None for trace in traces):
                raise ValueError("eligibility traces must be detached tensors")
            if not all(torch.isfinite(trace).all().item() for trace in traces):
                raise FloatingPointError("eligibility traces must be finite")
            devices.update(trace.device for trace in traces)
        if len(devices) != 1:
            raise ValueError("all eligibility traces must share one neural device")

    @classmethod
    def zeros_like(
        cls,
        *,
        acceleration: tuple[nn.Parameter, ...],
        consume: tuple[nn.Parameter, ...],
        signal: tuple[nn.Parameter, ...],
    ) -> ActionEligibilityTraces:
        """Create shape- and device-matched zero traces for three action heads."""

        return cls(
            acceleration=tuple(torch.zeros_like(parameter) for parameter in acceleration),
            consume=tuple(torch.zeros_like(parameter) for parameter in consume),
            signal=tuple(torch.zeros_like(parameter) for parameter in signal),
        )

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return (*self.acceleration, *self.consume, *self.signal)

    def advanced(
        self,
        *,
        acceleration_increment: tuple[Tensor, ...],
        consume_increment: tuple[Tensor, ...],
        signal_increment: tuple[Tensor, ...],
        decay: float,
    ) -> ActionEligibilityTraces:
        """Return ``decay * previous + detached increment`` for every trace."""

        decay_value = _non_negative_finite_real("decay", decay)
        if decay_value >= 1.0:
            raise ValueError("decay must be less than one")

        def advance(
            previous: tuple[Tensor, ...],
            increment: tuple[Tensor, ...],
        ) -> tuple[Tensor, ...]:
            if not isinstance(increment, tuple) or len(previous) != len(increment):
                raise ValueError("eligibility increments must match trace tensors")
            next_values: list[Tensor] = []
            for old, new in zip(previous, increment, strict=True):
                if not isinstance(new, Tensor):
                    raise TypeError("every eligibility increment must be a Tensor")
                if old.shape != new.shape or old.device != new.device:
                    raise ValueError("eligibility increment shape/device mismatch")
                next_values.append(
                    (old * decay_value + new.detach()).detach()
                )
            return tuple(next_values)

        return ActionEligibilityTraces(
            acceleration=advance(self.acceleration, acceleration_increment),
            consume=advance(self.consume, consume_increment),
            signal=advance(self.signal, signal_increment),
        )

    def snapshot(self) -> ActionEligibilityTraces:
        """Return detached copies without exposing mutable runtime storage."""

        return ActionEligibilityTraces(
            acceleration=tuple(trace.detach().clone() for trace in self.acceleration),
            consume=tuple(trace.detach().clone() for trace in self.consume),
            signal=tuple(trace.detach().clone() for trace in self.signal),
        )

    def combined_norm(self) -> float:
        flat = torch.cat(tuple(trace.reshape(-1) for trace in self.tensors))
        norm = float(torch.linalg.vector_norm(flat).item())
        if not math.isfinite(norm):
            raise FloatingPointError("eligibility norm is non-finite")
        return norm


@dataclass(frozen=True, slots=True)
class HomeostaticPlasticityResult:
    """One completed internal-modulation action-head update without identity."""

    homeostatic_drive_before: float
    homeostatic_drive_after: float
    modulation: float
    eligibility_norm: float
    applied_update_norm: float

    def __post_init__(self) -> None:
        for name in ("homeostatic_drive_before", "homeostatic_drive_after"):
            value = _non_negative_finite_real(name, getattr(self, name))
            if value > 1.0:
                raise ValueError(f"{name} cannot exceed one")
            object.__setattr__(self, name, value)
        modulation = getattr(self, "modulation")
        if isinstance(modulation, bool) or not isinstance(modulation, (int, float)):
            raise TypeError("modulation must be a real number")
        modulation_value = float(modulation)
        if not math.isfinite(modulation_value) or not -1.0 < modulation_value < 1.0:
            raise ValueError("modulation must be finite and strictly between -1 and 1")
        object.__setattr__(self, "modulation", modulation_value)
        for name in ("eligibility_norm", "applied_update_norm"):
            object.__setattr__(
                self,
                name,
                _non_negative_finite_real(name, getattr(self, name)),
            )
