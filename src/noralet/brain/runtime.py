"""Persistent per-Noralet inference state and explicit action sampling."""

from __future__ import annotations

from enum import StrEnum
import math
from statistics import NormalDist
from typing import Protocol

import torch
from torch import Tensor

from noralet.brain.config import NoraletBrainConfig
from noralet.brain.model import BrainActionParameters, NoraletBrainModel
from noralet.noralets.actions import ActionIntent
from noralet.noralets.actuators import NoraletActuatorConfig
from noralet.noralets.experience import NoraletExperience
from noralet.noralets.signals import (
    SignalDirection,
    SignalEmissionIntent,
    SignalType,
)


ACTION_RANDOM_DRAW_ORDER = (
    "acceleration_standard_normal",
    "consume_uniform",
    "signal_category_uniform",
)
_STANDARD_NORMAL = NormalDist()


class _RandomSource(Protocol):
    def random(self) -> float: ...


class SignalMotorChoice(StrEnum):
    """Exactly nine structurally available signal motor outcomes."""

    NONE = "none"
    A_LEFT = "a_left"
    A_RIGHT = "a_right"
    B_LEFT = "b_left"
    B_RIGHT = "b_right"
    C_LEFT = "c_left"
    C_RIGHT = "c_right"
    D_LEFT = "d_left"
    D_RIGHT = "d_right"


_SIGNAL_EMISSIONS: tuple[SignalEmissionIntent | None, ...] = (
    None,
    SignalEmissionIntent(SignalType.A, SignalDirection.LEFT),
    SignalEmissionIntent(SignalType.A, SignalDirection.RIGHT),
    SignalEmissionIntent(SignalType.B, SignalDirection.LEFT),
    SignalEmissionIntent(SignalType.B, SignalDirection.RIGHT),
    SignalEmissionIntent(SignalType.C, SignalDirection.LEFT),
    SignalEmissionIntent(SignalType.C, SignalDirection.RIGHT),
    SignalEmissionIntent(SignalType.D, SignalDirection.LEFT),
    SignalEmissionIntent(SignalType.D, SignalDirection.RIGHT),
)


class NoraletBrain:
    """One independent model copy and its persistent recurrent hidden state."""

    def __init__(
        self,
        *,
        model: NoraletBrainModel,
        config: NoraletBrainConfig,
        actuator_config: NoraletActuatorConfig,
        device: torch.device,
        action_random_source: object | None = None,
    ) -> None:
        if not isinstance(model, NoraletBrainModel):
            raise TypeError("model must be a NoraletBrainModel")
        if not isinstance(config, NoraletBrainConfig):
            raise TypeError("config must be a NoraletBrainConfig")
        if not isinstance(actuator_config, NoraletActuatorConfig):
            raise TypeError("actuator_config must be a NoraletActuatorConfig")
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        if action_random_source is not None and not callable(
            getattr(action_random_source, "random", None)
        ):
            raise TypeError("action_random_source must provide random()")
        self._model = model
        self._config = config
        self._actuator_config = actuator_config
        self._device = device
        self._action_random_source = action_random_source
        self._hidden_state = torch.zeros(
            config.hidden_size,
            dtype=next(model.parameters()).dtype,
            device=device,
        )
        self._activation_count = 0

    @property
    def model(self) -> NoraletBrainModel:
        """Return this individual's observer-accessible model instance."""

        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def hidden_state(self) -> Tensor:
        """Return a detached copy of the current persistent state."""

        return self._hidden_state.detach().clone()

    @property
    def activation_count(self) -> int:
        return self._activation_count

    def parameter_snapshot(self) -> tuple[Tensor, ...]:
        """Return detached CPU copies suitable for no-learning audits."""

        return tuple(
            parameter.detach().cpu().clone()
            for parameter in self._model.parameters()
        )

    def activate(self, experience: NoraletExperience) -> BrainActionParameters:
        """Advance recurrence once from exactly one brain-facing Experience."""

        if not isinstance(experience, NoraletExperience):
            raise TypeError("experience must be a NoraletExperience")
        self._model.eval()
        with torch.no_grad():
            hidden, acceleration_loc, consume_logit, signal_logits = self._model(
                experience,
                self._hidden_state,
            )
            if not all(
                torch.isfinite(value).all().item()
                for value in (
                    hidden,
                    acceleration_loc,
                    consume_logit,
                    signal_logits,
                )
            ):
                raise FloatingPointError("brain activation produced a non-finite value")
            self._hidden_state = hidden.detach()
            self._activation_count += 1
            return BrainActionParameters(
                acceleration_loc=acceleration_loc.item(),
                consume_logit=consume_logit.item(),
                signal_logits=tuple(signal_logits.detach().cpu().tolist()),
            )

    def sample_action(
        self,
        parameters: BrainActionParameters,
        random_source: _RandomSource,
    ) -> ActionIntent:
        """Consume exactly three ordered draws and construct one physical intent."""

        if not isinstance(parameters, BrainActionParameters):
            raise TypeError("parameters must be BrainActionParameters")
        if not callable(getattr(random_source, "random", None)):
            raise TypeError("random_source must provide random()")

        draws = tuple(float(random_source.random()) for _ in ACTION_RANDOM_DRAW_ORDER)
        if not all(0.0 <= draw < 1.0 and math.isfinite(draw) for draw in draws):
            raise ValueError("action random draws must be finite values in [0, 1)")
        normal_quantile = min(
            math.nextafter(1.0, 0.0),
            max(math.nextafter(0.0, 1.0), draws[0]),
        )
        standard_normal = _STANDARD_NORMAL.inv_cdf(normal_quantile)
        raw_motor_value = (
            parameters.acceleration_loc
            + self._config.acceleration_exploration_std * standard_normal
        )
        acceleration = (
            self._actuator_config.max_acceleration * math.tanh(raw_motor_value)
        )
        consume = draws[1] < parameters.consume_probability
        signal_index = self._categorical_index(
            parameters.signal_probabilities,
            draws[2],
        )
        return ActionIntent(
            acceleration=acceleration,
            consume=consume,
            signal_emission=_SIGNAL_EMISSIONS[signal_index],
        )

    def act(
        self,
        experience: NoraletExperience,
    ) -> ActionIntent:
        """Activate once and sample the corresponding low-level motor action."""

        if self._action_random_source is None:
            raise RuntimeError("this NoraletBrain has no action random stream")
        return self.sample_action(
            self.activate(experience),
            self._action_random_source,
        )

    @staticmethod
    def signal_motor_outcomes() -> tuple[SignalMotorChoice, ...]:
        return tuple(SignalMotorChoice)

    @staticmethod
    def _categorical_index(
        probabilities: tuple[float, ...],
        uniform: float,
    ) -> int:
        threshold = uniform
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if threshold < cumulative:
                return index
        return len(probabilities) - 1
