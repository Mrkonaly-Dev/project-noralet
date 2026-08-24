"""Persistent per-Noralet inference state and explicit action sampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from statistics import NormalDist
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from noralet.brain.config import NoraletBrainConfig
from noralet.brain.encoder import ExperienceEncoder
from noralet.brain.learning import (
    NoraletLearningConfig,
    PredictiveLearningResult,
)
from noralet.brain.model import (
    ACTION_VECTOR_SIZE,
    BrainActionParameters,
    NoraletBrainModel,
)
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


@dataclass(frozen=True, slots=True)
class BrainActionSelection:
    """One selected physical intent and its eleven-value neural motor vector."""

    action_intent: ActionIntent
    action_vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.action_intent, ActionIntent):
            raise TypeError("action_intent must be an ActionIntent")
        if not isinstance(self.action_vector, tuple):
            raise TypeError("action_vector must be an immutable tuple")
        if len(self.action_vector) != ACTION_VECTOR_SIZE:
            raise ValueError("action_vector must contain exactly eleven values")
        values = tuple(float(value) for value in self.action_vector)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("action_vector values must be finite")
        if not -1.0 <= values[0] <= 1.0:
            raise ValueError("normalized acceleration must be in [-1, 1]")
        if values[1] not in (0.0, 1.0):
            raise ValueError("consume command must be zero or one")
        signal_values = values[2:]
        if any(value not in (0.0, 1.0) for value in signal_values):
            raise ValueError("signal motor values must be zero or one")
        if math.fsum(signal_values) != 1.0:
            raise ValueError("signal motor representation must be one-hot")
        object.__setattr__(self, "action_vector", values)

    @property
    def normalized_acceleration_command(self) -> float:
        return self.action_vector[0]

    @property
    def consume_command(self) -> float:
        return self.action_vector[1]

    @property
    def signal_motor_index(self) -> int:
        return self.action_vector[2:].index(1.0)


@dataclass(slots=True)
class _PendingTransition:
    prediction: Tensor
    action_vector: tuple[float, ...]


class NoraletBrain:
    """One independent model copy and its persistent recurrent hidden state."""

    def __init__(
        self,
        *,
        model: NoraletBrainModel,
        config: NoraletBrainConfig,
        learning_config: NoraletLearningConfig | None,
        actuator_config: NoraletActuatorConfig,
        device: torch.device,
        action_random_source: object | None = None,
        target_experience_encoder: ExperienceEncoder | None = None,
    ) -> None:
        if not isinstance(model, NoraletBrainModel):
            raise TypeError("model must be a NoraletBrainModel")
        if not isinstance(config, NoraletBrainConfig):
            raise TypeError("config must be a NoraletBrainConfig")
        if learning_config is not None and not isinstance(
            learning_config,
            NoraletLearningConfig,
        ):
            raise TypeError("learning_config must be a NoraletLearningConfig")
        if not isinstance(actuator_config, NoraletActuatorConfig):
            raise TypeError("actuator_config must be a NoraletActuatorConfig")
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        if action_random_source is not None and not callable(
            getattr(action_random_source, "random", None)
        ):
            raise TypeError("action_random_source must provide random()")
        if learning_config is None:
            if model.prediction_model is not None:
                raise ValueError("a no-learning brain cannot contain a predictor")
            if target_experience_encoder is not None:
                raise ValueError("a no-learning brain cannot contain a target encoder")
        else:
            if model.prediction_model is None:
                raise ValueError("a learning brain requires a prediction model")
            if not isinstance(target_experience_encoder, ExperienceEncoder):
                raise TypeError("a learning brain requires an ExperienceEncoder target")
        self._model = model
        self._config = config
        self._learning_config = learning_config
        self._actuator_config = actuator_config
        self._device = device
        self._action_random_source = action_random_source
        self._target_experience_encoder = target_experience_encoder
        self._hidden_state = torch.zeros(
            config.hidden_size,
            dtype=next(model.parameters()).dtype,
            device=device,
        )
        self._activation_count = 0
        self._learning_update_count = 0
        self._pending_transition: _PendingTransition | None = None
        self._optimizer: torch.optim.Adam | None = None
        if learning_config is not None:
            for parameter in model.action_head_parameters():
                parameter.requires_grad_(False)
            plastic_parameters = model.predictive_plastic_parameters()
            self._optimizer = torch.optim.Adam(
                plastic_parameters,
                lr=learning_config.learning_rate,
            )

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

    @property
    def learning_enabled(self) -> bool:
        return self._learning_config is not None

    @property
    def learning_update_count(self) -> int:
        return self._learning_update_count

    @property
    def target_experience_encoder(self) -> ExperienceEncoder | None:
        """Return the observer-accessible permanently frozen sensory target."""

        return self._target_experience_encoder

    @property
    def optimizer(self) -> torch.optim.Adam | None:
        """Return this individual's observer-accessible optimizer."""

        return self._optimizer

    @property
    def has_pending_transition(self) -> bool:
        return self._pending_transition is not None

    @property
    def pending_action_vector(self) -> tuple[float, ...] | None:
        if self._pending_transition is None:
            return None
        return self._pending_transition.action_vector

    @property
    def pending_prediction(self) -> Tensor | None:
        if self._pending_transition is None:
            return None
        return self._pending_transition.prediction.detach().clone()

    def parameter_snapshot(self) -> tuple[Tensor, ...]:
        """Return detached CPU copies of this individual's online model."""

        return tuple(
            parameter.detach().cpu().clone()
            for parameter in self._model.parameters()
        )

    def plastic_parameter_snapshot(self) -> tuple[Tensor, ...]:
        return tuple(
            parameter.detach().cpu().clone()
            for parameter in self._model.predictive_plastic_parameters()
        )

    def action_head_parameter_snapshot(self) -> tuple[Tensor, ...]:
        return tuple(
            parameter.detach().cpu().clone()
            for parameter in self._model.action_head_parameters()
        )

    def target_parameter_snapshot(self) -> tuple[Tensor, ...]:
        if self._target_experience_encoder is None:
            return ()
        return tuple(
            parameter.detach().cpu().clone()
            for parameter in self._target_experience_encoder.parameters()
        )

    def activate(self, experience: NoraletExperience) -> BrainActionParameters:
        """Advance recurrence once from exactly one brain-facing Experience."""

        parameters, _ = self._activate(experience, track_gradient=False)
        return parameters

    def _activate(
        self,
        experience: NoraletExperience,
        *,
        track_gradient: bool,
    ) -> tuple[BrainActionParameters, Tensor]:
        if not isinstance(experience, NoraletExperience):
            raise TypeError("experience must be a NoraletExperience")
        self._model.eval()
        context = torch.enable_grad() if track_gradient else torch.no_grad()
        with context:
            hidden, acceleration_loc, consume_logit, signal_logits = self._model(
                experience,
                self._hidden_state.detach(),
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
            parameters = BrainActionParameters(
                acceleration_loc=acceleration_loc.item(),
                consume_logit=consume_logit.item(),
                signal_logits=tuple(signal_logits.detach().cpu().tolist()),
            )
        return parameters, hidden

    def sample_brain_action(
        self,
        parameters: BrainActionParameters,
        random_source: _RandomSource,
    ) -> BrainActionSelection:
        """Select one intent and its predictor-facing motor representation."""

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
        normalized_acceleration = math.tanh(raw_motor_value)
        acceleration = (
            self._actuator_config.max_acceleration * normalized_acceleration
        )
        consume = draws[1] < parameters.consume_probability
        signal_index = self._categorical_index(
            parameters.signal_probabilities,
            draws[2],
        )
        signal_one_hot = tuple(
            1.0 if index == signal_index else 0.0
            for index in range(len(_SIGNAL_EMISSIONS))
        )
        return BrainActionSelection(
            action_intent=ActionIntent(
                acceleration=acceleration,
                consume=consume,
                signal_emission=_SIGNAL_EMISSIONS[signal_index],
            ),
            action_vector=(
                normalized_acceleration,
                1.0 if consume else 0.0,
                *signal_one_hot,
            ),
        )

    def sample_action(
        self,
        parameters: BrainActionParameters,
        random_source: _RandomSource,
    ) -> ActionIntent:
        """Consume exactly three ordered draws and construct one physical intent."""

        return self.sample_brain_action(parameters, random_source).action_intent

    def act(
        self,
        experience: NoraletExperience,
    ) -> ActionIntent:
        """Activate once and sample the corresponding low-level motor action."""

        if self._action_random_source is None:
            raise RuntimeError("this NoraletBrain has no action random stream")
        if self._pending_transition is not None:
            raise RuntimeError("the previous lived transition is still unresolved")
        if not self.learning_enabled:
            return self.sample_action(
                self.activate(experience),
                self._action_random_source,
            )

        parameters, hidden = self._activate(experience, track_gradient=True)
        selection = self.sample_brain_action(
            parameters,
            self._action_random_source,
        )
        predictor = self._model.prediction_model
        assert predictor is not None
        action_tensor = hidden.new_tensor(selection.action_vector)
        prediction = predictor(hidden, action_tensor)
        if not torch.isfinite(prediction).all().item():
            raise FloatingPointError(
                "next-experience prediction produced a non-finite value"
            )
        self._pending_transition = _PendingTransition(
            prediction=prediction,
            action_vector=selection.action_vector,
        )
        return selection.action_intent

    def learn(self, next_experience: NoraletExperience) -> PredictiveLearningResult:
        """Apply one Adam update from one actually lived next Experience."""

        if not isinstance(next_experience, NoraletExperience):
            raise TypeError("next_experience must be a NoraletExperience")
        if not self.learning_enabled:
            raise RuntimeError("predictive lifetime learning is disabled")
        if self._pending_transition is None:
            raise RuntimeError("there is no pending lived transition to learn")
        assert self._learning_config is not None
        assert self._target_experience_encoder is not None
        assert self._optimizer is not None
        pending = self._pending_transition
        plastic_parameters = self._model.predictive_plastic_parameters()

        self._optimizer.zero_grad(set_to_none=True)
        try:
            self._target_experience_encoder.eval()
            with torch.no_grad():
                target = self._target_experience_encoder(next_experience)
            if not torch.isfinite(target).all().item():
                raise FloatingPointError(
                    "target Experience encoder produced a non-finite value"
                )
            loss = F.mse_loss(pending.prediction, target)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("prediction loss is non-finite")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
                for parameter in plastic_parameters
            ):
                raise FloatingPointError("predictive gradient is non-finite")
            try:
                raw_gradient_norm = nn.utils.clip_grad_norm_(
                    plastic_parameters,
                    self._learning_config.max_gradient_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                raise FloatingPointError(
                    "predictive gradient norm is non-finite"
                ) from error
            raw_gradient_norm_value = float(raw_gradient_norm.item())
            if not math.isfinite(raw_gradient_norm_value):
                raise FloatingPointError("predictive gradient norm is non-finite")
            self._optimizer.step()
            if not self._learning_state_is_finite(plastic_parameters):
                raise FloatingPointError(
                    "predictive optimizer produced non-finite learning state"
                )
            self._learning_update_count += 1
            return PredictiveLearningResult(
                prediction_loss=float(loss.detach().item()),
                gradient_norm=min(
                    raw_gradient_norm_value,
                    self._learning_config.max_gradient_norm,
                ),
            )
        finally:
            self._optimizer.zero_grad(set_to_none=True)
            self._pending_transition = None

    def discard_pending_transition(self) -> None:
        """Release an unlived terminal prediction without fabricating a target."""

        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)
        self._pending_transition = None

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

    def _learning_state_is_finite(
        self,
        plastic_parameters: tuple[nn.Parameter, ...],
    ) -> bool:
        if not all(
            torch.isfinite(parameter).all().item()
            for parameter in plastic_parameters
        ):
            return False
        assert self._optimizer is not None
        return all(
            not isinstance(value, Tensor) or torch.isfinite(value).all().item()
            for state in self._optimizer.state.values()
            for value in state.values()
        )
