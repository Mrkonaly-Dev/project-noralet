"""Compact recurrent neural model and brain-facing action parameters."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from noralet.brain.config import NoraletBrainConfig
from noralet.brain.encoder import ExperienceEncoder
from noralet.brain.learning import NoraletLearningConfig
from noralet.noralets.experience import NoraletExperience


SIGNAL_MOTOR_OUTCOME_COUNT = 9
ACTION_VECTOR_SIZE = 11


@dataclass(frozen=True, slots=True)
class BrainActionParameters:
    """CPU scalar distribution parameters produced by one neural activation."""

    acceleration_loc: float
    consume_logit: float
    signal_logits: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("acceleration_loc", "consume_logit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, converted)
        if not isinstance(self.signal_logits, tuple):
            raise TypeError("signal_logits must be an immutable tuple")
        if len(self.signal_logits) != SIGNAL_MOTOR_OUTCOME_COUNT:
            raise ValueError("signal_logits must contain exactly nine values")
        logits = tuple(float(value) for value in self.signal_logits)
        if not all(math.isfinite(value) for value in logits):
            raise ValueError("signal_logits must be finite")
        object.__setattr__(self, "signal_logits", logits)

    @property
    def consume_probability(self) -> float:
        """Return the numerically stable sigmoid probability."""

        if self.consume_logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-self.consume_logit))
        exponential = math.exp(self.consume_logit)
        return exponential / (1.0 + exponential)

    @property
    def signal_probabilities(self) -> tuple[float, ...]:
        """Return the numerically stable nine-way softmax probabilities."""

        maximum = max(self.signal_logits)
        weights = tuple(math.exp(value - maximum) for value in self.signal_logits)
        total = math.fsum(weights)
        return tuple(weight / total for weight in weights)


class PredictionModel(nn.Module):
    """Compact action-conditioned predictor of the next sensory embedding."""

    def __init__(
        self,
        *,
        hidden_size: int,
        predictor_hidden_size: int,
        experience_embedding_size: int,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(
            hidden_size + ACTION_VECTOR_SIZE,
            predictor_hidden_size,
        )
        self.output_layer = nn.Linear(
            predictor_hidden_size,
            experience_embedding_size,
        )

    def forward(self, hidden_state: Tensor, action_vector: Tensor) -> Tensor:
        if hidden_state.ndim != 1:
            raise ValueError("hidden_state must be a one-dimensional tensor")
        if action_vector.ndim != 1 or action_vector.shape[0] != ACTION_VECTOR_SIZE:
            raise ValueError("action_vector must contain exactly eleven values")
        if hidden_state.device != action_vector.device:
            raise ValueError("hidden_state and action_vector must share a device")
        combined = torch.tanh(
            self.input_layer(torch.cat((hidden_state, action_vector)))
        )
        return self.output_layer(combined)


class NoraletBrainModel(nn.Module):
    """Recurrent action model with an optional inherited forward predictor."""

    def __init__(
        self,
        config: NoraletBrainConfig,
        *,
        external_pattern_length: int,
        signal_pattern_length: int,
        learning_config: NoraletLearningConfig | None = None,
    ) -> None:
        super().__init__()
        if learning_config is not None and not isinstance(
            learning_config,
            NoraletLearningConfig,
        ):
            raise TypeError("learning_config must be a NoraletLearningConfig")
        self.encoder = ExperienceEncoder(
            config,
            external_pattern_length=external_pattern_length,
            signal_pattern_length=signal_pattern_length,
        )
        self.recurrent_core = nn.GRUCell(
            config.experience_embedding_size,
            config.hidden_size,
        )
        self.acceleration_head = nn.Linear(config.hidden_size, 1)
        self.consume_head = nn.Linear(config.hidden_size, 1)
        self.signal_head = nn.Linear(
            config.hidden_size,
            SIGNAL_MOTOR_OUTCOME_COUNT,
        )
        self.prediction_model = (
            None
            if learning_config is None
            else PredictionModel(
                hidden_size=config.hidden_size,
                predictor_hidden_size=learning_config.predictor_hidden_size,
                experience_embedding_size=config.experience_embedding_size,
            )
        )

    def iteration_8_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return pre-predictor parameters in their original initialization order."""

        modules = (
            self.encoder,
            self.recurrent_core,
            self.acceleration_head,
            self.consume_head,
            self.signal_head,
        )
        return tuple(
            parameter
            for module in modules
            for parameter in module.parameters()
        )

    def predictive_plastic_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return exactly encoder, recurrent-core and predictor parameters."""

        if self.prediction_model is None:
            return ()
        return tuple(
            parameter
            for module in (
                self.encoder,
                self.recurrent_core,
                self.prediction_model,
            )
            for parameter in module.parameters()
        )

    def action_head_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the inherited motor-head parameters excluded from learning."""

        return tuple(
            parameter
            for module in (
                self.acceleration_head,
                self.consume_head,
                self.signal_head,
            )
            for parameter in module.parameters()
        )

    def forward(
        self,
        experience: NoraletExperience,
        hidden_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        embedding = self.encoder(experience)
        hidden = self.recurrent_core(embedding, hidden_state)
        return (
            hidden,
            self.acceleration_head(hidden).squeeze(-1),
            self.consume_head(hidden).squeeze(-1),
            self.signal_head(hidden),
        )
