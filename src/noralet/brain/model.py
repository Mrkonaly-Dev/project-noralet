"""Compact recurrent neural model and brain-facing action parameters."""

from __future__ import annotations

from dataclasses import dataclass
import math

from torch import Tensor, nn

from noralet.brain.config import NoraletBrainConfig
from noralet.brain.encoder import ExperienceEncoder
from noralet.noralets.experience import NoraletExperience


SIGNAL_MOTOR_OUTCOME_COUNT = 9


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


class NoraletBrainModel(nn.Module):
    """Experience encoder, one GRUCell and three low-level action heads."""

    def __init__(
        self,
        config: NoraletBrainConfig,
        *,
        external_pattern_length: int,
        signal_pattern_length: int,
    ) -> None:
        super().__init__()
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
