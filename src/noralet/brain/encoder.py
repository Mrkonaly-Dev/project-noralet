"""Permutation-invariant neural encoding of brain-facing Experience values."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from noralet.brain.config import NoraletBrainConfig
from noralet.noralets.experience import (
    ExternalPercept,
    NoraletExperience,
    SignalPercept,
)


def _small_mlp(input_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.Tanh(),
        nn.Linear(output_size, output_size),
        nn.Tanh(),
    )


class ExperienceEncoder(nn.Module):
    """Encode heterogeneous variable-length Experience into one fixed vector."""

    def __init__(
        self,
        config: NoraletBrainConfig,
        *,
        external_pattern_length: int,
        signal_pattern_length: int,
    ) -> None:
        super().__init__()
        if not isinstance(config, NoraletBrainConfig):
            raise TypeError("config must be a NoraletBrainConfig")
        for name, value in (
            ("external_pattern_length", external_pattern_length),
            ("signal_pattern_length", signal_pattern_length),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.external_pattern_length = external_pattern_length
        self.signal_pattern_length = signal_pattern_length
        self.external_percept_mlp = _small_mlp(
            external_pattern_length + 2,
            config.external_percept_embedding_size,
        )
        self.signal_percept_mlp = _small_mlp(
            signal_pattern_length + 2,
            config.signal_percept_embedding_size,
        )
        self.interoception_mlp = _small_mlp(
            3,
            config.interoception_embedding_size,
        )
        self.sensorimotor_mlp = _small_mlp(
            signal_pattern_length + 6,
            config.sensorimotor_embedding_size,
        )
        fused_size = (
            config.external_percept_embedding_size
            + config.signal_percept_embedding_size
            + config.interoception_embedding_size
            + config.sensorimotor_embedding_size
        )
        self.fusion = _small_mlp(
            fused_size,
            config.experience_embedding_size,
        )

    def forward(self, experience: NoraletExperience) -> Tensor:
        if not isinstance(experience, NoraletExperience):
            raise TypeError("experience must be a NoraletExperience")
        external = self.encode_external(experience.external_percepts)
        signals = self.encode_signals(experience.signal_percepts)
        interoception = experience.interoception
        interoceptive = self.interoception_mlp(
            self._tensor(
                (
                    interoception.energy_distress,
                    interoception.condition_distress,
                    interoception.energetic_exertion,
                ),
                expected_length=3,
            )
        )
        feedback = experience.sensorimotor_feedback
        sensorimotor = self.sensorimotor_mlp(
            self._tensor(
                (
                    feedback.motor_direction,
                    feedback.motor_effort,
                    feedback.consume_activation,
                    feedback.ingestion_signal,
                    feedback.signal_emission_activation,
                    *feedback.signal_emission_pattern,
                    feedback.signal_emission_direction,
                ),
                expected_length=self.signal_pattern_length + 6,
            )
        )
        return self.fusion(
            torch.cat((external, signals, interoceptive, sensorimotor), dim=0)
        )

    def encode_external(
        self,
        percepts: Sequence[ExternalPercept],
    ) -> Tensor:
        """Return the shared-MLP, sum-pooled external-field summary."""

        if not isinstance(percepts, (tuple, list)):
            raise TypeError("external percepts must be a finite sequence")
        if not all(isinstance(percept, ExternalPercept) for percept in percepts):
            raise TypeError("every external percept must be an ExternalPercept")
        embeddings = tuple(
            self.external_percept_mlp(
                self._tensor(
                    (
                        *percept.appearance_pattern,
                        percept.direction_signal,
                        percept.proximity_signal,
                    ),
                    expected_length=self.external_pattern_length + 2,
                )
            )
            for percept in percepts
        )
        return self._sum_or_zero(
            embeddings,
            self.external_percept_mlp[-2].out_features,
        )

    def encode_signals(
        self,
        percepts: Sequence[SignalPercept],
    ) -> Tensor:
        """Return the separate shared-MLP, sum-pooled signal summary."""

        if not isinstance(percepts, (tuple, list)):
            raise TypeError("signal percepts must be a finite sequence")
        if not all(isinstance(percept, SignalPercept) for percept in percepts):
            raise TypeError("every signal percept must be a SignalPercept")
        embeddings = tuple(
            self.signal_percept_mlp(
                self._tensor(
                    (
                        *percept.signal_pattern,
                        percept.direction_signal,
                        percept.strength_signal,
                    ),
                    expected_length=self.signal_pattern_length + 2,
                )
            )
            for percept in percepts
        )
        return self._sum_or_zero(
            embeddings,
            self.signal_percept_mlp[-2].out_features,
        )

    def _sum_or_zero(
        self,
        embeddings: tuple[Tensor, ...],
        size: int,
    ) -> Tensor:
        if embeddings:
            return torch.stack(embeddings, dim=0).sum(dim=0)
        parameter = next(self.parameters())
        return torch.zeros(size, dtype=parameter.dtype, device=parameter.device)

    def _tensor(
        self,
        values: Sequence[float],
        *,
        expected_length: int,
    ) -> Tensor:
        if len(values) != expected_length:
            raise ValueError(
                f"sensory vector must have length {expected_length}, "
                f"received {len(values)}"
            )
        parameter = next(self.parameters())
        return torch.tensor(
            values,
            dtype=parameter.dtype,
            device=parameter.device,
        )
