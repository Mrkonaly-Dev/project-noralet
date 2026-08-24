"""Focused immutable configuration for the Iteration 8 neural substrate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


_SUPPORTED_DEVICES = frozenset(("cpu", "cuda", "auto"))


@dataclass(frozen=True, slots=True)
class NoraletBrainConfig:
    """Dimensions, exploration and deterministic prototype configuration."""

    base_brain_seed: int
    external_percept_embedding_size: int
    signal_percept_embedding_size: int
    interoception_embedding_size: int
    sensorimotor_embedding_size: int
    experience_embedding_size: int
    hidden_size: int
    acceleration_exploration_std: float
    device: str = "cpu"

    def __post_init__(self) -> None:
        if type(self.base_brain_seed) is not int:
            raise TypeError("base_brain_seed must be an integer")
        for name in (
            "external_percept_embedding_size",
            "signal_percept_embedding_size",
            "interoception_embedding_size",
            "sensorimotor_embedding_size",
            "experience_embedding_size",
            "hidden_size",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        exploration = self.acceleration_exploration_std
        if isinstance(exploration, bool) or not isinstance(
            exploration,
            (int, float),
        ):
            raise TypeError("acceleration_exploration_std must be a real number")
        exploration = float(exploration)
        if not math.isfinite(exploration):
            raise ValueError("acceleration_exploration_std must be finite")
        if exploration < 0.0:
            raise ValueError("acceleration_exploration_std cannot be negative")

        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        device = self.device.strip().lower()
        if device not in _SUPPORTED_DEVICES:
            choices = ", ".join(sorted(_SUPPORTED_DEVICES))
            raise ValueError(f"device must be one of: {choices}")

        object.__setattr__(self, "acceleration_exploration_std", exploration)
        object.__setattr__(self, "device", device)


def resolve_brain_device(choice: str) -> torch.device:
    """Resolve an explicit neural device without silent CUDA fallback."""

    if not isinstance(choice, str):
        raise TypeError("device choice must be a string")
    normalized = choice.strip().lower()
    if normalized not in _SUPPORTED_DEVICES:
        choices = ", ".join(sorted(_SUPPORTED_DEVICES))
        raise ValueError(f"device must be one of: {choices}")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        return torch.device("cuda")
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")
