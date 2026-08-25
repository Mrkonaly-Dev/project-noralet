"""Focused immutable configuration for the Iteration 8 neural substrate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import torch


_SUPPORTED_DEVICES = frozenset(("cpu", "cuda", "auto"))
BASE_BRAIN_INITIALIZATION_VERSION = "002-neutral-actuator-baselines"


def _open_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 < converted < 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1)")
    return converted


@dataclass(frozen=True, slots=True)
class BaseBrainInitializationConfig:
    """Versioned minimal priors for stable newborn neural operation."""

    version: str = BASE_BRAIN_INITIALIZATION_VERSION
    acceleration_output_weight_scale: float = 0.01
    initial_consume_probability: float = 0.05
    initial_signal_probability: float = 0.05

    def __post_init__(self) -> None:
        if self.version != BASE_BRAIN_INITIALIZATION_VERSION:
            raise ValueError(
                "version must identify the current neutral initializer: "
                f"{BASE_BRAIN_INITIALIZATION_VERSION}"
            )
        scale = self.acceleration_output_weight_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError(
                "acceleration_output_weight_scale must be a real number"
            )
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "acceleration_output_weight_scale must be finite and positive"
            )
        object.__setattr__(self, "acceleration_output_weight_scale", scale)
        object.__setattr__(
            self,
            "initial_consume_probability",
            _open_probability(
                "initial_consume_probability",
                self.initial_consume_probability,
            ),
        )
        object.__setattr__(
            self,
            "initial_signal_probability",
            _open_probability(
                "initial_signal_probability",
                self.initial_signal_probability,
            ),
        )


def base_brain_initialization_manifest(
    config: BaseBrainInitializationConfig | None = None,
) -> dict[str, object]:
    """Return compact serializable provenance for a random BaseBrain birth."""

    selected = BaseBrainInitializationConfig() if config is None else config
    if not isinstance(selected, BaseBrainInitializationConfig):
        raise TypeError("config must be a BaseBrainInitializationConfig")
    return dict(asdict(selected))


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
    initialization: BaseBrainInitializationConfig = field(
        default_factory=BaseBrainInitializationConfig
    )

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
        if not isinstance(self.initialization, BaseBrainInitializationConfig):
            raise TypeError(
                "initialization must be a BaseBrainInitializationConfig"
            )

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
