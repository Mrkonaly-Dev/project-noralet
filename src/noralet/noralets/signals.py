"""Meaningless physical signal channels and emission configuration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
import math


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _finite_vector(name: str, values: Iterable[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, dict, set, frozenset)):
        raise TypeError(f"{name} must be an ordered numeric iterable")
    try:
        vector = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an ordered numeric iterable") from error
    return tuple(_finite_float(f"{name} value", value) for value in vector)


class SignalType(StrEnum):
    """Exactly four engine/observer signal identifiers without semantics."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class SignalDirection(StrEnum):
    """The two physical half-lines available to signal emission."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class SignalEmissionIntent:
    """Request exactly one signal channel in exactly one direction."""

    signal_type: SignalType
    direction: SignalDirection

    def __post_init__(self) -> None:
        if not isinstance(self.signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType")
        if not isinstance(self.direction, SignalDirection):
            raise TypeError("direction must be a SignalDirection")


@dataclass(frozen=True, slots=True)
class NoraletSignalConfig:
    """Physical range, cost and sensory patterns for the four channels."""

    signal_radius: float
    signal_energy_cost: float
    signal_pattern_a: tuple[float, ...]
    signal_pattern_b: tuple[float, ...]
    signal_pattern_c: tuple[float, ...]
    signal_pattern_d: tuple[float, ...]

    def __post_init__(self) -> None:
        radius = _finite_float("signal_radius", self.signal_radius)
        cost = _finite_float("signal_energy_cost", self.signal_energy_cost)
        if radius <= 0.0:
            raise ValueError("signal_radius must be positive")
        if cost < 0.0:
            raise ValueError("signal_energy_cost cannot be negative")

        pattern_names = (
            "signal_pattern_a",
            "signal_pattern_b",
            "signal_pattern_c",
            "signal_pattern_d",
        )
        patterns = tuple(
            _finite_vector(name, getattr(self, name)) for name in pattern_names
        )
        if any(not pattern for pattern in patterns):
            raise ValueError("signal patterns cannot be empty")
        if len({len(pattern) for pattern in patterns}) != 1:
            raise ValueError("signal patterns must have equal length")
        if len(set(patterns)) != len(patterns):
            raise ValueError("signal patterns must be pairwise distinguishable")

        object.__setattr__(self, "signal_radius", radius)
        object.__setattr__(self, "signal_energy_cost", cost)
        for name, pattern in zip(pattern_names, patterns):
            object.__setattr__(self, name, pattern)

    @property
    def signal_pattern_length(self) -> int:
        """Return the common sensory-pattern length."""

        return len(self.signal_pattern_a)

    def pattern_for(self, signal_type: SignalType) -> tuple[float, ...]:
        """Return the configured sensory pattern for one engine channel."""

        if not isinstance(signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType")
        return {
            SignalType.A: self.signal_pattern_a,
            SignalType.B: self.signal_pattern_b,
            SignalType.C: self.signal_pattern_c,
            SignalType.D: self.signal_pattern_d,
        }[signal_type]
