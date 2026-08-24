"""Immutable Noralet-facing sensory values and experience configuration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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


def _unit_signal(name: str, value: float, *, upper_inclusive: bool) -> float:
    signal = _finite_float(name, value)
    valid = 0.0 <= signal <= 1.0 if upper_inclusive else 0.0 <= signal < 1.0
    if not valid:
        interval = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must be in {interval}")
    return signal


def _direction_signal(name: str, value: float) -> float:
    direction = _finite_float(name, value)
    if direction not in (-1.0, 0.0, 1.0):
        raise ValueError(f"{name} must be -1, 0, or 1")
    return direction


@dataclass(frozen=True, slots=True)
class NoraletExperienceConfig:
    """External-sensory and bodily-sensation experiment parameters."""

    vision_radius: float
    consumable_base_appearance: tuple[float, ...]
    noralet_base_appearance: tuple[float, ...]
    boundary_base_appearance: tuple[float, ...]
    signature_length: int
    energy_distress_exponent: float
    condition_distress_exponent: float
    motor_effort_scale: float
    ingestion_sensation_scale: float
    exertion_sensation_scale: float

    def __post_init__(self) -> None:
        vision_radius = _finite_float("vision_radius", self.vision_radius)
        if vision_radius <= 0.0:
            raise ValueError("vision_radius must be positive")

        pattern_names = (
            "consumable_base_appearance",
            "noralet_base_appearance",
            "boundary_base_appearance",
        )
        patterns = tuple(
            _finite_vector(name, getattr(self, name)) for name in pattern_names
        )
        if any(not pattern for pattern in patterns):
            raise ValueError("base appearance patterns cannot be empty")
        if len({len(pattern) for pattern in patterns}) != 1:
            raise ValueError("base appearance patterns must have equal length")
        if len(set(patterns)) != len(patterns):
            raise ValueError("base appearance patterns must be distinguishable")

        if type(self.signature_length) is not int:
            raise TypeError("signature_length must be an integer")
        if self.signature_length <= 0:
            raise ValueError("signature_length must be positive")

        energy_exponent = _finite_float(
            "energy_distress_exponent",
            self.energy_distress_exponent,
        )
        condition_exponent = _finite_float(
            "condition_distress_exponent",
            self.condition_distress_exponent,
        )
        motor_scale = _finite_float("motor_effort_scale", self.motor_effort_scale)
        ingestion_scale = _finite_float(
            "ingestion_sensation_scale",
            self.ingestion_sensation_scale,
        )
        exertion_scale = _finite_float(
            "exertion_sensation_scale",
            self.exertion_sensation_scale,
        )
        if energy_exponent <= 1.0:
            raise ValueError("energy_distress_exponent must be greater than 1")
        if condition_exponent <= 0.0:
            raise ValueError("condition_distress_exponent must be positive")
        if motor_scale <= 0.0:
            raise ValueError("motor_effort_scale must be positive")
        if ingestion_scale <= 0.0:
            raise ValueError("ingestion_sensation_scale must be positive")
        if exertion_scale <= 0.0:
            raise ValueError("exertion_sensation_scale must be positive")

        object.__setattr__(self, "vision_radius", vision_radius)
        for name, pattern in zip(pattern_names, patterns):
            object.__setattr__(self, name, pattern)
        object.__setattr__(self, "energy_distress_exponent", energy_exponent)
        object.__setattr__(self, "condition_distress_exponent", condition_exponent)
        object.__setattr__(self, "motor_effort_scale", motor_scale)
        object.__setattr__(self, "ingestion_sensation_scale", ingestion_scale)
        object.__setattr__(self, "exertion_sensation_scale", exertion_scale)

    @property
    def base_pattern_length(self) -> int:
        """Return the shared configured base-appearance length."""

        return len(self.consumable_base_appearance)

    @property
    def appearance_length(self) -> int:
        """Return the uniform complete external-appearance length."""

        return self.base_pattern_length + self.signature_length


@dataclass(frozen=True, slots=True)
class ExternalPercept:
    """Uniform semantic-free appearance and relative spatial sensation."""

    appearance_pattern: tuple[float, ...]
    direction_signal: float
    proximity_signal: float

    def __post_init__(self) -> None:
        appearance = _finite_vector("appearance_pattern", self.appearance_pattern)
        if not appearance:
            raise ValueError("appearance_pattern cannot be empty")
        object.__setattr__(self, "appearance_pattern", appearance)
        object.__setattr__(
            self,
            "direction_signal",
            _direction_signal("direction_signal", self.direction_signal),
        )
        object.__setattr__(
            self,
            "proximity_signal",
            _unit_signal(
                "proximity_signal",
                self.proximity_signal,
                upper_inclusive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class SignalPercept:
    """Meaningless signal pattern with source direction and bounded strength."""

    signal_pattern: tuple[float, ...]
    direction_signal: float
    strength_signal: float

    def __post_init__(self) -> None:
        pattern = _finite_vector("signal_pattern", self.signal_pattern)
        if not pattern:
            raise ValueError("signal_pattern cannot be empty")
        object.__setattr__(self, "signal_pattern", pattern)
        object.__setattr__(
            self,
            "direction_signal",
            _direction_signal("direction_signal", self.direction_signal),
        )
        object.__setattr__(
            self,
            "strength_signal",
            _unit_signal(
                "strength_signal",
                self.strength_signal,
                upper_inclusive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class Interoception:
    """Derived current bodily distress and recent expenditure sensation."""

    energy_distress: float
    condition_distress: float
    energetic_exertion: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "energy_distress",
            _unit_signal(
                "energy_distress",
                self.energy_distress,
                upper_inclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "condition_distress",
            _unit_signal(
                "condition_distress",
                self.condition_distress,
                upper_inclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "energetic_exertion",
            _unit_signal(
                "energetic_exertion",
                self.energetic_exertion,
                upper_inclusive=False,
            ),
        )


@dataclass(frozen=True, slots=True)
class SensorimotorFeedback:
    """Derived feedback from the transition that produced this experience."""

    motor_direction: float
    motor_effort: float
    consume_activation: float
    ingestion_signal: float
    signal_emission_activation: float = 0.0
    signal_emission_pattern: tuple[float, ...] = ()
    signal_emission_direction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "motor_direction",
            _direction_signal("motor_direction", self.motor_direction),
        )
        object.__setattr__(
            self,
            "motor_effort",
            _unit_signal(
                "motor_effort",
                self.motor_effort,
                upper_inclusive=False,
            ),
        )
        consume_activation = _finite_float(
            "consume_activation",
            self.consume_activation,
        )
        if consume_activation not in (0.0, 1.0):
            raise ValueError("consume_activation must be 0 or 1")
        object.__setattr__(self, "consume_activation", consume_activation)
        object.__setattr__(
            self,
            "ingestion_signal",
            _unit_signal(
                "ingestion_signal",
                self.ingestion_signal,
                upper_inclusive=False,
            ),
        )
        emission_activation = _finite_float(
            "signal_emission_activation",
            self.signal_emission_activation,
        )
        if emission_activation not in (0.0, 1.0):
            raise ValueError("signal_emission_activation must be 0 or 1")
        emission_pattern = _finite_vector(
            "signal_emission_pattern",
            self.signal_emission_pattern,
        )
        emission_direction = _direction_signal(
            "signal_emission_direction",
            self.signal_emission_direction,
        )
        if emission_activation == 0.0:
            if any(value != 0.0 for value in emission_pattern):
                raise ValueError(
                    "inactive signal emission pattern must be neutral"
                )
            if emission_direction != 0.0:
                raise ValueError(
                    "inactive signal emission direction must be neutral"
                )
        else:
            if not emission_pattern:
                raise ValueError("active signal emission pattern cannot be empty")
            if emission_direction not in (-1.0, 1.0):
                raise ValueError(
                    "active signal emission direction must be -1 or 1"
                )
        object.__setattr__(
            self,
            "signal_emission_activation",
            emission_activation,
        )
        object.__setattr__(
            self,
            "signal_emission_pattern",
            emission_pattern,
        )
        object.__setattr__(
            self,
            "signal_emission_direction",
            emission_direction,
        )


@dataclass(frozen=True, slots=True)
class NoraletExperience:
    """Complete immutable brain-facing experience without routing identity."""

    external_percepts: tuple[ExternalPercept, ...]
    signal_percepts: tuple[SignalPercept, ...]
    interoception: Interoception
    sensorimotor_feedback: SensorimotorFeedback

    def __post_init__(self) -> None:
        if not isinstance(self.external_percepts, tuple):
            raise TypeError("external_percepts must be an immutable tuple")
        if not all(
            isinstance(percept, ExternalPercept)
            for percept in self.external_percepts
        ):
            raise TypeError("every external percept must be an ExternalPercept")
        if not isinstance(self.signal_percepts, tuple):
            raise TypeError("signal_percepts must be an immutable tuple")
        if not all(
            isinstance(percept, SignalPercept) for percept in self.signal_percepts
        ):
            raise TypeError("every signal percept must be a SignalPercept")
        if not isinstance(self.interoception, Interoception):
            raise TypeError("interoception must be an Interoception")
        if not isinstance(self.sensorimotor_feedback, SensorimotorFeedback):
            raise TypeError(
                "sensorimotor_feedback must be a SensorimotorFeedback"
            )
