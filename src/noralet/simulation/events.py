"""Structured descriptions of completed simulation transitions."""

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import TypeAlias


def _validate_transition(tick_before: int, tick_after: int) -> None:
    if type(tick_before) is not int or type(tick_after) is not int:
        raise TypeError("event ticks must be integers")
    if tick_before < 0:
        raise ValueError("tick_before cannot be negative")
    if tick_after != tick_before + 1:
        raise ValueError("event must describe exactly one tick")


def _validate_noralet_id(noralet_id: int) -> None:
    if type(noralet_id) is not int:
        raise TypeError("noralet_id must be an integer")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class TickAdvanced:
    """Records the authoritative clock advancing by one tick."""

    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_transition(self.tick_before, self.tick_after)


@dataclass(frozen=True, slots=True)
class NoraletAccelerated:
    """Records a non-zero acceleration applied during a transition."""

    noralet_id: int
    acceleration: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_noralet_id(self.noralet_id)
        _validate_finite("acceleration", self.acceleration)
        if self.acceleration == 0.0:
            raise ValueError("NoraletAccelerated requires non-zero acceleration")
        _validate_transition(self.tick_before, self.tick_after)


@dataclass(frozen=True, slots=True)
class NoraletMoved:
    """Records a body's resolved position and velocity change."""

    noralet_id: int
    position_before: float
    position_after: float
    velocity_after: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_noralet_id(self.noralet_id)
        _validate_finite("position_before", self.position_before)
        _validate_finite("position_after", self.position_after)
        _validate_finite("velocity_after", self.velocity_after)
        if self.position_before == self.position_after:
            raise ValueError("NoraletMoved requires an actual position change")
        _validate_transition(self.tick_before, self.tick_after)


class NoraletDeathCause(StrEnum):
    """Machine-readable causes currently implemented by the simulation."""

    WORLD_BOUNDARY = "world_boundary"


@dataclass(frozen=True, slots=True)
class NoraletDied:
    """Records removal of a body that crossed a world boundary."""

    noralet_id: int
    cause: NoraletDeathCause
    resolved_position: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_noralet_id(self.noralet_id)
        if not isinstance(self.cause, NoraletDeathCause):
            raise TypeError("cause must be a NoraletDeathCause")
        _validate_finite("resolved_position", self.resolved_position)
        _validate_transition(self.tick_before, self.tick_after)


SimulationEvent: TypeAlias = (
    TickAdvanced | NoraletAccelerated | NoraletMoved | NoraletDied
)
