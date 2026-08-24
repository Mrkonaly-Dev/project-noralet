"""Structured descriptions of completed simulation transitions."""

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import TypeAlias

from noralet.noralets.signals import SignalDirection, SignalType


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


def _validate_point_id(point_id: int) -> None:
    if type(point_id) is not int:
        raise TypeError("point_id must be an integer")
    if point_id < 0:
        raise ValueError("point_id cannot be negative")


def _validate_region_id(region_id: str) -> None:
    if not isinstance(region_id, str):
        raise TypeError("region_id must be a string")
    if not region_id:
        raise ValueError("region_id cannot be empty")


def _validate_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
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
    ENERGY_DEPLETION = "energy_depletion"
    NATURAL = "natural"


@dataclass(frozen=True, slots=True)
class NoraletDied:
    """Records removal of a body during one resolved transition."""

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


@dataclass(frozen=True, slots=True)
class EnergyPointFormed:
    """Records Environmental Energy transferred into a new point."""

    region_id: str
    point_id: int
    position: float
    energy: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_region_id(self.region_id)
        _validate_point_id(self.point_id)
        _validate_finite("position", self.position)
        _validate_finite("energy", self.energy)
        if self.energy <= 0.0:
            raise ValueError("formed energy must be positive")
        _validate_transition(self.tick_before, self.tick_after)


@dataclass(frozen=True, slots=True)
class EnergyPointDecayed:
    """Records proportional energy returned by one existing point."""

    region_id: str
    point_id: int
    energy_returned: float
    remaining_energy: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_region_id(self.region_id)
        _validate_point_id(self.point_id)
        _validate_finite("energy_returned", self.energy_returned)
        _validate_finite("remaining_energy", self.remaining_energy)
        if self.energy_returned <= 0.0:
            raise ValueError("decayed energy transfer must be positive")
        if self.remaining_energy < 0.0:
            raise ValueError("remaining energy cannot be negative")
        _validate_transition(self.tick_before, self.tick_after)


@dataclass(frozen=True, slots=True)
class EnergyPointDissolved:
    """Records point removal and return of its complete final remainder."""

    region_id: str
    point_id: int
    energy_returned: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_region_id(self.region_id)
        _validate_point_id(self.point_id)
        _validate_finite("energy_returned", self.energy_returned)
        if self.energy_returned < 0.0:
            raise ValueError("dissolved energy transfer cannot be negative")
        _validate_transition(self.tick_before, self.tick_after)


@dataclass(frozen=True, slots=True)
class EnergyConsumed:
    """Records Consumable Energy transferred into one Noralet."""

    noralet_id: int
    point_id: int
    energy_transferred: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_noralet_id(self.noralet_id)
        _validate_point_id(self.point_id)
        _validate_finite("energy_transferred", self.energy_transferred)
        if self.energy_transferred <= 0.0:
            raise ValueError("consumed energy transfer must be positive")
        _validate_transition(self.tick_before, self.tick_after)


class NoraletEnergyExpenditureReason(StrEnum):
    """Machine-readable reasons for Noralet-to-environment transfers."""

    EXISTENCE = "existence"
    ACCELERATION = "acceleration"
    SIGNAL = "signal"


@dataclass(frozen=True, slots=True)
class NoraletEnergySpent:
    """Records Noralet Energy expended into a local environmental pool."""

    noralet_id: int
    region_id: str
    reason: NoraletEnergyExpenditureReason
    energy_transferred: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_noralet_id(self.noralet_id)
        _validate_region_id(self.region_id)
        if not isinstance(self.reason, NoraletEnergyExpenditureReason):
            raise TypeError("reason must be a NoraletEnergyExpenditureReason")
        _validate_finite("energy_transferred", self.energy_transferred)
        if self.energy_transferred <= 0.0:
            raise ValueError("spent energy transfer must be positive")
        _validate_transition(self.tick_before, self.tick_after)


@dataclass(frozen=True, slots=True)
class NoraletEnergyReleased:
    """Records remaining Noralet Energy returned on death."""

    noralet_id: int
    region_id: str
    energy_transferred: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_noralet_id(self.noralet_id)
        _validate_region_id(self.region_id)
        _validate_finite("energy_transferred", self.energy_transferred)
        if self.energy_transferred <= 0.0:
            raise ValueError("released energy transfer must be positive")
        _validate_transition(self.tick_before, self.tick_after)


@dataclass(frozen=True, slots=True)
class SignalEmitted:
    """Records one successfully executed physical signal emission."""

    noralet_id: int
    signal_type: SignalType
    emission_direction: SignalDirection
    origin: float
    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        _validate_noralet_id(self.noralet_id)
        if not isinstance(self.signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType")
        if not isinstance(self.emission_direction, SignalDirection):
            raise TypeError("emission_direction must be a SignalDirection")
        _validate_finite("origin", self.origin)
        _validate_transition(self.tick_before, self.tick_after)


SimulationEvent: TypeAlias = (
    TickAdvanced
    | NoraletAccelerated
    | NoraletMoved
    | NoraletDied
    | EnergyPointFormed
    | EnergyPointDecayed
    | EnergyPointDissolved
    | EnergyConsumed
    | NoraletEnergySpent
    | NoraletEnergyReleased
    | SignalEmitted
)
