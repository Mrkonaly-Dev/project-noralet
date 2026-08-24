"""Immutable physical actuator limits for Noralet bodies."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class NoraletActuatorConfig:
    """Physical acceleration capability shared by configured bodies."""

    max_acceleration: float

    def __post_init__(self) -> None:
        value = self.max_acceleration
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("max_acceleration must be a real number")
        maximum = float(value)
        if not math.isfinite(maximum):
            raise ValueError("max_acceleration must be finite")
        if maximum <= 0.0:
            raise ValueError("max_acceleration must be positive")
        object.__setattr__(self, "max_acceleration", maximum)
