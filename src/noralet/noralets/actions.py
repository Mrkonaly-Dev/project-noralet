"""Immutable intentions proposed for Noralet bodies."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ActionIntent:
    """Request acceleration along the canonical one-dimensional axis."""

    acceleration: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.acceleration, bool) or not isinstance(
            self.acceleration, (int, float)
        ):
            raise TypeError("acceleration must be a real number")

        acceleration = float(self.acceleration)
        if not math.isfinite(acceleration):
            raise ValueError("acceleration must be finite")
        object.__setattr__(self, "acceleration", acceleration)

