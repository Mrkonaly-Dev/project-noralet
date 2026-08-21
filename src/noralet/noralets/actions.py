"""Immutable intentions proposed for Noralet bodies."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ActionIntent:
    """Request acceleration and optionally attempt local energy consumption."""

    acceleration: float = 0.0
    consume: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.acceleration, bool) or not isinstance(
            self.acceleration, (int, float)
        ):
            raise TypeError("acceleration must be a real number")

        acceleration = float(self.acceleration)
        if not math.isfinite(acceleration):
            raise ValueError("acceleration must be finite")
        if type(self.consume) is not bool:
            raise TypeError("consume must be a boolean")
        object.__setattr__(self, "acceleration", acceleration)
