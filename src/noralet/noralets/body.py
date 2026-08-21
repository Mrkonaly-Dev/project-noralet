"""Immutable physical state of a living Noralet."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class NoraletBodyState:
    """Simulation identity, position and velocity for one living body."""

    noralet_id: int
    position: float
    velocity: float = 0.0

    def __post_init__(self) -> None:
        if type(self.noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")

        position = self._finite_float("position", self.position)
        velocity = self._finite_float("velocity", self.velocity)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        return converted
