"""Immutable physical state of a living Noralet."""

from collections.abc import Iterable
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class NoraletBodyState:
    """Immutable objective physical and slow physiological body state."""

    noralet_id: int
    position: float
    velocity: float = 0.0
    energy: float = 0.0
    age_ticks: int = 0
    condition: float = 1.0
    perceptual_signature: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if type(self.noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")

        position = self._finite_float("position", self.position)
        velocity = self._finite_float("velocity", self.velocity)
        energy = self._finite_float("energy", self.energy)
        if energy < 0.0:
            raise ValueError("Noralet energy cannot be negative")
        if type(self.age_ticks) is not int:
            raise TypeError("age_ticks must be an integer")
        if self.age_ticks < 0:
            raise ValueError("age_ticks cannot be negative")
        condition = self._finite_float("condition", self.condition)
        if not 0.0 <= condition <= 1.0:
            raise ValueError("condition must be in [0, 1]")
        signature = self._finite_signature(self.perceptual_signature)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "perceptual_signature", signature)

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        return converted

    @classmethod
    def _finite_signature(
        cls,
        values: Iterable[float],
    ) -> tuple[float, ...]:
        if isinstance(values, (str, bytes, dict, set, frozenset)):
            raise TypeError("perceptual_signature must be an ordered numeric iterable")
        try:
            signature = tuple(values)
        except TypeError as error:
            raise TypeError(
                "perceptual_signature must be an ordered numeric iterable"
            ) from error
        return tuple(
            cls._finite_float("perceptual_signature value", value)
            for value in signature
        )
