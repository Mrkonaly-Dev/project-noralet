"""Objective physical state for one simulation tick."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorldState:
    """Immutable Iteration 1 world state containing only its tick."""

    tick: int = 0

    def __post_init__(self) -> None:
        if type(self.tick) is not int:
            raise TypeError("tick must be an integer")
        if self.tick < 0:
            raise ValueError("tick cannot be negative")

