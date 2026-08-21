"""Structured descriptions of completed simulation transitions."""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class TickAdvanced:
    """Records the authoritative clock advancing by one tick."""

    tick_before: int
    tick_after: int

    def __post_init__(self) -> None:
        if self.tick_before < 0:
            raise ValueError("tick_before cannot be negative")
        if self.tick_after != self.tick_before + 1:
            raise ValueError("TickAdvanced must describe exactly one tick")


SimulationEvent: TypeAlias = TickAdvanced

