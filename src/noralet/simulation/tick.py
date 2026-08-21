"""Observer-facing result of one completed simulation transition."""

from dataclasses import dataclass

from noralet.simulation.events import SimulationEvent


@dataclass(frozen=True, slots=True)
class TickResult:
    """Immutable structured record returned by ``Simulation.step()``."""

    tick_before: int
    tick_after: int
    events: tuple[SimulationEvent, ...]

    def __post_init__(self) -> None:
        if self.tick_before < 0:
            raise ValueError("tick_before cannot be negative")
        if self.tick_after != self.tick_before + 1:
            raise ValueError("TickResult must describe exactly one tick")
        if not isinstance(self.events, tuple):
            raise TypeError("events must be an immutable tuple")

