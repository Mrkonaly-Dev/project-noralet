"""Configuration for a simulation run."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Immutable settings required by the Iteration 1 runtime."""

    master_seed: int

    def __post_init__(self) -> None:
        if type(self.master_seed) is not int:
            raise TypeError("master_seed must be an integer")

