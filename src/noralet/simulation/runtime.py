"""Authoritative deterministic simulation runtime."""

from __future__ import annotations

from noralet.simulation.config import SimulationConfig
from noralet.simulation.events import TickAdvanced
from noralet.simulation.randomness import DeterministicRandomStreams
from noralet.simulation.state import WorldState
from noralet.simulation.tick import TickResult


class Simulation:
    """Own the world state, random streams, and authoritative clock."""

    __slots__ = ("_config", "_random_streams", "_state")

    def __init__(self, config: SimulationConfig) -> None:
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")

        self._config = config
        self._state = WorldState()
        self._random_streams = DeterministicRandomStreams(config.master_seed)

    @property
    def config(self) -> SimulationConfig:
        """Return this run's immutable configuration."""

        return self._config

    @property
    def state(self) -> WorldState:
        """Return the current immutable objective world state."""

        return self._state

    @property
    def random_streams(self) -> DeterministicRandomStreams:
        """Return the simulation-owned deterministic stream facility."""

        return self._random_streams

    def step(self) -> TickResult:
        """Advance the empty universe through one explicit transition."""

        # Read/calculation phase: this reference remains logically immutable.
        state_before = self._state

        # Resolution boundary: Iteration 1 resolves only the simulation clock.
        state_after = self._resolve_next_state(state_before)
        event = TickAdvanced(
            tick_before=state_before.tick,
            tick_after=state_after.tick,
        )
        result = TickResult(
            tick_before=state_before.tick,
            tick_after=state_after.tick,
            events=(event,),
        )

        # Publish the new canonical state only after the transition is complete.
        self._state = state_after
        return result

    @staticmethod
    def _resolve_next_state(state_before: WorldState) -> WorldState:
        return WorldState(tick=state_before.tick + 1)

