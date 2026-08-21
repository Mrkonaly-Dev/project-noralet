"""Authoritative deterministic simulation runtime."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

from noralet.noralets.actions import ActionIntent
from noralet.noralets.body import NoraletBodyState
from noralet.simulation.config import SimulationConfig
from noralet.simulation.events import (
    NoraletAccelerated,
    NoraletDeathCause,
    NoraletDied,
    NoraletMoved,
    SimulationEvent,
    TickAdvanced,
)
from noralet.simulation.randomness import DeterministicRandomStreams
from noralet.simulation.state import WorldState
from noralet.simulation.tick import TickResult


class Simulation:
    """Own the world state, random streams, and authoritative clock."""

    __slots__ = ("_config", "_random_streams", "_state")

    def __init__(
        self,
        config: SimulationConfig,
        initial_bodies: Iterable[NoraletBodyState] = (),
    ) -> None:
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")

        try:
            bodies = tuple(initial_bodies)
        except TypeError as error:
            raise TypeError("initial_bodies must be iterable") from error

        self._config = config
        self._state = WorldState(bodies=bodies)
        self._validate_initial_positions(self._state.bodies)
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

    def step(
        self,
        action_intents: Mapping[int, ActionIntent] | None = None,
    ) -> TickResult:
        """Resolve one lockstep physical transition from external intentions."""

        # Read/calculation phase: this reference remains logically immutable.
        state_before = self._state
        intents = self._validate_action_intents(state_before, action_intents)

        # Resolution boundary: all bodies are calculated from state_before.
        state_after, physical_events = self._resolve_next_state(state_before, intents)
        tick_event = TickAdvanced(
            tick_before=state_before.tick,
            tick_after=state_after.tick,
        )
        result = TickResult(
            tick_before=state_before.tick,
            tick_after=state_after.tick,
            events=(*physical_events, tick_event),
        )

        # Publish the new canonical state only after the transition is complete.
        self._state = state_after
        return result

    def _validate_initial_positions(
        self,
        bodies: tuple[NoraletBodyState, ...],
    ) -> None:
        for body in bodies:
            if not self._is_inside_world(body.position):
                raise ValueError(
                    f"initial position for Noralet {body.noralet_id} is outside "
                    "the traversable world"
                )

    @staticmethod
    def _validate_action_intents(
        state_before: WorldState,
        action_intents: Mapping[int, ActionIntent] | None,
    ) -> dict[int, ActionIntent]:
        if action_intents is None:
            return {}
        if not isinstance(action_intents, Mapping):
            raise TypeError("action_intents must be a mapping from ID to ActionIntent")

        intents = dict(action_intents)
        if any(type(noralet_id) is not int for noralet_id in intents):
            raise TypeError("action-intent target IDs must be integers")
        if any(not isinstance(intent, ActionIntent) for intent in intents.values()):
            raise TypeError("every action intent must be an ActionIntent")

        living_ids = {body.noralet_id for body in state_before.bodies}
        unknown_ids = sorted(set(intents) - living_ids)
        if unknown_ids:
            unknown = ", ".join(str(noralet_id) for noralet_id in unknown_ids)
            raise ValueError(f"action intent targets non-living Noralet ID(s): {unknown}")

        return intents

    def _resolve_next_state(
        self,
        state_before: WorldState,
        action_intents: Mapping[int, ActionIntent],
    ) -> tuple[WorldState, tuple[SimulationEvent, ...]]:
        tick_after = state_before.tick + 1

        accelerations = {
            body.noralet_id: (
                action_intents[body.noralet_id].acceleration
                if body.noralet_id in action_intents
                else 0.0
            )
            for body in state_before.bodies
        }
        velocities_after = {
            body.noralet_id: body.velocity + accelerations[body.noralet_id]
            for body in state_before.bodies
        }
        positions_after = {
            body.noralet_id: body.position + velocities_after[body.noralet_id]
            for body in state_before.bodies
        }

        self._validate_resolved_values(velocities_after, positions_after)

        acceleration_events = tuple(
            NoraletAccelerated(
                noralet_id=body.noralet_id,
                acceleration=accelerations[body.noralet_id],
                tick_before=state_before.tick,
                tick_after=tick_after,
            )
            for body in state_before.bodies
            if accelerations[body.noralet_id] != 0.0
        )
        movement_events = tuple(
            NoraletMoved(
                noralet_id=body.noralet_id,
                position_before=body.position,
                position_after=positions_after[body.noralet_id],
                velocity_after=velocities_after[body.noralet_id],
                tick_before=state_before.tick,
                tick_after=tick_after,
            )
            for body in state_before.bodies
            if positions_after[body.noralet_id] != body.position
        )
        death_events = tuple(
            NoraletDied(
                noralet_id=body.noralet_id,
                cause=NoraletDeathCause.WORLD_BOUNDARY,
                resolved_position=positions_after[body.noralet_id],
                tick_before=state_before.tick,
                tick_after=tick_after,
            )
            for body in state_before.bodies
            if not self._is_inside_world(positions_after[body.noralet_id])
        )
        surviving_bodies = tuple(
            NoraletBodyState(
                noralet_id=body.noralet_id,
                position=positions_after[body.noralet_id],
                velocity=velocities_after[body.noralet_id],
            )
            for body in state_before.bodies
            if self._is_inside_world(positions_after[body.noralet_id])
        )

        state_after = WorldState(tick=tick_after, bodies=surviving_bodies)
        events: tuple[SimulationEvent, ...] = (
            *acceleration_events,
            *movement_events,
            *death_events,
        )
        return state_after, events

    @staticmethod
    def _validate_resolved_values(
        velocities_after: Mapping[int, float],
        positions_after: Mapping[int, float],
    ) -> None:
        for noralet_id in sorted(velocities_after):
            if not math.isfinite(velocities_after[noralet_id]):
                raise OverflowError(
                    f"resolved velocity for Noralet {noralet_id} is not finite"
                )
            if not math.isfinite(positions_after[noralet_id]):
                raise OverflowError(
                    f"resolved position for Noralet {noralet_id} is not finite"
                )

    def _is_inside_world(self, position: float) -> bool:
        return self.config.left_boundary <= position <= self.config.right_boundary
