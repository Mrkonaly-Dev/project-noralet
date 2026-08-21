"""Authoritative deterministic simulation runtime."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

from noralet.noralets.actions import ActionIntent
from noralet.noralets.body import NoraletBodyState
from noralet.simulation.config import SimulationConfig
from noralet.simulation.events import (
    EnergyPointDecayed,
    EnergyPointDissolved,
    EnergyPointFormed,
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
from noralet.world.energy import (
    ConsumableEnergyPoint,
    EnergyConservationError,
    EnvironmentalEnergyPool,
)
from noralet.world.regions import RegionDefinition


class Simulation:
    """Own the world state, random streams, and authoritative clock."""

    __slots__ = (
        "_config",
        "_initial_total_energy",
        "_random_streams",
        "_state",
    )

    ENERGY_CONSERVATION_ABS_TOLERANCE = 1e-9

    def __init__(
        self,
        config: SimulationConfig,
        initial_bodies: Iterable[NoraletBodyState] = (),
        initial_energy_points: Iterable[ConsumableEnergyPoint] = (),
    ) -> None:
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")

        try:
            bodies = tuple(initial_bodies)
        except TypeError as error:
            raise TypeError("initial_bodies must be iterable") from error
        try:
            energy_points = tuple(initial_energy_points)
        except TypeError as error:
            raise TypeError("initial_energy_points must be iterable") from error
        if not all(
            isinstance(point, ConsumableEnergyPoint) for point in energy_points
        ):
            raise TypeError(
                "every initial energy point must be a ConsumableEnergyPoint"
            )

        self._config = config
        environmental_energy = self._initial_environmental_energy(energy_points)
        next_point_id = max(
            (point.point_id for point in energy_points),
            default=-1,
        ) + 1
        self._state = WorldState(
            bodies=bodies,
            environmental_energy=environmental_energy,
            energy_points=energy_points,
            next_energy_point_id=next_point_id,
        )
        self._validate_initial_positions(self._state.bodies)
        self._validate_initial_energy_points(self._state.energy_points)
        self._random_streams = DeterministicRandomStreams(config.master_seed)
        self._initial_total_energy = self._state.energy_totals.total_energy
        if not math.isfinite(self._initial_total_energy):
            raise ValueError("initial total energy must be finite")
        self.audit_energy_conservation()

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

    @property
    def initial_total_energy(self) -> float:
        """Return the immutable closed-universe energy baseline."""

        return self._initial_total_energy

    def step(
        self,
        action_intents: Mapping[int, ActionIntent] | None = None,
    ) -> TickResult:
        """Resolve one lockstep world transition from external intentions."""

        # Read/calculation phase: this reference remains logically immutable.
        state_before = self._state
        intents = self._validate_action_intents(state_before, action_intents)

        # Resolution boundary: all bodies are calculated from state_before.
        state_after, physical_events = self._resolve_next_state(state_before, intents)
        self.audit_energy_conservation(state_after)
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

    def audit_energy_conservation(self, state: WorldState | None = None) -> None:
        """Raise before publication if a state violates the energy baseline."""

        candidate = self._state if state is None else state
        if not isinstance(candidate, WorldState):
            raise TypeError("state must be a WorldState")
        current_total = candidate.energy_totals.total_energy
        if not math.isfinite(current_total) or not math.isclose(
            current_total,
            self._initial_total_energy,
            rel_tol=0.0,
            abs_tol=self.ENERGY_CONSERVATION_ABS_TOLERANCE,
        ):
            difference = current_total - self._initial_total_energy
            raise EnergyConservationError(
                "closed energy invariant violated: "
                f"expected {self._initial_total_energy!r} eU, "
                f"observed {current_total!r} eU "
                f"(difference {difference!r} eU)"
            )

    def _initial_environmental_energy(
        self,
        energy_points: tuple[ConsumableEnergyPoint, ...],
    ) -> tuple[EnvironmentalEnergyPool, ...]:
        ecology = self.config.energy_ecology
        if ecology is None:
            if energy_points:
                raise ValueError(
                    "initial energy points require an EnergyEcologyConfig"
                )
            return ()
        return ecology.initial_environmental_energy

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

    def _validate_initial_energy_points(
        self,
        energy_points: tuple[ConsumableEnergyPoint, ...],
    ) -> None:
        ecology = self.config.energy_ecology
        if ecology is None:
            return
        for point in energy_points:
            if not self._is_inside_world(point.position):
                raise ValueError(
                    f"initial energy point {point.point_id} is outside "
                    "the traversable world"
                )
            ecology.region_for(point.position)

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

        (
            environmental_energy,
            energy_points,
            next_energy_point_id,
            ecology_events,
        ) = self._resolve_ecology(state_before, tick_after)

        state_after = WorldState(
            tick=tick_after,
            bodies=surviving_bodies,
            environmental_energy=environmental_energy,
            energy_points=energy_points,
            next_energy_point_id=next_energy_point_id,
        )
        events: tuple[SimulationEvent, ...] = (
            *acceleration_events,
            *movement_events,
            *death_events,
            *ecology_events,
        )
        return state_after, events

    def _resolve_ecology(
        self,
        state_before: WorldState,
        tick_after: int,
    ) -> tuple[
        tuple[EnvironmentalEnergyPool, ...],
        tuple[ConsumableEnergyPoint, ...],
        int,
        tuple[SimulationEvent, ...],
    ]:
        ecology = self.config.energy_ecology
        if ecology is None:
            return (
                state_before.environmental_energy,
                state_before.energy_points,
                state_before.next_energy_point_id,
                (),
            )

        pools_before = {
            pool.region_id: pool.energy for pool in state_before.environmental_energy
        }
        returned_by_region: dict[str, list[float]] = {
            region.region_id: [] for region in ecology.regions
        }
        decayed_points: list[ConsumableEnergyPoint] = []
        decay_events: list[EnergyPointDecayed] = []
        dissolution_events: list[EnergyPointDissolved] = []

        for point in state_before.energy_points:
            region = ecology.region_for(point.position)
            remaining_energy = point.energy * (1.0 - ecology.decay_rate)
            decayed_energy = point.energy - remaining_energy

            if decayed_energy > 0.0:
                returned_by_region[region.region_id].append(decayed_energy)
                decay_events.append(
                    EnergyPointDecayed(
                        region_id=region.region_id,
                        point_id=point.point_id,
                        energy_returned=decayed_energy,
                        remaining_energy=remaining_energy,
                        tick_before=state_before.tick,
                        tick_after=tick_after,
                    )
                )

            if remaining_energy <= ecology.point_removal_threshold:
                returned_by_region[region.region_id].append(remaining_energy)
                dissolution_events.append(
                    EnergyPointDissolved(
                        region_id=region.region_id,
                        point_id=point.point_id,
                        energy_returned=remaining_energy,
                        tick_before=state_before.tick,
                        tick_after=tick_after,
                    )
                )
            else:
                decayed_points.append(
                    ConsumableEnergyPoint(
                        point_id=point.point_id,
                        position=point.position,
                        energy=remaining_energy,
                    )
                )

        pools_after_decay = {
            region.region_id: math.fsum(
                (
                    pools_before[region.region_id],
                    *returned_by_region[region.region_id],
                )
            )
            for region in ecology.regions
        }

        next_point_id = state_before.next_energy_point_id
        formed_points: list[ConsumableEnergyPoint] = []
        formation_events: list[EnergyPointFormed] = []
        for region in ecology.regions:
            probability = ecology.formation_probabilities.for_kind(region.kind)
            trigger = self.random_streams.stream(
                self._energy_stream_name(region.region_id, "trigger")
            ).random()
            available_energy = pools_after_decay[region.region_id]
            if trigger >= probability:
                continue
            if available_energy < ecology.formation_energy_min:
                continue

            maximum = min(ecology.formation_energy_max, available_energy)
            amount = self._formation_amount(region, ecology.formation_energy_min, maximum)
            position = self._formation_position(region)
            point = ConsumableEnergyPoint(
                point_id=next_point_id,
                position=position,
                energy=amount,
            )
            next_point_id += 1
            formed_points.append(point)
            pools_after_decay[region.region_id] = available_energy - amount
            formation_events.append(
                EnergyPointFormed(
                    region_id=region.region_id,
                    point_id=point.point_id,
                    position=point.position,
                    energy=point.energy,
                    tick_before=state_before.tick,
                    tick_after=tick_after,
                )
            )

        environmental_energy = tuple(
            EnvironmentalEnergyPool(
                region_id=region.region_id,
                energy=pools_after_decay[region.region_id],
            )
            for region in ecology.regions
        )
        energy_points = tuple((*decayed_points, *formed_points))
        events: tuple[SimulationEvent, ...] = (
            *decay_events,
            *dissolution_events,
            *formation_events,
        )
        return environmental_energy, energy_points, next_point_id, events

    def _formation_amount(
        self,
        region: RegionDefinition,
        minimum: float,
        maximum: float,
    ) -> float:
        if minimum == maximum:
            return minimum
        fraction = self.random_streams.stream(
            self._energy_stream_name(region.region_id, "amount")
        ).random()
        return min(minimum + (maximum - minimum) * fraction, maximum)

    def _formation_position(self, region: RegionDefinition) -> float:
        fraction = self.random_streams.stream(
            self._energy_stream_name(region.region_id, "position")
        ).random()
        position = region.left + (region.right - region.left) * fraction
        ecology = self.config.energy_ecology
        assert ecology is not None
        if region is not ecology.regions[-1] and position >= region.right:
            return math.nextafter(region.right, region.left)
        return position

    @staticmethod
    def _energy_stream_name(region_id: str, purpose: str) -> str:
        return f"energy:region:{len(region_id)}:{region_id}:formation:{purpose}"

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
