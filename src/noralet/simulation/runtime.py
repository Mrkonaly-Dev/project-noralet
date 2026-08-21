"""Authoritative deterministic simulation runtime."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

from noralet.noralets.actions import ActionIntent
from noralet.noralets.body import NoraletBodyState
from noralet.noralets.energy import NoraletEnergyConfig
from noralet.simulation.config import SimulationConfig
from noralet.simulation.events import (
    EnergyConsumed,
    EnergyPointDecayed,
    EnergyPointDissolved,
    EnergyPointFormed,
    NoraletAccelerated,
    NoraletDeathCause,
    NoraletDied,
    NoraletEnergyExpenditureReason,
    NoraletEnergyReleased,
    NoraletEnergySpent,
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


@dataclass(frozen=True, slots=True)
class _FormationCandidate:
    """Uncommitted region formation result awaiting spacing resolution."""

    region: RegionDefinition
    position: float
    energy: float


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
        self._validate_initial_noralet_energy(self._state.bodies)
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

        state_before = self._state
        intents = self._validate_action_intents(state_before, action_intents)

        state_after, transition_events = self._resolve_next_state(
            state_before,
            intents,
        )
        self.audit_energy_conservation(state_after)
        tick_event = TickAdvanced(
            tick_before=state_before.tick,
            tick_after=state_after.tick,
        )
        result = TickResult(
            tick_before=state_before.tick,
            tick_after=state_after.tick,
            events=(*transition_events, tick_event),
        )

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

    def _validate_initial_noralet_energy(
        self,
        bodies: tuple[NoraletBodyState, ...],
    ) -> None:
        energy_config = self.config.noralet_energy
        if energy_config is None:
            if any(body.energy != 0.0 for body in bodies):
                raise ValueError(
                    "positive initial Noralet Energy requires a "
                    "NoraletEnergyConfig"
                )
            return

        for body in bodies:
            if body.energy > energy_config.energy_capacity:
                raise ValueError(
                    f"initial energy for Noralet {body.noralet_id} exceeds "
                    "energy_capacity"
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

        ordered_points = tuple(
            sorted(energy_points, key=lambda point: (point.position, point.point_id))
        )
        for previous, current in zip(ordered_points, ordered_points[1:]):
            if (
                current.position - previous.position
                < ecology.minimum_energy_point_spacing
            ):
                raise ValueError(
                    "initial energy points violate minimum_energy_point_spacing"
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
        if self.config.noralet_energy is None:
            return self._resolve_legacy_next_state(state_before, action_intents)
        return self._resolve_energy_enabled_next_state(state_before, action_intents)

    def _resolve_legacy_next_state(
        self,
        state_before: WorldState,
        action_intents: Mapping[int, ActionIntent],
    ) -> tuple[WorldState, tuple[SimulationEvent, ...]]:
        """Preserve Iteration 1-3 motion when Noralet Energy is disabled."""

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
                energy=body.energy,
            )
            for body in state_before.bodies
            if self._is_inside_world(positions_after[body.noralet_id])
        )

        (
            environmental_energy,
            energy_points,
            next_energy_point_id,
            ecology_events,
        ) = self._resolve_ecology(
            state_before=state_before,
            tick_after=tick_after,
            environmental_energy=state_before.environmental_energy,
            energy_points=state_before.energy_points,
            next_energy_point_id=state_before.next_energy_point_id,
        )

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

    def _resolve_energy_enabled_next_state(
        self,
        state_before: WorldState,
        action_intents: Mapping[int, ActionIntent],
    ) -> tuple[WorldState, tuple[SimulationEvent, ...]]:
        """Resolve the complete Iteration 4 transfer and physics sequence."""

        energy_config = self.config.noralet_energy
        ecology = self.config.energy_ecology
        assert energy_config is not None
        assert ecology is not None
        tick_after = state_before.tick + 1

        (
            noralet_energy,
            post_consumption_points,
            consumption_events,
        ) = self._resolve_consumption(
            state_before,
            action_intents,
            energy_config,
            tick_after,
        )

        environmental_transfers: dict[str, list[float]] = {
            region.region_id: [] for region in ecology.regions
        }
        existence_events: list[NoraletEnergySpent] = []
        for body in state_before.bodies:
            region = ecology.region_for(body.position)
            energy_spent = min(
                noralet_energy[body.noralet_id],
                energy_config.existence_energy_cost_per_tick,
            )
            if energy_spent > 0.0:
                noralet_energy[body.noralet_id] -= energy_spent
                environmental_transfers[region.region_id].append(energy_spent)
                existence_events.append(
                    NoraletEnergySpent(
                        noralet_id=body.noralet_id,
                        region_id=region.region_id,
                        reason=NoraletEnergyExpenditureReason.EXISTENCE,
                        energy_transferred=energy_spent,
                        tick_before=state_before.tick,
                        tick_after=tick_after,
                    )
                )

        applied_accelerations: dict[int, float] = {}
        acceleration_events_spent: list[NoraletEnergySpent] = []
        coefficient = energy_config.acceleration_energy_cost_per_unit
        for body in state_before.bodies:
            requested = (
                action_intents[body.noralet_id].acceleration
                if body.noralet_id in action_intents
                else 0.0
            )
            available = noralet_energy[body.noralet_id]
            if coefficient == 0.0:
                applied = requested
                energy_spent = 0.0
            else:
                requested_cost = coefficient * abs(requested)
                if requested_cost <= available:
                    applied = requested
                    energy_spent = requested_cost
                elif available > 0.0 and requested != 0.0:
                    applied = math.copysign(available / coefficient, requested)
                    energy_spent = available if applied != 0.0 else 0.0
                else:
                    applied = 0.0
                    energy_spent = 0.0

            applied_accelerations[body.noralet_id] = applied
            if energy_spent > 0.0:
                noralet_energy[body.noralet_id] -= energy_spent
                region = ecology.region_for(body.position)
                environmental_transfers[region.region_id].append(energy_spent)
                acceleration_events_spent.append(
                    NoraletEnergySpent(
                        noralet_id=body.noralet_id,
                        region_id=region.region_id,
                        reason=NoraletEnergyExpenditureReason.ACCELERATION,
                        energy_transferred=energy_spent,
                        tick_before=state_before.tick,
                        tick_after=tick_after,
                    )
                )

        velocities_after = {
            body.noralet_id: body.velocity + applied_accelerations[body.noralet_id]
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
                acceleration=applied_accelerations[body.noralet_id],
                tick_before=state_before.tick,
                tick_after=tick_after,
            )
            for body in state_before.bodies
            if applied_accelerations[body.noralet_id] != 0.0
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

        death_causes: dict[int, NoraletDeathCause] = {}
        for body in state_before.bodies:
            resolved_position = positions_after[body.noralet_id]
            if not self._is_inside_world(resolved_position):
                death_causes[body.noralet_id] = NoraletDeathCause.WORLD_BOUNDARY
            elif noralet_energy[body.noralet_id] <= 0.0:
                death_causes[body.noralet_id] = NoraletDeathCause.ENERGY_DEPLETION

        death_events = tuple(
            NoraletDied(
                noralet_id=body.noralet_id,
                cause=death_causes[body.noralet_id],
                resolved_position=positions_after[body.noralet_id],
                tick_before=state_before.tick,
                tick_after=tick_after,
            )
            for body in state_before.bodies
            if body.noralet_id in death_causes
        )

        release_events: list[NoraletEnergyReleased] = []
        for body in state_before.bodies:
            if body.noralet_id not in death_causes:
                continue
            energy_released = noralet_energy[body.noralet_id]
            if energy_released <= 0.0:
                continue
            resolved_position = positions_after[body.noralet_id]
            if resolved_position < self.config.left_boundary:
                destination = ecology.regions[0]
            elif resolved_position > self.config.right_boundary:
                destination = ecology.regions[-1]
            else:
                destination = ecology.region_for(resolved_position)
            environmental_transfers[destination.region_id].append(energy_released)
            noralet_energy[body.noralet_id] = 0.0
            release_events.append(
                NoraletEnergyReleased(
                    noralet_id=body.noralet_id,
                    region_id=destination.region_id,
                    energy_transferred=energy_released,
                    tick_before=state_before.tick,
                    tick_after=tick_after,
                )
            )

        surviving_bodies = tuple(
            NoraletBodyState(
                noralet_id=body.noralet_id,
                position=positions_after[body.noralet_id],
                velocity=velocities_after[body.noralet_id],
                energy=noralet_energy[body.noralet_id],
            )
            for body in state_before.bodies
            if body.noralet_id not in death_causes
        )

        pools_before = {
            pool.region_id: pool.energy for pool in state_before.environmental_energy
        }
        environmental_energy = tuple(
            EnvironmentalEnergyPool(
                region_id=region.region_id,
                energy=math.fsum(
                    (
                        pools_before[region.region_id],
                        *environmental_transfers[region.region_id],
                    )
                ),
            )
            for region in ecology.regions
        )

        (
            environmental_energy,
            energy_points,
            next_energy_point_id,
            ecology_events,
        ) = self._resolve_ecology(
            state_before=state_before,
            tick_after=tick_after,
            environmental_energy=environmental_energy,
            energy_points=post_consumption_points,
            next_energy_point_id=state_before.next_energy_point_id,
        )

        state_after = WorldState(
            tick=tick_after,
            bodies=surviving_bodies,
            environmental_energy=environmental_energy,
            energy_points=energy_points,
            next_energy_point_id=next_energy_point_id,
        )
        events: tuple[SimulationEvent, ...] = (
            *consumption_events,
            *existence_events,
            *acceleration_events_spent,
            *acceleration_events,
            *movement_events,
            *death_events,
            *release_events,
            *ecology_events,
        )
        return state_after, events

    def _resolve_consumption(
        self,
        state_before: WorldState,
        action_intents: Mapping[int, ActionIntent],
        energy_config: NoraletEnergyConfig,
        tick_after: int,
    ) -> tuple[
        dict[int, float],
        tuple[ConsumableEnergyPoint, ...],
        tuple[EnergyConsumed, ...],
    ]:
        """Resolve all tick-start consume targets with fair water filling."""

        targets_by_point: dict[int, list[int]] = {}
        for body in state_before.bodies:
            intent = action_intents.get(body.noralet_id)
            if intent is None or not intent.consume:
                continue
            target = self._select_consumption_target(
                body,
                state_before.energy_points,
                energy_config.consume_radius,
            )
            if target is not None:
                targets_by_point.setdefault(target.point_id, []).append(
                    body.noralet_id
                )

        noralet_energy = {
            body.noralet_id: body.energy for body in state_before.bodies
        }
        surviving_points: list[ConsumableEnergyPoint] = []
        events: list[EnergyConsumed] = []
        for point in state_before.energy_points:
            consumer_ids = tuple(sorted(targets_by_point.get(point.point_id, ())))
            capacities = {
                noralet_id: (
                    energy_config.energy_capacity - noralet_energy[noralet_id]
                )
                for noralet_id in consumer_ids
            }
            allocations = self._fair_allocations(point.energy, capacities)
            actual_allocations: dict[int, float] = {}
            for noralet_id in consumer_ids:
                allocation = min(
                    allocations.get(noralet_id, 0.0),
                    energy_config.energy_capacity - noralet_energy[noralet_id],
                )
                if allocation <= 0.0:
                    continue
                noralet_energy[noralet_id] += allocation
                actual_allocations[noralet_id] = allocation
                events.append(
                    EnergyConsumed(
                        noralet_id=noralet_id,
                        point_id=point.point_id,
                        energy_transferred=allocation,
                        tick_before=state_before.tick,
                        tick_after=tick_after,
                    )
                )

            consumed = math.fsum(actual_allocations.values())
            if self._aggregate_capacity_reaches(point.energy, capacities.values()):
                continue
            remaining_energy = point.energy - consumed
            if remaining_energy > 0.0:
                surviving_points.append(
                    ConsumableEnergyPoint(
                        point_id=point.point_id,
                        position=point.position,
                        energy=remaining_energy,
                    )
                )

        return noralet_energy, tuple(surviving_points), tuple(events)

    @staticmethod
    def _select_consumption_target(
        body: NoraletBodyState,
        energy_points: tuple[ConsumableEnergyPoint, ...],
        consume_radius: float,
    ) -> ConsumableEnergyPoint | None:
        accessible = tuple(
            point
            for point in energy_points
            if abs(point.position - body.position) <= consume_radius
        )
        if not accessible:
            return None
        return min(
            accessible,
            key=lambda point: (
                abs(point.position - body.position),
                point.point_id,
            ),
        )

    @staticmethod
    def _fair_allocations(
        available_energy: float,
        capacities: Mapping[int, float],
    ) -> dict[int, float]:
        """Return order-independent capacity-limited equal-share allocations."""

        allocations = {noralet_id: 0.0 for noralet_id in capacities}
        remaining_capacities = {
            noralet_id: capacity
            for noralet_id, capacity in capacities.items()
            if capacity > 0.0
        }
        remaining_energy = available_energy
        while remaining_capacities and remaining_energy > 0.0:
            equal_share = remaining_energy / len(remaining_capacities)
            saturated = tuple(
                noralet_id
                for noralet_id in sorted(remaining_capacities)
                if remaining_capacities[noralet_id] <= equal_share
            )
            if not saturated:
                for noralet_id in remaining_capacities:
                    allocations[noralet_id] += equal_share
                remaining_energy = 0.0
                break

            for noralet_id in saturated:
                allocations[noralet_id] += remaining_capacities.pop(noralet_id)
            remaining_energy = max(
                0.0,
                available_energy - math.fsum(allocations.values()),
            )

        return allocations

    @staticmethod
    def _aggregate_capacity_reaches(
        required_energy: float,
        capacities: Iterable[float],
    ) -> bool:
        """Check aggregate capacity without overflowing an unnecessary sum."""

        accumulated = 0.0
        for capacity in capacities:
            accumulated += min(capacity, required_energy - accumulated)
            if accumulated >= required_energy:
                return True
        return False

    def _resolve_ecology(
        self,
        *,
        state_before: WorldState,
        tick_after: int,
        environmental_energy: tuple[EnvironmentalEnergyPool, ...],
        energy_points: tuple[ConsumableEnergyPoint, ...],
        next_energy_point_id: int,
    ) -> tuple[
        tuple[EnvironmentalEnergyPool, ...],
        tuple[ConsumableEnergyPoint, ...],
        int,
        tuple[SimulationEvent, ...],
    ]:
        ecology = self.config.energy_ecology
        if ecology is None:
            return (
                environmental_energy,
                energy_points,
                next_energy_point_id,
                (),
            )

        pools_before = {
            pool.region_id: pool.energy for pool in environmental_energy
        }
        returned_by_region: dict[str, list[float]] = {
            region.region_id: [] for region in ecology.regions
        }
        decayed_points: list[ConsumableEnergyPoint] = []
        decay_events: list[EnergyPointDecayed] = []
        dissolution_events: list[EnergyPointDissolved] = []

        for point in energy_points:
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

        candidates: list[_FormationCandidate] = []
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
            amount = self._formation_amount(
                region,
                ecology.formation_energy_min,
                maximum,
            )
            position = self._formation_position(region)
            candidates.append(
                _FormationCandidate(
                    region=region,
                    position=position,
                    energy=amount,
                )
            )

        rejected_candidates = {
            index
            for index, candidate in enumerate(candidates)
            if any(
                abs(candidate.position - point.position)
                < ecology.minimum_energy_point_spacing
                for point in decayed_points
            )
        }
        for first_index, first in enumerate(candidates):
            for second_index in range(first_index + 1, len(candidates)):
                second = candidates[second_index]
                if (
                    abs(first.position - second.position)
                    < ecology.minimum_energy_point_spacing
                ):
                    rejected_candidates.add(first_index)
                    rejected_candidates.add(second_index)

        formed_points: list[ConsumableEnergyPoint] = []
        formation_events: list[EnergyPointFormed] = []
        for index, candidate in enumerate(candidates):
            if index in rejected_candidates:
                continue
            point = ConsumableEnergyPoint(
                point_id=next_energy_point_id,
                position=candidate.position,
                energy=candidate.energy,
            )
            next_energy_point_id += 1
            formed_points.append(point)
            pools_after_decay[candidate.region.region_id] -= candidate.energy
            formation_events.append(
                EnergyPointFormed(
                    region_id=candidate.region.region_id,
                    point_id=point.point_id,
                    position=point.position,
                    energy=point.energy,
                    tick_before=state_before.tick,
                    tick_after=tick_after,
                )
            )

        resulting_environmental_energy = tuple(
            EnvironmentalEnergyPool(
                region_id=region.region_id,
                energy=pools_after_decay[region.region_id],
            )
            for region in ecology.regions
        )
        resulting_energy_points = tuple((*decayed_points, *formed_points))
        events: tuple[SimulationEvent, ...] = (
            *decay_events,
            *dissolution_events,
            *formation_events,
        )
        return (
            resulting_environmental_energy,
            resulting_energy_points,
            next_energy_point_id,
            events,
        )

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
