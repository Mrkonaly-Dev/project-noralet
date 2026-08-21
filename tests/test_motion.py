"""Tests for lockstep one-dimensional Noralet motion."""

from __future__ import annotations

import dataclasses
import unittest

from noralet.simulation import (
    ActionIntent,
    NoraletAccelerated,
    NoraletBodyState,
    NoraletDeathCause,
    NoraletDied,
    NoraletMoved,
    Simulation,
    SimulationConfig,
    TickAdvanced,
)


def simulation_with(
    *bodies: NoraletBodyState,
    left_boundary: float = -100.0,
    right_boundary: float = 100.0,
    seed: int = 123,
) -> Simulation:
    return Simulation(
        SimulationConfig(
            master_seed=seed,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        ),
        initial_bodies=bodies,
    )


class MotionTests(unittest.TestCase):
    def test_stationary_noralet_remains_stationary(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=0, velocity=0)
        )

        result = simulation.step({1: ActionIntent(acceleration=0)})

        self.assertEqual(simulation.state.body(1).position, 0.0)
        self.assertEqual(simulation.state.body(1).velocity, 0.0)
        self.assertEqual(result.events, (TickAdvanced(tick_before=0, tick_after=1),))

    def test_acceleration_from_rest_uses_updated_velocity_for_position(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=0, velocity=0)
        )

        simulation.step({1: ActionIntent(acceleration=1)})

        body = simulation.state.body(1)
        self.assertEqual(body.velocity, 1.0)
        self.assertEqual(body.position, 1.0)

    def test_velocity_persists_without_friction(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=0, velocity=2)
        )

        simulation.step()
        self.assertEqual(simulation.state.body(1).velocity, 2.0)
        self.assertEqual(simulation.state.body(1).position, 2.0)

        simulation.step()
        self.assertEqual(simulation.state.body(1).velocity, 2.0)
        self.assertEqual(simulation.state.body(1).position, 4.0)

    def test_opposing_acceleration_reduces_velocity_before_movement(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=0, velocity=3)
        )

        simulation.step({1: ActionIntent(acceleration=-1)})

        self.assertEqual(simulation.state.body(1).velocity, 2.0)
        self.assertEqual(simulation.state.body(1).position, 2.0)

    def test_opposing_acceleration_can_reverse_direction(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=0, velocity=1)
        )

        simulation.step({1: ActionIntent(acceleration=-3)})

        self.assertEqual(simulation.state.body(1).velocity, -2.0)
        self.assertEqual(simulation.state.body(1).position, -2.0)

    def test_multiple_noralets_move_independently_in_one_tick(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=-10, velocity=1),
            NoraletBodyState(noralet_id=2, position=10, velocity=-1),
            NoraletBodyState(noralet_id=3, position=20, velocity=0),
        )

        simulation.step(
            {
                1: ActionIntent(acceleration=0.5),
                2: ActionIntent(acceleration=-0.5),
            }
        )

        self.assertEqual(simulation.state.body(1).position, -8.5)
        self.assertEqual(simulation.state.body(1).velocity, 1.5)
        self.assertEqual(simulation.state.body(2).position, 8.5)
        self.assertEqual(simulation.state.body(2).velocity, -1.5)
        self.assertEqual(simulation.state.body(3).position, 20.0)
        self.assertEqual(simulation.state.body(3).velocity, 0.0)

    def test_noralets_can_share_a_position_and_cross_without_collision(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=-1, velocity=1),
            NoraletBodyState(noralet_id=2, position=1, velocity=-1),
        )

        simulation.step()

        self.assertEqual(simulation.state.body(1).position, 0.0)
        self.assertEqual(simulation.state.body(2).position, 0.0)
        self.assertEqual(simulation.state.body(1).velocity, 1.0)
        self.assertEqual(simulation.state.body(2).velocity, -1.0)

    def test_body_and_intent_insertion_order_cannot_change_result(self) -> None:
        bodies = (
            NoraletBodyState(noralet_id=1, position=-5, velocity=0.5),
            NoraletBodyState(noralet_id=2, position=5, velocity=-0.5),
        )
        first = simulation_with(*bodies)
        second = simulation_with(*reversed(bodies))

        first_result = first.step(
            {1: ActionIntent(acceleration=1), 2: ActionIntent(acceleration=-1)}
        )
        second_result = second.step(
            {2: ActionIntent(acceleration=-1), 1: ActionIntent(acceleration=1)}
        )

        self.assertEqual(first.state, second.state)
        self.assertEqual(first_result, second_result)

    def test_missing_intent_means_zero_acceleration(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=0, velocity=1),
            NoraletBodyState(noralet_id=2, position=10, velocity=1),
        )

        simulation.step({1: ActionIntent(acceleration=1)})

        self.assertEqual(simulation.state.body(1).velocity, 2.0)
        self.assertEqual(simulation.state.body(1).position, 2.0)
        self.assertEqual(simulation.state.body(2).velocity, 1.0)
        self.assertEqual(simulation.state.body(2).position, 11.0)

    def test_reaching_exact_boundaries_does_not_kill(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=-9, velocity=0),
            NoraletBodyState(noralet_id=2, position=9, velocity=0),
            left_boundary=-10,
            right_boundary=10,
        )

        result = simulation.step(
            {1: ActionIntent(acceleration=-1), 2: ActionIntent(acceleration=1)}
        )

        self.assertEqual(simulation.state.body(1).position, -10.0)
        self.assertEqual(simulation.state.body(2).position, 10.0)
        self.assertFalse(any(isinstance(event, NoraletDied) for event in result.events))

    def test_crossing_boundaries_removes_bodies_without_clamp_or_bounce(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=-9, velocity=-2),
            NoraletBodyState(noralet_id=2, position=9, velocity=2),
            left_boundary=-10,
            right_boundary=10,
        )

        result = simulation.step()

        self.assertEqual(simulation.state.bodies, ())
        deaths = tuple(event for event in result.events if isinstance(event, NoraletDied))
        self.assertEqual(
            deaths,
            (
                NoraletDied(
                    noralet_id=1,
                    cause=NoraletDeathCause.WORLD_BOUNDARY,
                    resolved_position=-11.0,
                    tick_before=0,
                    tick_after=1,
                ),
                NoraletDied(
                    noralet_id=2,
                    cause=NoraletDeathCause.WORLD_BOUNDARY,
                    resolved_position=11.0,
                    tick_before=0,
                    tick_after=1,
                ),
            ),
        )

    def test_terminal_events_follow_physical_phase_and_identity_order(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=2, position=0, velocity=0),
            NoraletBodyState(noralet_id=1, position=0, velocity=0),
            left_boundary=-1,
            right_boundary=1,
        )

        result = simulation.step(
            {2: ActionIntent(acceleration=-2), 1: ActionIntent(acceleration=2)}
        )

        self.assertEqual(
            tuple(type(event) for event in result.events),
            (
                NoraletAccelerated,
                NoraletAccelerated,
                NoraletMoved,
                NoraletMoved,
                NoraletDied,
                NoraletDied,
                TickAdvanced,
            ),
        )
        physical_ids = tuple(
            event.noralet_id
            for event in result.events
            if not isinstance(event, TickAdvanced)
        )
        self.assertEqual(physical_ids, (1, 2, 1, 2, 1, 2))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.events[0].acceleration = 0.0  # type: ignore[union-attr,misc]

    def test_old_world_and_contained_bodies_remain_unchanged(self) -> None:
        simulation = simulation_with(
            NoraletBodyState(noralet_id=1, position=0, velocity=1)
        )
        old_state = simulation.state
        old_body = old_state.body(1)

        simulation.step({1: ActionIntent(acceleration=1)})

        self.assertEqual(old_state.tick, 0)
        self.assertEqual(old_body.position, 0.0)
        self.assertEqual(old_body.velocity, 1.0)
        self.assertIsNot(old_state, simulation.state)
        self.assertIsNot(old_body, simulation.state.body(1))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            old_body.velocity = 99.0  # type: ignore[misc]

    def test_multi_tick_world_and_event_history_is_deterministic(self) -> None:
        initial_bodies = (
            NoraletBodyState(noralet_id=1, position=-2, velocity=0),
            NoraletBodyState(noralet_id=2, position=2, velocity=0),
        )
        first = simulation_with(*initial_bodies, left_boundary=-5, right_boundary=5)
        second = simulation_with(
            *reversed(initial_bodies), left_boundary=-5, right_boundary=5
        )
        action_sequence = (
            {1: ActionIntent(1), 2: ActionIntent(-1)},
            {},
            {1: ActionIntent(2), 2: ActionIntent(-2)},
            {},
        )

        first_history = [first.step(actions) for actions in action_sequence]
        second_history = [
            second.step(dict(reversed(tuple(actions.items()))))
            for actions in action_sequence
        ]

        self.assertEqual(first_history, second_history)
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.state.tick, 4)
        self.assertEqual(first.state.bodies, ())


if __name__ == "__main__":
    unittest.main()
