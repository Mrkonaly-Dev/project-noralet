"""Tests for explicit, fair and capacity-limited energy consumption."""

from __future__ import annotations

import unittest

from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    EnergyConsumed,
    EnergyPointDecayed,
    EnvironmentalEnergyPool,
    NoraletBodyState,
    Simulation,
)
from noralet_energy_test_support import noralet_energy_simulation


class ConsumptionTests(unittest.TestCase):
    def test_one_noralet_consumes_one_accessible_point(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=10),),
            points=(ConsumableEnergyPoint(4, 0.5, 30),),
        )

        result = simulation.step({1: ActionIntent(consume=True)})

        self.assertEqual(simulation.state.body(1).energy, 40.0)
        self.assertEqual(simulation.state.energy_points, ())
        self.assertEqual(simulation.state.energy_totals.total_energy, 40.0)
        self.assertIn(EnergyConsumed(1, 4, 30.0, 0, 1), result.events)

    def test_consumption_never_happens_without_explicit_action(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=10),),
            points=(ConsumableEnergyPoint(4, 0, 30),),
        )

        result = simulation.step()

        self.assertEqual(simulation.state.body(1).energy, 10.0)
        self.assertEqual(simulation.state.energy_point(4).energy, 30.0)
        self.assertFalse(any(isinstance(event, EnergyConsumed) for event in result.events))

    def test_out_of_range_consume_attempt_transfers_nothing(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=10),),
            points=(ConsumableEnergyPoint(4, 1.01, 30),),
            consume_radius=1,
        )

        result = simulation.step({1: ActionIntent(consume=True)})

        self.assertEqual(simulation.state.body(1).energy, 10.0)
        self.assertEqual(simulation.state.energy_point(4).energy, 30.0)
        self.assertFalse(any(isinstance(event, EnergyConsumed) for event in result.events))

    def test_consume_radius_is_inclusive(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=10),),
            points=(ConsumableEnergyPoint(4, 1, 5),),
            consume_radius=1,
        )

        simulation.step({1: ActionIntent(consume=True)})

        self.assertEqual(simulation.state.body(1).energy, 15.0)
        self.assertEqual(simulation.state.energy_points, ())

    def test_consumption_uses_tick_start_position(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, velocity=1, energy=10),),
            points=(ConsumableEnergyPoint(4, 1, 5),),
            consume_radius=0.2,
            minimum_spacing=0.5,
        )

        first = simulation.step({1: ActionIntent(consume=True)})
        self.assertEqual(simulation.state.body(1).position, 1.0)
        self.assertEqual(simulation.state.body(1).energy, 10.0)
        self.assertFalse(any(isinstance(event, EnergyConsumed) for event in first.events))

        second = simulation.step({1: ActionIntent(consume=True)})
        self.assertEqual(simulation.state.body(1).energy, 15.0)
        self.assertTrue(any(isinstance(event, EnergyConsumed) for event in second.events))

    def test_nearest_target_and_point_id_tie_break_are_deterministic(self) -> None:
        body = NoraletBodyState(1, 0, energy=1)
        points = (
            ConsumableEnergyPoint(8, 0.5, 1),
            ConsumableEnergyPoint(2, -0.5, 1),
            ConsumableEnergyPoint(1, 0.75, 1),
        )

        target = Simulation._select_consumption_target(body, points, 1)

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.point_id, 2)

    def test_three_consumers_receive_equal_shares(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(3, 0.5, energy=0),
                NoraletBodyState(1, -0.5, energy=0),
                NoraletBodyState(2, 0, energy=0),
            ),
            points=(ConsumableEnergyPoint(4, 0, 60),),
        )

        result = simulation.step(
            {
                3: ActionIntent(consume=True),
                1: ActionIntent(consume=True),
                2: ActionIntent(consume=True),
            }
        )

        self.assertEqual(
            tuple(body.energy for body in simulation.state.bodies),
            (20.0, 20.0, 20.0),
        )
        transfers = tuple(
            event.energy_transferred
            for event in result.events
            if isinstance(event, EnergyConsumed)
        )
        self.assertEqual(transfers, (20.0, 20.0, 20.0))

    def test_unused_share_is_redistributed_to_consumers_with_capacity(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(1, -0.5, energy=29),
                NoraletBodyState(2, 0, energy=0),
                NoraletBodyState(3, 0.5, energy=0),
            ),
            points=(ConsumableEnergyPoint(4, 0, 31),),
            energy_capacity=30,
        )

        simulation.step(
            {
                body.noralet_id: ActionIntent(consume=True)
                for body in simulation.state.bodies
            }
        )

        self.assertEqual(simulation.state.body(1).energy, 30.0)
        self.assertEqual(simulation.state.body(2).energy, 15.0)
        self.assertEqual(simulation.state.body(3).energy, 15.0)
        self.assertEqual(simulation.state.energy_points, ())

    def test_overflow_remains_in_point_when_aggregate_capacity_is_insufficient(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(1, -0.5, energy=8),
                NoraletBodyState(2, 0.5, energy=9),
            ),
            points=(ConsumableEnergyPoint(4, 0, 10),),
            energy_capacity=10,
        )

        simulation.step(
            {1: ActionIntent(consume=True), 2: ActionIntent(consume=True)}
        )

        self.assertEqual(simulation.state.body(1).energy, 10.0)
        self.assertEqual(simulation.state.body(2).energy, 10.0)
        self.assertEqual(simulation.state.energy_point(4).energy, 7.0)
        self.assertEqual(simulation.state.energy_totals.total_energy, 27.0)

    def test_consumer_insertion_order_cannot_change_allocations_or_events(self) -> None:
        bodies = (
            NoraletBodyState(1, -0.5, energy=29),
            NoraletBodyState(2, 0, energy=0),
            NoraletBodyState(3, 0.5, energy=0),
        )
        point = (ConsumableEnergyPoint(4, 0, 31),)
        first = noralet_energy_simulation(
            bodies=bodies,
            points=point,
            energy_capacity=30,
        )
        second = noralet_energy_simulation(
            bodies=tuple(reversed(bodies)),
            points=point,
            energy_capacity=30,
        )

        first_result = first.step(
            {
                1: ActionIntent(consume=True),
                2: ActionIntent(consume=True),
                3: ActionIntent(consume=True),
            }
        )
        second_result = second.step(
            {
                3: ActionIntent(consume=True),
                2: ActionIntent(consume=True),
                1: ActionIntent(consume=True),
            }
        )

        self.assertEqual(first.state, second.state)
        self.assertEqual(first_result, second_result)

    def test_partially_consumed_point_decays_after_consumption(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=9),),
            points=(ConsumableEnergyPoint(4, 0, 10),),
            pools=(EnvironmentalEnergyPool("all", 0),),
            energy_capacity=10,
            decay_rate=0.5,
        )

        result = simulation.step({1: ActionIntent(consume=True)})

        self.assertEqual(simulation.state.body(1).energy, 10.0)
        self.assertEqual(simulation.state.energy_point(4).energy, 4.5)
        self.assertEqual(simulation.state.environmental_energy_for("all"), 4.5)
        self.assertIn(EnergyPointDecayed("all", 4, 4.5, 4.5, 0, 1), result.events)

    def test_fully_consumed_point_has_no_decay_event(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=0),),
            points=(ConsumableEnergyPoint(4, 0, 10),),
            decay_rate=0.5,
        )

        result = simulation.step({1: ActionIntent(consume=True)})

        self.assertEqual(simulation.state.energy_points, ())
        self.assertFalse(
            any(isinstance(event, EnergyPointDecayed) for event in result.events)
        )


if __name__ == "__main__":
    unittest.main()
