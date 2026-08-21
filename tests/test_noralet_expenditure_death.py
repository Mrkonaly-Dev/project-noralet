"""Tests for Noralet expenditure, affordable motion and deterministic death."""

from __future__ import annotations

import unittest

from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    EnergyConsumed,
    EnvironmentalEnergyPool,
    NoraletAccelerated,
    NoraletBodyState,
    NoraletDeathCause,
    NoraletDied,
    NoraletEnergyExpenditureReason,
    NoraletEnergyReleased,
    NoraletEnergySpent,
    NoraletMoved,
    RegionDefinition,
    RegionKind,
    TickAdvanced,
)
from noralet_energy_test_support import noralet_energy_simulation


class NoraletExpenditureAndDeathTests(unittest.TestCase):
    def test_every_living_noralet_pays_existence_cost_to_its_local_region(self) -> None:
        regions = (
            RegionDefinition("left", -10, 0, RegionKind.INFERTILE),
            RegionDefinition("right", 0, 10, RegionKind.INFERTILE),
        )
        simulation = noralet_energy_simulation(
            regions=regions,
            pools=(
                EnvironmentalEnergyPool("left", 10),
                EnvironmentalEnergyPool("right", 20),
            ),
            bodies=(
                NoraletBodyState(2, 5, energy=10),
                NoraletBodyState(1, -5, energy=10),
            ),
            existence_cost=2,
        )

        result = simulation.step()

        self.assertEqual(simulation.state.body(1).energy, 8.0)
        self.assertEqual(simulation.state.body(2).energy, 8.0)
        self.assertEqual(simulation.state.environmental_energy_for("left"), 12.0)
        self.assertEqual(simulation.state.environmental_energy_for("right"), 22.0)
        spent = tuple(
            event
            for event in result.events
            if isinstance(event, NoraletEnergySpent)
        )
        self.assertEqual(tuple(event.noralet_id for event in spent), (1, 2))
        self.assertTrue(
            all(
                event.reason is NoraletEnergyExpenditureReason.EXISTENCE
                for event in spent
            )
        )

    def test_insufficient_existence_energy_transfers_only_available_amount(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=0.4),),
            existence_cost=1,
        )

        result = simulation.step()

        self.assertEqual(simulation.state.bodies, ())
        self.assertEqual(simulation.state.environmental_energy_for("all"), 0.4)
        self.assertEqual(simulation.state.energy_totals.total_energy, 0.4)
        self.assertIn(
            NoraletEnergySpent(
                1,
                "all",
                NoraletEnergyExpenditureReason.EXISTENCE,
                0.4,
                0,
                1,
            ),
            result.events,
        )
        self.assertIn(
            NoraletDied(1, NoraletDeathCause.ENERGY_DEPLETION, 0.0, 0, 1),
            result.events,
        )

    def test_consumption_before_existence_cost_can_prevent_death(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=0),),
            points=(ConsumableEnergyPoint(4, 0, 2),),
            existence_cost=1,
        )

        result = simulation.step({1: ActionIntent(consume=True)})

        self.assertEqual(simulation.state.body(1).energy, 1.0)
        self.assertEqual(simulation.state.energy_points, ())
        self.assertFalse(any(isinstance(event, NoraletDied) for event in result.events))

    def test_coasting_moves_but_has_no_acceleration_cost(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, velocity=2, energy=10),),
            existence_cost=1,
            acceleration_cost=5,
        )

        result = simulation.step()

        body = simulation.state.body(1)
        self.assertEqual(body.position, 2.0)
        self.assertEqual(body.velocity, 2.0)
        self.assertEqual(body.energy, 9.0)
        reasons = tuple(
            event.reason
            for event in result.events
            if isinstance(event, NoraletEnergySpent)
        )
        self.assertEqual(reasons, (NoraletEnergyExpenditureReason.EXISTENCE,))

    def test_affordable_acceleration_spends_linear_cost_and_drives_motion(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=10),),
            acceleration_cost=2,
        )

        result = simulation.step({1: ActionIntent(acceleration=-3)})

        body = simulation.state.body(1)
        self.assertEqual(body.energy, 4.0)
        self.assertEqual(body.velocity, -3.0)
        self.assertEqual(body.position, -3.0)
        self.assertEqual(simulation.state.environmental_energy_for("all"), 6.0)
        self.assertIn(
            NoraletEnergySpent(
                1,
                "all",
                NoraletEnergyExpenditureReason.ACCELERATION,
                6.0,
                0,
                1,
            ),
            result.events,
        )

    def test_unaffordable_acceleration_is_reduced_and_terminal_motion_occurs(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=3),),
            acceleration_cost=2,
        )

        result = simulation.step({1: ActionIntent(acceleration=4)})

        self.assertEqual(simulation.state.bodies, ())
        self.assertEqual(simulation.state.environmental_energy_for("all"), 3.0)
        self.assertIn(NoraletAccelerated(1, 1.5, 0, 1), result.events)
        self.assertIn(NoraletMoved(1, 0.0, 1.5, 1.5, 0, 1), result.events)
        self.assertIn(
            NoraletDied(1, NoraletDeathCause.ENERGY_DEPLETION, 1.5, 0, 1),
            result.events,
        )

    def test_zero_acceleration_cost_preserves_full_free_acceleration(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=1),),
            acceleration_cost=0,
        )

        result = simulation.step({1: ActionIntent(acceleration=4)})

        body = simulation.state.body(1)
        self.assertEqual(body.energy, 1.0)
        self.assertEqual(body.position, 4.0)
        self.assertIn(NoraletAccelerated(1, 4.0, 0, 1), result.events)
        self.assertFalse(
            any(
                isinstance(event, NoraletEnergySpent)
                and event.reason is NoraletEnergyExpenditureReason.ACCELERATION
                for event in result.events
            )
        )

    def test_expenditure_returns_to_tick_start_region_even_after_crossing_region(self) -> None:
        regions = (
            RegionDefinition("left", -10, 0, RegionKind.INFERTILE),
            RegionDefinition("right", 0, 10, RegionKind.INFERTILE),
        )
        simulation = noralet_energy_simulation(
            regions=regions,
            pools=(
                EnvironmentalEnergyPool("left", 0),
                EnvironmentalEnergyPool("right", 0),
            ),
            bodies=(NoraletBodyState(1, -0.5, energy=10),),
            existence_cost=1,
            acceleration_cost=1,
        )

        simulation.step({1: ActionIntent(acceleration=1)})

        self.assertEqual(simulation.state.body(1).position, 0.5)
        self.assertEqual(simulation.state.environmental_energy_for("left"), 2.0)
        self.assertEqual(simulation.state.environmental_energy_for("right"), 0.0)

    def test_zero_energy_in_bounds_causes_deterministic_depletion_death(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=0),),
        )

        result = simulation.step()

        self.assertEqual(simulation.state.bodies, ())
        self.assertIn(
            NoraletDied(1, NoraletDeathCause.ENERGY_DEPLETION, 0.0, 0, 1),
            result.events,
        )
        self.assertEqual(simulation.state.energy_totals.total_energy, 0.0)

    def test_boundary_death_returns_remaining_energy_to_each_crossed_edge(self) -> None:
        regions = (
            RegionDefinition("left", -10, 0, RegionKind.INFERTILE),
            RegionDefinition("right", 0, 10, RegionKind.INFERTILE),
        )
        simulation = noralet_energy_simulation(
            regions=regions,
            pools=(
                EnvironmentalEnergyPool("left", 0),
                EnvironmentalEnergyPool("right", 0),
            ),
            bodies=(
                NoraletBodyState(2, 9, velocity=2, energy=10),
                NoraletBodyState(1, -9, velocity=-2, energy=10),
            ),
            existence_cost=1,
        )

        result = simulation.step()

        self.assertEqual(simulation.state.bodies, ())
        self.assertEqual(simulation.state.environmental_energy_for("left"), 10.0)
        self.assertEqual(simulation.state.environmental_energy_for("right"), 10.0)
        releases = tuple(
            event
            for event in result.events
            if isinstance(event, NoraletEnergyReleased)
        )
        self.assertEqual(
            releases,
            (
                NoraletEnergyReleased(1, "left", 9.0, 0, 1),
                NoraletEnergyReleased(2, "right", 9.0, 0, 1),
            ),
        )

    def test_boundary_death_precedes_simultaneous_energy_depletion(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 9, velocity=2, energy=1),),
            existence_cost=1,
        )

        result = simulation.step()
        death = next(event for event in result.events if isinstance(event, NoraletDied))

        self.assertEqual(death.cause, NoraletDeathCause.WORLD_BOUNDARY)
        self.assertEqual(death.resolved_position, 11.0)

    def test_complete_event_order_matches_transition_phases(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 9, velocity=1, energy=1),),
            points=(ConsumableEnergyPoint(4, 9, 5),),
            existence_cost=1,
            acceleration_cost=1,
        )

        result = simulation.step({1: ActionIntent(acceleration=1, consume=True)})

        self.assertEqual(
            tuple(type(event) for event in result.events),
            (
                EnergyConsumed,
                NoraletEnergySpent,
                NoraletEnergySpent,
                NoraletAccelerated,
                NoraletMoved,
                NoraletDied,
                NoraletEnergyReleased,
                TickAdvanced,
            ),
        )
        self.assertEqual(simulation.state.environmental_energy_for("all"), 6.0)
        self.assertEqual(simulation.state.energy_totals.total_energy, 6.0)


if __name__ == "__main__":
    unittest.main()
