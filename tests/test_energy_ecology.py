"""Focused tests for closed region-local energy transfers."""

from __future__ import annotations

import dataclasses
import unittest

from energy_test_support import ecology_config
from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    EnergyPointDecayed,
    EnergyPointDissolved,
    EnergyPointFormed,
    EnvironmentalEnergyPool,
    FormationProbabilities,
    NoraletAccelerated,
    NoraletBodyState,
    NoraletDied,
    NoraletMoved,
    RegionDefinition,
    RegionKind,
    Simulation,
    SimulationConfig,
    TickAdvanced,
)


PROBABILITIES = FormationProbabilities(0.0, 0.5, 1.0)


def make_simulation(
    regions: tuple[RegionDefinition, ...],
    pools: tuple[EnvironmentalEnergyPool, ...],
    *,
    points: tuple[ConsumableEnergyPoint, ...] = (),
    bodies: tuple[NoraletBodyState, ...] = (),
    seed: int = 123,
    formation_min: float = 2.0,
    formation_max: float = 4.0,
    decay_rate: float = 0.25,
    removal_threshold: float = 0.1,
    probabilities: FormationProbabilities = PROBABILITIES,
) -> Simulation:
    ecology = ecology_config(
        regions,
        pools,
        probabilities=probabilities,
        formation_min=formation_min,
        formation_max=formation_max,
        decay_rate=decay_rate,
        removal_threshold=removal_threshold,
    )
    return Simulation(
        SimulationConfig(
            master_seed=seed,
            left_boundary=regions[0].left,
            right_boundary=regions[-1].right,
            energy_ecology=ecology,
        ),
        initial_bodies=bodies,
        initial_energy_points=points,
    )


class EnergyEcologyTests(unittest.TestCase):
    def test_explicit_initial_energy_totals_are_observable(self) -> None:
        simulation = make_simulation(
            (
                RegionDefinition("left", -10, 0, RegionKind.INFERTILE),
                RegionDefinition("right", 0, 10, RegionKind.INFERTILE),
            ),
            (
                EnvironmentalEnergyPool("left", 4.5),
                EnvironmentalEnergyPool("right", 7.5),
            ),
            points=(
                ConsumableEnergyPoint(8, -2, 1.25),
                ConsumableEnergyPoint(3, 4, 2.75),
            ),
            decay_rate=0,
        )

        totals = simulation.state.energy_totals
        self.assertEqual(totals.environmental_energy, 12.0)
        self.assertEqual(totals.consumable_energy, 4.0)
        self.assertEqual(totals.total_energy, 16.0)
        self.assertEqual(simulation.initial_total_energy, 16.0)

    def test_zero_energy_universe_stays_exactly_zero(self) -> None:
        simulation = make_simulation(
            (RegionDefinition("fertile", -1, 1, RegionKind.FERTILE),),
            (EnvironmentalEnergyPool("fertile", 0),),
            formation_min=1,
            formation_max=1,
        )

        for _ in range(100):
            result = simulation.step()
            self.assertEqual(simulation.state.energy_totals.total_energy, 0.0)
            self.assertEqual(simulation.state.energy_points, ())
            self.assertFalse(
                any(isinstance(event, EnergyPointFormed) for event in result.events)
            )

    def test_guaranteed_formation_transfers_energy_without_creation(self) -> None:
        simulation = make_simulation(
            (RegionDefinition("fertile", -1, 1, RegionKind.FERTILE),),
            (EnvironmentalEnergyPool("fertile", 10),),
            formation_min=2,
            formation_max=2,
        )

        result = simulation.step()

        self.assertEqual(simulation.state.environmental_energy_for("fertile"), 8.0)
        self.assertEqual(simulation.state.energy_point(0).energy, 2.0)
        self.assertEqual(simulation.state.energy_totals.total_energy, 10.0)
        formation = next(
            event for event in result.events if isinstance(event, EnergyPointFormed)
        )
        self.assertEqual(formation.region_id, "fertile")
        self.assertEqual(formation.point_id, 0)
        self.assertEqual(formation.energy, 2.0)
        self.assertEqual(
            simulation.config.energy_ecology.region_for(formation.position).region_id,
            "fertile",
        )

    def test_formation_never_exceeds_available_environmental_energy(self) -> None:
        simulation = make_simulation(
            (RegionDefinition("fertile", -1, 1, RegionKind.FERTILE),),
            (EnvironmentalEnergyPool("fertile", 3),),
            formation_min=2,
            formation_max=4,
        )

        simulation.step()

        formed = simulation.state.energy_points[0]
        self.assertGreaterEqual(formed.energy, 2.0)
        self.assertLessEqual(formed.energy, 3.0)
        self.assertGreaterEqual(
            simulation.state.environmental_energy_for("fertile"),
            0.0,
        )
        self.assertEqual(simulation.state.energy_totals.total_energy, 3.0)

    def test_formation_is_impossible_at_zero_probability_or_below_minimum(self) -> None:
        zero_probability = make_simulation(
            (RegionDefinition("infertile", -1, 1, RegionKind.INFERTILE),),
            (EnvironmentalEnergyPool("infertile", 10),),
            formation_min=2,
            formation_max=2,
        )
        insufficient = make_simulation(
            (RegionDefinition("fertile", -1, 1, RegionKind.FERTILE),),
            (EnvironmentalEnergyPool("fertile", 1.5),),
            formation_min=2,
            formation_max=2,
        )

        for simulation in (zero_probability, insufficient):
            result = simulation.step()
            self.assertEqual(simulation.state.energy_points, ())
            self.assertFalse(
                any(isinstance(event, EnergyPointFormed) for event in result.events)
            )
            self.assertEqual(
                simulation.state.energy_totals.total_energy,
                simulation.initial_total_energy,
            )

    def test_region_kind_selects_its_configured_formation_probability(self) -> None:
        probabilities = FormationProbabilities(0.0, 0.000000000001, 1.0)
        simulation = make_simulation(
            (
                RegionDefinition("infertile", -3, -1, RegionKind.INFERTILE),
                RegionDefinition("sparse", -1, 1, RegionKind.SPARSE),
                RegionDefinition("fertile", 1, 3, RegionKind.FERTILE),
            ),
            (
                EnvironmentalEnergyPool("infertile", 10),
                EnvironmentalEnergyPool("sparse", 10),
                EnvironmentalEnergyPool("fertile", 10),
            ),
            probabilities=probabilities,
            formation_min=1,
            formation_max=1,
            seed=20260821,
        )

        result = simulation.step()
        formed_regions = tuple(
            event.region_id
            for event in result.events
            if isinstance(event, EnergyPointFormed)
        )

        self.assertEqual(formed_regions, ("fertile",))

    def test_every_formed_position_belongs_to_its_reported_region(self) -> None:
        regions = (
            RegionDefinition("sparse", -10, 0, RegionKind.SPARSE),
            RegionDefinition("fertile", 0, 10, RegionKind.FERTILE),
        )
        simulation = make_simulation(
            regions,
            (
                EnvironmentalEnergyPool("sparse", 100),
                EnvironmentalEnergyPool("fertile", 100),
            ),
            formation_min=1,
            formation_max=1,
            decay_rate=0,
        )

        formations: list[EnergyPointFormed] = []
        for _ in range(20):
            formations.extend(
                event
                for event in simulation.step().events
                if isinstance(event, EnergyPointFormed)
            )

        self.assertTrue(formations)
        ecology = simulation.config.energy_ecology
        self.assertIsNotNone(ecology)
        assert ecology is not None
        for event in formations:
            self.assertEqual(ecology.region_for(event.position).region_id, event.region_id)
        self.assertEqual(ecology.region_for(0).region_id, "fertile")

    def test_proportional_decay_returns_energy_to_local_pool(self) -> None:
        simulation = make_simulation(
            (RegionDefinition("local", -1, 1, RegionKind.INFERTILE),),
            (EnvironmentalEnergyPool("local", 10),),
            points=(ConsumableEnergyPoint(4, 0, 8),),
            decay_rate=0.25,
        )

        result = simulation.step()

        self.assertEqual(simulation.state.environmental_energy_for("local"), 12.0)
        self.assertEqual(simulation.state.energy_point(4).energy, 6.0)
        self.assertEqual(simulation.state.energy_totals.total_energy, 18.0)
        self.assertIn(
            EnergyPointDecayed("local", 4, 2.0, 6.0, 0, 1),
            result.events,
        )

    def test_dissolution_returns_the_entire_final_remainder(self) -> None:
        simulation = make_simulation(
            (RegionDefinition("local", -1, 1, RegionKind.INFERTILE),),
            (EnvironmentalEnergyPool("local", 10),),
            points=(ConsumableEnergyPoint(4, 0, 1.5),),
            decay_rate=0.5,
            removal_threshold=1,
            formation_min=2,
        )

        result = simulation.step()

        self.assertEqual(simulation.state.environmental_energy_for("local"), 11.5)
        self.assertEqual(simulation.state.energy_points, ())
        self.assertEqual(simulation.state.energy_totals.total_energy, 11.5)
        self.assertIn(
            EnergyPointDecayed("local", 4, 0.75, 0.75, 0, 1),
            result.events,
        )
        self.assertIn(
            EnergyPointDissolved("local", 4, 0.75, 0, 1),
            result.events,
        )

    def test_new_point_first_decays_on_the_following_transition(self) -> None:
        simulation = make_simulation(
            (RegionDefinition("fertile", -1, 1, RegionKind.FERTILE),),
            (EnvironmentalEnergyPool("fertile", 10),),
            formation_min=2,
            formation_max=2,
            decay_rate=0.5,
        )

        first = simulation.step()
        self.assertEqual(simulation.state.energy_point(0).energy, 2.0)
        self.assertFalse(
            any(isinstance(event, EnergyPointDecayed) for event in first.events)
        )

        second = simulation.step()
        self.assertEqual(simulation.state.energy_point(0).energy, 1.0)
        self.assertEqual(simulation.state.energy_point(1).energy, 2.0)
        self.assertTrue(
            any(
                isinstance(event, EnergyPointDecayed) and event.point_id == 0
                for event in second.events
            )
        )

    def test_decay_returns_to_each_points_own_region(self) -> None:
        simulation = make_simulation(
            (
                RegionDefinition("left", -2, 0, RegionKind.INFERTILE),
                RegionDefinition("right", 0, 2, RegionKind.INFERTILE),
            ),
            (
                EnvironmentalEnergyPool("left", 10),
                EnvironmentalEnergyPool("right", 20),
            ),
            points=(
                ConsumableEnergyPoint(9, 1, 8),
                ConsumableEnergyPoint(2, -1, 4),
            ),
            decay_rate=0.25,
        )

        simulation.step()

        self.assertEqual(simulation.state.environmental_energy_for("left"), 11.0)
        self.assertEqual(simulation.state.environmental_energy_for("right"), 22.0)
        self.assertEqual(simulation.state.energy_point(2).energy, 3.0)
        self.assertEqual(simulation.state.energy_point(9).energy, 6.0)

    def test_environmental_energy_does_not_diffuse_between_regions(self) -> None:
        simulation = make_simulation(
            (
                RegionDefinition("left", -2, 0, RegionKind.INFERTILE),
                RegionDefinition("right", 0, 2, RegionKind.INFERTILE),
            ),
            (
                EnvironmentalEnergyPool("left", 100),
                EnvironmentalEnergyPool("right", 0),
            ),
            decay_rate=0,
        )

        for _ in range(20):
            simulation.step()

        self.assertEqual(simulation.state.environmental_energy_for("left"), 100.0)
        self.assertEqual(simulation.state.environmental_energy_for("right"), 0.0)
        self.assertEqual(simulation.state.energy_points, ())

    def test_prior_energy_state_is_not_mutated(self) -> None:
        simulation = make_simulation(
            (RegionDefinition("local", -1, 1, RegionKind.INFERTILE),),
            (EnvironmentalEnergyPool("local", 10),),
            points=(ConsumableEnergyPoint(0, 0, 4),),
            decay_rate=0.5,
        )
        before = simulation.state
        old_pool = before.environmental_energy[0]
        old_point = before.energy_points[0]

        simulation.step()

        self.assertEqual(old_pool.energy, 10.0)
        self.assertEqual(old_point.energy, 4.0)
        self.assertEqual(before.tick, 0)
        self.assertIsNot(before, simulation.state)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            old_point.energy = 99.0  # type: ignore[misc]

    def test_event_order_follows_physics_then_ecology_then_clock(self) -> None:
        simulation = make_simulation(
            (RegionDefinition("fertile", -1, 1, RegionKind.FERTILE),),
            (EnvironmentalEnergyPool("fertile", 10),),
            points=(ConsumableEnergyPoint(7, 0, 0.2),),
            bodies=(NoraletBodyState(1, 0, 0),),
            formation_min=2,
            formation_max=2,
            decay_rate=0.5,
            removal_threshold=0.1,
        )

        result = simulation.step({1: ActionIntent(2)})

        self.assertEqual(
            tuple(type(event) for event in result.events),
            (
                NoraletAccelerated,
                NoraletMoved,
                NoraletDied,
                EnergyPointDecayed,
                EnergyPointDissolved,
                EnergyPointFormed,
                TickAdvanced,
            ),
        )
        self.assertEqual(simulation.state.bodies, ())
        self.assertEqual(simulation.state.energy_totals.total_energy, 10.2)


if __name__ == "__main__":
    unittest.main()
