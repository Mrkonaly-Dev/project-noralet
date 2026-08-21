"""Determinism, ordering and conservation tests for Iteration 3."""

from __future__ import annotations

import unittest

from energy_test_support import ecology_config
from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    EnergyConservationError,
    EnergyPointDecayed,
    EnergyPointDissolved,
    EnergyPointFormed,
    EnvironmentalEnergyPool,
    FormationProbabilities,
    NoraletBodyState,
    RegionDefinition,
    RegionKind,
    Simulation,
    SimulationConfig,
    WorldState,
)


PROBABILITIES = FormationProbabilities(0.0, 0.5, 1.0)


def configured_simulation(
    *,
    regions: tuple[RegionDefinition, ...],
    pools: tuple[EnvironmentalEnergyPool, ...],
    points: tuple[ConsumableEnergyPoint, ...] = (),
    bodies: tuple[NoraletBodyState, ...] = (),
    seed: int = 20260821,
    decay_rate: float = 0.2,
) -> Simulation:
    ecology = ecology_config(
        regions,
        pools,
        probabilities=PROBABILITIES,
        formation_min=1.0,
        formation_max=3.0,
        decay_rate=decay_rate,
        removal_threshold=0.05,
    )
    return Simulation(
        SimulationConfig(seed, -6, 6, ecology),
        initial_bodies=bodies,
        initial_energy_points=points,
    )


class EnergyDeterminismTests(unittest.TestCase):
    def test_multiple_region_and_point_results_ignore_insertion_order(self) -> None:
        regions = (
            RegionDefinition("left", -6, -2, RegionKind.SPARSE),
            RegionDefinition("middle", -2, 2, RegionKind.FERTILE),
            RegionDefinition("right", 2, 6, RegionKind.INFERTILE),
        )
        pools = (
            EnvironmentalEnergyPool("left", 15),
            EnvironmentalEnergyPool("middle", 20),
            EnvironmentalEnergyPool("right", 25),
        )
        points = (
            ConsumableEnergyPoint(8, 4, 2.0),
            ConsumableEnergyPoint(1, -4, 4.0),
            ConsumableEnergyPoint(5, 0, 3.0),
        )
        bodies = (
            NoraletBodyState(2, 2, -0.25),
            NoraletBodyState(1, -2, 0.25),
        )
        first = configured_simulation(
            regions=regions,
            pools=pools,
            points=points,
            bodies=bodies,
        )
        second = configured_simulation(
            regions=tuple(reversed(regions)),
            pools=tuple(reversed(pools)),
            points=tuple(reversed(points)),
            bodies=tuple(reversed(bodies)),
        )
        first_actions = {1: ActionIntent(0.5), 2: ActionIntent(-0.5)}
        second_actions = {2: ActionIntent(-0.5), 1: ActionIntent(0.5)}

        first_result = first.step(first_actions)
        second_result = second.step(second_actions)

        self.assertEqual(first.state, second.state)
        self.assertEqual(first_result, second_result)
        self.assertEqual(first.state.energy_totals.total_energy, 69.0)
        formed_ids = tuple(
            event.point_id
            for event in first_result.events
            if isinstance(event, EnergyPointFormed)
        )
        self.assertTrue(formed_ids)
        self.assertEqual(formed_ids[0], 9)
        decay_ids = tuple(
            event.point_id
            for event in first_result.events
            if isinstance(event, EnergyPointDecayed)
        )
        self.assertEqual(decay_ids, (1, 5, 8))

    def test_identical_seed_state_and_actions_reproduce_multi_tick_history(self) -> None:
        regions = (
            RegionDefinition("left", -6, -2, RegionKind.SPARSE),
            RegionDefinition("middle", -2, 2, RegionKind.FERTILE),
            RegionDefinition("right", 2, 6, RegionKind.INFERTILE),
        )
        pools = (
            EnvironmentalEnergyPool("left", 25),
            EnvironmentalEnergyPool("middle", 25),
            EnvironmentalEnergyPool("right", 25),
        )
        points = (
            ConsumableEnergyPoint(3, -3, 2.25),
            ConsumableEnergyPoint(7, 3, 1.75),
        )
        bodies = (
            NoraletBodyState(1, -1, 0),
            NoraletBodyState(2, 1, 0),
        )
        first = configured_simulation(
            regions=regions,
            pools=pools,
            points=points,
            bodies=bodies,
        )
        second = configured_simulation(
            regions=regions,
            pools=pools,
            points=points,
            bodies=bodies,
        )
        actions = (
            {1: ActionIntent(0.1), 2: ActionIntent(-0.1)},
            {},
            {1: ActionIntent(-0.1), 2: ActionIntent(0.1)},
            {},
            {},
            {1: ActionIntent(0.2), 2: ActionIntent(-0.2)},
        )

        first_history = [first.step(item) for item in actions]
        second_history = [second.step(item) for item in actions]

        self.assertEqual(first_history, second_history)
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.initial_total_energy, 79.0)
        self.assertEqual(first.state.energy_totals.total_energy, 79.0)

    def test_one_regions_formation_does_not_shift_another_regions_sequence(self) -> None:
        inactive_a = (
            RegionDefinition("A", -6, 0, RegionKind.INFERTILE),
            RegionDefinition("B", 0, 6, RegionKind.FERTILE),
        )
        active_a = (
            RegionDefinition("A", -6, 0, RegionKind.FERTILE),
            RegionDefinition("B", 0, 6, RegionKind.FERTILE),
        )
        pools = (
            EnvironmentalEnergyPool("A", 100),
            EnvironmentalEnergyPool("B", 100),
        )
        first = configured_simulation(
            regions=inactive_a,
            pools=pools,
            decay_rate=0,
        )
        second = configured_simulation(
            regions=active_a,
            pools=pools,
            decay_rate=0,
        )

        def b_sequence(simulation: Simulation) -> list[tuple[int, int, float, float]]:
            sequence: list[tuple[int, int, float, float]] = []
            for _ in range(15):
                for event in simulation.step().events:
                    if isinstance(event, EnergyPointFormed) and event.region_id == "B":
                        sequence.append(
                            (
                                event.tick_before,
                                event.tick_after,
                                event.position,
                                event.energy,
                            )
                        )
            return sequence

        first_b = b_sequence(first)
        second_b = b_sequence(second)

        self.assertEqual(first_b, second_b)
        self.assertEqual(
            first.state.environmental_energy_for("B"),
            second.state.environmental_energy_for("B"),
        )
        self.assertNotEqual(len(first.state.energy_points), len(second.state.energy_points))

    def test_energy_draws_do_not_shift_existing_named_streams(self) -> None:
        ecology_simulation = configured_simulation(
            regions=(RegionDefinition("all", -6, 6, RegionKind.FERTILE),),
            pools=(EnvironmentalEnergyPool("all", 100),),
            decay_rate=0,
        )
        reference = Simulation(
            SimulationConfig(20260821, -6, 6),
        )

        for _ in range(20):
            ecology_simulation.step()
            reference.step()

        for stream_name in ("world", "mortality", "future-system"):
            with self.subTest(stream_name=stream_name):
                energy_draws = [
                    ecology_simulation.random_streams.stream(stream_name).random()
                    for _ in range(10)
                ]
                reference_draws = [
                    reference.random_streams.stream(stream_name).random()
                    for _ in range(10)
                ]
                self.assertEqual(energy_draws, reference_draws)

    def test_conservation_failure_is_loud_and_does_not_publish_state(self) -> None:
        simulation = configured_simulation(
            regions=(RegionDefinition("all", -6, 6, RegionKind.INFERTILE),),
            pools=(EnvironmentalEnergyPool("all", 10),),
            points=(ConsumableEnergyPoint(0, 0, 2),),
            decay_rate=0,
        )
        published = simulation.state
        invalid = WorldState(
            tick=published.tick + 1,
            bodies=published.bodies,
            environmental_energy=(EnvironmentalEnergyPool("all", 10.01),),
            energy_points=published.energy_points,
            next_energy_point_id=published.next_energy_point_id,
        )

        with self.assertRaisesRegex(EnergyConservationError, "invariant violated"):
            simulation.audit_energy_conservation(invalid)

        self.assertIs(simulation.state, published)
        self.assertEqual(simulation.state.tick, 0)
        self.assertEqual(simulation.state.energy_totals.total_energy, 12.0)

    def test_long_mixed_run_keeps_the_initial_total_within_runtime_audit(self) -> None:
        simulation = configured_simulation(
            regions=(
                RegionDefinition("left", -6, -2, RegionKind.SPARSE),
                RegionDefinition("middle", -2, 2, RegionKind.FERTILE),
                RegionDefinition("right", 2, 6, RegionKind.INFERTILE),
            ),
            pools=(
                EnvironmentalEnergyPool("left", 100.1),
                EnvironmentalEnergyPool("middle", 200.2),
                EnvironmentalEnergyPool("right", 300.3),
            ),
            points=(
                ConsumableEnergyPoint(2, -4, 7.7),
                ConsumableEnergyPoint(8, 0, 8.8),
                ConsumableEnergyPoint(11, 4, 9.9),
            ),
            seed=8675309,
            decay_rate=0.13,
        )
        baseline = simulation.initial_total_energy

        for _ in range(250):
            simulation.step()
            simulation.audit_energy_conservation()

        self.assertAlmostEqual(
            simulation.state.energy_totals.total_energy,
            baseline,
            delta=Simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )

    def test_dissolution_order_is_canonical_by_point_identity(self) -> None:
        simulation = configured_simulation(
            regions=(RegionDefinition("all", -6, 6, RegionKind.INFERTILE),),
            pools=(EnvironmentalEnergyPool("all", 0),),
            points=(
                ConsumableEnergyPoint(9, 1, 0.06),
                ConsumableEnergyPoint(2, -1, 0.06),
            ),
            decay_rate=0.5,
        )

        result = simulation.step()
        dissolved_ids = tuple(
            event.point_id
            for event in result.events
            if isinstance(event, EnergyPointDissolved)
        )

        self.assertEqual(dissolved_ids, (2, 9))
        self.assertEqual(simulation.state.energy_points, ())


if __name__ == "__main__":
    unittest.main()
