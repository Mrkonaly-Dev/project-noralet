"""Formation-spacing, determinism and long-run Iteration 4 tests."""

from __future__ import annotations

import unittest

from energy_test_support import ecology_config
from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    DeterministicRandomStreams,
    EnergyConservationError,
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
from noralet_energy_test_support import noralet_energy_simulation


PROBABILITIES = FormationProbabilities(0.0, 0.5, 1.0)


class Iteration4DeterminismTests(unittest.TestCase):
    def test_formation_too_close_to_existing_point_is_rejected_without_transfer(self) -> None:
        region = RegionDefinition("fertile", -1, 1, RegionKind.FERTILE)
        ecology = ecology_config(
            (region,),
            (EnvironmentalEnergyPool("fertile", 10),),
            probabilities=PROBABILITIES,
            formation_min=2,
            formation_max=2,
            decay_rate=0,
            removal_threshold=0,
            minimum_spacing=3,
        )
        simulation = Simulation(
            SimulationConfig(7, -1, 1, ecology),
            initial_energy_points=(ConsumableEnergyPoint(0, 0, 2),),
        )

        result = simulation.step()

        self.assertEqual(simulation.state.environmental_energy_for("fertile"), 10.0)
        self.assertEqual(simulation.state.energy_points, (ConsumableEnergyPoint(0, 0, 2),))
        self.assertFalse(
            any(isinstance(event, EnergyPointFormed) for event in result.events)
        )

    def test_spacing_rejection_does_not_resample_formation_position(self) -> None:
        seed = 1234
        region_id = "fertile"
        region = RegionDefinition(region_id, -1, 1, RegionKind.FERTILE)
        ecology = ecology_config(
            (region,),
            (EnvironmentalEnergyPool(region_id, 10),),
            probabilities=PROBABILITIES,
            formation_min=2,
            formation_max=2,
            decay_rate=0,
            removal_threshold=0,
            minimum_spacing=3,
        )
        simulation = Simulation(
            SimulationConfig(seed, -1, 1, ecology),
            initial_energy_points=(ConsumableEnergyPoint(0, 0, 2),),
        )
        position_stream_name = Simulation._energy_stream_name(
            region_id,
            "position",
        )
        reference = DeterministicRandomStreams(seed).stream(position_stream_name)
        reference.random()
        expected_next_draw = reference.random()

        simulation.step()
        actual_next_draw = simulation.random_streams.stream(
            position_stream_name
        ).random()

        self.assertEqual(actual_next_draw, expected_next_draw)

    def test_same_tick_candidate_conflict_rejects_both_without_region_priority(self) -> None:
        regions = (
            RegionDefinition("left", -1, 0, RegionKind.FERTILE),
            RegionDefinition("right", 0, 1, RegionKind.FERTILE),
        )
        pools = (
            EnvironmentalEnergyPool("left", 10),
            EnvironmentalEnergyPool("right", 20),
        )

        def build(
            configured_regions: tuple[RegionDefinition, ...],
            configured_pools: tuple[EnvironmentalEnergyPool, ...],
        ) -> Simulation:
            ecology = ecology_config(
                configured_regions,
                configured_pools,
                probabilities=PROBABILITIES,
                formation_min=2,
                formation_max=2,
                decay_rate=0,
                removal_threshold=0,
                minimum_spacing=2,
            )
            return Simulation(SimulationConfig(99, -1, 1, ecology))

        first = build(regions, pools)
        second = build(tuple(reversed(regions)), tuple(reversed(pools)))

        first_result = first.step()
        second_result = second.step()

        self.assertEqual(first.state, second.state)
        self.assertEqual(first_result, second_result)
        self.assertEqual(first.state.energy_points, ())
        self.assertEqual(first.state.environmental_energy_for("left"), 10.0)
        self.assertEqual(first.state.environmental_energy_for("right"), 20.0)

    def test_all_generated_points_preserve_global_minimum_spacing(self) -> None:
        regions = (
            RegionDefinition("fertile", -100, 100, RegionKind.FERTILE),
        )
        simulation = noralet_energy_simulation(
            regions=regions,
            pools=(EnvironmentalEnergyPool("fertile", 500),),
            minimum_spacing=3,
            consume_radius=1,
            formation_min=2,
            formation_max=2,
            decay_rate=0,
        )

        for _ in range(50):
            simulation.step()
            points = tuple(
                sorted(simulation.state.energy_points, key=lambda point: point.position)
            )
            for previous, current in zip(points, points[1:]):
                self.assertGreaterEqual(current.position - previous.position, 3.0)

    def test_identical_mixed_runs_ignore_initial_insertion_order(self) -> None:
        regions = (
            RegionDefinition("left", -20, -5, RegionKind.INFERTILE),
            RegionDefinition("middle", -5, 5, RegionKind.FERTILE),
            RegionDefinition("right", 5, 20, RegionKind.SPARSE),
        )
        pools = (
            EnvironmentalEnergyPool("left", 60.1),
            EnvironmentalEnergyPool("middle", 70.2),
            EnvironmentalEnergyPool("right", 80.3),
        )
        bodies = (
            NoraletBodyState(1, -18, velocity=-0.5, energy=8),
            NoraletBodyState(2, 0, energy=2),
            NoraletBodyState(3, 18, velocity=0.5, energy=8),
        )
        points = (
            ConsumableEnergyPoint(2, -10, 6),
            ConsumableEnergyPoint(7, 0, 6),
            ConsumableEnergyPoint(11, 10, 6),
        )

        def build(reverse: bool) -> Simulation:
            return noralet_energy_simulation(
                regions=tuple(reversed(regions)) if reverse else regions,
                pools=tuple(reversed(pools)) if reverse else pools,
                bodies=tuple(reversed(bodies)) if reverse else bodies,
                points=tuple(reversed(points)) if reverse else points,
                energy_capacity=20,
                existence_cost=0.15,
                acceleration_cost=0.4,
                consume_radius=1.5,
                minimum_spacing=3.1,
                formation_min=1.5,
                formation_max=3,
                decay_rate=0.08,
                removal_threshold=0.05,
                seed=8675309,
            )

        first = build(False)
        second = build(True)
        first_history = []
        second_history = []
        for tick in range(80):
            first_actions = {
                body.noralet_id: ActionIntent(
                    acceleration=self._mixed_acceleration(body.noralet_id, tick),
                    consume=True,
                )
                for body in first.state.bodies
            }
            second_actions = dict(reversed(tuple(first_actions.items())))
            first_history.append(first.step(first_actions))
            second_history.append(second.step(second_actions))

        self.assertEqual(first_history, second_history)
        self.assertEqual(first.state, second.state)

    def test_long_mixed_run_preserves_complete_three_form_total(self) -> None:
        regions = (
            RegionDefinition("left", -20, -5, RegionKind.INFERTILE),
            RegionDefinition("middle", -5, 5, RegionKind.FERTILE),
            RegionDefinition("right", 5, 20, RegionKind.SPARSE),
        )
        simulation = noralet_energy_simulation(
            regions=regions,
            pools=(
                EnvironmentalEnergyPool("left", 60.1),
                EnvironmentalEnergyPool("middle", 70.2),
                EnvironmentalEnergyPool("right", 80.3),
            ),
            bodies=(
                NoraletBodyState(1, -18, velocity=-0.5, energy=8),
                NoraletBodyState(2, 0, energy=2),
                NoraletBodyState(3, 18, velocity=0.5, energy=8),
            ),
            points=(
                ConsumableEnergyPoint(2, -10, 6),
                ConsumableEnergyPoint(7, 0, 6),
                ConsumableEnergyPoint(11, 10, 6),
            ),
            energy_capacity=20,
            existence_cost=0.15,
            acceleration_cost=0.4,
            consume_radius=1.5,
            minimum_spacing=3.1,
            formation_min=1.5,
            formation_max=3,
            decay_rate=0.08,
            removal_threshold=0.05,
            seed=8675309,
        )
        baseline = simulation.initial_total_energy

        for tick in range(300):
            actions = {
                body.noralet_id: ActionIntent(
                    acceleration=self._mixed_acceleration(body.noralet_id, tick),
                    consume=True,
                )
                for body in simulation.state.bodies
            }
            simulation.step(actions)
            simulation.audit_energy_conservation()

        self.assertAlmostEqual(
            simulation.state.energy_totals.total_energy,
            baseline,
            delta=Simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )

    def test_conservation_audit_includes_noralet_energy_and_prevents_publication(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=5),),
            pools=(EnvironmentalEnergyPool("all", 10),),
        )
        published = simulation.state
        invalid = WorldState(
            tick=1,
            bodies=(NoraletBodyState(1, 0, energy=5.01),),
            environmental_energy=published.environmental_energy,
            energy_points=published.energy_points,
            next_energy_point_id=published.next_energy_point_id,
        )

        with self.assertRaises(EnergyConservationError):
            simulation.audit_energy_conservation(invalid)

        self.assertIs(simulation.state, published)
        self.assertEqual(simulation.state.tick, 0)

    def test_consumption_and_expenditure_do_not_shift_unrelated_random_streams(self) -> None:
        seed = 2468
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=5),),
            points=(ConsumableEnergyPoint(4, 0, 5),),
            existence_cost=0.1,
            acceleration_cost=0.2,
            seed=seed,
        )
        reference = DeterministicRandomStreams(seed)

        for _ in range(5):
            if simulation.state.bodies:
                simulation.step(
                    {1: ActionIntent(acceleration=0.1, consume=True)}
                )
            else:
                simulation.step()

        observed = [
            simulation.random_streams.stream("world").random()
            for _ in range(10)
        ]
        expected = [reference.stream("world").random() for _ in range(10)]
        self.assertEqual(observed, expected)

    @staticmethod
    def _mixed_acceleration(noralet_id: int, tick: int) -> float:
        if noralet_id == 1:
            return -0.2
        if noralet_id == 3:
            return 0.2
        return 0.1 if tick % 2 == 0 else -0.1


if __name__ == "__main__":
    unittest.main()
