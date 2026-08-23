"""Mortality-stream isolation and deterministic life-history experiments."""

from __future__ import annotations

import statistics
import unittest

from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    DeterministicRandomStreams,
    EnvironmentalEnergyPool,
    EnergyConsumed,
    EnergyPointDecayed,
    EnergyPointDissolved,
    EnergyPointFormed,
    FormationProbabilities,
    NoraletBodyState,
    NoraletDeathCause,
    NoraletDied,
    NoraletEnergySpent,
    RegionDefinition,
    RegionKind,
    Simulation,
)
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config


class MortalityDeterminismTests(unittest.TestCase):
    def test_prior_deterministic_death_does_not_consume_mortality_draw(self) -> None:
        seed = 24680
        config = physiology_config(baseline_loss=0, base_hazard=100)
        for body, existence_cost in (
            (NoraletBodyState(1, 9, velocity=2, energy=1), 0),
            (NoraletBodyState(1, 0, energy=1), 1),
        ):
            with self.subTest(body=body, existence_cost=existence_cost):
                simulation = noralet_energy_simulation(
                    bodies=(body,),
                    existence_cost=existence_cost,
                    seed=seed,
                    physiology=config,
                )
                stream_name = Simulation._mortality_stream_name(1)
                expected = DeterministicRandomStreams(seed).stream(
                    stream_name
                ).random()

                simulation.step()
                observed = simulation.random_streams.stream(stream_name).random()

                self.assertEqual(observed, expected)

    def test_body_insertion_order_cannot_change_state_or_events(self) -> None:
        bodies = (
            NoraletBodyState(3, 3, energy=100, age_ticks=300, condition=0.7),
            NoraletBodyState(1, -3, energy=100, age_ticks=10, condition=1),
            NoraletBodyState(2, 0, energy=100, age_ticks=100, condition=0.9),
        )
        config = physiology_config(
            baseline_loss=0,
            base_hazard=0.001,
            age_scale=200,
            age_hazard_scale=0.08,
            condition_hazard_scale=0.1,
            interaction_hazard_scale=0.2,
        )
        first = noralet_energy_simulation(
            bodies=bodies,
            seed=112233,
            physiology=config,
        )
        second = noralet_energy_simulation(
            bodies=tuple(reversed(bodies)),
            seed=112233,
            physiology=config,
        )

        first_history = [first.step() for _ in range(12)]
        second_history = [second.step() for _ in range(12)]

        self.assertEqual(first_history, second_history)
        self.assertEqual(first.state, second.state)

    def test_extra_draws_for_one_noralet_do_not_shift_anothers_stream(self) -> None:
        seed = 778899
        focal_name = Simulation._mortality_stream_name(2)
        noisy = DeterministicRandomStreams(seed)
        reference = DeterministicRandomStreams(seed)

        for _ in range(100):
            noisy.stream(Simulation._mortality_stream_name(1)).random()

        self.assertEqual(
            [noisy.stream(focal_name).random() for _ in range(20)],
            [reference.stream(focal_name).random() for _ in range(20)],
        )

    def test_added_noralet_does_not_shift_existing_mortality_history(self) -> None:
        config = physiology_config(
            baseline_loss=0,
            base_hazard=0.03,
            age_hazard_scale=0,
            condition_hazard_scale=0,
        )
        focal = NoraletBodyState(7, 0, energy=100)
        alone = noralet_energy_simulation(
            bodies=(focal,),
            seed=998877,
            physiology=config,
        )
        with_added = noralet_energy_simulation(
            bodies=(NoraletBodyState(2, -5, energy=100), focal),
            seed=998877,
            physiology=config,
        )

        def focal_death_tick(simulation: Simulation) -> int | None:
            for _ in range(200):
                result = simulation.step()
                if any(
                    isinstance(event, NoraletDied) and event.noralet_id == 7
                    for event in result.events
                ):
                    return result.tick_after
            return None

        self.assertEqual(focal_death_tick(alone), focal_death_tick(with_added))

    def test_same_seed_reproduces_complete_natural_death_history(self) -> None:
        config = physiology_config(
            baseline_loss=0.0002,
            deprivation_scale=0.002,
            base_hazard=0.002,
            age_scale=50,
            age_hazard_scale=0.02,
            condition_hazard_scale=0.04,
            interaction_hazard_scale=0.08,
        )
        bodies = tuple(
            NoraletBodyState(
                noralet_id=index,
                position=0,
                energy=20 + index,
                age_ticks=index * 5,
                condition=1 - index * 0.02,
            )
            for index in range(1, 11)
        )
        first = noralet_energy_simulation(
            bodies=bodies,
            energy_capacity=40,
            seed=13579,
            physiology=config,
        )
        second = noralet_energy_simulation(
            bodies=bodies,
            energy_capacity=40,
            seed=13579,
            physiology=config,
        )

        first_history = [first.step() for _ in range(80)]
        second_history = [second.step() for _ in range(80)]

        self.assertEqual(first_history, second_history)
        self.assertEqual(first.state, second.state)


class LifeHistorySeparationTests(unittest.TestCase):
    def test_chronic_low_energy_causes_worse_condition_and_shorter_lives(self) -> None:
        count = 64
        config = physiology_config(
            baseline_loss=0,
            deprivation_scale=0.02,
            deprivation_exponent=2,
            base_hazard=0,
            age_hazard_scale=0,
            condition_hazard_scale=0.2,
            condition_exponent=2,
            interaction_hazard_scale=0,
        )
        well = noralet_energy_simulation(
            bodies=tuple(
                NoraletBodyState(index, 0, energy=100)
                for index in range(count)
            ),
            seed=424242,
            physiology=config,
        )
        deprived = noralet_energy_simulation(
            bodies=tuple(
                NoraletBodyState(index, 0, energy=10)
                for index in range(count)
            ),
            seed=424242,
            physiology=config,
        )

        poor_death_ticks: list[int] = []
        for _ in range(15):
            well.step()
            result = deprived.step()
            poor_death_ticks.extend(
                event.tick_after
                for event in result.events
                if isinstance(event, NoraletDied)
                and event.cause is NoraletDeathCause.NATURAL
            )
        self.assertEqual(well.state.body(0).condition, 1.0)
        self.assertLess(deprived.state.body(0).condition, 0.82)

        for _ in range(105):
            well.step()
            result = deprived.step()
            poor_death_ticks.extend(
                event.tick_after
                for event in result.events
                if isinstance(event, NoraletDied)
                and event.cause is NoraletDeathCause.NATURAL
            )

        self.assertEqual(len(well.state.bodies), count)
        self.assertGreater(len(poor_death_ticks), 56)
        self.assertLess(statistics.fmean(poor_death_ticks), 90)

    def test_advanced_age_materially_increases_mortality(self) -> None:
        count = 64
        config = physiology_config(
            baseline_loss=0,
            deprivation_scale=0,
            base_hazard=0,
            age_scale=1_000,
            age_exponent=3,
            age_hazard_scale=0.15,
            condition_hazard_scale=0,
            interaction_hazard_scale=0,
        )
        young = noralet_energy_simulation(
            bodies=tuple(
                NoraletBodyState(index, 0, energy=100, age_ticks=0)
                for index in range(count)
            ),
            seed=515151,
            physiology=config,
        )
        old = noralet_energy_simulation(
            bodies=tuple(
                NoraletBodyState(index, 0, energy=100, age_ticks=1_000)
                for index in range(count)
            ),
            seed=515151,
            physiology=config,
        )

        for _ in range(20):
            young.step()
            old.step()

        young_deaths = count - len(young.state.bodies)
        old_deaths = count - len(old.state.bodies)
        self.assertLessEqual(young_deaths, 1)
        self.assertGreater(old_deaths, 50)
        self.assertGreater(old_deaths, young_deaths + 45)

    def test_long_mixed_run_preserves_energy_with_all_mortality_phases(self) -> None:
        regions = (
            RegionDefinition("left", -20, -5, RegionKind.INFERTILE),
            RegionDefinition("middle", -5, 5, RegionKind.FERTILE),
            RegionDefinition("right", 5, 20, RegionKind.SPARSE),
        )
        simulation = noralet_energy_simulation(
            regions=regions,
            pools=(
                EnvironmentalEnergyPool("left", 80),
                EnvironmentalEnergyPool("middle", 90),
                EnvironmentalEnergyPool("right", 70),
            ),
            bodies=(
                NoraletBodyState(1, -18, energy=0.2),
                NoraletBodyState(
                    2,
                    0,
                    energy=10,
                    age_ticks=200,
                    condition=0.1,
                ),
                NoraletBodyState(3, 18, velocity=0.1, energy=15),
                NoraletBodyState(4, -2, energy=12),
            ),
            points=(
                ConsumableEnergyPoint(2, -10, 6),
                ConsumableEnergyPoint(7, 0, 6),
                ConsumableEnergyPoint(11, 10, 6),
            ),
            energy_capacity=20,
            existence_cost=0.2,
            acceleration_cost=0.4,
            consume_radius=1.5,
            minimum_spacing=3.1,
            formation_min=1.5,
            formation_max=3,
            decay_rate=0.08,
            removal_threshold=0.05,
            probabilities=FormationProbabilities(0.05, 0.4, 0.9),
            seed=8675309,
            physiology=physiology_config(
                baseline_loss=0.0005,
                deprivation_scale=0.01,
                base_hazard=0.0001,
                age_scale=100,
                age_exponent=2,
                age_hazard_scale=0.08,
                condition_hazard_scale=0.2,
                interaction_hazard_scale=0.4,
            ),
        )
        baseline = simulation.initial_total_energy
        causes: set[NoraletDeathCause] = set()
        observed_event_types: set[type[object]] = set()

        for tick in range(300):
            actions = {
                body.noralet_id: ActionIntent(
                    acceleration=(0.1 if body.noralet_id % 2 == 0 else -0.1),
                    consume=True,
                )
                for body in simulation.state.bodies
            }
            result = simulation.step(actions)
            causes.update(
                event.cause
                for event in result.events
                if isinstance(event, NoraletDied)
            )
            observed_event_types.update(type(event) for event in result.events)
            simulation.audit_energy_conservation()

        self.assertIn(NoraletDeathCause.ENERGY_DEPLETION, causes)
        self.assertIn(NoraletDeathCause.NATURAL, causes)
        self.assertTrue(
            {
                EnergyConsumed,
                NoraletEnergySpent,
                EnergyPointDecayed,
                EnergyPointDissolved,
                EnergyPointFormed,
            }.issubset(observed_event_types)
        )
        self.assertAlmostEqual(
            simulation.state.energy_totals.total_energy,
            baseline,
            delta=Simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
