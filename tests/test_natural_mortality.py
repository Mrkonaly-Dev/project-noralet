"""Pure mortality model and natural-death resolution tests."""

from __future__ import annotations

import math
import unittest

from noralet.simulation import (
    DeterministicRandomStreams,
    NoraletBodyState,
    NoraletDeathCause,
    NoraletDied,
    NoraletEnergyReleased,
    NoraletMoved,
    RegionDefinition,
    RegionKind,
    EnvironmentalEnergyPool,
    mortality_hazard,
    natural_death_probability,
    Simulation,
)
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config


class MortalityFunctionTests(unittest.TestCase):
    def test_exact_hazard_equation_and_probability_transform(self) -> None:
        config = physiology_config(
            base_hazard=0.001,
            age_scale=100,
            age_exponent=2,
            age_hazard_scale=0.1,
            condition_hazard_scale=0.2,
            condition_exponent=2,
            interaction_hazard_scale=0.3,
        )

        hazard = mortality_hazard(100, 0.25, config)
        probability = natural_death_probability(100, 0.25, config)

        self.assertAlmostEqual(hazard, 0.38225)
        self.assertAlmostEqual(probability, 1.0 - math.exp(-hazard))

    def test_age_condition_interaction_and_strong_state_separation(self) -> None:
        config = physiology_config(
            base_hazard=1e-9,
            age_scale=100,
            age_exponent=3,
            age_hazard_scale=0.02,
            condition_hazard_scale=0.03,
            condition_exponent=2,
            interaction_hazard_scale=0.5,
        )
        young_good = mortality_hazard(1, 1.0, config)
        old_good = mortality_hazard(100, 1.0, config)
        young_poor = mortality_hazard(1, 0.0, config)
        old_poor = mortality_hazard(100, 0.0, config)
        without_interaction = 1e-9 + 0.02 + 0.03

        self.assertLess(young_good, 1e-7)
        self.assertGreater(old_good, young_good)
        self.assertGreater(young_poor, young_good)
        self.assertGreater(old_poor, old_good)
        self.assertGreater(old_poor, young_poor)
        self.assertGreater(old_poor, without_interaction)
        self.assertGreater(old_poor / young_good, 1_000_000)

    def test_extreme_valid_age_has_finite_hazard_and_probability_below_one(self) -> None:
        config = physiology_config(
            base_hazard=1,
            age_scale=1e-300,
            age_exponent=10,
            age_hazard_scale=1e300,
            condition_hazard_scale=1e300,
            interaction_hazard_scale=1e300,
        )

        hazard = mortality_hazard(10**10_000, 0.0, config)
        probability = natural_death_probability(10**10_000, 0.0, config)

        self.assertTrue(math.isfinite(hazard))
        self.assertGreaterEqual(hazard, 0.0)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLess(probability, 1.0)


class NaturalMortalityResolutionTests(unittest.TestCase):
    def test_controlled_high_probability_causes_natural_death(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=7),),
            physiology=physiology_config(
                baseline_loss=0,
                base_hazard=100,
            ),
        )

        result = simulation.step()

        self.assertEqual(simulation.state.bodies, ())
        self.assertIn(
            NoraletDied(1, NoraletDeathCause.NATURAL, 0.0, 0, 1),
            result.events,
        )

    def test_known_draw_above_probability_survives_and_draws_exactly_once(self) -> None:
        seed = 20260821
        stream_name = Simulation._mortality_stream_name(1)
        reference = DeterministicRandomStreams(seed).stream(stream_name)
        first_roll = reference.random()
        expected_next_roll = reference.random()
        controlled_probability = first_roll / 2.0
        controlled_hazard = -math.log1p(-controlled_probability)
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=100, age_ticks=40),),
            seed=seed,
            physiology=physiology_config(
                baseline_loss=0,
                base_hazard=controlled_hazard,
            ),
        )

        result = simulation.step()

        self.assertFalse(any(isinstance(event, NoraletDied) for event in result.events))
        self.assertEqual(simulation.state.body(1).age_ticks, 41)
        self.assertEqual(simulation.state.body(1).condition, 1.0)
        self.assertEqual(
            simulation.random_streams.stream(stream_name).random(),
            expected_next_roll,
        )

    def test_movement_then_natural_death_releases_energy_at_resolved_region(self) -> None:
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
            bodies=(NoraletBodyState(1, -1, velocity=2, energy=7),),
            physiology=physiology_config(baseline_loss=0, base_hazard=100),
        )
        baseline = simulation.initial_total_energy

        result = simulation.step()

        self.assertEqual(simulation.state.bodies, ())
        self.assertEqual(simulation.state.environmental_energy_for("left"), 0.0)
        self.assertEqual(simulation.state.environmental_energy_for("right"), 7.0)
        self.assertEqual(simulation.state.energy_totals.total_energy, baseline)
        self.assertEqual(
            tuple(type(event) for event in result.events[:3]),
            (NoraletMoved, NoraletDied, NoraletEnergyReleased),
        )
        self.assertEqual(result.events[1].cause, NoraletDeathCause.NATURAL)
        self.assertEqual(result.events[1].resolved_position, 1.0)
        self.assertEqual(result.events[2].region_id, "right")

    def test_boundary_and_depletion_take_precedence_over_natural_death(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(1, 9, velocity=2, energy=1),
                NoraletBodyState(2, 0, energy=1),
                NoraletBodyState(3, -5, energy=2),
            ),
            existence_cost=1,
            physiology=physiology_config(baseline_loss=0, base_hazard=100),
        )

        result = simulation.step()
        deaths = tuple(
            event for event in result.events if isinstance(event, NoraletDied)
        )

        self.assertEqual(
            tuple((event.noralet_id, event.cause) for event in deaths),
            (
                (1, NoraletDeathCause.WORLD_BOUNDARY),
                (2, NoraletDeathCause.ENERGY_DEPLETION),
                (3, NoraletDeathCause.NATURAL),
            ),
        )

    def test_natural_phase_follows_prior_death_and_release_phases(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(1, 9, velocity=2, energy=2),
                NoraletBodyState(2, 0, energy=1),
                NoraletBodyState(3, -5, energy=2),
            ),
            existence_cost=1,
            physiology=physiology_config(baseline_loss=0, base_hazard=100),
        )

        result = simulation.step()
        terminal = tuple(
            event
            for event in result.events
            if isinstance(event, (NoraletDied, NoraletEnergyReleased))
        )

        self.assertEqual(
            tuple(type(event) for event in terminal),
            (
                NoraletDied,
                NoraletDied,
                NoraletEnergyReleased,
                NoraletDied,
                NoraletEnergyReleased,
            ),
        )
        self.assertEqual(
            tuple(event.noralet_id for event in terminal),
            (1, 2, 1, 3, 3),
        )
        self.assertEqual(terminal[0].cause, NoraletDeathCause.WORLD_BOUNDARY)
        self.assertEqual(terminal[1].cause, NoraletDeathCause.ENERGY_DEPLETION)
        self.assertEqual(terminal[3].cause, NoraletDeathCause.NATURAL)


if __name__ == "__main__":
    unittest.main()
