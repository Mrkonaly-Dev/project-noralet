"""Age advancement and irreversible slow-condition tests."""

from __future__ import annotations

import unittest

from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    NoraletBodyState,
    Simulation,
    SimulationConfig,
    condition_after_tick,
)
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config


class ConditionAndAgeingTests(unittest.TestCase):
    def test_legacy_mode_preserves_explicit_physiology_without_advancing_it(self) -> None:
        simulation = Simulation(
            SimulationConfig(1),
            initial_bodies=(
                NoraletBodyState(1, 0, age_ticks=25, condition=0.75),
            ),
        )

        simulation.step()

        self.assertEqual(simulation.state.body(1).age_ticks, 25)
        self.assertEqual(simulation.state.body(1).condition, 0.75)

    def test_survivor_advances_exactly_one_tick_from_explicit_age(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=100, age_ticks=700),),
            physiology=physiology_config(baseline_loss=0),
        )
        state_before = simulation.state

        simulation.step()

        self.assertEqual(simulation.state.body(1).age_ticks, 701)
        self.assertEqual(state_before.body(1).age_ticks, 700)

    def test_safe_energy_applies_only_baseline_wear(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=50, condition=0.8),),
            physiology=physiology_config(baseline_loss=0.002),
        )

        simulation.step()

        self.assertAlmostEqual(simulation.state.body(1).condition, 0.798)

    def test_mild_and_severe_deprivation_are_nonlinearly_separated(self) -> None:
        config = physiology_config(
            baseline_loss=0,
            deprivation_scale=0.1,
            deprivation_exponent=2,
        )
        mild = condition_after_tick(1, 40, 100, config)
        severe = condition_after_tick(1, 10, 100, config)

        mild_loss = 1.0 - mild
        severe_loss = 1.0 - severe
        self.assertGreater(mild_loss, 0.0)
        self.assertGreater(severe_loss, mild_loss)
        self.assertAlmostEqual(mild_loss, 0.004)
        self.assertAlmostEqual(severe_loss, 0.064)
        self.assertGreater(severe_loss, 10 * mild_loss)

    def test_persistent_deprivation_accumulates_life_history(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(1, -2, energy=100),
                NoraletBodyState(2, 2, energy=10),
            ),
            physiology=physiology_config(
                baseline_loss=0.0005,
                deprivation_scale=0.01,
                deprivation_exponent=2,
            ),
        )

        for _ in range(30):
            simulation.step()

        supplied = simulation.state.body(1)
        deprived = simulation.state.body(2)
        self.assertEqual(supplied.age_ticks, deprived.age_ticks)
        self.assertAlmostEqual(supplied.condition, 0.985)
        self.assertLess(deprived.condition, supplied.condition - 0.15)

    def test_safe_energy_does_not_restore_prior_damage(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=10),),
            points=(ConsumableEnergyPoint(5, 0, 90),),
            physiology=physiology_config(
                baseline_loss=0,
                deprivation_scale=0.1,
                deprivation_exponent=2,
            ),
        )

        simulation.step()
        damaged_condition = simulation.state.body(1).condition
        simulation.step({1: ActionIntent(consume=True)})
        after_restoration = simulation.state.body(1).condition
        simulation.step()

        self.assertLess(damaged_condition, 1.0)
        self.assertEqual(after_restoration, damaged_condition)
        self.assertEqual(simulation.state.body(1).condition, damaged_condition)
        self.assertEqual(simulation.state.body(1).energy, 100.0)

    def test_condition_never_increases_across_varied_surviving_transitions(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(NoraletBodyState(1, 0, energy=80, condition=0.9),),
            points=(ConsumableEnergyPoint(5, 0, 20),),
            acceleration_cost=0.2,
            physiology=physiology_config(
                baseline_loss=0.0001,
                deprivation_scale=0.005,
            ),
        )
        conditions = [simulation.state.body(1).condition]

        for tick in range(12):
            simulation.step(
                {
                    1: ActionIntent(
                        acceleration=0.05 if tick % 2 == 0 else -0.05,
                        consume=tick == 2,
                    )
                }
            )
            conditions.append(simulation.state.body(1).condition)

        self.assertTrue(
            all(after <= before for before, after in zip(conditions, conditions[1:]))
        )


if __name__ == "__main__":
    unittest.main()
