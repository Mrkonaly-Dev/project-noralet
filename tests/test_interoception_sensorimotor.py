"""Interoception and transition-feedback tests for Iteration 6."""

from __future__ import annotations

import math
import unittest

from experience_test_support import experience_config, experience_simulation
from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    NoraletAccelerated,
    NoraletBodyState,
)
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config


class InteroceptionTests(unittest.TestCase):
    def test_initial_distress_uses_only_bounded_derived_sensations(self) -> None:
        config = experience_config(
            energy_exponent=2,
            condition_exponent=0.5,
        )
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(
                    1,
                    0,
                    energy=25,
                    condition=0.36,
                    perceptual_signature=(0, 0),
                ),
            ),
        )

        interoception = simulation.experience_for(1).interoception

        self.assertEqual(interoception.energy_distress, 0.75**2)
        self.assertEqual(interoception.condition_distress, 0.8)
        self.assertEqual(interoception.energetic_exertion, 0.0)
        self.assertFalse(hasattr(interoception, "energy"))
        self.assertFalse(hasattr(interoception, "energy_ratio"))
        self.assertFalse(hasattr(interoception, "condition"))
        self.assertFalse(hasattr(interoception, "age_ticks"))

    def test_distress_reaches_its_endpoint_values(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(1, -1, energy=100, perceptual_signature=(0, 0)),
                NoraletBodyState(
                    2,
                    1,
                    energy=0,
                    condition=0,
                    perceptual_signature=(1, 1),
                ),
            )
        )

        healthy = simulation.experience_for(1).interoception
        depleted = simulation.experience_for(2).interoception

        self.assertEqual((healthy.energy_distress, healthy.condition_distress), (0, 0))
        self.assertEqual((depleted.energy_distress, depleted.condition_distress), (1, 1))

    def test_lower_energy_and_condition_produce_stronger_distress(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(
                    1,
                    -1,
                    energy=75,
                    condition=0.8,
                    perceptual_signature=(0, 0),
                ),
                NoraletBodyState(
                    2,
                    1,
                    energy=25,
                    condition=0.2,
                    perceptual_signature=(1, 1),
                ),
            )
        )

        stronger = simulation.experience_for(2).interoception
        weaker = simulation.experience_for(1).interoception

        self.assertGreater(stronger.energy_distress, weaker.energy_distress)
        self.assertGreater(stronger.condition_distress, weaker.condition_distress)

    def test_condition_distress_reads_the_current_published_body(self) -> None:
        config = experience_config(condition_exponent=2)
        simulation = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(
                    1,
                    0,
                    energy=100,
                    condition=0.8,
                    perceptual_signature=(0, 0),
                ),
            ),
            physiology=physiology_config(baseline_loss=0.1),
            experience=config,
        )
        before = simulation.experience_for(1).interoception.condition_distress

        simulation.step()
        body_after = simulation.state.body(1)
        after = simulation.experience_for(1).interoception.condition_distress

        self.assertAlmostEqual(before, (1 - 0.8) ** 2)
        self.assertAlmostEqual(body_after.condition, 0.7)
        self.assertAlmostEqual(after, (1 - body_after.condition) ** 2)


class SensorimotorFeedbackTests(unittest.TestCase):
    def test_initial_feedback_is_neutral_even_while_body_is_moving(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(
                    1,
                    0,
                    velocity=3,
                    energy=50,
                    perceptual_signature=(0, 0),
                ),
            )
        )

        experience = simulation.experience_for(1)

        self.assertEqual(
            (
                experience.sensorimotor_feedback.motor_direction,
                experience.sensorimotor_feedback.motor_effort,
                experience.sensorimotor_feedback.consume_activation,
                experience.sensorimotor_feedback.ingestion_signal,
                experience.interoception.energetic_exertion,
            ),
            (0, 0, 0, 0, 0),
        )

    def test_applied_acceleration_and_combined_actual_expenditure_feed_t_plus_one(self) -> None:
        config = experience_config(motor_scale=2, exertion_scale=4)
        simulation = experience_simulation(
            experience=config,
            existence_cost=2,
            acceleration_cost=3,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
            ),
        )

        before = simulation.experience_for(1)
        simulation.step({1: ActionIntent(acceleration=-2)})
        after = simulation.experience_for(1)

        self.assertEqual(before.sensorimotor_feedback.motor_effort, 0)
        self.assertEqual(after.sensorimotor_feedback.motor_direction, -1)
        self.assertAlmostEqual(after.sensorimotor_feedback.motor_effort, 1 - math.exp(-1))
        self.assertAlmostEqual(after.interoception.energetic_exertion, 1 - math.exp(-2))
        self.assertEqual(simulation.state.body(1).energy, 42)

    def test_unaffordable_acceleration_is_reduced_but_death_has_no_feedback(self) -> None:
        simulation = experience_simulation(
            acceleration_cost=2,
            bodies=(
                NoraletBodyState(1, 0, energy=3, perceptual_signature=(0, 0)),
            ),
        )

        result = simulation.step({1: ActionIntent(acceleration=4)})

        self.assertIn(NoraletAccelerated(1, 1.5, 0, 1), result.events)
        with self.assertRaises(KeyError):
            simulation.experience_for(1)

    def test_consume_activation_reports_attempt_even_when_it_fails(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
            )
        )

        simulation.step({1: ActionIntent(consume=True)})
        feedback = simulation.experience_for(1).sensorimotor_feedback

        self.assertEqual(feedback.consume_activation, 1)
        self.assertEqual(feedback.ingestion_signal, 0)
        self.assertFalse(hasattr(feedback, "consume_success"))
        self.assertFalse(hasattr(feedback, "consumed_energy"))

    def test_successful_ingestion_is_bounded_and_uses_actual_transfer(self) -> None:
        config = experience_config(ingestion_scale=5)
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=98, perceptual_signature=(0, 0)),
            ),
            points=(ConsumableEnergyPoint(1, 0, 9),),
        )

        simulation.step({1: ActionIntent(consume=True)})
        feedback = simulation.experience_for(1).sensorimotor_feedback

        self.assertEqual(simulation.state.body(1).energy, 100)
        self.assertEqual(feedback.consume_activation, 1)
        self.assertAlmostEqual(feedback.ingestion_signal, 1 - math.exp(-2 / 5))
        self.assertGreater(feedback.ingestion_signal, 0)
        self.assertLess(feedback.ingestion_signal, 1)
        self.assertEqual(simulation.experience_for(1).interoception.energy_distress, 0)

    def test_available_food_without_consume_action_produces_no_ingestion(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
            ),
            points=(ConsumableEnergyPoint(1, 0, 5),),
        )

        simulation.step()
        feedback = simulation.experience_for(1).sensorimotor_feedback

        self.assertEqual(feedback.consume_activation, 0)
        self.assertEqual(feedback.ingestion_signal, 0)
        self.assertEqual(simulation.state.energy_point(1).energy, 5)

    def test_feedback_is_replaced_by_each_new_transition(self) -> None:
        simulation = experience_simulation(
            acceleration_cost=0,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
            ),
        )

        simulation.step({1: ActionIntent(acceleration=1, consume=True)})
        first = simulation.experience_for(1).sensorimotor_feedback
        simulation.step()
        second = simulation.experience_for(1).sensorimotor_feedback

        self.assertEqual(first.motor_direction, 1)
        self.assertGreater(first.motor_effort, 0)
        self.assertEqual(first.consume_activation, 1)
        self.assertEqual(simulation.state.body(1).velocity, 1)
        self.assertEqual(second.motor_direction, 0)
        self.assertEqual(second.motor_effort, 0)
        self.assertEqual(second.consume_activation, 0)

    def test_coasting_has_no_motor_effort(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(
                    1,
                    0,
                    velocity=1,
                    energy=50,
                    perceptual_signature=(0, 0),
                ),
            )
        )

        simulation.step()
        feedback = simulation.experience_for(1).sensorimotor_feedback

        self.assertEqual(simulation.state.body(1).position, 1)
        self.assertEqual(feedback.motor_direction, 0)
        self.assertEqual(feedback.motor_effort, 0)

    def test_existence_expenditure_creates_exertion_without_motor_effort(self) -> None:
        config = experience_config(exertion_scale=2)
        simulation = experience_simulation(
            experience=config,
            existence_cost=2,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
            ),
        )

        simulation.step()
        experience = simulation.experience_for(1)

        self.assertEqual(experience.sensorimotor_feedback.motor_effort, 0)
        self.assertAlmostEqual(
            experience.interoception.energetic_exertion,
            1 - math.exp(-1),
        )
        self.assertFalse(hasattr(experience.interoception, "expenditure_reason"))
        self.assertFalse(hasattr(experience.interoception, "energy_spent"))

    def test_larger_actual_expenditure_produces_stronger_bounded_exertion(self) -> None:
        body = NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0))
        lower = experience_simulation(bodies=(body,), existence_cost=1)
        higher = experience_simulation(bodies=(body,), existence_cost=4)

        lower.step()
        higher.step()
        lower_signal = lower.experience_for(1).interoception.energetic_exertion
        higher_signal = higher.experience_for(1).interoception.energetic_exertion

        self.assertGreater(higher_signal, lower_signal)
        self.assertGreaterEqual(lower_signal, 0)
        self.assertLess(higher_signal, 1)

    def test_each_existing_death_cause_has_no_next_experience(self) -> None:
        config = experience_config()
        boundary = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(
                    1,
                    9,
                    velocity=2,
                    energy=10,
                    perceptual_signature=(0, 0),
                ),
            ),
        )
        depletion = experience_simulation(
            experience=config,
            existence_cost=1,
            bodies=(
                NoraletBodyState(1, 0, energy=1, perceptual_signature=(0, 0)),
            ),
        )
        natural = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(1, 0, energy=10, perceptual_signature=(0, 0)),
            ),
            physiology=physiology_config(base_hazard=1_000),
            experience=config,
            seed=1,
        )

        for simulation in (boundary, depletion, natural):
            simulation.step()
            self.assertEqual(simulation.experiences_for_all(), ())
            with self.assertRaises(KeyError):
                simulation.experience_for(1)

    def test_saturating_feedback_remains_strictly_below_one(self) -> None:
        config = experience_config(motor_scale=1e-300)
        simulation = experience_simulation(
            experience=config,
            acceleration_cost=0,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
            ),
        )

        simulation.step({1: ActionIntent(acceleration=1)})
        effort = simulation.experience_for(1).sensorimotor_feedback.motor_effort

        self.assertGreater(effort, 0)
        self.assertLess(effort, 1)


if __name__ == "__main__":
    unittest.main()
