"""Autonomous lockstep coordination, routing and brain-lifecycle tests."""

from __future__ import annotations

from dataclasses import fields
import unittest

import torch

from brain_test_support import (
    actuator_config,
    autonomous_setup,
    brain_body,
    brain_config,
)
from noralet import (
    ActionIntent,
    AutonomousSimulationRunner,
    BaseBrain,
    NoraletExperience,
    RoutedNoraletExperience,
)
from signal_test_support import signal_simulation


class AutonomousCoordinatorValidationTests(unittest.TestCase):
    def test_autonomy_requires_experience_signals_and_physical_actuator(self) -> None:
        simulation = signal_simulation(
            bodies=(brain_body(1, 0),),
        )
        assert simulation.config.noralet_experience is not None
        assert simulation.config.noralet_signals is not None
        base = BaseBrain(
            brain_config(),
            simulation.config.noralet_experience,
            simulation.config.noralet_signals,
            actuator_config(),
        )

        with self.assertRaisesRegex(ValueError, "requires Experience"):
            AutonomousSimulationRunner(simulation, base)

    def test_base_brain_interface_must_match_simulation(self) -> None:
        runner, _ = autonomous_setup(actuator=actuator_config(0.25))
        simulation = runner.simulation
        assert simulation.config.noralet_experience is not None
        assert simulation.config.noralet_signals is not None
        mismatched = BaseBrain(
            brain_config(),
            simulation.config.noralet_experience,
            simulation.config.noralet_signals,
            actuator_config(0.5),
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            AutonomousSimulationRunner(simulation, mismatched)

    def test_routed_identity_remains_outside_noralet_experience(self) -> None:
        runner, _ = autonomous_setup()
        routed = runner.simulation.routed_experiences_for_all()

        self.assertEqual(
            tuple(field.name for field in fields(RoutedNoraletExperience)),
            ("noralet_id", "experience"),
        )
        self.assertEqual(tuple(item.noralet_id for item in routed), (1, 2))
        self.assertTrue(
            all(
                isinstance(item.experience, NoraletExperience)
                for item in routed
            )
        )
        self.assertFalse(hasattr(routed[0].experience, "noralet_id"))

    def test_world_state_contains_no_neural_model_or_tensor_state(self) -> None:
        runner, _ = autonomous_setup()
        state = runner.simulation.state

        self.assertFalse(
            any(
                isinstance(getattr(state, field.name), (torch.Tensor, torch.nn.Module))
                for field in fields(state)
            )
        )
        for forbidden in ("brain", "model", "hidden_state", "device"):
            self.assertFalse(hasattr(state, forbidden))


class AutonomousCoordinatorTests(unittest.TestCase):
    def test_each_living_brain_activates_exactly_once_per_autonomous_tick(self) -> None:
        runner, _ = autonomous_setup()

        first = runner.step()
        second = runner.step()

        self.assertEqual(len(first.action_intents), 2)
        self.assertEqual(len(second.action_intents), 2)
        self.assertEqual(runner.brain_for(1).activation_count, 2)
        self.assertEqual(runner.brain_for(2).activation_count, 2)
        self.assertEqual(runner.simulation.state.tick, 2)

    def test_current_experiences_all_precede_one_world_step(self) -> None:
        coordinated, _ = autonomous_setup(simulation_seed=77)
        manual, _ = autonomous_setup(simulation_seed=77)
        routed = manual.simulation.routed_experiences_for_all()
        manual_actions = tuple(
            (
                item.noralet_id,
                manual.brain_for(item.noralet_id).act(item.experience),
            )
            for item in routed
        )
        manual_tick = manual.simulation.step(dict(manual_actions))

        result = coordinated.step()

        self.assertEqual(result.action_intents, manual_actions)
        self.assertEqual(result.tick_result, manual_tick)
        self.assertEqual(coordinated.simulation.state, manual.simulation.state)

    def test_observation_reads_do_not_activate_brains_or_shift_actions(self) -> None:
        observed, _ = autonomous_setup(simulation_seed=91)
        control, _ = autonomous_setup(simulation_seed=91)

        for _ in range(20):
            observed.simulation.experiences_for_all()
            observed.simulation.routed_experiences_for_all()
        self.assertEqual(observed.brain_for(1).activation_count, 0)
        observed_result = observed.step()
        control_result = control.step()

        self.assertEqual(observed_result, control_result)
        self.assertEqual(observed.simulation.state, control.simulation.state)

    def test_death_removes_brain_and_prevents_future_activation(self) -> None:
        runner, _ = autonomous_setup(
            bodies=(
                brain_body(1, 9.5, velocity=2),
                brain_body(2, 0),
            ),
            actuator=actuator_config(0.1),
        )
        doomed = runner.brain_for(1)

        runner.step()

        self.assertEqual(runner.brain_ids, (2,))
        self.assertEqual(doomed.activation_count, 1)
        with self.assertRaises(KeyError):
            runner.brain_for(1)
        runner.step()
        self.assertEqual(doomed.activation_count, 1)
        self.assertEqual(runner.brain_for(2).activation_count, 2)

    def test_one_death_does_not_change_other_brain_or_action_stream(self) -> None:
        population, _ = autonomous_setup(
            bodies=(
                brain_body(1, -9.5, velocity=-2),
                brain_body(2, 2),
            ),
            actuator=actuator_config(0.05),
            simulation_seed=101,
        )
        solo, _ = autonomous_setup(
            bodies=(brain_body(2, 2),),
            actuator=actuator_config(0.05),
            simulation_seed=101,
        )

        for _ in range(3):
            population_result = population.step()
            solo_result = solo.step()
            self.assertEqual(
                population_result.action_for(2),
                solo_result.action_for(2),
            )
            self.assertTrue(
                torch.equal(
                    population.brain_for(2).hidden_state,
                    solo.brain_for(2).hidden_state,
                )
            )

        self.assertEqual(population.brain_ids, (2,))

    def test_manual_simulation_step_remains_available_beside_runner(self) -> None:
        runner, _ = autonomous_setup()

        manual_result = runner.simulation.step(
            {1: ActionIntent(acceleration=10)}
        )

        self.assertEqual(manual_result.tick_after, 1)
        self.assertEqual(runner.brain_for(1).activation_count, 0)
        self.assertLessEqual(abs(runner.simulation.state.body(1).velocity), 0.25)


if __name__ == "__main__":
    unittest.main()
