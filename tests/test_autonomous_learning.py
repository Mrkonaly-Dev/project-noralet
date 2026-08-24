"""Lockstep autonomous lifetime-learning integration and death semantics."""

from __future__ import annotations

from dataclasses import asdict, fields
from types import MethodType
import unittest

import torch

from brain_test_support import (
    actuator_config,
    autonomous_setup,
    brain_body,
    brain_config,
    learning_config,
)
from noralet import NoraletLearningResult
from physiology_test_support import physiology_config


class AutonomousLearningTests(unittest.TestCase):
    def test_each_surviving_transition_produces_one_learning_result(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )

        result = runner.step()

        self.assertEqual(
            tuple(item.noralet_id for item in result.learning_results),
            (1, 2),
        )
        self.assertEqual(runner.brain_for(1).learning_update_count, 1)
        self.assertEqual(runner.brain_for(2).learning_update_count, 1)
        self.assertIsInstance(result.learning_for(1), NoraletLearningResult)
        self.assertFalse(runner.brain_for(1).has_pending_transition)

    def test_all_actions_precede_world_step_and_all_learning(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )
        order: list[str] = []
        for identity in runner.brain_ids:
            brain = runner.brain_for(identity)
            original_act = brain.act
            original_learn = brain.learn

            def act(
                self,
                experience,
                *,
                _id=identity,
                _call=original_act,
                _case=self,
            ):
                self_tick = runner.simulation.state.tick
                _case.assertEqual(self_tick, 0)
                order.append(f"act:{_id}")
                return _call(experience)

            def learn(
                self,
                experience,
                *,
                _id=identity,
                _call=original_learn,
                _case=self,
            ):
                self_tick = runner.simulation.state.tick
                _case.assertEqual(self_tick, 1)
                order.append(f"learn:{_id}")
                return _call(experience)

            brain.act = MethodType(act, brain)  # type: ignore[method-assign]
            brain.learn = MethodType(learn, brain)  # type: ignore[method-assign]

        runner.step()

        self.assertEqual(
            order,
            ["act:1", "act:2", "learn:1", "learn:2"],
        )

    def test_learning_cannot_retroactively_change_resolved_action_or_tick(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(learning_rate=0.1),
            actuator=actuator_config(0.001),
        )

        result = runner.step()
        recorded_actions = result.action_intents
        recorded_tick = result.tick_result
        for _ in range(4):
            runner.step()

        self.assertEqual(result.action_intents, recorded_actions)
        self.assertEqual(result.tick_result, recorded_tick)
        self.assertEqual(recorded_tick.tick_after, 1)

    def test_multi_tick_lifetime_changes_only_plastic_neural_state(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )
        brain = runner.brain_for(1)
        plastic_before = brain.plastic_parameter_snapshot()
        target_before = brain.target_parameter_snapshot()
        heads_before = brain.action_head_parameter_snapshot()
        hidden_before = brain.hidden_state
        initial_energy = runner.simulation.state.energy_totals.total_energy

        results = tuple(runner.step() for _ in range(20))

        self.assertEqual(runner.simulation.state.tick, 20)
        self.assertEqual(brain.activation_count, 20)
        self.assertEqual(brain.learning_update_count, 20)
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(
                    plastic_before,
                    brain.plastic_parameter_snapshot(),
                )
            )
        )
        self.assertFalse(torch.equal(hidden_before, brain.hidden_state))
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    target_before,
                    brain.target_parameter_snapshot(),
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    heads_before,
                    brain.action_head_parameter_snapshot(),
                )
            )
        )
        self.assertTrue(
            all(len(result.learning_results) == 2 for result in results)
        )
        self.assertAlmostEqual(
            runner.simulation.state.energy_totals.total_energy,
            initial_energy,
            delta=runner.simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )

    def test_reading_learning_metrics_is_non_causal(self) -> None:
        observed, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
            simulation_seed=444,
        )
        control, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
            simulation_seed=444,
        )

        for _ in range(5):
            observed_result = observed.step()
            tuple(asdict(item) for item in observed_result.learning_results)
            control_result = control.step()
            self.assertEqual(observed_result, control_result)

        self.assertEqual(observed.simulation.state, control.simulation.state)
        for identity in observed.brain_ids:
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(
                        observed.brain_for(identity).parameter_snapshot(),
                        control.brain_for(identity).parameter_snapshot(),
                    )
                )
            )

    def test_body_insertion_order_cannot_change_per_id_learning_history(self) -> None:
        forward, _ = autonomous_setup(
            bodies=(brain_body(1, -2), brain_body(2, 2)),
            learning=learning_config(),
            actuator=actuator_config(0.001),
            simulation_seed=445,
        )
        reverse, _ = autonomous_setup(
            bodies=(brain_body(2, 2), brain_body(1, -2)),
            learning=learning_config(),
            actuator=actuator_config(0.001),
            simulation_seed=445,
        )

        forward_history = tuple(forward.step() for _ in range(6))
        reverse_history = tuple(reverse.step() for _ in range(6))

        self.assertEqual(forward_history, reverse_history)
        self.assertEqual(forward.simulation.state, reverse.simulation.state)
        for identity in forward.brain_ids:
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(
                        forward.brain_for(identity).parameter_snapshot(),
                        reverse.brain_for(identity).parameter_snapshot(),
                    )
                )
            )


class TerminalLearningTests(unittest.TestCase):
    def _assert_terminal_transition_does_not_learn(self, runner) -> None:
        doomed = runner.brain_for(1)

        result = runner.step()

        self.assertNotIn(1, runner.brain_ids)
        self.assertEqual(doomed.learning_update_count, 0)
        self.assertFalse(doomed.has_pending_transition)
        self.assertNotIn(
            1,
            tuple(item.noralet_id for item in result.learning_results),
        )
        with self.assertRaises(KeyError):
            runner.simulation.experience_for(1)

    def test_boundary_death_discards_pending_prediction(self) -> None:
        runner, _ = autonomous_setup(
            bodies=(brain_body(1, 9.9, velocity=1.0), brain_body(2, 0)),
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )
        self._assert_terminal_transition_does_not_learn(runner)

    def test_energy_depletion_death_discards_pending_prediction(self) -> None:
        runner, _ = autonomous_setup(
            bodies=(brain_body(1, 0, energy=0.0), brain_body(2, 2)),
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )
        self._assert_terminal_transition_does_not_learn(runner)

    def test_natural_death_discards_pending_prediction(self) -> None:
        runner, _ = autonomous_setup(
            bodies=(brain_body(1, 0), brain_body(2, 2)),
            learning=learning_config(),
            actuator=actuator_config(0.001),
            physiology=physiology_config(baseline_loss=0, base_hazard=100),
        )
        doomed = runner.brain_for(1)

        result = runner.step()

        self.assertEqual(runner.brain_ids, ())
        self.assertEqual(doomed.learning_update_count, 0)
        self.assertFalse(doomed.has_pending_transition)
        self.assertEqual(result.learning_results, ())

    def test_pending_context_has_no_death_or_world_target_fields(self) -> None:
        import noralet.brain.runtime as runtime

        self.assertEqual(
            tuple(field.name for field in fields(runtime._PendingTransition)),
            ("prediction", "action_vector"),
        )


if __name__ == "__main__":
    unittest.main()
