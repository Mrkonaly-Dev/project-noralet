"""Predictive-plasticity reproducibility, RNG isolation and CUDA smokes."""

from __future__ import annotations

import random
import unittest

import torch

from brain_test_support import (
    actuator_config,
    autonomous_setup,
    brain_config,
    learning_config,
)


class LearningDeterminismTests(unittest.TestCase):
    def test_identical_cpu_lifetimes_reproduce_complete_learning_history(self) -> None:
        first, _ = autonomous_setup(
            brain=brain_config(device="cpu"),
            learning=learning_config(),
            actuator=actuator_config(0.001),
            simulation_seed=991,
        )
        second, _ = autonomous_setup(
            brain=brain_config(device="cpu"),
            learning=learning_config(),
            actuator=actuator_config(0.001),
            simulation_seed=991,
        )

        first_history = tuple(first.step() for _ in range(10))
        second_history = tuple(second.step() for _ in range(10))

        self.assertEqual(first_history, second_history)
        self.assertEqual(first.simulation.state, second.simulation.state)
        for identity in first.brain_ids:
            first_brain = first.brain_for(identity)
            second_brain = second.brain_for(identity)
            self.assertTrue(
                torch.equal(first_brain.hidden_state, second_brain.hidden_state)
            )
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(
                        first_brain.parameter_snapshot(),
                        second_brain.parameter_snapshot(),
                    )
                )
            )

    def test_learning_step_does_not_advance_global_rngs(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()

        runner.step()

        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_learning_adds_no_simulation_or_action_stream_draw(self) -> None:
        learning_runner, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
            simulation_seed=992,
        )
        control_runner, _ = autonomous_setup(
            actuator=actuator_config(0.001),
            simulation_seed=992,
        )

        learning_result = learning_runner.step()
        control_result = control_runner.step()

        self.assertEqual(learning_result.action_intents, control_result.action_intents)
        self.assertEqual(learning_result.tick_result, control_result.tick_result)
        learning_streams = learning_runner.simulation.random_streams._streams
        control_streams = control_runner.simulation.random_streams._streams
        self.assertEqual(set(learning_streams), set(control_streams))
        self.assertTrue(
            all(
                learning_streams[name].getstate()
                == control_streams[name].getstate()
                for name in learning_streams
            )
        )

    def test_no_learning_path_preserves_known_iteration_8_history(self) -> None:
        first, _ = autonomous_setup(
            brain=brain_config(seed=808),
            actuator=actuator_config(0.01),
            simulation_seed=1234,
        )
        second, _ = autonomous_setup(
            brain=brain_config(seed=808),
            actuator=actuator_config(0.01),
            simulation_seed=1234,
        )

        first_history = tuple(first.step() for _ in range(8))
        second_history = tuple(second.step() for _ in range(8))

        self.assertEqual(first_history, second_history)
        self.assertTrue(all(result.learning_results == () for result in first_history))
        self.assertTrue(
            all(
                first.brain_for(identity).learning_update_count == 0
                for identity in first.brain_ids
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaLifetimeLearningTests(unittest.TestCase):
    def test_cuda_learning_step_runs_real_backward_and_adam_update(self) -> None:
        runner, base = autonomous_setup(
            brain=brain_config(device="cuda"),
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )
        brain = runner.brain_for(1)
        plastic_before = brain.plastic_parameter_snapshot()
        target_before = brain.target_parameter_snapshot()
        heads_before = brain.action_head_parameter_snapshot()

        result = runner.step().learning_for(1)

        prototype_weight = base.prototype_model.encoder.fusion[0].weight
        self.assertEqual(prototype_weight.device.type, "cpu")
        self.assertEqual(brain.hidden_state.device.type, "cuda")
        self.assertTrue(torch.isfinite(torch.tensor(result.prediction_loss)))
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(
                    plastic_before,
                    brain.plastic_parameter_snapshot(),
                )
            )
        )
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
        target = brain.target_experience_encoder
        predictor = brain.model.prediction_model
        assert target is not None
        assert predictor is not None
        self.assertEqual(next(target.parameters()).device.type, "cuda")
        self.assertEqual(next(predictor.parameters()).device.type, "cuda")
        optimizer = brain.optimizer
        assert optimizer is not None
        moment_tensors = tuple(
            value
            for state in optimizer.state.values()
            for name, value in state.items()
            if name in ("exp_avg", "exp_avg_sq")
        )
        self.assertTrue(moment_tensors)
        self.assertTrue(all(value.device.type == "cuda" for value in moment_tensors))

    def test_cuda_multi_tick_learning_smoke_is_finite_and_conservative(self) -> None:
        runner, _ = autonomous_setup(
            brain=brain_config(device="cuda"),
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )
        initial_energy = runner.simulation.state.energy_totals.total_energy

        results = tuple(runner.step() for _ in range(6))

        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(item.prediction_loss))
                for result in results
                for item in result.learning_results
            )
        )
        for identity in runner.brain_ids:
            brain = runner.brain_for(identity)
            self.assertTrue(
                all(
                    torch.isfinite(parameter).all()
                    for parameter in brain.model.parameters()
                )
            )
        self.assertAlmostEqual(
            runner.simulation.state.energy_totals.total_energy,
            initial_energy,
            delta=runner.simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
