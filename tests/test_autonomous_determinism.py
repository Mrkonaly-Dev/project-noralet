"""Autonomous history determinism, RNG isolation and no-learning tests."""

from __future__ import annotations

import unittest

import torch

from brain_test_support import actuator_config, autonomous_setup, brain_body


def hidden_snapshot(runner) -> tuple[tuple[int, tuple[float, ...]], ...]:
    return tuple(
        (
            identity,
            tuple(runner.brain_for(identity).hidden_state.cpu().tolist()),
        )
        for identity in runner.brain_ids
    )


class AutonomousDeterminismTests(unittest.TestCase):
    def test_same_cpu_config_reproduces_autonomous_history(self) -> None:
        actuator = actuator_config(0.02)
        first, _ = autonomous_setup(actuator=actuator, simulation_seed=5150)
        second, _ = autonomous_setup(actuator=actuator, simulation_seed=5150)
        first_history = []
        second_history = []

        for _ in range(15):
            first_result = first.step()
            second_result = second.step()
            first_history.append(
                (first_result, first.simulation.state, hidden_snapshot(first))
            )
            second_history.append(
                (second_result, second.simulation.state, hidden_snapshot(second))
            )

        self.assertEqual(first_history, second_history)

    def test_initial_body_insertion_order_cannot_change_per_id_history(self) -> None:
        bodies = (
            brain_body(3, 4),
            brain_body(1, -4),
            brain_body(2, 0),
        )
        first, _ = autonomous_setup(
            bodies=bodies,
            actuator=actuator_config(0.01),
            simulation_seed=44,
        )
        second, _ = autonomous_setup(
            bodies=tuple(reversed(bodies)),
            actuator=actuator_config(0.01),
            simulation_seed=44,
        )

        for _ in range(10):
            self.assertEqual(first.step(), second.step())
            self.assertEqual(first.simulation.state, second.simulation.state)
            self.assertEqual(hidden_snapshot(first), hidden_snapshot(second))

    def test_extra_action_draws_for_one_noralet_do_not_shift_another(self) -> None:
        shifted, _ = autonomous_setup(simulation_seed=202)
        control, _ = autonomous_setup(simulation_seed=202)
        shifted_a_stream = shifted.simulation.random_streams.stream(
            shifted.action_stream_name(1)
        )
        for _ in range(30):
            shifted_a_stream.random()

        shifted_result = shifted.step()
        control_result = control.step()

        self.assertEqual(shifted_result.action_for(2), control_result.action_for(2))
        self.assertTrue(
            torch.equal(
                shifted.brain_for(2).hidden_state,
                control.brain_for(2).hidden_state,
            )
        )

    def test_action_sampling_does_not_shift_ecology_or_mortality_streams(self) -> None:
        first, _ = autonomous_setup(simulation_seed=303)
        second, _ = autonomous_setup(simulation_seed=303)
        action_stream = first.simulation.random_streams.stream(
            first.action_stream_name(1)
        )
        for _ in range(40):
            action_stream.random()

        first.step()
        second.step()
        stream_names = (
            "energy:region:3:all:formation:trigger",
            "mortality:noralet:1:1",
            "mortality:noralet:1:2",
        )

        for name in stream_names:
            with self.subTest(name=name):
                self.assertEqual(
                    first.simulation.random_streams.stream(name).getstate(),
                    second.simulation.random_streams.stream(name).getstate(),
                )

    def test_many_autonomous_ticks_keep_weights_fixed_and_gradients_empty(self) -> None:
        runner, _ = autonomous_setup(
            actuator=actuator_config(0.005),
            simulation_seed=818,
        )
        initial = {
            identity: runner.brain_for(identity).parameter_snapshot()
            for identity in runner.brain_ids
        }

        for _ in range(30):
            runner.step()

        for identity in runner.brain_ids:
            brain = runner.brain_for(identity)
            self.assertTrue(
                all(
                    torch.equal(before, after)
                    for before, after in zip(
                        initial[identity],
                        brain.parameter_snapshot(),
                    )
                )
            )
            self.assertTrue(
                all(parameter.grad is None for parameter in brain.model.parameters())
            )
        self.assertFalse(hasattr(runner, "optimizer"))

    def test_finite_autonomous_smoke_preserves_energy_and_valid_actions(self) -> None:
        runner, _ = autonomous_setup(
            actuator=actuator_config(0.01),
            signal_energy_cost=0.1,
            existence_cost=0.01,
            acceleration_cost=0.05,
            simulation_seed=919,
        )
        initial_total = runner.simulation.initial_total_energy
        consume_was_requested = False
        signal_was_requested = False

        for _ in range(20):
            result = runner.step()
            for _, action in result.action_intents:
                self.assertLessEqual(abs(action.acceleration), 0.01)
                self.assertIsInstance(action.consume, bool)
                consume_was_requested |= action.consume
                signal_was_requested |= action.signal_emission is not None
            runner.simulation.audit_energy_conservation()

        self.assertTrue(consume_was_requested)
        self.assertTrue(signal_was_requested)
        self.assertAlmostEqual(
            runner.simulation.state.energy_totals.total_energy,
            initial_total,
            delta=runner.simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaAutonomousSmokeTests(unittest.TestCase):
    def test_cuda_brains_encode_act_and_run_without_device_mismatch(self) -> None:
        from brain_test_support import brain_config

        runner, base = autonomous_setup(
            brain=brain_config(device="cuda"),
            actuator=actuator_config(0.01),
        )

        result = runner.step()

        self.assertEqual(
            base.prototype_model.acceleration_head.weight.device.type,
            "cpu",
        )
        self.assertEqual(runner.brain_for(1).hidden_state.device.type, "cuda")
        self.assertTrue(torch.isfinite(runner.brain_for(1).hidden_state).all())
        self.assertEqual(result.tick_result.tick_after, 1)


if __name__ == "__main__":
    unittest.main()
