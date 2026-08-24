"""Brain configuration, device and deterministic BaseBrain tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

import torch

from brain_test_support import autonomous_setup, brain_config
from noralet import BaseBrain, NoraletBrainConfig, resolve_brain_device


class BrainConfigurationTests(unittest.TestCase):
    def test_configuration_is_canonical_immutable_and_focused(self) -> None:
        config = brain_config(exploration_std=0, device=" CPU ")

        self.assertEqual(config.acceleration_exploration_std, 0.0)
        self.assertEqual(config.device, "cpu")
        with self.assertRaises(FrozenInstanceError):
            config.hidden_size = 10  # type: ignore[misc]
        for forbidden in (
            "learning_rate",
            "optimizer",
            "reward_coefficient",
            "discount_factor",
            "prediction_loss",
            "training_buffer",
        ):
            self.assertFalse(hasattr(config, forbidden))

    def test_every_neural_dimension_must_be_a_positive_integer(self) -> None:
        config = brain_config()
        names = (
            "external_percept_embedding_size",
            "signal_percept_embedding_size",
            "interoception_embedding_size",
            "sensorimotor_embedding_size",
            "experience_embedding_size",
            "hidden_size",
        )
        for name in names:
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(config, **{name: 0})
            with self.subTest(name=name), self.assertRaises(TypeError):
                replace(config, **{name: True})

    def test_seed_exploration_and_device_are_strictly_validated(self) -> None:
        config = brain_config()
        with self.assertRaises(TypeError):
            replace(config, base_brain_seed=1.5)
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(config, acceleration_exploration_std=value)
        with self.assertRaises(TypeError):
            replace(config, acceleration_exploration_std=True)
        with self.assertRaises(ValueError):
            replace(config, device="metal")

    def test_cpu_device_resolves_explicitly(self) -> None:
        self.assertEqual(resolve_brain_device("cpu"), torch.device("cpu"))
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(resolve_brain_device("auto").type, expected)

    def test_explicit_cuda_never_silently_falls_back(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(resolve_brain_device("cuda").type, "cuda")
        else:
            with self.assertRaisesRegex(RuntimeError, "explicitly requested"):
                resolve_brain_device("cuda")

    def test_public_torch_and_brain_api_load(self) -> None:
        self.assertTrue(torch.__version__)
        self.assertIsInstance(brain_config(), NoraletBrainConfig)
        runner, base = autonomous_setup()
        self.assertIsInstance(base, BaseBrain)
        self.assertEqual(runner.brain_ids, (1, 2))


class BaseBrainTests(unittest.TestCase):
    def test_same_seed_and_architecture_produce_exactly_equal_parameters(self) -> None:
        _, first = autonomous_setup(brain=brain_config(seed=81))
        _, second = autonomous_setup(brain=brain_config(seed=81))

        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    first.parameter_snapshot(),
                    second.parameter_snapshot(),
                )
            )
        )

    def test_different_seed_changes_prototype_parameters(self) -> None:
        _, first = autonomous_setup(brain=brain_config(seed=81))
        _, second = autonomous_setup(brain=brain_config(seed=82))

        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(
                    first.parameter_snapshot(),
                    second.parameter_snapshot(),
                )
            )
        )

    def test_base_initialization_does_not_consume_global_torch_rng(self) -> None:
        torch.manual_seed(987654)
        state_before = torch.random.get_rng_state().clone()

        autonomous_setup(brain=brain_config(seed=123))

        self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))

    def test_spawned_brains_have_equal_independent_parameters(self) -> None:
        runner, base = autonomous_setup()
        first = runner.brain_for(1)
        second = runner.brain_for(2)
        base_before = base.parameter_snapshot()
        second_before = second.parameter_snapshot()

        with torch.no_grad():
            next(first.model.parameters()).add_(1.0)

        self.assertFalse(
            torch.equal(first.parameter_snapshot()[0], second_before[0])
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(
                    second_before,
                    second.parameter_snapshot(),
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(base_before, base.parameter_snapshot())
            )
        )
        self.assertNotEqual(
            next(first.model.parameters()).data_ptr(),
            next(second.model.parameters()).data_ptr(),
        )

    def test_each_spawn_starts_with_zero_independent_hidden_state(self) -> None:
        runner, _ = autonomous_setup(brain=brain_config(hidden_size=5))
        first = runner.brain_for(1)
        second = runner.brain_for(2)

        self.assertEqual(tuple(first.hidden_state.shape), (5,))
        self.assertEqual(first.hidden_state.device.type, "cpu")
        self.assertTrue(torch.equal(first.hidden_state, torch.zeros(5)))
        self.assertTrue(torch.equal(second.hidden_state, torch.zeros(5)))
        self.assertNotEqual(
            first.hidden_state.data_ptr(),
            second.hidden_state.data_ptr(),
        )

    def test_model_has_one_gru_and_no_lstm_or_transformer(self) -> None:
        _, base = autonomous_setup()
        modules = tuple(base.prototype_model.modules())

        self.assertEqual(
            sum(isinstance(module, torch.nn.GRUCell) for module in modules),
            1,
        )
        self.assertFalse(any(isinstance(module, torch.nn.LSTM) for module in modules))
        self.assertFalse(
            any(isinstance(module, torch.nn.Transformer) for module in modules)
        )
        self.assertTrue(torch.are_deterministic_algorithms_enabled())


if __name__ == "__main__":
    unittest.main()
