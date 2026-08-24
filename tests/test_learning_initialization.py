"""Predictive-learning configuration, inheritance and frozen-target tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import math
import unittest

import torch

from brain_test_support import (
    autonomous_setup,
    brain_config,
    learning_config,
)
from noralet import NoraletLearningConfig


class LearningConfigurationTests(unittest.TestCase):
    def test_configuration_is_focused_immutable_and_canonical(self) -> None:
        config = NoraletLearningConfig(0.01, 2, 9)

        self.assertEqual(
            tuple(field.name for field in fields(config)),
            ("learning_rate", "max_gradient_norm", "predictor_hidden_size"),
        )
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.max_gradient_norm, 2.0)
        with self.assertRaises(FrozenInstanceError):
            config.learning_rate = 0.2  # type: ignore[misc]

    def test_real_controls_must_be_positive_and_finite(self) -> None:
        for name in ("learning_rate", "max_gradient_norm"):
            for value in (0, -1, math.inf, -math.inf, math.nan):
                values = {
                    "learning_rate": 0.01,
                    "max_gradient_norm": 1.0,
                    "predictor_hidden_size": 8,
                }
                values[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        NoraletLearningConfig(**values)
            values = {
                "learning_rate": 0.01,
                "max_gradient_norm": 1.0,
                "predictor_hidden_size": 8,
            }
            values[name] = True
            with self.subTest(name=name, value=True):
                with self.assertRaises(TypeError):
                    NoraletLearningConfig(**values)

    def test_predictor_hidden_size_must_be_positive_integer(self) -> None:
        for value, error in ((0, ValueError), (-1, ValueError), (1.5, TypeError)):
            with self.subTest(value=value):
                with self.assertRaises(error):
                    NoraletLearningConfig(0.01, 1.0, value)  # type: ignore[arg-type]


class LearningInheritanceTests(unittest.TestCase):
    def test_online_and_target_encoders_begin_equal_but_independent(self) -> None:
        runner, _ = autonomous_setup(learning=learning_config())
        brain = runner.brain_for(1)
        target = brain.target_experience_encoder
        self.assertIsNotNone(target)
        assert target is not None

        online_parameters = tuple(brain.model.encoder.parameters())
        target_parameters = tuple(target.parameters())
        self.assertEqual(len(online_parameters), len(target_parameters))
        for online, frozen in zip(online_parameters, target_parameters, strict=True):
            self.assertTrue(torch.equal(online, frozen))
            self.assertNotEqual(online.data_ptr(), frozen.data_ptr())

    def test_target_encoder_is_permanently_frozen_and_gradient_free(self) -> None:
        runner, _ = autonomous_setup(learning=learning_config())
        brain = runner.brain_for(1)
        before = brain.target_parameter_snapshot()

        for _ in range(12):
            runner.step()

        after = brain.target_parameter_snapshot()
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(before, after))
        )
        target = brain.target_experience_encoder
        assert target is not None
        self.assertTrue(
            all(
                not parameter.requires_grad and parameter.grad is None
                for parameter in target.parameters()
            )
        )

    def test_predictor_initialization_is_seed_deterministic(self) -> None:
        first, _ = autonomous_setup(
            brain=brain_config(seed=901),
            learning=learning_config(),
        )
        second, _ = autonomous_setup(
            brain=brain_config(seed=901),
            learning=learning_config(),
        )
        different, _ = autonomous_setup(
            brain=brain_config(seed=902),
            learning=learning_config(),
        )
        first_predictor = first.brain_for(1).model.prediction_model
        second_predictor = second.brain_for(1).model.prediction_model
        different_predictor = different.brain_for(1).model.prediction_model
        assert first_predictor is not None
        assert second_predictor is not None
        assert different_predictor is not None

        first_values = tuple(first_predictor.parameters())
        second_values = tuple(second_predictor.parameters())
        different_values = tuple(different_predictor.parameters())
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(first_values, second_values))
        )
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(first_values, different_values))
        )

    def test_predictor_does_not_shift_iteration_8_initialization(self) -> None:
        disabled, _ = autonomous_setup(brain=brain_config(seed=903))
        enabled, _ = autonomous_setup(
            brain=brain_config(seed=903),
            learning=learning_config(),
        )
        disabled_model = disabled.brain_for(1).model
        enabled_model = enabled.brain_for(1).model

        disabled_values = disabled_model.iteration_8_parameters()
        enabled_values = enabled_model.iteration_8_parameters()
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(disabled_values, enabled_values))
        )

    def test_no_learning_mode_creates_no_training_machinery(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)

        self.assertFalse(brain.learning_enabled)
        self.assertIsNone(brain.model.prediction_model)
        self.assertIsNone(brain.target_experience_encoder)
        self.assertIsNone(brain.optimizer)
        self.assertEqual(brain.learning_update_count, 0)

    def test_each_brain_owns_an_independent_optimizer_and_empty_state(self) -> None:
        runner, _ = autonomous_setup(learning=learning_config())
        first = runner.brain_for(1)
        second = runner.brain_for(2)

        self.assertIsNot(first.optimizer, second.optimizer)
        assert first.optimizer is not None
        assert second.optimizer is not None
        self.assertEqual(len(first.optimizer.state), 0)
        self.assertEqual(len(second.optimizer.state), 0)

    def test_optimizer_membership_is_exactly_the_plastic_path(self) -> None:
        runner, _ = autonomous_setup(learning=learning_config())
        brain = runner.brain_for(1)
        optimizer = brain.optimizer
        target = brain.target_experience_encoder
        assert optimizer is not None
        assert target is not None
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        plastic = {
            id(parameter)
            for parameter in brain.model.predictive_plastic_parameters()
        }
        action_heads = {
            id(parameter) for parameter in brain.model.action_head_parameters()
        }
        frozen_target = {id(parameter) for parameter in target.parameters()}

        self.assertEqual(optimized, plastic)
        self.assertTrue(optimized.isdisjoint(action_heads))
        self.assertTrue(optimized.isdisjoint(frozen_target))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in brain.model.action_head_parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
