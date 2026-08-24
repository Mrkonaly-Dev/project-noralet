"""Brain-native intention vectors and compact prediction-model tests."""

from __future__ import annotations

import inspect
import math
import unittest

import torch

from brain_test_support import (
    actuator_config,
    autonomous_setup,
    brain_body,
    brain_config,
    learning_config,
    sample_experience,
)
from noralet import ACTION_VECTOR_SIZE, BrainActionParameters, PredictionModel


class FixedDraws:
    def __init__(self, *draws: float) -> None:
        self._draws = iter(draws)

    def random(self) -> float:
        return next(self._draws)


class ActionRepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        runner, _ = autonomous_setup(
            brain=brain_config(exploration_std=0.0),
            learning=learning_config(),
            actuator=actuator_config(0.4),
        )
        self.brain = runner.brain_for(1)

    def test_acceleration_is_normalized_before_physical_scaling(self) -> None:
        parameters = BrainActionParameters(
            acceleration_loc=math.atanh(0.5),
            consume_logit=0.0,
            signal_logits=(0.0,) * 9,
        )

        selection = self.brain.sample_brain_action(
            parameters,
            FixedDraws(0.5, 0.9, 0.01),
        )

        self.assertAlmostEqual(selection.normalized_acceleration_command, 0.5)
        self.assertAlmostEqual(selection.action_intent.acceleration, 0.2)
        self.assertNotEqual(
            selection.normalized_acceleration_command,
            selection.action_intent.acceleration,
        )

    def test_consume_intentions_have_distinct_binary_values(self) -> None:
        parameters = BrainActionParameters(0.0, 0.0, (0.0,) * 9)

        consume = self.brain.sample_brain_action(
            parameters,
            FixedDraws(0.5, 0.0, 0.0),
        )
        abstain = self.brain.sample_brain_action(
            parameters,
            FixedDraws(0.5, 0.999, 0.0),
        )

        self.assertEqual(consume.consume_command, 1.0)
        self.assertEqual(abstain.consume_command, 0.0)
        self.assertTrue(consume.action_intent.consume)
        self.assertFalse(abstain.action_intent.consume)

    def test_signal_motor_representation_is_exact_nine_way_one_hot(self) -> None:
        parameters = BrainActionParameters(0.0, 0.0, (0.0,) * 9)

        for expected_index in range(9):
            with self.subTest(expected_index=expected_index):
                selection = self.brain.sample_brain_action(
                    parameters,
                    FixedDraws(
                        0.5,
                        0.5,
                        (expected_index + 0.5) / 9.0,
                    ),
                )
                signal_values = selection.action_vector[2:]
                self.assertEqual(len(signal_values), 9)
                self.assertEqual(signal_values.count(1.0), 1)
                self.assertEqual(selection.signal_motor_index, expected_index)

    def test_every_selected_action_vector_has_fixed_eleven_values(self) -> None:
        parameters = BrainActionParameters(0.3, -0.4, tuple(range(9)))
        selection = self.brain.sample_brain_action(
            parameters,
            FixedDraws(0.75, 0.25, 0.75),
        )

        self.assertEqual(ACTION_VECTOR_SIZE, 11)
        self.assertEqual(len(selection.action_vector), ACTION_VECTOR_SIZE)

    def test_intention_survives_unaffordable_signal_execution(self) -> None:
        runner, _ = autonomous_setup(
            bodies=(brain_body(1, 0.0, energy=1.0),),
            brain=brain_config(exploration_std=0.0),
            learning=learning_config(),
            signal_energy_cost=2.0,
            actuator=actuator_config(0.01),
        )
        brain = runner.brain_for(1)
        with torch.no_grad():
            brain.model.signal_head.weight.zero_()
            brain.model.signal_head.bias.fill_(-100.0)
            brain.model.signal_head.bias[1] = 100.0

        action = brain.act(runner.simulation.experience_for(1))
        intended = brain.pending_action_vector
        self.assertIsNotNone(intended)
        assert intended is not None
        self.assertEqual(intended[3], 1.0)
        self.assertIsNotNone(action.signal_emission)

        runner.simulation.step({1: action})
        actual_experience = runner.simulation.experience_for(1)

        self.assertEqual(
            actual_experience.sensorimotor_feedback.signal_emission_activation,
            0.0,
        )
        brain.learn(actual_experience)


class PredictionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        runner, _ = autonomous_setup(learning=learning_config())
        self.brain = runner.brain_for(1)

    def test_output_shape_matches_target_encoder_embedding(self) -> None:
        predictor = self.brain.model.prediction_model
        target = self.brain.target_experience_encoder
        assert predictor is not None
        assert target is not None
        hidden = torch.zeros(7)
        action = torch.tensor((0.0, 0.0, 1.0, *([0.0] * 8)))

        prediction = predictor(hidden, action)
        target_embedding = target(sample_experience())

        self.assertEqual(tuple(prediction.shape), tuple(target_embedding.shape))
        self.assertEqual(tuple(prediction.shape), (6,))

    def test_valid_hidden_and_action_produce_finite_prediction(self) -> None:
        predictor = self.brain.model.prediction_model
        assert predictor is not None
        prediction = predictor(
            torch.linspace(-0.5, 0.5, 7),
            torch.tensor((0.2, 1.0, 0.0, 1.0, *([0.0] * 7))),
        )

        self.assertTrue(torch.isfinite(prediction).all())

    def test_predictor_api_accepts_only_hidden_and_action_tensors(self) -> None:
        signature = inspect.signature(PredictionModel.forward)

        self.assertEqual(
            tuple(signature.parameters),
            ("self", "hidden_state", "action_vector"),
        )
        self.assertNotIn("WorldState", inspect.getsource(PredictionModel))

    def test_predictor_has_one_hidden_tanh_and_unrestricted_linear_output(self) -> None:
        predictor = self.brain.model.prediction_model
        assert predictor is not None

        self.assertIsInstance(predictor.input_layer, torch.nn.Linear)
        self.assertIsInstance(predictor.output_layer, torch.nn.Linear)
        self.assertEqual(
            sum(isinstance(module, torch.nn.Linear) for module in predictor.modules()),
            2,
        )


if __name__ == "__main__":
    unittest.main()
