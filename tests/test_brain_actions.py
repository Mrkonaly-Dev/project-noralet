"""Stochastic low-level neural action-head and fixed-RNG tests."""

from __future__ import annotations

import math
from statistics import NormalDist
import unittest

import torch

from brain_test_support import autonomous_setup, brain_config
from noralet import (
    ACTION_RANDOM_DRAW_ORDER,
    BrainActionParameters,
    SignalDirection,
    SignalMotorChoice,
    SignalType,
)


class CountingRandom:
    def __init__(self, draws: tuple[float, ...]) -> None:
        self.draws = list(draws)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.draws.pop(0)


def neutral_parameters(
    *,
    acceleration_loc: float = 0.0,
    consume_logit: float = 0.0,
    signal_logits: tuple[float, ...] = (0.0,) * 9,
) -> BrainActionParameters:
    return BrainActionParameters(
        acceleration_loc=acceleration_loc,
        consume_logit=consume_logit,
        signal_logits=signal_logits,
    )


class ActionDistributionTests(unittest.TestCase):
    def test_action_parameter_probabilities_are_valid_and_stable(self) -> None:
        parameters = neutral_parameters(consume_logit=1000)

        self.assertEqual(parameters.consume_probability, 1.0)
        self.assertEqual(len(parameters.signal_probabilities), 9)
        self.assertAlmostEqual(math.fsum(parameters.signal_probabilities), 1.0)
        self.assertTrue(all(value > 0 for value in parameters.signal_probabilities))

    def test_acceleration_uses_location_exploration_and_tanh_bound(self) -> None:
        runner, _ = autonomous_setup(
            brain=brain_config(exploration_std=0.4),
        )
        brain = runner.brain_for(1)
        random_source = CountingRandom((0.75, 0.5, 0.0))

        action = brain.sample_action(
            neutral_parameters(acceleration_loc=0.3),
            random_source,
        )

        z = NormalDist().inv_cdf(0.75)
        expected = 0.25 * math.tanh(0.3 + 0.4 * z)
        self.assertAlmostEqual(action.acceleration, expected)
        self.assertLessEqual(abs(action.acceleration), 0.25)

    def test_zero_exploration_still_consumes_normal_draw(self) -> None:
        runner, _ = autonomous_setup(
            brain=brain_config(exploration_std=0),
        )
        brain = runner.brain_for(1)
        first_source = CountingRandom((0.01, 0.5, 0.5))
        second_source = CountingRandom((0.99, 0.5, 0.5))
        parameters = neutral_parameters(acceleration_loc=-0.7)

        first = brain.sample_action(parameters, first_source)
        second = brain.sample_action(parameters, second_source)

        self.assertEqual(first.acceleration, second.acceleration)
        self.assertEqual(first_source.calls, 3)
        self.assertEqual(second_source.calls, 3)

    def test_consume_logit_uses_seeded_bernoulli_choice(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)
        parameters = neutral_parameters(consume_logit=0)

        selected = brain.sample_action(
            parameters,
            CountingRandom((0.5, 0.499, 0.0)),
        )
        rejected = brain.sample_action(
            parameters,
            CountingRandom((0.5, 0.5, 0.0)),
        )

        self.assertTrue(selected.consume)
        self.assertFalse(rejected.consume)

    def test_signal_motor_space_is_exactly_the_required_nine_categories(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)
        self.assertEqual(
            brain.signal_motor_outcomes(),
            (
                SignalMotorChoice.NONE,
                SignalMotorChoice.A_LEFT,
                SignalMotorChoice.A_RIGHT,
                SignalMotorChoice.B_LEFT,
                SignalMotorChoice.B_RIGHT,
                SignalMotorChoice.C_LEFT,
                SignalMotorChoice.C_RIGHT,
                SignalMotorChoice.D_LEFT,
                SignalMotorChoice.D_RIGHT,
            ),
        )

    def test_each_signal_category_maps_to_one_valid_existing_action(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)
        expected = (
            None,
            (SignalType.A, SignalDirection.LEFT),
            (SignalType.A, SignalDirection.RIGHT),
            (SignalType.B, SignalDirection.LEFT),
            (SignalType.B, SignalDirection.RIGHT),
            (SignalType.C, SignalDirection.LEFT),
            (SignalType.C, SignalDirection.RIGHT),
            (SignalType.D, SignalDirection.LEFT),
            (SignalType.D, SignalDirection.RIGHT),
        )
        parameters = neutral_parameters()

        for index, expected_emission in enumerate(expected):
            with self.subTest(index=index):
                uniform = (index + 0.5) / 9.0
                action = brain.sample_action(
                    parameters,
                    CountingRandom((0.5, 0.5, uniform)),
                )
                if expected_emission is None:
                    self.assertIsNone(action.signal_emission)
                else:
                    assert action.signal_emission is not None
                    self.assertEqual(
                        (
                            action.signal_emission.signal_type,
                            action.signal_emission.direction,
                        ),
                        expected_emission,
                    )

    def test_finite_parameters_produce_bounded_acceleration(self) -> None:
        runner, _ = autonomous_setup(
            brain=brain_config(exploration_std=1e100),
        )
        brain = runner.brain_for(1)

        for location, normal_draw in ((-1e100, 0.0), (0.0, 0.5), (1e100, 0.999999)):
            with self.subTest(location=location, draw=normal_draw):
                action = brain.sample_action(
                    neutral_parameters(acceleration_loc=location),
                    CountingRandom((normal_draw, 0.5, 0.5)),
                )
                self.assertTrue(math.isfinite(action.acceleration))
                self.assertLessEqual(abs(action.acceleration), 0.25)

    def test_fixed_draw_order_is_three_unconditional_uniform_reads(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)
        source = CountingRandom((0.5, 0.0, 0.0))

        action = brain.sample_action(
            neutral_parameters(
                consume_logit=1000,
                signal_logits=(1000.0, *(-1000.0 for _ in range(8))),
            ),
            source,
        )

        self.assertEqual(
            ACTION_RANDOM_DRAW_ORDER,
            (
                "acceleration_standard_normal",
                "consume_uniform",
                "signal_category_uniform",
            ),
        )
        self.assertEqual(source.calls, 3)
        self.assertTrue(action.consume)
        self.assertIsNone(action.signal_emission)

    def test_action_sampling_does_not_use_or_advance_global_torch_rng(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)
        torch.manual_seed(77)
        before = torch.random.get_rng_state().clone()

        brain.sample_action(
            neutral_parameters(),
            CountingRandom((0.2, 0.4, 0.6)),
        )

        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
