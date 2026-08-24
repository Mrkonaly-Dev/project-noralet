"""Controlled lifetime adaptation diagnostics for homeostatic plasticity."""

from __future__ import annotations

import random
import unittest

import torch

from brain_test_support import (
    autonomous_setup,
    brain_config,
    homeostatic_config,
    sample_experience,
)


def binary_action_adaptation(*, selected_action_improves: bool) -> tuple[float, float]:
    """Run a continuous two-state bodily-consequence experiment."""

    _, base = autonomous_setup(
        brain=brain_config(exploration_std=0.2),
        homeostatic=homeostatic_config(
            energy_distress_weight=1,
            condition_distress_weight=0,
            homeostatic_modulation_scale=0.2,
            eligibility_decay=0.7,
            action_learning_rate=0.02,
            max_homeostatic_update_norm=1,
        ),
    )
    brain = base.spawn(action_random_source=random.Random(123))
    with torch.no_grad():
        brain.model.consume_head.weight.zero_()
        brain.model.consume_head.bias.zero_()
    evaluation_context = torch.zeros(brain.model.consume_head.in_features)
    initial_probability = float(
        torch.sigmoid(brain.model.consume_head(evaluation_context)).item()
    )
    distress = 0.8 if selected_action_improves else 0.2

    for _ in range(300):
        action = brain.act(
            sample_experience(
                energy_distress=distress,
                condition_distress=0,
            )
        )
        if selected_action_improves:
            next_distress = 0.2 if action.consume else 0.8
        else:
            next_distress = 0.8 if action.consume else 0.2
        brain.apply_homeostatic_update(
            sample_experience(
                energy_distress=next_distress,
                condition_distress=0,
            )
        )
        distress = next_distress

    final_probability = float(
        torch.sigmoid(brain.model.consume_head(evaluation_context)).item()
    )
    return initial_probability, final_probability


class LifetimeActionAdaptationDiagnosticTests(unittest.TestCase):
    def test_action_followed_by_lower_distress_becomes_more_probable(self) -> None:
        initial, final = binary_action_adaptation(selected_action_improves=True)

        self.assertEqual(initial, 0.5)
        self.assertGreater(final, initial + 0.02)

    def test_action_followed_by_higher_distress_becomes_less_probable(self) -> None:
        initial, final = binary_action_adaptation(selected_action_improves=False)

        self.assertEqual(initial, 0.5)
        self.assertLess(final, initial - 0.02)


if __name__ == "__main__":
    unittest.main()
