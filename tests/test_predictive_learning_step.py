"""One-transition predictive optimizer, truncation and capability tests."""

from __future__ import annotations

import inspect
import unittest

import torch

from brain_test_support import (
    autonomous_setup,
    brain_config,
    external_percept,
    learning_config,
    sample_experience,
    signal_percept,
)


class ConstantRandom:
    def __init__(self, value: float = 0.5) -> None:
        self.value = value
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return self.value


def snapshots_differ(
    before: tuple[torch.Tensor, ...],
    after: tuple[torch.Tensor, ...],
) -> bool:
    return any(not torch.equal(left, right) for left, right in zip(before, after))


def module_snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in module.parameters())


def controlled_brain(*, rate: float = 0.01, max_norm: float = 1.0):
    _, base = autonomous_setup(
        brain=brain_config(exploration_std=0.0),
        learning=learning_config(
            learning_rate=rate,
            max_gradient_norm=max_norm,
        ),
    )
    source = ConstantRandom()
    brain = base.spawn(action_random_source=source)
    with torch.no_grad():
        for head in (
            brain.model.acceleration_head,
            brain.model.consume_head,
            brain.model.signal_head,
        ):
            head.weight.zero_()
            head.bias.zero_()
    return brain, base, source


class PredictiveUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = sample_experience(
            external_percepts=(external_percept(proximity=0.7),),
            signal_percepts=(signal_percept(strength=0.4),),
            energy_distress=0.5,
            motor_direction=-1.0,
        )
        self.next = sample_experience(
            external_percepts=(external_percept(direction=-1.0, proximity=0.2),),
            energy_distress=0.15,
            condition_distress=0.25,
            energetic_exertion=0.45,
            motor_direction=1.0,
        )

    def test_one_transition_updates_each_plastic_component_once(self) -> None:
        brain, _, _ = controlled_brain()
        encoder_before = module_snapshot(brain.model.encoder)
        gru_before = module_snapshot(brain.model.recurrent_core)
        predictor = brain.model.prediction_model
        assert predictor is not None
        predictor_before = module_snapshot(predictor)

        brain.act(self.current)
        result = brain.learn(self.next)

        self.assertEqual(brain.learning_update_count, 1)
        self.assertGreaterEqual(result.prediction_loss, 0.0)
        self.assertTrue(
            snapshots_differ(
                encoder_before,
                module_snapshot(brain.model.encoder),
            )
        )
        self.assertTrue(
            snapshots_differ(
                gru_before,
                module_snapshot(brain.model.recurrent_core),
            )
        )
        self.assertTrue(snapshots_differ(predictor_before, module_snapshot(predictor)))

    def test_action_heads_and_base_prototype_remain_exactly_fixed(self) -> None:
        brain, base, _ = controlled_brain()
        heads_before = brain.action_head_parameter_snapshot()
        base_before = base.parameter_snapshot()

        for _ in range(8):
            brain.act(self.current)
            brain.learn(self.next)

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
            all(
                torch.equal(left, right)
                for left, right in zip(base_before, base.parameter_snapshot())
            )
        )

    def test_training_one_brain_cannot_modify_another(self) -> None:
        _, base = autonomous_setup(learning=learning_config())
        first = base.spawn(action_random_source=ConstantRandom())
        second = base.spawn(action_random_source=ConstantRandom())
        second_online = second.plastic_parameter_snapshot()
        second_target = second.target_parameter_snapshot()

        for _ in range(6):
            first.act(self.current)
            first.learn(self.next)

        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    second_online,
                    second.plastic_parameter_snapshot(),
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    second_target,
                    second.target_parameter_snapshot(),
                )
            )
        )

    def test_same_inheritance_plus_different_lives_diverges(self) -> None:
        _, base = autonomous_setup(learning=learning_config())
        first = base.spawn(action_random_source=ConstantRandom())
        second = base.spawn(action_random_source=ConstantRandom())
        target_before = first.target_parameter_snapshot()
        alternate = sample_experience(
            energy_distress=0.95,
            condition_distress=0.8,
            energetic_exertion=0.7,
            motor_direction=-1.0,
        )

        for _ in range(12):
            first.act(self.current)
            first.learn(self.next)
            second.act(alternate)
            second.learn(alternate)

        self.assertTrue(
            snapshots_differ(
                first.plastic_parameter_snapshot(),
                second.plastic_parameter_snapshot(),
            )
        )
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    target_before,
                    first.target_parameter_snapshot(),
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    first.target_parameter_snapshot(),
                    second.target_parameter_snapshot(),
                )
            )
        )

    def test_repeated_predictable_transition_materially_reduces_loss(self) -> None:
        brain, _, _ = controlled_brain(rate=0.01)
        losses: list[float] = []

        for _ in range(100):
            brain.act(self.current)
            losses.append(brain.learn(self.next).prediction_loss)

        initial_mean = sum(losses[:10]) / 10
        final_mean = sum(losses[-10:]) / 10
        self.assertLess(final_mean, initial_mean * 0.25)

    def test_persistent_hidden_state_and_completed_context_are_detached(self) -> None:
        brain, _, _ = controlled_brain()

        brain.act(self.current)
        pending = brain.pending_prediction
        self.assertIsNotNone(pending)
        self.assertTrue(brain.has_pending_transition)
        brain.learn(self.next)

        hidden = brain.hidden_state
        self.assertFalse(hidden.requires_grad)
        self.assertIsNone(hidden.grad_fn)
        self.assertFalse(brain.has_pending_transition)
        self.assertIsNone(brain.pending_prediction)
        self.assertIsNone(brain.pending_action_vector)

    def test_unresolved_transition_blocks_the_next_action(self) -> None:
        brain, _, _ = controlled_brain()
        brain.act(self.current)

        with self.assertRaisesRegex(RuntimeError, "still unresolved"):
            brain.act(self.current)

        brain.discard_pending_transition()
        self.assertFalse(brain.has_pending_transition)

    def test_completed_transition_cannot_be_replayed_or_updated_twice(self) -> None:
        brain, _, _ = controlled_brain()
        brain.act(self.current)
        brain.learn(self.next)

        with self.assertRaisesRegex(RuntimeError, "no pending"):
            brain.learn(self.next)
        self.assertEqual(brain.learning_update_count, 1)
        self.assertFalse(
            any(
                "replay" in name.lower() or "buffer" in name.lower()
                for name in vars(brain)
            )
        )

    def test_gradient_norm_is_clipped_and_gradients_are_cleared(self) -> None:
        maximum = 1e-5
        brain, _, _ = controlled_brain(max_norm=maximum)

        for _ in range(2):
            brain.act(self.current)
            result = brain.learn(self.next)
            self.assertLessEqual(result.gradient_norm, maximum)
            self.assertTrue(
                all(
                    parameter.grad is None
                    for parameter in brain.model.predictive_plastic_parameters()
                )
            )

    def test_non_finite_loss_is_rejected_before_optimizer_step(self) -> None:
        brain, _, _ = controlled_brain()
        predictor = brain.model.prediction_model
        assert predictor is not None
        with torch.no_grad():
            predictor.output_layer.weight.zero_()
            predictor.output_layer.bias.fill_(1e20)
        before = brain.plastic_parameter_snapshot()

        brain.act(self.current)
        with self.assertRaisesRegex(FloatingPointError, "loss is non-finite"):
            brain.learn(self.next)

        self.assertEqual(brain.learning_update_count, 0)
        self.assertFalse(brain.has_pending_transition)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(before, brain.plastic_parameter_snapshot())
            )
        )

    def test_non_finite_gradient_is_rejected_before_optimizer_step(self) -> None:
        brain, _, _ = controlled_brain()
        predictor = brain.model.prediction_model
        assert predictor is not None
        before = brain.plastic_parameter_snapshot()
        hook = predictor.output_layer.weight.register_hook(
            lambda gradient: torch.full_like(gradient, torch.inf)
        )
        try:
            brain.act(self.current)
            with self.assertRaisesRegex(
                FloatingPointError,
                "gradient is non-finite",
            ):
                brain.learn(self.next)
        finally:
            hook.remove()

        self.assertEqual(brain.learning_update_count, 0)
        self.assertFalse(brain.has_pending_transition)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(before, brain.plastic_parameter_snapshot())
            )
        )

    def test_learning_api_accepts_only_next_experience(self) -> None:
        brain, _, _ = controlled_brain()
        signature = inspect.signature(brain.learn)

        self.assertEqual(tuple(signature.parameters), ("next_experience",))
        brain.act(self.current)
        with self.assertRaisesRegex(TypeError, "NoraletExperience"):
            brain.learn(object())  # type: ignore[arg-type]
        brain.discard_pending_transition()

    def test_learning_adds_no_action_random_draws(self) -> None:
        brain, _, source = controlled_brain()

        brain.act(self.current)
        self.assertEqual(source.draw_count, 3)
        brain.learn(self.next)
        self.assertEqual(source.draw_count, 3)


if __name__ == "__main__":
    unittest.main()
