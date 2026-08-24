"""Eligibility traces and direct three-factor action-head plasticity tests."""

from __future__ import annotations

import math
from statistics import NormalDist
import unittest

import torch

from brain_test_support import (
    autonomous_setup,
    brain_config,
    homeostatic_config,
    learning_config,
    sample_experience,
)
from noralet import ActionEligibilityTraces


class SequenceRandom:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return next(self._values)


def module_snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in module.parameters())


def snapshots_equal(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def controlled_brain(
    *,
    modulation_scale: float = 0.2,
    decay: float = 0.8,
    rate: float = 0.05,
    maximum_norm: float = 10.0,
    draws: tuple[float, ...] = (0.8413447460685429, 0.1, 0.5),
    predictive: bool = False,
):
    _, base = autonomous_setup(
        brain=brain_config(exploration_std=0.25),
        learning=learning_config() if predictive else None,
        homeostatic=homeostatic_config(
            energy_distress_weight=1,
            condition_distress_weight=0,
            homeostatic_modulation_scale=modulation_scale,
            eligibility_decay=decay,
            action_learning_rate=rate,
            max_homeostatic_update_norm=maximum_norm,
        ),
    )
    source = SequenceRandom(draws)
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


class EligibilityTraceTests(unittest.TestCase):
    def test_spawned_traces_are_zero_shape_matched_and_on_parameter_device(self) -> None:
        brain, _, _ = controlled_brain()
        traces = brain.eligibility_traces
        assert traces is not None
        parameter_groups = (
            tuple(brain.model.acceleration_head.parameters()),
            tuple(brain.model.consume_head.parameters()),
            tuple(brain.model.signal_head.parameters()),
        )

        for trace_group, parameter_group in zip(
            (traces.acceleration, traces.consume, traces.signal),
            parameter_groups,
            strict=True,
        ):
            self.assertEqual(len(trace_group), len(parameter_group))
            for trace, parameter in zip(trace_group, parameter_group, strict=True):
                self.assertEqual(trace.shape, parameter.shape)
                self.assertEqual(trace.device, parameter.device)
                self.assertTrue(torch.count_nonzero(trace).item() == 0)
                self.assertFalse(trace.requires_grad)
                self.assertIsNone(trace.grad_fn)

    def test_one_selected_action_creates_detached_eligibility_in_every_head(self) -> None:
        brain, _, source = controlled_brain()

        brain.act(sample_experience(energy_distress=0.5))
        traces = brain.eligibility_traces
        assert traces is not None

        self.assertEqual(source.draw_count, 3)
        for group in (traces.acceleration, traces.consume, traces.signal):
            self.assertTrue(any(torch.count_nonzero(trace).item() for trace in group))
        self.assertTrue(
            all(
                not trace.requires_grad and trace.grad_fn is None
                for trace in traces.tensors
            )
        )

    def test_synthetic_trace_decay_and_zero_decay_are_exact(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        zero = ActionEligibilityTraces.zeros_like(
            acceleration=(parameter,),
            consume=(parameter,),
            signal=(parameter,),
        )
        ones = (torch.ones(2),)
        first = zero.advanced(
            acceleration_increment=ones,
            consume_increment=ones,
            signal_increment=ones,
            decay=0.7,
        )
        no_increment = (torch.zeros(2),)

        decayed = first.advanced(
            acceleration_increment=no_increment,
            consume_increment=no_increment,
            signal_increment=no_increment,
            decay=0.7,
        )
        replaced = first.advanced(
            acceleration_increment=no_increment,
            consume_increment=no_increment,
            signal_increment=no_increment,
            decay=0,
        )

        self.assertTrue(torch.equal(decayed.acceleration[0], torch.full((2,), 0.7)))
        self.assertTrue(torch.equal(replaced.acceleration[0], torch.zeros(2)))

    def test_trace_persists_after_modulation_and_decays_on_next_action(self) -> None:
        draws = (
            0.8413447460685429,
            0.1,
            0.5,
            0.5,
            0.1,
            0.5,
        )
        brain, _, _ = controlled_brain(decay=0.5, draws=draws)
        neutral = sample_experience(energy_distress=0.5)

        brain.act(neutral)
        first = brain.eligibility_traces
        assert first is not None
        brain.apply_homeostatic_update(neutral)
        retained = brain.eligibility_traces
        assert retained is not None
        self.assertTrue(
            all(
                torch.equal(a, b)
                for a, b in zip(first.tensors, retained.tensors, strict=True)
            )
        )

        brain.act(neutral)
        second = brain.eligibility_traces
        assert second is not None
        first_acceleration = first.acceleration
        for old, new in zip(first_acceleration, second.acceleration, strict=True):
            self.assertTrue(torch.allclose(new, old * 0.5, atol=1e-7))


class ActionPlasticityPolarityTests(unittest.TestCase):
    signal_index = 4

    @staticmethod
    def action_outputs(brain, hidden: torch.Tensor) -> tuple[float, float, float]:
        with torch.no_grad():
            acceleration_loc = float(brain.model.acceleration_head(hidden).item())
            consume_probability = float(
                torch.sigmoid(brain.model.consume_head(hidden)).item()
            )
            signal_probability = float(
                torch.softmax(brain.model.signal_head(hidden), dim=0)[4].item()
            )
        return acceleration_loc, consume_probability, signal_probability

    def _run_polarity(self, *, favorable: bool) -> None:
        signal_draw = (self.signal_index + 0.5) / 9
        acceleration_draw = 0.8413447460685429
        brain, _, _ = controlled_brain(
            draws=(acceleration_draw, 0.1, signal_draw),
            maximum_norm=100,
        )
        before_drive, after_drive = ((0.8, 0.2) if favorable else (0.2, 0.8))
        brain.act(sample_experience(energy_distress=before_drive))
        hidden = brain.hidden_state
        before = self.action_outputs(brain, hidden)
        result = brain.apply_homeostatic_update(
            sample_experience(energy_distress=after_drive)
        )
        after = self.action_outputs(brain, hidden)
        selected_raw = 0.25 * NormalDist().inv_cdf(acceleration_draw)

        before_acceleration_distance = abs(selected_raw - before[0])
        after_acceleration_distance = abs(selected_raw - after[0])
        if favorable:
            self.assertGreater(result.modulation, 0)
            self.assertLess(after_acceleration_distance, before_acceleration_distance)
            self.assertGreater(after[1], before[1])
            self.assertGreater(after[2], before[2])
        else:
            self.assertLess(result.modulation, 0)
            self.assertGreater(after_acceleration_distance, before_acceleration_distance)
            self.assertLess(after[1], before[1])
            self.assertLess(after[2], before[2])

    def test_positive_modulation_increases_selected_action_likelihoods(self) -> None:
        self._run_polarity(favorable=True)

    def test_negative_modulation_decreases_selected_action_likelihoods(self) -> None:
        self._run_polarity(favorable=False)

    def test_neutral_modulation_retains_traces_without_parameter_change(self) -> None:
        brain, _, _ = controlled_brain()
        experience = sample_experience(energy_distress=0.5)
        brain.act(experience)
        traces = brain.eligibility_traces
        assert traces is not None
        before = brain.action_head_parameter_snapshot()

        result = brain.apply_homeostatic_update(experience)

        self.assertEqual(result.modulation, 0.0)
        self.assertEqual(result.applied_update_norm, 0.0)
        self.assertTrue(snapshots_equal(before, brain.action_head_parameter_snapshot()))
        after_traces = brain.eligibility_traces
        assert after_traces is not None
        self.assertTrue(
            all(
                torch.equal(a, b)
                for a, b in zip(traces.tensors, after_traces.tensors, strict=True)
            )
        )

    def test_update_direction_is_clipped_before_learning_rate(self) -> None:
        maximum = 0.01
        rate = 0.2
        brain, _, _ = controlled_brain(rate=rate, maximum_norm=maximum)
        brain.act(sample_experience(energy_distress=0.9))

        result = brain.apply_homeostatic_update(
            sample_experience(energy_distress=0.1)
        )

        self.assertLessEqual(result.applied_update_norm, maximum * rate * 1.00001)
        self.assertGreater(result.applied_update_norm, 0.0)


class PlasticityBoundaryTests(unittest.TestCase):
    def test_homeostatic_update_changes_only_action_heads(self) -> None:
        brain, base, _ = controlled_brain(predictive=True)
        predictor = brain.model.prediction_model
        target = brain.target_experience_encoder
        assert predictor is not None
        assert target is not None
        current = sample_experience(energy_distress=0.8)
        next_experience = sample_experience(
            energy_distress=0.2,
            condition_distress=0.7,
        )
        base_before = base.parameter_snapshot()
        target_before = brain.target_parameter_snapshot()
        brain.act(current)
        brain.learn(next_experience)
        encoder_before = module_snapshot(brain.model.encoder)
        gru_before = module_snapshot(brain.model.recurrent_core)
        predictor_before = module_snapshot(predictor)
        heads_before = brain.action_head_parameter_snapshot()

        result = brain.apply_homeostatic_update(next_experience)

        self.assertNotEqual(result.modulation, 0.0)
        self.assertTrue(snapshots_equal(encoder_before, module_snapshot(brain.model.encoder)))
        self.assertTrue(snapshots_equal(gru_before, module_snapshot(brain.model.recurrent_core)))
        self.assertTrue(snapshots_equal(predictor_before, module_snapshot(predictor)))
        self.assertTrue(snapshots_equal(target_before, brain.target_parameter_snapshot()))
        self.assertTrue(snapshots_equal(base_before, base.parameter_snapshot()))
        self.assertFalse(snapshots_equal(heads_before, brain.action_head_parameter_snapshot()))

    def test_predictive_optimizer_still_excludes_trainable_action_heads(self) -> None:
        brain, _, _ = controlled_brain(predictive=True)
        optimizer = brain.optimizer
        assert optimizer is not None
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        action_parameters = set(map(id, brain.model.action_head_parameters()))

        self.assertTrue(optimized.isdisjoint(action_parameters))
        self.assertTrue(
            all(parameter.requires_grad for parameter in brain.model.action_head_parameters())
        )

    def test_eligibility_calculation_leaves_all_parameter_grad_fields_empty(self) -> None:
        brain, _, _ = controlled_brain(predictive=True)

        brain.act(sample_experience(energy_distress=0.5))

        self.assertTrue(
            all(parameter.grad is None for parameter in brain.model.parameters())
        )
        brain.discard_pending_transition()

    def test_non_finite_action_parameter_is_rejected_before_update(self) -> None:
        brain, _, _ = controlled_brain()
        brain.act(sample_experience(energy_distress=0.8))
        with torch.no_grad():
            brain.model.consume_head.bias.fill_(torch.inf)

        with self.assertRaisesRegex(FloatingPointError, "before modulation"):
            brain.apply_homeostatic_update(
                sample_experience(energy_distress=0.2)
            )

        self.assertFalse(brain.has_pending_transition)


class DelayedEligibilityTests(unittest.TestCase):
    @staticmethod
    def delayed_run(decay: float):
        positive_normal = 0.8413447460685429
        draws = (
            positive_normal,
            0.1,
            0.5,
            0.5,
            0.1,
            0.5,
            0.5,
            0.1,
            0.5,
        )
        brain, _, _ = controlled_brain(decay=decay, draws=draws, maximum_norm=100)
        neutral = sample_experience(energy_distress=0.5)
        for _ in range(2):
            brain.act(neutral)
            brain.apply_homeostatic_update(neutral)
        before = tuple(
            parameter.detach().clone()
            for parameter in brain.model.acceleration_head.parameters()
        )
        brain.act(neutral)
        traces = brain.eligibility_traces
        assert traces is not None
        acceleration_eligibility_norm = torch.linalg.vector_norm(
            torch.cat(tuple(trace.reshape(-1) for trace in traces.acceleration))
        ).item()
        result = brain.apply_homeostatic_update(
            sample_experience(energy_distress=0.1)
        )
        after = tuple(
            parameter.detach().clone()
            for parameter in brain.model.acceleration_head.parameters()
        )
        changed = not snapshots_equal(before, after)
        return acceleration_eligibility_norm, changed, result

    def test_delayed_improvement_modifies_residual_earlier_acceleration_path(self) -> None:
        persistent_norm, persistent_changed, persistent_result = self.delayed_run(0.5)
        immediate_norm, immediate_changed, immediate_result = self.delayed_run(0.0)

        self.assertGreater(persistent_norm, 0.0)
        self.assertTrue(persistent_changed)
        self.assertEqual(immediate_norm, 0.0)
        self.assertFalse(immediate_changed)
        self.assertGreater(persistent_result.modulation, 0.0)
        self.assertEqual(persistent_result.modulation, immediate_result.modulation)


if __name__ == "__main__":
    unittest.main()
