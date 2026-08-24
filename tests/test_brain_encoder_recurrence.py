"""Set encoding, Experience boundary and recurrent-state tests."""

from __future__ import annotations

import inspect
import math
import unittest

import torch

from brain_test_support import (
    autonomous_setup,
    external_percept,
    sample_experience,
    signal_percept,
)
from noralet import NoraletBrain, WorldState
from noralet.brain.encoder import ExperienceEncoder
from noralet.brain.model import NoraletBrainModel


class ExperienceEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        _, base = autonomous_setup()
        self.encoder = base.prototype_model.encoder

    def test_zero_one_and_many_external_percepts_have_fixed_output_shape(self) -> None:
        experiences = (
            sample_experience(),
            sample_experience(external_percepts=(external_percept(),)),
            sample_experience(
                external_percepts=tuple(
                    external_percept(
                        direction=-1.0 if index % 2 else 1.0,
                        proximity=index / 10.0,
                    )
                    for index in range(10)
                )
            ),
        )

        with torch.no_grad():
            embeddings = tuple(self.encoder(experience) for experience in experiences)

        self.assertTrue(all(tuple(value.shape) == (6,) for value in embeddings))
        self.assertTrue(all(torch.isfinite(value).all() for value in embeddings))

    def test_external_summary_is_permutation_invariant(self) -> None:
        percepts = (
            external_percept(direction=-1, proximity=0.2),
            external_percept(direction=0, proximity=0.7),
            external_percept(direction=1, proximity=0.4),
        )

        with torch.no_grad():
            forward = self.encoder.encode_external(percepts)
            reversed_summary = self.encoder.encode_external(tuple(reversed(percepts)))

        self.assertTrue(torch.allclose(forward, reversed_summary, atol=1e-6, rtol=0))

    def test_external_sum_pooling_preserves_multiplicity(self) -> None:
        percept = external_percept()

        with torch.no_grad():
            single = self.encoder.encode_external((percept,))
            duplicate = self.encoder.encode_external((percept, percept))

        self.assertTrue(torch.allclose(duplicate, single * 2, atol=1e-7, rtol=0))
        self.assertFalse(torch.equal(single, duplicate))

    def test_signal_summary_is_permutation_invariant(self) -> None:
        percepts = (
            signal_percept(direction=-1, strength=0.2),
            signal_percept(direction=0, strength=0.7),
            signal_percept(direction=1, strength=0.4),
        )

        with torch.no_grad():
            forward = self.encoder.encode_signals(percepts)
            reversed_summary = self.encoder.encode_signals(tuple(reversed(percepts)))

        self.assertTrue(torch.allclose(forward, reversed_summary, atol=1e-6, rtol=0))

    def test_signal_sum_pooling_preserves_multiplicity(self) -> None:
        percept = signal_percept()

        with torch.no_grad():
            single = self.encoder.encode_signals((percept,))
            duplicate = self.encoder.encode_signals((percept, percept))

        self.assertTrue(torch.allclose(duplicate, single * 2, atol=1e-7, rtol=0))
        self.assertFalse(torch.equal(single, duplicate))

    def test_empty_external_and_signal_sets_are_exact_zero_vectors(self) -> None:
        with torch.no_grad():
            external = self.encoder.encode_external(())
            signals = self.encoder.encode_signals(())

        self.assertEqual(tuple(external.shape), (4,))
        self.assertEqual(tuple(signals.shape), (4,))
        self.assertTrue(torch.equal(external, torch.zeros_like(external)))
        self.assertTrue(torch.equal(signals, torch.zeros_like(signals)))

    def test_encoder_has_no_semantic_lookup_or_objective_input(self) -> None:
        source = inspect.getsource(ExperienceEncoder)
        for forbidden in (
            "WorldState",
            "NoraletBodyState",
            "SignalType",
            "SignalDirection",
            "noralet_id",
            "point_id",
        ):
            self.assertNotIn(forbidden, source)
        model_source = inspect.getsource(NoraletBrainModel)
        for forbidden in (
            "WorldState",
            "NoraletBodyState",
            "noralet_id",
            "position",
            "velocity",
            "energy",
            "condition",
            "age",
            "region",
        ):
            self.assertNotIn(forbidden, model_source)

        with torch.no_grad():
            result = self.encoder(
                sample_experience(
                    external_percepts=(external_percept(),),
                    signal_percepts=(signal_percept(),),
                )
            )
        self.assertEqual(tuple(result.shape), (6,))


class RecurrentStateTests(unittest.TestCase):
    def test_activation_accepts_experience_and_rejects_world_state(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)
        signature = inspect.signature(NoraletBrain.activate)
        action_signature = inspect.signature(NoraletBrain.act)

        self.assertEqual(tuple(signature.parameters), ("self", "experience"))
        self.assertEqual(
            tuple(action_signature.parameters),
            ("self", "experience"),
        )
        with self.assertRaises(TypeError):
            brain.activate(WorldState())  # type: ignore[arg-type]

    def test_one_activation_updates_hidden_to_correct_finite_shape(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)
        before = brain.hidden_state

        parameters = brain.activate(sample_experience())

        self.assertEqual(tuple(brain.hidden_state.shape), (7,))
        self.assertTrue(torch.isfinite(brain.hidden_state).all())
        self.assertFalse(torch.equal(brain.hidden_state, before))
        self.assertTrue(math.isfinite(parameters.acceleration_loc))
        self.assertEqual(len(parameters.signal_logits), 9)

    def test_persistent_recurrence_uses_previous_hidden_state(self) -> None:
        runner, base = autonomous_setup()
        persistent = runner.brain_for(1)
        fresh = base.spawn()
        first_experience = sample_experience(energy_distress=0.8)
        second_experience = sample_experience(energy_distress=0.2)

        persistent.activate(first_experience)
        persistent_parameters = persistent.activate(second_experience)
        fresh_parameters = fresh.activate(second_experience)

        self.assertFalse(torch.equal(persistent.hidden_state, fresh.hidden_state))
        self.assertNotEqual(persistent_parameters, fresh_parameters)
        self.assertEqual(persistent.activation_count, 2)

    def test_activating_one_brain_does_not_change_another_hidden_state(self) -> None:
        runner, _ = autonomous_setup()
        first = runner.brain_for(1)
        second = runner.brain_for(2)
        second_before = second.hidden_state

        first.activate(sample_experience())

        self.assertFalse(torch.equal(first.hidden_state, torch.zeros(7)))
        self.assertTrue(torch.equal(second.hidden_state, second_before))
        self.assertEqual(second.activation_count, 0)

    def test_equal_clones_produce_equal_pre_sampling_parameters(self) -> None:
        runner, _ = autonomous_setup()
        experience = sample_experience(
            external_percepts=(external_percept(),),
            signal_percepts=(signal_percept(),),
        )

        first = runner.brain_for(1).activate(experience)
        second = runner.brain_for(2).activate(experience)

        self.assertEqual(first, second)
        self.assertTrue(
            torch.equal(
                runner.brain_for(1).hidden_state,
                runner.brain_for(2).hidden_state,
            )
        )

    def test_different_experience_histories_diverge_hidden_not_weights(self) -> None:
        runner, _ = autonomous_setup()
        first = runner.brain_for(1)
        second = runner.brain_for(2)
        first.activate(sample_experience(energy_distress=0.0))
        second.activate(sample_experience(energy_distress=1.0))

        self.assertFalse(torch.equal(first.hidden_state, second.hidden_state))
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    first.parameter_snapshot(),
                    second.parameter_snapshot(),
                )
            )
        )

    def test_inference_changes_no_parameters_or_gradients(self) -> None:
        runner, _ = autonomous_setup()
        brain = runner.brain_for(1)
        initial = brain.parameter_snapshot()

        for index in range(30):
            brain.activate(sample_experience(energy_distress=(index % 10) / 10))

        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(initial, brain.parameter_snapshot())
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in brain.model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
