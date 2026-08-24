"""Autonomous integration, lifecycle, determinism and CUDA plasticity tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import fields
from types import MethodType
import random
import unittest

import torch

from brain_test_support import (
    actuator_config,
    autonomous_setup,
    brain_body,
    brain_config,
    homeostatic_config,
    learning_config,
    sample_experience,
)
from noralet import NoraletHomeostaticLearningResult
from physiology_test_support import physiology_config


def snapshots_equal(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


class HomeostaticCoordinatorTests(unittest.TestCase):
    def test_pending_modulatory_context_contains_only_current_drive(self) -> None:
        import noralet.brain.runtime as runtime

        self.assertEqual(
            tuple(field.name for field in fields(runtime._PendingHomeostaticTransition)),
            ("homeostatic_drive",),
        )

    def test_survivors_receive_separate_predictive_and_homeostatic_results(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(),
            homeostatic=homeostatic_config(),
            actuator=actuator_config(0.001),
            existence_cost=0.1,
        )

        result = runner.step()

        self.assertEqual(tuple(item.noralet_id for item in result.learning_results), (1, 2))
        self.assertEqual(
            tuple(item.noralet_id for item in result.homeostatic_learning_results),
            (1, 2),
        )
        self.assertIsInstance(
            result.homeostatic_learning_for(1),
            NoraletHomeostaticLearningResult,
        )
        self.assertEqual(runner.brain_for(1).learning_update_count, 1)
        self.assertEqual(runner.brain_for(1).homeostatic_update_count, 1)
        self.assertFalse(runner.brain_for(1).has_pending_transition)
        with self.assertRaises(FrozenInstanceError):
            result.homeostatic_learning_results[0].modulation = 0  # type: ignore[misc]

    def test_every_action_precedes_predictive_then_homeostatic_learning(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(),
            homeostatic=homeostatic_config(),
            actuator=actuator_config(0.001),
            existence_cost=0.1,
        )
        order: list[str] = []
        for identity in runner.brain_ids:
            brain = runner.brain_for(identity)
            original_act = brain.act
            original_learn = brain.learn
            original_homeostatic = brain.apply_homeostatic_update

            def act(self, experience, *, _id=identity, _call=original_act):
                order.append(f"act:{_id}")
                return _call(experience)

            def learn(self, experience, *, _id=identity, _call=original_learn):
                order.append(f"predict:{_id}")
                return _call(experience)

            def homeostatic(
                self,
                experience,
                *,
                _id=identity,
                _call=original_homeostatic,
            ):
                order.append(f"homeostatic:{_id}")
                return _call(experience)

            brain.act = MethodType(act, brain)  # type: ignore[method-assign]
            brain.learn = MethodType(learn, brain)  # type: ignore[method-assign]
            brain.apply_homeostatic_update = MethodType(  # type: ignore[method-assign]
                homeostatic,
                brain,
            )

        runner.step()

        self.assertEqual(
            order,
            [
                "act:1",
                "act:2",
                "predict:1",
                "homeostatic:1",
                "predict:2",
                "homeostatic:2",
            ],
        )

    def test_combined_learning_changes_disjoint_online_parameter_sets(self) -> None:
        runner, base = autonomous_setup(
            learning=learning_config(learning_rate=0.02),
            homeostatic=homeostatic_config(action_learning_rate=0.2),
            actuator=actuator_config(0.001),
            existence_cost=0.2,
        )
        brain = runner.brain_for(1)
        plastic_before = brain.plastic_parameter_snapshot()
        heads_before = brain.action_head_parameter_snapshot()
        target_before = brain.target_parameter_snapshot()
        base_before = base.parameter_snapshot()

        result = runner.step()

        self.assertFalse(snapshots_equal(plastic_before, brain.plastic_parameter_snapshot()))
        homeostatic_result = result.homeostatic_learning_for(1)
        self.assertNotEqual(homeostatic_result.modulation, 0.0)
        self.assertFalse(snapshots_equal(heads_before, brain.action_head_parameter_snapshot()))
        self.assertTrue(snapshots_equal(target_before, brain.target_parameter_snapshot()))
        self.assertTrue(snapshots_equal(base_before, base.parameter_snapshot()))

    def test_individual_bodily_histories_diverge_action_heads(self) -> None:
        _, base = autonomous_setup(homeostatic=homeostatic_config())

        class ConstantRandom:
            def random(self) -> float:
                return 0.7

        first = base.spawn(action_random_source=ConstantRandom())
        second = base.spawn(action_random_source=ConstantRandom())
        current = sample_experience(energy_distress=0.5)
        first.act(current)
        second.act(current)
        first.apply_homeostatic_update(sample_experience(energy_distress=0.1))
        second.apply_homeostatic_update(sample_experience(energy_distress=0.9))

        self.assertFalse(
            snapshots_equal(
                first.action_head_parameter_snapshot(),
                second.action_head_parameter_snapshot(),
            )
        )

    def test_mixed_autonomous_smoke_is_finite_and_energy_conservative(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(),
            homeostatic=homeostatic_config(action_learning_rate=0.01),
            actuator=actuator_config(0.001),
            existence_cost=0.05,
            acceleration_cost=0.02,
            signal_energy_cost=0.01,
        )
        initial_energy = runner.simulation.state.energy_totals.total_energy

        results = tuple(runner.step() for _ in range(10))

        self.assertTrue(all(result.learning_results for result in results))
        self.assertTrue(
            all(result.homeostatic_learning_results for result in results)
        )
        for result in results:
            for metric in result.homeostatic_learning_results:
                self.assertTrue(torch.isfinite(torch.tensor(metric.modulation)))
        for identity in runner.brain_ids:
            brain = runner.brain_for(identity)
            traces = brain.eligibility_traces
            assert traces is not None
            self.assertTrue(all(torch.isfinite(trace).all() for trace in traces.tensors))
            self.assertTrue(all(torch.isfinite(p).all() for p in brain.model.parameters()))
        self.assertAlmostEqual(
            runner.simulation.state.energy_totals.total_energy,
            initial_energy,
            delta=runner.simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )


class HomeostaticCompatibilityAndRngTests(unittest.TestCase):
    def test_disabled_mode_preserves_iteration_9_action_heads_and_results(self) -> None:
        runner, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
        )
        brain = runner.brain_for(1)
        heads_before = brain.action_head_parameter_snapshot()

        results = tuple(runner.step() for _ in range(6))

        self.assertIsNone(brain.eligibility_traces)
        self.assertEqual(brain.homeostatic_update_count, 0)
        self.assertTrue(
            all(result.homeostatic_learning_results == () for result in results)
        )
        self.assertTrue(snapshots_equal(heads_before, brain.action_head_parameter_snapshot()))

    def test_homeostatic_machinery_adds_no_rng_draw_before_first_update(self) -> None:
        enabled, _ = autonomous_setup(
            learning=learning_config(),
            homeostatic=homeostatic_config(),
            actuator=actuator_config(0.001),
            simulation_seed=337,
        )
        control, _ = autonomous_setup(
            learning=learning_config(),
            actuator=actuator_config(0.001),
            simulation_seed=337,
        )
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()

        enabled_result = enabled.step()
        control_result = control.step()

        self.assertEqual(enabled_result.action_intents, control_result.action_intents)
        self.assertEqual(enabled_result.tick_result, control_result.tick_result)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        enabled_streams = enabled.simulation.random_streams._streams
        control_streams = control.simulation.random_streams._streams
        self.assertEqual(set(enabled_streams), set(control_streams))
        self.assertTrue(
            all(
                enabled_streams[name].getstate() == control_streams[name].getstate()
                for name in enabled_streams
            )
        )

    def test_zero_acceleration_exploration_is_allowed_only_when_disabled(self) -> None:
        autonomous_setup(
            brain=brain_config(exploration_std=0),
            learning=learning_config(),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            autonomous_setup(
                brain=brain_config(exploration_std=0),
                homeostatic=homeostatic_config(),
            )

    def test_identical_cpu_lifetimes_reproduce_plasticity_and_traces(self) -> None:
        kwargs = dict(
            brain=brain_config(device="cpu"),
            learning=learning_config(),
            homeostatic=homeostatic_config(),
            actuator=actuator_config(0.001),
            existence_cost=0.1,
            simulation_seed=338,
        )
        first, _ = autonomous_setup(**kwargs)
        second, _ = autonomous_setup(**kwargs)

        first_history = tuple(first.step() for _ in range(8))
        second_history = tuple(second.step() for _ in range(8))

        self.assertEqual(first_history, second_history)
        self.assertEqual(first.simulation.state, second.simulation.state)
        for identity in first.brain_ids:
            first_brain = first.brain_for(identity)
            second_brain = second.brain_for(identity)
            self.assertTrue(
                snapshots_equal(first_brain.parameter_snapshot(), second_brain.parameter_snapshot())
            )
            self.assertTrue(torch.equal(first_brain.hidden_state, second_brain.hidden_state))
            first_traces = first_brain.eligibility_traces
            second_traces = second_brain.eligibility_traces
            assert first_traces is not None
            assert second_traces is not None
            self.assertTrue(
                all(
                    torch.equal(a, b)
                    for a, b in zip(first_traces.tensors, second_traces.tensors, strict=True)
                )
            )


class HomeostaticDeathTests(unittest.TestCase):
    def _assert_no_posthumous_update(self, runner) -> None:
        doomed = runner.brain_for(1)

        result = runner.step()

        self.assertNotIn(1, runner.brain_ids)
        self.assertEqual(doomed.homeostatic_update_count, 0)
        self.assertFalse(doomed.has_pending_transition)
        self.assertIsNone(doomed.eligibility_traces)
        self.assertNotIn(
            1,
            tuple(item.noralet_id for item in result.homeostatic_learning_results),
        )

    def test_boundary_death_discards_modulatory_context_and_eligibility(self) -> None:
        runner, _ = autonomous_setup(
            bodies=(brain_body(1, 9.9, velocity=1.0), brain_body(2, 0)),
            learning=learning_config(),
            homeostatic=homeostatic_config(),
            actuator=actuator_config(0.001),
        )
        self._assert_no_posthumous_update(runner)

    def test_energy_depletion_discards_modulatory_context_and_eligibility(self) -> None:
        runner, _ = autonomous_setup(
            bodies=(brain_body(1, 0, energy=0), brain_body(2, 2)),
            learning=learning_config(),
            homeostatic=homeostatic_config(),
            actuator=actuator_config(0.001),
        )
        self._assert_no_posthumous_update(runner)

    def test_natural_death_discards_modulatory_context_and_eligibility(self) -> None:
        runner, _ = autonomous_setup(
            bodies=(brain_body(1, 0), brain_body(2, 2)),
            learning=learning_config(),
            homeostatic=homeostatic_config(),
            actuator=actuator_config(0.001),
            physiology=physiology_config(baseline_loss=0, base_hazard=100),
        )
        doomed = runner.brain_for(1)

        result = runner.step()

        self.assertEqual(runner.brain_ids, ())
        self.assertEqual(result.homeostatic_learning_results, ())
        self.assertEqual(doomed.homeostatic_update_count, 0)
        self.assertIsNone(doomed.eligibility_traces)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaHomeostaticLearningTests(unittest.TestCase):
    def test_real_cuda_action_and_predictive_plasticity_smoke(self) -> None:
        runner, _ = autonomous_setup(
            brain=brain_config(device="cuda"),
            learning=learning_config(),
            homeostatic=homeostatic_config(action_learning_rate=0.1),
            actuator=actuator_config(0.001),
            existence_cost=0.1,
        )
        brain = runner.brain_for(1)
        target_before = brain.target_parameter_snapshot()
        heads_before = brain.action_head_parameter_snapshot()
        initial_energy = runner.simulation.state.energy_totals.total_energy

        results = tuple(runner.step() for _ in range(3))

        self.assertTrue(all(result.learning_results for result in results))
        self.assertTrue(all(result.homeostatic_learning_results for result in results))
        self.assertFalse(snapshots_equal(heads_before, brain.action_head_parameter_snapshot()))
        self.assertTrue(snapshots_equal(target_before, brain.target_parameter_snapshot()))
        traces = brain.eligibility_traces
        assert traces is not None
        self.assertTrue(all(trace.device.type == "cuda" for trace in traces.tensors))
        self.assertTrue(all(torch.isfinite(trace).all() for trace in traces.tensors))
        self.assertTrue(all(torch.isfinite(p).all() for p in brain.model.parameters()))
        self.assertEqual(brain.hidden_state.device.type, "cuda")
        self.assertAlmostEqual(
            runner.simulation.state.energy_totals.total_energy,
            initial_energy,
            delta=runner.simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
