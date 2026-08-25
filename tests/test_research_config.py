"""Protocol, seed and fair-initialization tests for Research 001."""

from __future__ import annotations

import unittest

import torch

from noralet.brain import AutonomousSimulationRunner
from noralet.research.config import (
    PROTOCOL_CONDITIONS,
    BaselineExperimentConfig,
    LearningCondition,
    build_run_components,
    seed_mapping,
)


class ResearchConfigurationTests(unittest.TestCase):
    def test_pilot_defaults_define_the_complete_four_condition_matrix(self) -> None:
        config = BaselineExperimentConfig()

        self.assertEqual(config.replicate_seeds, tuple(range(1, 11)))
        self.assertEqual(config.max_ticks, 5_000)
        self.assertEqual(config.sample_every_ticks, 10)
        self.assertEqual(config.initial_population, 6)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.conditions, PROTOCOL_CONDITIONS)
        self.assertEqual(config.total_runs, 40)

    def test_seed_mapping_is_fixed_and_role_separated(self) -> None:
        expected = seed_mapping(1)

        self.assertEqual(expected, seed_mapping(1))
        self.assertEqual(expected.simulation_seed, 872351993654427070)
        self.assertEqual(expected.base_brain_seed, 4389058043275550208)
        self.assertNotEqual(expected.simulation_seed, expected.base_brain_seed)
        self.assertNotEqual(expected, seed_mapping(2))

    def test_protocol_requires_multiple_unique_replicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            BaselineExperimentConfig(replicate_seeds=(1,))
        with self.assertRaisesRegex(ValueError, "unique"):
            BaselineExperimentConfig(replicate_seeds=(1, 1))

    def test_conditions_share_world_inherited_brain_hidden_and_action_rng(self) -> None:
        config = BaselineExperimentConfig(
            replicate_seeds=(5, 9),
            max_ticks=2,
            sample_every_ticks=1,
            initial_population=3,
            device="cpu",
        )
        seeds = config.seed_mappings[0]
        built = [
            build_run_components(config, condition, seeds)
            for condition in PROTOCOL_CONDITIONS
        ]
        reference_simulation, reference_base = built[0]
        reference_parameters = tuple(
            parameter.detach().cpu().clone()
            for parameter in reference_base.prototype_model.iteration_8_parameters()
        )
        reference_runner = AutonomousSimulationRunner(
            reference_simulation,
            reference_base,
        )
        action_stream_name = reference_runner.action_stream_name(1)
        reference_rng_state = reference_simulation.random_streams.stream(
            action_stream_name
        ).getstate()

        for (simulation, base), condition in zip(
            built[1:],
            PROTOCOL_CONDITIONS[1:],
            strict=True,
        ):
            self.assertEqual(simulation.state, reference_simulation.state)
            self.assertEqual(
                tuple(body.perceptual_signature for body in simulation.state.bodies),
                tuple(
                    body.perceptual_signature
                    for body in reference_simulation.state.bodies
                ),
            )
            compared = base.prototype_model.iteration_8_parameters()
            self.assertEqual(len(compared), len(reference_parameters))
            for actual, expected in zip(compared, reference_parameters, strict=True):
                self.assertTrue(torch.equal(actual.detach().cpu(), expected))

            runner = AutonomousSimulationRunner(simulation, base)
            for identity in runner.brain_ids:
                self.assertTrue(
                    torch.equal(
                        runner.brain_for(identity).hidden_state,
                        reference_runner.brain_for(identity).hidden_state,
                    )
                )
            self.assertEqual(
                simulation.random_streams.stream(action_stream_name).getstate(),
                reference_rng_state,
                condition,
            )

        self.assertFalse(reference_base.learning_config)
        self.assertFalse(reference_base.homeostatic_plasticity_config)
        self.assertTrue(built[1][1].learning_config)
        self.assertFalse(built[1][1].homeostatic_plasticity_config)
        self.assertTrue(built[2][1].learning_config)
        self.assertTrue(built[2][1].homeostatic_plasticity_config)
        self.assertFalse(built[3][1].learning_config)
        self.assertTrue(built[3][1].homeostatic_plasticity_config)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_research_components_complete_a_tick(self) -> None:
        config = BaselineExperimentConfig(
            replicate_seeds=(1, 2),
            max_ticks=1,
            sample_every_ticks=1,
            initial_population=2,
            device="cuda",
            conditions=(LearningCondition.FULL_CURRENT_BRAIN,),
        )
        simulation, base = build_run_components(
            config,
            LearningCondition.FULL_CURRENT_BRAIN,
            config.seed_mappings[0],
        )
        runner = AutonomousSimulationRunner(simulation, base)

        result = runner.step()

        self.assertEqual(result.tick_result.tick_after, 1)
        self.assertEqual(str(runner.brain_for(1).device), "cuda")


if __name__ == "__main__":
    unittest.main()
