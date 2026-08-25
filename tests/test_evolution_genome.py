"""Inherited-genome purity, mutation and mutation-only selection tests."""

from __future__ import annotations

import unittest

import torch

from noralet.brain import AutonomousSimulationRunner
from noralet.evolution.config import EvolutionConfig
from noralet.evolution.evaluation import (
    CandidateEvaluation,
    build_evolution_components,
    evaluate_candidate,
    evaluate_generation,
)
from noralet.evolution.genome import BaseBrainGenome, mutate_genome
from noralet.evolution.selection import (
    create_next_generation,
    initialize_generation_zero,
)


def tiny_config(*, device: str = "cpu") -> EvolutionConfig:
    return EvolutionConfig(
        generation_count=2,
        device=device,
        population_size=3,
        elite_count=1,
        parent_pool_size=2,
        mutation_sigma=0.02,
        training_world_seeds=(101,),
        validation_world_seeds=(202,),
        noralets_per_world=2,
        max_ticks=2,
        initial_seed=7,
    )


class GenomeInheritanceTests(unittest.TestCase):
    def test_protocol_defaults_and_genome_cover_current_inherited_modules(self) -> None:
        config = EvolutionConfig()
        self.assertEqual(
            (
                config.population_size,
                config.elite_count,
                config.parent_pool_size,
                config.worlds_per_candidate,
                config.noralets_per_world,
                config.max_ticks,
                config.initial_body_energy,
                config.mutation_sigma,
            ),
            (32, 4, 8, 4, 6, 2_000, 10.0, 0.02),
        )
        genome = initialize_generation_zero(tiny_config())[0].genome
        names = set(genome.state())
        for prefix in (
            "encoder.",
            "recurrent_core.",
            "prediction_model.",
            "acceleration_head.",
            "consume_head.",
            "signal_head.",
        ):
            self.assertTrue(any(name.startswith(prefix) for name in names), prefix)

    def test_fresh_spawns_match_genome_and_evaluation_cannot_change_it(self) -> None:
        config = tiny_config()
        candidate = initialize_generation_zero(config)[0]
        original = BaseBrainGenome.from_state(candidate.genome.state())
        _, base_brain = build_evolution_components(
            config,
            candidate.genome,
            simulation_seed=101,
        )
        first = base_brain.spawn()
        second = base_brain.spawn()
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    first.parameter_snapshot(),
                    second.parameter_snapshot(),
                    strict=True,
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(value, dict(first.model.named_parameters())[name].cpu())
                for name, value in candidate.genome.state().items()
            )
        )

        evaluate_candidate(
            candidate,
            config,
            world_seeds=config.training_world_seeds,
        )
        self.assertTrue(candidate.genome.exactly_equals(original))
        _, later_base = build_evolution_components(
            config,
            candidate.genome,
            simulation_seed=101,
        )
        self.assertTrue(
            all(
                torch.equal(
                    value,
                    dict(later_base.spawn().model.named_parameters())[name].cpu(),
                )
                for name, value in original.state().items()
            )
        )

    def test_adult_learning_is_discarded_instead_of_inherited(self) -> None:
        config = tiny_config()
        candidate = initialize_generation_zero(config)[0]
        simulation, base_brain = build_evolution_components(
            config,
            candidate.genome,
            simulation_seed=101,
        )
        runner = AutonomousSimulationRunner(simulation, base_brain)
        inherited = runner.brain_for(1).parameter_snapshot()
        result = runner.step()
        self.assertTrue(result.learning_results)
        adult = runner.brain_for(1).parameter_snapshot()
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(inherited, adult))
        )

        fresh_simulation, fresh_base = build_evolution_components(
            config,
            candidate.genome,
            simulation_seed=101,
        )
        fresh = AutonomousSimulationRunner(fresh_simulation, fresh_base)
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(
                    inherited,
                    fresh.brain_for(1).parameter_snapshot(),
                    strict=True,
                )
            )
        )

    def test_mutation_is_reproducible_and_preserves_tensor_metadata(self) -> None:
        genome = initialize_generation_zero(tiny_config())[0].genome
        first = mutate_genome(genome, sigma=0.02, seed=123)
        second = mutate_genome(genome, sigma=0.02, seed=123)
        third = mutate_genome(genome, sigma=0.02, seed=124)
        self.assertTrue(first.exactly_equals(second))
        self.assertFalse(first.exactly_equals(third))
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(genome.tensors(), first.tensors(), strict=True)
            )
        )
        for before, after in zip(genome.tensors(), first.tensors(), strict=True):
            self.assertEqual(before.shape, after.shape)
            self.assertEqual(before.dtype, after.dtype)
            self.assertEqual(before.device, after.device)

    def test_generation_zero_is_independently_diverse(self) -> None:
        candidates = initialize_generation_zero(tiny_config())
        self.assertTrue(
            any(
                not candidates[0].genome.exactly_equals(candidate.genome)
                for candidate in candidates[1:]
            )
        )
        self.assertTrue(all(value.source.startswith("independent-seed:") for value in candidates))


class EvolutionSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tiny_config()
        self.candidates = initialize_generation_zero(self.config)
        self.evaluations = tuple(
            CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                world_seeds=self.config.training_world_seeds,
                lifetimes=(3 - index, 3 - index),
                boundary_death_count=0,
                energy_death_count=0,
                natural_death_count=0,
                consumed_energy=0.0,
            )
            for index, candidate in enumerate(self.candidates)
        )

    def test_elite_is_copied_exactly_and_children_are_mutated(self) -> None:
        next_generation = create_next_generation(
            0,
            self.candidates,
            self.evaluations,
            self.config,
        )
        self.assertTrue(next_generation[0].elite_copied)
        self.assertEqual(next_generation[0].parent_id, self.candidates[0].candidate_id)
        self.assertTrue(
            next_generation[0].genome.exactly_equals(self.candidates[0].genome)
        )
        self.assertTrue(
            all(value.mutation_sigma == self.config.mutation_sigma for value in next_generation[1:])
        )

    def test_all_candidates_receive_the_same_training_world_seeds(self) -> None:
        evaluations = evaluate_generation(
            0,
            self.candidates,
            self.config,
            progress=None,
        )
        self.assertEqual(
            {evaluation.world_seeds for evaluation in evaluations},
            {self.config.training_world_seeds},
        )

    def test_validation_run_cannot_change_selection_result(self) -> None:
        before = create_next_generation(
            0,
            self.candidates,
            self.evaluations,
            self.config,
        )
        evaluate_candidate(
            self.candidates[0],
            self.config,
            world_seeds=self.config.validation_world_seeds,
        )
        after = create_next_generation(
            0,
            self.candidates,
            self.evaluations,
            self.config,
        )
        self.assertEqual(
            tuple(value.parent_id for value in before),
            tuple(value.parent_id for value in after),
        )
        self.assertTrue(
            all(
                left.genome.exactly_equals(right.genome)
                for left, right in zip(before, after, strict=True)
            )
        )

    def test_fitness_is_exact_mean_observed_lifetime(self) -> None:
        evaluation = CandidateEvaluation(
            candidate_id="candidate",
            world_seeds=(1,),
            lifetimes=(1, 3, 5),
            boundary_death_count=1,
            energy_death_count=1,
            natural_death_count=0,
            consumed_energy=0.0,
        )
        self.assertEqual(evaluation.fitness, 3.0)
        self.assertEqual(evaluation.median_lifetime, 3.0)


if __name__ == "__main__":
    unittest.main()
