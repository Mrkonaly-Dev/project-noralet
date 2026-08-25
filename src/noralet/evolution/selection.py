"""Independent initialization, ranking, elitism and mutation-only selection."""

from __future__ import annotations

from noralet.evolution.config import EvolutionConfig, derived_seed
from noralet.evolution.evaluation import CandidateEvaluation, EvolutionCandidate
from noralet.evolution.genome import BaseBrainGenome, mutate_genome
from noralet.research.config import LearningCondition, build_baseline_components


def initialize_generation_zero(
    config: EvolutionConfig,
) -> tuple[EvolutionCandidate, ...]:
    """Create independently initialized prototypes, never one mutated founder."""

    if not isinstance(config, EvolutionConfig):
        raise TypeError("config must be an EvolutionConfig")
    candidates: list[EvolutionCandidate] = []
    for index in range(config.population_size):
        candidate_seed = derived_seed(config.initial_seed, "generation-0", index)
        _, base_brain = build_baseline_components(
            initial_population=config.noralets_per_world,
            device=config.device,
            condition=LearningCondition.FULL_CURRENT_BRAIN,
            simulation_seed=config.training_world_seeds[0],
            base_brain_seed=candidate_seed,
            initial_body_energy=config.initial_body_energy,
        )
        candidates.append(
            EvolutionCandidate(
                candidate_id=f"g000-c{index:03d}",
                genome=BaseBrainGenome.from_base_brain(base_brain),
                parent_id=None,
                source=f"independent-seed:{candidate_seed}",
                elite_copied=False,
                mutation_sigma=0.0,
            )
        )
    return tuple(candidates)


def ranked_candidates(
    candidates: tuple[EvolutionCandidate, ...],
    evaluations: tuple[CandidateEvaluation, ...],
) -> tuple[tuple[EvolutionCandidate, CandidateEvaluation], ...]:
    """Rank on training fitness only with a stable candidate-ID tie break."""

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    evaluation_ids = {evaluation.candidate_id for evaluation in evaluations}
    if evaluation_ids != set(by_id) or len(evaluations) != len(candidates):
        raise ValueError("evaluations must cover every candidate exactly once")
    pairs = tuple((by_id[value.candidate_id], value) for value in evaluations)
    return tuple(sorted(pairs, key=lambda pair: (-pair[1].fitness, pair[0].candidate_id)))


def create_next_generation(
    generation: int,
    candidates: tuple[EvolutionCandidate, ...],
    evaluations: tuple[CandidateEvaluation, ...],
    config: EvolutionConfig,
) -> tuple[EvolutionCandidate, ...]:
    """Copy elites and fill the generation with mutated parent-pool children."""

    if type(generation) is not int or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    ranked = ranked_candidates(candidates, evaluations)
    parents = tuple(candidate for candidate, _ in ranked[: config.parent_pool_size])
    next_generation = generation + 1
    created: list[EvolutionCandidate] = []
    for index in range(config.population_size):
        candidate_id = f"g{next_generation:03d}-c{index:03d}"
        if index < config.elite_count:
            parent = ranked[index][0]
            created.append(
                EvolutionCandidate(
                    candidate_id=candidate_id,
                    genome=BaseBrainGenome.from_state(parent.genome.state()),
                    parent_id=parent.candidate_id,
                    source="elite-copy",
                    elite_copied=True,
                    mutation_sigma=0.0,
                )
            )
            continue
        parent_index = derived_seed(
            config.initial_seed,
            "parent",
            next_generation,
            index,
        ) % len(parents)
        parent = parents[parent_index]
        child = mutate_genome(
            parent.genome,
            sigma=config.mutation_sigma,
            seed=derived_seed(
                config.initial_seed,
                "mutation",
                next_generation,
                index,
            ),
        )
        created.append(
            EvolutionCandidate(
                candidate_id=candidate_id,
                genome=child,
                parent_id=parent.candidate_id,
                source="gaussian-mutation",
                elite_copied=False,
                mutation_sigma=config.mutation_sigma,
            )
        )
    return tuple(created)
