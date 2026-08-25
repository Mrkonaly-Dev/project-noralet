"""Fresh-life evaluation in Evolution Bootstrap Environment v1."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from collections.abc import Callable

from noralet.brain import AutonomousSimulationRunner, BaseBrain
from noralet.evolution.config import EvolutionConfig, derived_seed
from noralet.evolution.genome import BaseBrainGenome
from noralet.research.config import LearningCondition, build_baseline_components
from noralet.simulation.events import EnergyConsumed, NoraletDeathCause, NoraletDied
from noralet.simulation.runtime import Simulation


@dataclass(frozen=True, slots=True)
class EvolutionCandidate:
    candidate_id: str
    genome: BaseBrainGenome
    parent_id: str | None
    source: str
    elite_copied: bool
    mutation_sigma: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise TypeError("candidate_id must be a non-empty string")
        if not isinstance(self.genome, BaseBrainGenome):
            raise TypeError("genome must be a BaseBrainGenome")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id
        ):
            raise TypeError("parent_id must be None or a non-empty string")
        if not isinstance(self.source, str) or not self.source:
            raise TypeError("source must be a non-empty string")
        if type(self.elite_copied) is not bool:
            raise TypeError("elite_copied must be a bool")
        if isinstance(self.mutation_sigma, bool) or not isinstance(
            self.mutation_sigma,
            (int, float),
        ):
            raise TypeError("mutation_sigma must be a real number")
        sigma = float(self.mutation_sigma)
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("mutation_sigma must be finite and non-negative")
        object.__setattr__(self, "mutation_sigma", sigma)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    world_seeds: tuple[int, ...]
    lifetimes: tuple[int, ...]
    boundary_death_count: int
    energy_death_count: int
    natural_death_count: int
    consumed_energy: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise TypeError("candidate_id must be a non-empty string")
        if not isinstance(self.world_seeds, tuple) or not self.world_seeds:
            raise TypeError("world_seeds must be a non-empty tuple")
        if any(type(seed) is not int for seed in self.world_seeds):
            raise TypeError("world seeds must be integers")
        if not isinstance(self.lifetimes, tuple) or not self.lifetimes:
            raise TypeError("lifetimes must be a non-empty tuple")
        if any(type(value) is not int or value < 0 for value in self.lifetimes):
            raise ValueError("lifetimes must be non-negative integers")
        for name in (
            "boundary_death_count",
            "energy_death_count",
            "natural_death_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.boundary_death_count
            + self.energy_death_count
            + self.natural_death_count
            > len(self.lifetimes)
        ):
            raise ValueError("death counts cannot exceed evaluated individuals")
        if isinstance(self.consumed_energy, bool) or not isinstance(
            self.consumed_energy,
            (int, float),
        ):
            raise TypeError("consumed_energy must be a real number")
        consumed = float(self.consumed_energy)
        if not math.isfinite(consumed) or consumed < 0.0:
            raise ValueError("consumed_energy must be finite and non-negative")
        object.__setattr__(self, "consumed_energy", consumed)

    @property
    def fitness(self) -> float:
        return math.fsum(self.lifetimes) / len(self.lifetimes)

    @property
    def median_lifetime(self) -> float:
        return float(statistics.median(self.lifetimes))

    @property
    def total_individuals(self) -> int:
        return len(self.lifetimes)


def build_evolution_components(
    config: EvolutionConfig,
    genome: BaseBrainGenome,
    *,
    simulation_seed: int,
) -> tuple[Simulation, BaseBrain]:
    """Build the baseline laws with the explicit 10 eU evolution birth state."""

    if not isinstance(config, EvolutionConfig):
        raise TypeError("config must be an EvolutionConfig")
    if not isinstance(genome, BaseBrainGenome):
        raise TypeError("genome must be a BaseBrainGenome")
    simulation, base_brain = build_baseline_components(
        initial_population=config.noralets_per_world,
        device=config.device,
        condition=LearningCondition.FULL_CURRENT_BRAIN,
        simulation_seed=simulation_seed,
        base_brain_seed=derived_seed(config.initial_seed, "evaluation-prototype"),
        initial_body_energy=config.initial_body_energy,
    )
    genome.apply_to(base_brain)
    return simulation, base_brain


def evaluate_candidate(
    candidate: EvolutionCandidate,
    config: EvolutionConfig,
    *,
    world_seeds: tuple[int, ...],
) -> CandidateEvaluation:
    """Evaluate fresh learned lives while leaving the inherited genome untouched."""

    if not isinstance(candidate, EvolutionCandidate):
        raise TypeError("candidate must be an EvolutionCandidate")
    if not isinstance(config, EvolutionConfig):
        raise TypeError("config must be an EvolutionConfig")
    if not isinstance(world_seeds, tuple) or not world_seeds:
        raise TypeError("world_seeds must be a non-empty tuple")
    if any(type(seed) is not int for seed in world_seeds):
        raise TypeError("every world seed must be an integer")

    lifetimes: list[int] = []
    boundary_deaths = 0
    energy_deaths = 0
    natural_deaths = 0
    consumed_energy = 0.0
    for world_seed in world_seeds:
        simulation, base_brain = build_evolution_components(
            config,
            candidate.genome,
            simulation_seed=world_seed,
        )
        initial_ids = tuple(body.noralet_id for body in simulation.state.bodies)
        world_lifetimes: dict[int, int] = {}
        runner = AutonomousSimulationRunner(simulation, base_brain)
        while runner.simulation.state.tick < config.max_ticks and runner.brain_ids:
            result = runner.step()
            for event in result.tick_result.events:
                if isinstance(event, EnergyConsumed):
                    consumed_energy += event.energy_transferred
                elif isinstance(event, NoraletDied):
                    world_lifetimes[event.noralet_id] = event.tick_after
                    if event.cause is NoraletDeathCause.WORLD_BOUNDARY:
                        boundary_deaths += 1
                    elif event.cause is NoraletDeathCause.ENERGY_DEPLETION:
                        energy_deaths += 1
                    elif event.cause is NoraletDeathCause.NATURAL:
                        natural_deaths += 1
        observed_tick = runner.simulation.state.tick
        for noralet_id in initial_ids:
            lifetimes.append(world_lifetimes.get(noralet_id, observed_tick))

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        world_seeds=world_seeds,
        lifetimes=tuple(lifetimes),
        boundary_death_count=boundary_deaths,
        energy_death_count=energy_deaths,
        natural_death_count=natural_deaths,
        consumed_energy=consumed_energy,
    )


def evaluate_generation(
    generation: int,
    candidates: tuple[EvolutionCandidate, ...],
    config: EvolutionConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[CandidateEvaluation, ...]:
    """Evaluate every candidate on the exact same configured training worlds."""

    evaluations: list[CandidateEvaluation] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        if progress is not None:
            progress(
                f"Generation {generation} candidate [{index}/{total}] "
                f"{candidate.candidate_id}"
            )
        evaluations.append(
            evaluate_candidate(
                candidate,
                config,
                world_seeds=config.training_world_seeds,
            )
        )
    return tuple(evaluations)
