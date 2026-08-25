"""Evolution v2 with fresh shared selection worlds and fixed benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch

from noralet.evolution.config import EVOLUTION_ID, derived_seed
from noralet.evolution.engine import (
    _deserialize_candidate,
    _fresh_population_initialization,
    _provenance,
    _save_checkpoint,
    _saved_population_initialization,
    _serialize_candidate,
    _write_csv,
)
from noralet.evolution.evaluation import (
    CandidateEvaluation,
    EvolutionCandidate,
    evaluate_candidate,
)
from noralet.evolution.genome import BaseBrainGenome, mutate_genome
from noralet.evolution.selection import ranked_candidates
from noralet.research.config import (
    LearningCondition,
    SeedMapping,
    baseline_configuration_manifest,
    build_baseline_components,
)


DISTRIBUTIONAL_EVOLUTION_ID = "002-basebrain-distributional-evolution"
DISTRIBUTIONAL_SCHEMA_VERSION = "1.0"
_SEED_DOMAIN = b"project-noralet:evolution-002:seed:v1\0"

GENERATION_COLUMNS = (
    "generation",
    "selection_world_seeds",
    "best_selection_fitness",
    "mean_selection_fitness",
    "median_selection_fitness",
    "selection_champion_id",
    "boundary_death_fraction",
    "energy_depletion_death_fraction",
    "natural_death_fraction",
    "benchmark_performed",
    "benchmark_mean_lifetime",
    "benchmark_median_lifetime",
    "benchmark_best_so_far_mean",
)

CANDIDATE_COLUMNS = (
    "generation",
    "candidate_id",
    "parent_id",
    "source",
    "elite_copied",
    "mutation_sigma",
    "selection_fitness",
    "mean_lifetime",
    "median_lifetime",
    "boundary_death_count",
    "energy_death_count",
    "natural_death_count",
    "total_individuals_evaluated",
    "mean_consumed_energy",
)

BENCHMARK_COLUMNS = (
    "generation",
    "candidate_id",
    "selection_fitness",
    "benchmark_world_seeds",
    "mean_lifetime",
    "median_lifetime",
    "world_mean_stddev",
    "boundary_death_fraction",
    "energy_depletion_death_fraction",
    "natural_death_fraction",
    "mean_consumed_energy",
    "benchmark_best_so_far",
)


def distributional_seed(initial_seed: int, *roles: object) -> int:
    if type(initial_seed) is not int:
        raise TypeError("initial_seed must be an integer")
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(str(initial_seed).encode("ascii"))
    for role in roles:
        encoded = str(role).encode("utf-8")
        digest.update(b"\0")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class DistributionalEvolutionConfig:
    generation_count: int = 20
    device: str = "cpu"
    population_size: int = 8
    elite_count: int = 2
    parent_pool_size: int = 4
    mutation_sigma: float = 0.02
    selection_world_count: int = 4
    benchmark_world_count: int = 8
    benchmark_interval: int = 5
    noralets_per_world: int = 4
    max_ticks: int = 1_000
    initial_body_energy: float = 10.0
    initial_seed: int = 1
    output_root: Path = Path("evolution-results")

    def __post_init__(self) -> None:
        for name in (
            "generation_count",
            "population_size",
            "elite_count",
            "parent_pool_size",
            "selection_world_count",
            "benchmark_world_count",
            "benchmark_interval",
            "noralets_per_world",
            "max_ticks",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.elite_count > self.parent_pool_size:
            raise ValueError("elite_count cannot exceed parent_pool_size")
        if self.parent_pool_size > self.population_size:
            raise ValueError("parent_pool_size cannot exceed population_size")
        if type(self.initial_seed) is not int:
            raise TypeError("initial_seed must be an integer")
        device = self.device.strip().lower()
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        for name in ("mutation_sigma", "initial_body_energy"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.initial_body_energy > 100.0:
            raise ValueError("initial_body_energy cannot exceed body capacity 100")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "output_root", Path(self.output_root))

    def state(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_root"] = str(self.output_root)
        return values

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> DistributionalEvolutionConfig:
        values = dict(state)
        values["output_root"] = Path(values["output_root"])
        return cls(**values)

    def with_resume_target(
        self,
        *,
        generation_count: int,
        device: str | None = None,
    ) -> DistributionalEvolutionConfig:
        return replace(
            self,
            generation_count=generation_count,
            device=self.device if device is None else device,
        )

    def scientific_configuration(self) -> dict[str, Any]:
        manifest_seed = SeedMapping(
            replicate_seed=self.initial_seed,
            simulation_seed=selection_world_seeds(self, 0)[0],
            base_brain_seed=distributional_seed(
                self.initial_seed,
                "manifest-brain",
            ),
        )
        return {
            "evolution_id": DISTRIBUTIONAL_EVOLUTION_ID,
            "schema_version": DISTRIBUTIONAL_SCHEMA_VERSION,
            "generation_count_target": self.generation_count,
            "device": self.device,
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "parent_pool_size": self.parent_pool_size,
            "mutation_sigma": self.mutation_sigma,
            "selection_worlds_per_generation": self.selection_world_count,
            "benchmark_world_count": self.benchmark_world_count,
            "benchmark_interval": self.benchmark_interval,
            "noralets_per_world": self.noralets_per_world,
            "max_ticks": self.max_ticks,
            "initial_body_energy": self.initial_body_energy,
            "initial_seed": self.initial_seed,
            "learning_mode": LearningCondition.FULL_CURRENT_BRAIN.value,
            "fitness": "mean observed lifetime in current-generation selection worlds",
            "selection_fitness_comparability": "within-generation only",
            "benchmark_policy": "selection champion on fixed observer-only bank",
            "evolution_environment_configuration": baseline_configuration_manifest(
                population=self.noralets_per_world,
                device=self.device,
                seeds=manifest_seed,
                initial_body_energy=self.initial_body_energy,
            ),
        }


def selection_world_seeds(
    config: DistributionalEvolutionConfig,
    generation: int,
) -> tuple[int, ...]:
    if not isinstance(config, DistributionalEvolutionConfig):
        raise TypeError("config must be a DistributionalEvolutionConfig")
    if type(generation) is not int or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    return tuple(
        distributional_seed(
            config.initial_seed,
            "selection-world",
            generation,
            slot,
        )
        & ~1
        for slot in range(config.selection_world_count)
    )


def fixed_benchmark_world_seeds(
    config: DistributionalEvolutionConfig,
) -> tuple[int, ...]:
    if not isinstance(config, DistributionalEvolutionConfig):
        raise TypeError("config must be a DistributionalEvolutionConfig")
    return tuple(
        distributional_seed(config.initial_seed, "benchmark-world", slot) | 1
        for slot in range(config.benchmark_world_count)
    )


def initialize_distributional_population(
    config: DistributionalEvolutionConfig,
) -> tuple[EvolutionCandidate, ...]:
    seeds = selection_world_seeds(config, 0)
    candidates: list[EvolutionCandidate] = []
    for index in range(config.population_size):
        candidate_seed = derived_seed(config.initial_seed, "generation-0", index)
        _, base_brain = build_baseline_components(
            initial_population=config.noralets_per_world,
            device=config.device,
            condition=LearningCondition.FULL_CURRENT_BRAIN,
            simulation_seed=seeds[0],
            base_brain_seed=candidate_seed,
            initial_body_energy=config.initial_body_energy,
        )
        candidates.append(
            EvolutionCandidate(
                candidate_id=f"g000-c{index:03d}",
                genome=BaseBrainGenome.from_base_brain(base_brain),
                parent_id=None,
                source=f"independent-v2-seed:{candidate_seed}",
                elite_copied=False,
                mutation_sigma=0.0,
            )
        )
    return tuple(candidates)


def create_distributional_next_generation(
    generation: int,
    candidates: tuple[EvolutionCandidate, ...],
    evaluations: tuple[CandidateEvaluation, ...],
    config: DistributionalEvolutionConfig,
) -> tuple[EvolutionCandidate, ...]:
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
        genome = mutate_genome(
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
                genome=genome,
                parent_id=parent.candidate_id,
                source="gaussian-mutation",
                elite_copied=False,
                mutation_sigma=config.mutation_sigma,
            )
        )
    return tuple(created)


def _evaluation_config(
    config: DistributionalEvolutionConfig,
    world_seeds: tuple[int, ...],
):
    from noralet.evolution.config import EvolutionConfig

    unused_validation_seed = distributional_seed(
        config.initial_seed,
        "evaluation-config-unused-validation",
    )
    while unused_validation_seed in world_seeds:
        unused_validation_seed += 1

    return EvolutionConfig(
        generation_count=config.generation_count,
        device=config.device,
        population_size=config.population_size,
        elite_count=config.elite_count,
        parent_pool_size=config.parent_pool_size,
        mutation_sigma=config.mutation_sigma,
        training_world_seeds=world_seeds,
        validation_world_seeds=(unused_validation_seed,),
        noralets_per_world=config.noralets_per_world,
        max_ticks=config.max_ticks,
        initial_body_energy=config.initial_body_energy,
        initial_seed=config.initial_seed,
        champion_checkpoint_interval=config.benchmark_interval,
        output_root=config.output_root,
    )


def evaluate_distributional_generation(
    generation: int,
    candidates: tuple[EvolutionCandidate, ...],
    config: DistributionalEvolutionConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[tuple[CandidateEvaluation, ...], tuple[int, ...]]:
    seeds = selection_world_seeds(config, generation)
    evaluation_config = _evaluation_config(config, seeds)
    evaluations: list[CandidateEvaluation] = []
    for index, candidate in enumerate(candidates, start=1):
        if progress is not None:
            progress(
                f"Generation {generation} candidate [{index}/{len(candidates)}] "
                f"{candidate.candidate_id}"
            )
        evaluations.append(
            evaluate_candidate(candidate, evaluation_config, world_seeds=seeds)
        )
    return tuple(evaluations), seeds


def _benchmark_evaluation(
    candidate: EvolutionCandidate,
    config: DistributionalEvolutionConfig,
    benchmark_seeds: tuple[int, ...],
) -> tuple[CandidateEvaluation, float]:
    evaluation_config = _evaluation_config(config, benchmark_seeds)
    evaluations = tuple(
        evaluate_candidate(candidate, evaluation_config, world_seeds=(seed,))
        for seed in benchmark_seeds
    )
    combined = CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        world_seeds=benchmark_seeds,
        lifetimes=tuple(
            lifetime for value in evaluations for lifetime in value.lifetimes
        ),
        boundary_death_count=sum(value.boundary_death_count for value in evaluations),
        energy_death_count=sum(value.energy_death_count for value in evaluations),
        natural_death_count=sum(value.natural_death_count for value in evaluations),
        consumed_energy=math.fsum(value.consumed_energy for value in evaluations),
    )
    world_means = tuple(value.fitness for value in evaluations)
    return combined, float(statistics.pstdev(world_means))


def _genome_sha256(genome: BaseBrainGenome) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(genome.state().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(
            bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())
        )
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fork_population(
    path: Path,
    config: DistributionalEvolutionConfig,
) -> tuple[tuple[EvolutionCandidate, ...], dict[str, Any]]:
    resolved = Path(path).resolve()
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    if state.get("evolution_id") != EVOLUTION_ID:
        raise ValueError("fork source must be an Evolution Bootstrap v1 checkpoint")
    source_population = tuple(
        _deserialize_candidate(value) for value in state["population"]
    )
    if len(source_population) != config.population_size:
        raise ValueError(
            "fork source population size does not match the v2 population size"
        )
    population: list[EvolutionCandidate] = []
    identities: list[dict[str, Any]] = []
    for index, source in enumerate(source_population):
        v2_id = f"g000-c{index:03d}"
        copied = BaseBrainGenome.from_state(source.genome.state())
        population.append(
            EvolutionCandidate(
                candidate_id=v2_id,
                genome=copied,
                parent_id=source.candidate_id,
                source="forked-v1-population",
                elite_copied=False,
                mutation_sigma=0.0,
            )
        )
        identities.append(
            {
                "source_candidate_id": source.candidate_id,
                "v2_candidate_id": v2_id,
                "genome_sha256": _genome_sha256(copied),
            }
        )
    return tuple(population), {
        "source_evolution_id": state["evolution_id"],
        "source_run_id": resolved.parent.name,
        "source_checkpoint_path": str(resolved),
        "source_checkpoint_sha256": _file_sha256(resolved),
        "source_completed_generation": int(state["next_generation"]),
        "source_candidate_identities": identities,
        "source_population_initialization": _saved_population_initialization(
            state
        ),
        "v2_start_generation": 0,
    }


def _champion_payload(
    candidate: EvolutionCandidate,
    config: DistributionalEvolutionConfig,
    benchmark_seeds: tuple[int, ...],
    row: dict[str, Any],
    champion_kind: str,
    population_initialization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "evolution_id": DISTRIBUTIONAL_EVOLUTION_ID,
        "schema_version": DISTRIBUTIONAL_SCHEMA_VERSION,
        "champion_kind": champion_kind,
        "candidate_id": candidate.candidate_id,
        "generation": row["generation"],
        "selection_fitness_at_benchmark_generation": row["selection_fitness"],
        "benchmark_mean_lifetime": row["mean_lifetime"],
        "benchmark_median_lifetime": row["median_lifetime"],
        "benchmark_boundary_death_fraction": row["boundary_death_fraction"],
        "benchmark_energy_depletion_death_fraction": row[
            "energy_depletion_death_fraction"
        ],
        "benchmark_natural_death_fraction": row["natural_death_fraction"],
        "benchmark_mean_consumed_energy": row["mean_consumed_energy"],
        "benchmark_world_seeds": list(benchmark_seeds),
        "genome": candidate.genome.state(),
        "configuration": config.state(),
        "learning_mode": LearningCondition.FULL_CURRENT_BRAIN.value,
        "initial_body_energy": config.initial_body_energy,
        "population_initialization": population_initialization,
    }


def _save_distributional_champion(
    path: Path,
    candidate: EvolutionCandidate,
    config: DistributionalEvolutionConfig,
    benchmark_seeds: tuple[int, ...],
    row: dict[str, Any],
    *,
    champion_kind: str,
    population_initialization: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _champion_payload(
            candidate,
            config,
            benchmark_seeds,
            row,
            champion_kind,
            population_initialization,
        ),
        path,
    )


def _benchmark_row(
    *,
    generation: int,
    candidate: EvolutionCandidate,
    selection_fitness: float,
    evaluation: CandidateEvaluation,
    world_mean_stddev: float,
    benchmark_seeds: tuple[int, ...],
) -> dict[str, Any]:
    total = evaluation.total_individuals
    return {
        "generation": generation,
        "candidate_id": candidate.candidate_id,
        "selection_fitness": selection_fitness,
        "benchmark_world_seeds": json.dumps(benchmark_seeds, separators=(",", ":")),
        "mean_lifetime": evaluation.fitness,
        "median_lifetime": evaluation.median_lifetime,
        "world_mean_stddev": world_mean_stddev,
        "boundary_death_fraction": evaluation.boundary_death_count / total,
        "energy_depletion_death_fraction": evaluation.energy_death_count / total,
        "natural_death_fraction": evaluation.natural_death_count / total,
        "mean_consumed_energy": evaluation.consumed_energy / total,
        "benchmark_best_so_far": False,
    }


def _benchmark_is_better(
    row: dict[str, Any],
    best_state: dict[str, Any] | None,
) -> bool:
    return (
        best_state is None
        or row["mean_lifetime"] > best_state["row"]["mean_lifetime"]
        or (
            row["mean_lifetime"] == best_state["row"]["mean_lifetime"]
            and row["candidate_id"] < best_state["row"]["candidate_id"]
        )
    )


def _rebuild_benchmark_best(
    benchmark_history: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best_state = None
    rows_by_key = {
        (row["generation"], row["candidate_id"]): row for row in benchmark_rows
    }
    for item in benchmark_history:
        row = item["row"]
        became_best = _benchmark_is_better(row, best_state)
        row["benchmark_best_so_far"] = became_best
        rows_by_key[(row["generation"], row["candidate_id"])][
            "benchmark_best_so_far"
        ] = became_best
        if became_best:
            best_state = item
    return best_state


def _write_summary(
    destination: Path,
    config: DistributionalEvolutionConfig,
    generation_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    best_state: dict[str, Any] | None,
) -> None:
    lines = [
        "# Distributional Evolution v2 Summary",
        "",
        f"- Evolution ID: `{DISTRIBUTIONAL_EVOLUTION_ID}`",
        f"- Generations completed: {len(generation_rows)} / {config.generation_count}",
        f"- Device: {config.device}",
        f"- Population: {config.population_size}",
        f"- Selection worlds per generation: {config.selection_world_count}",
        f"- Fixed benchmark worlds: {config.benchmark_world_count}",
        f"- Benchmark interval: {config.benchmark_interval}",
        "- Fitness: mean observed lifetime in current-generation selection worlds",
        "",
        "## Interpretation guardrail",
        "",
        "Selection fitness between generations is not standardized because every "
        "generation receives different selection environments. Longitudinal "
        "progress must be read from the fixed benchmark measurements below.",
        "",
        "## Fixed benchmark progression",
        "",
        "| Generation | Candidate | Mean | Median | Boundary | Energy | Natural |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark_rows:
        lines.append(
            f"| {row['generation']} | {row['candidate_id']} | "
            f"{row['mean_lifetime']:.6g} | {row['median_lifetime']:.6g} | "
            f"{row['boundary_death_fraction']:.6g} | "
            f"{row['energy_depletion_death_fraction']:.6g} | "
            f"{row['natural_death_fraction']:.6g} |"
        )
    lines.extend(("", "## Benchmark-best", ""))
    if best_state is None:
        lines.append("No benchmark has completed.")
    else:
        row = best_state["row"]
        lines.extend(
            (
                f"- Candidate: `{row['candidate_id']}`",
                f"- Source generation: {row['generation']}",
                f"- Benchmark mean lifetime: {row['mean_lifetime']:.6g}",
                f"- Benchmark median lifetime: {row['median_lifetime']:.6g}",
            )
        )
    (destination / "summary.md").write_text("\n".join(lines) + "\n", "utf-8")


def _write_outputs(
    destination: Path,
    config: DistributionalEvolutionConfig,
    generation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    best_state: dict[str, Any] | None,
) -> None:
    _write_csv(destination / "generations.csv", GENERATION_COLUMNS, generation_rows)
    _write_csv(destination / "candidates.csv", CANDIDATE_COLUMNS, candidate_rows)
    _write_csv(destination / "benchmarks.csv", BENCHMARK_COLUMNS, benchmark_rows)
    _write_summary(destination, config, generation_rows, benchmark_rows, best_state)


def _checkpoint_state(
    *,
    config: DistributionalEvolutionConfig,
    next_generation: int,
    population: tuple[EvolutionCandidate, ...],
    generation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    benchmark_history: list[dict[str, Any]],
    benchmark_best_state: dict[str, Any] | None,
    current_selection_champion: dict[str, Any] | None,
    benchmark_seeds: tuple[int, ...],
    fork_provenance: dict[str, Any] | None,
    population_initialization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "evolution_id": DISTRIBUTIONAL_EVOLUTION_ID,
        "schema_version": DISTRIBUTIONAL_SCHEMA_VERSION,
        "configuration": config.state(),
        "next_generation": next_generation,
        "population": [_serialize_candidate(value) for value in population],
        "generation_rows": generation_rows,
        "candidate_rows": candidate_rows,
        "benchmark_rows": benchmark_rows,
        "benchmark_history": benchmark_history,
        "benchmark_best_state": benchmark_best_state,
        "current_selection_champion": current_selection_champion,
        "benchmark_world_seeds": list(benchmark_seeds),
        "fork_provenance": fork_provenance,
        "population_initialization": population_initialization,
        "rng_state": {
            "scheme": "stateless SHA-256 domain-separated seeds",
            "initial_seed": config.initial_seed,
            "continuation_requires_no_ambient_rng_state": True,
        },
    }


def _manifest(
    config: DistributionalEvolutionConfig,
    destination: Path,
    created_at: datetime,
    benchmark_seeds: tuple[int, ...],
    fork_provenance: dict[str, Any] | None,
    cli_arguments: Sequence[str] | None,
    population_initialization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "evolution_id": DISTRIBUTIONAL_EVOLUTION_ID,
        "schema_version": DISTRIBUTIONAL_SCHEMA_VERSION,
        "run_id": destination.name,
        "created_at_utc": created_at.isoformat(),
        "completed_at_utc": None,
        "status": "running",
        "result_directory": str(destination),
        "cli_arguments": list(cli_arguments) if cli_arguments is not None else None,
        "full_evolution_configuration": config.state(),
        **_provenance(),
        **config.scientific_configuration(),
        "fixed_benchmark_world_seeds": list(benchmark_seeds),
        "fork_provenance": fork_provenance,
        "population_initialization": population_initialization,
        "seed_derivation": {
            "algorithm": "SHA-256",
            "domain": _SEED_DOMAIN.rstrip(b"\0").decode("utf-8"),
            "selection_seed_parity": "even",
            "benchmark_seed_parity": "odd",
            "uses_python_hash": False,
        },
        "inheritance_rules": {
            "genome": "all named BaseBrain prototype parameters",
            "adult_learned_state_inherited": False,
            "hidden_state_inherited": False,
            "optimizer_state_inherited": False,
            "eligibility_traces_inherited": False,
            "world_state_inherited": False,
        },
        "selection_rule": {
            "fitness": "mean observed lifetime on current selection worlds",
            "benchmark_affects_selection": False,
            "elite_count": config.elite_count,
            "parent_pool_size": config.parent_pool_size,
            "mutation": "deterministic additive Gaussian",
            "crossover": False,
        },
    }


def _run_identifier(
    config: DistributionalEvolutionConfig,
    created_at: datetime,
    fork_provenance: dict[str, Any] | None,
) -> str:
    encoded = json.dumps(
        {
            "config": config.scientific_configuration(),
            "fork": fork_provenance,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:10]
    return f"{created_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{fingerprint}"


def _run_loop(
    *,
    destination: Path,
    config: DistributionalEvolutionConfig,
    manifest: dict[str, Any],
    next_generation: int,
    population: tuple[EvolutionCandidate, ...],
    generation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    benchmark_history: list[dict[str, Any]],
    benchmark_best_state: dict[str, Any] | None,
    current_selection_champion: dict[str, Any] | None,
    benchmark_seeds: tuple[int, ...],
    fork_provenance: dict[str, Any] | None,
    population_initialization: dict[str, Any],
    progress: Callable[[str], None] | None,
) -> Path:
    if next_generation > config.generation_count:
        raise ValueError("resume target precedes the checkpoint generation")
    champion_directory = destination / "champion"
    champion_directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"Evolution directory: {destination}")

    for generation in range(next_generation, config.generation_count):
        evaluations, selection_seeds = evaluate_distributional_generation(
            generation,
            population,
            config,
            progress=progress,
        )
        ranked = ranked_candidates(population, evaluations)
        selection_champion, champion_evaluation = ranked[0]

        # Benchmark observation cannot affect this selection operation.
        next_population = create_distributional_next_generation(
            generation,
            population,
            evaluations,
            config,
        )

        evaluation_by_id = {value.candidate_id: value for value in evaluations}
        for candidate in population:
            evaluation = evaluation_by_id[candidate.candidate_id]
            candidate_rows.append(
                {
                    "generation": generation,
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "source": candidate.source,
                    "elite_copied": candidate.elite_copied,
                    "mutation_sigma": candidate.mutation_sigma,
                    "selection_fitness": evaluation.fitness,
                    "mean_lifetime": evaluation.fitness,
                    "median_lifetime": evaluation.median_lifetime,
                    "boundary_death_count": evaluation.boundary_death_count,
                    "energy_death_count": evaluation.energy_death_count,
                    "natural_death_count": evaluation.natural_death_count,
                    "total_individuals_evaluated": evaluation.total_individuals,
                    "mean_consumed_energy": (
                        evaluation.consumed_energy / evaluation.total_individuals
                    ),
                }
            )

        current_selection_champion = {
            "generation": generation,
            "candidate": _serialize_candidate(selection_champion),
            "selection_fitness": champion_evaluation.fitness,
            "selection_world_seeds": list(selection_seeds),
        }
        benchmark_performed = (
            generation % config.benchmark_interval == 0
            or generation == config.generation_count - 1
        )
        benchmark_row = None
        if benchmark_performed:
            if progress is not None:
                progress(
                    f"Generation {generation} benchmark champion "
                    f"{selection_champion.candidate_id}"
                )
            benchmark_evaluation, world_sd = _benchmark_evaluation(
                selection_champion,
                config,
                benchmark_seeds,
            )
            benchmark_row = _benchmark_row(
                generation=generation,
                candidate=selection_champion,
                selection_fitness=champion_evaluation.fitness,
                evaluation=benchmark_evaluation,
                world_mean_stddev=world_sd,
                benchmark_seeds=benchmark_seeds,
            )
            better = _benchmark_is_better(benchmark_row, benchmark_best_state)
            if better:
                benchmark_row["benchmark_best_so_far"] = True
                benchmark_best_state = {
                    "candidate": _serialize_candidate(selection_champion),
                    "row": dict(benchmark_row),
                }
            benchmark_rows.append(benchmark_row)
            benchmark_history.append(
                {
                    "candidate": _serialize_candidate(selection_champion),
                    "row": dict(benchmark_row),
                }
            )
            _save_distributional_champion(
                champion_directory / f"benchmark-generation-{generation:03d}.pt",
                selection_champion,
                config,
                benchmark_seeds,
                benchmark_row,
                champion_kind="benchmark-evaluated-selection-champion",
                population_initialization=population_initialization,
            )
            assert benchmark_best_state is not None
            best_candidate = _deserialize_candidate(
                benchmark_best_state["candidate"]
            )
            _save_distributional_champion(
                champion_directory / "best.pt",
                best_candidate,
                config,
                benchmark_seeds,
                benchmark_best_state["row"],
                champion_kind="benchmark-best",
                population_initialization=population_initialization,
            )

        fitnesses = tuple(value.fitness for value in evaluations)
        total = sum(value.total_individuals for value in evaluations)
        generation_rows.append(
            {
                "generation": generation,
                "selection_world_seeds": json.dumps(
                    selection_seeds,
                    separators=(",", ":"),
                ),
                "best_selection_fitness": max(fitnesses),
                "mean_selection_fitness": math.fsum(fitnesses) / len(fitnesses),
                "median_selection_fitness": float(statistics.median(fitnesses)),
                "selection_champion_id": selection_champion.candidate_id,
                "boundary_death_fraction": (
                    sum(value.boundary_death_count for value in evaluations) / total
                ),
                "energy_depletion_death_fraction": (
                    sum(value.energy_death_count for value in evaluations) / total
                ),
                "natural_death_fraction": (
                    sum(value.natural_death_count for value in evaluations) / total
                ),
                "benchmark_performed": benchmark_performed,
                "benchmark_mean_lifetime": (
                    None if benchmark_row is None else benchmark_row["mean_lifetime"]
                ),
                "benchmark_median_lifetime": (
                    None
                    if benchmark_row is None
                    else benchmark_row["median_lifetime"]
                ),
                "benchmark_best_so_far_mean": (
                    None
                    if benchmark_best_state is None
                    else benchmark_best_state["row"]["mean_lifetime"]
                ),
            }
        )

        population = next_population
        state = _checkpoint_state(
            config=config,
            next_generation=generation + 1,
            population=population,
            generation_rows=generation_rows,
            candidate_rows=candidate_rows,
            benchmark_rows=benchmark_rows,
            benchmark_history=benchmark_history,
            benchmark_best_state=benchmark_best_state,
            current_selection_champion=current_selection_champion,
            benchmark_seeds=benchmark_seeds,
            fork_provenance=fork_provenance,
            population_initialization=population_initialization,
        )
        _save_checkpoint(destination / "evolution-state.pt", state)
        _write_outputs(
            destination,
            config,
            generation_rows,
            candidate_rows,
            benchmark_rows,
            benchmark_best_state,
        )
        manifest["generations_completed"] = generation + 1
        manifest["generation_count_target"] = config.generation_count
        manifest["current_selection_champion"] = {
            "generation": generation,
            "candidate_id": selection_champion.candidate_id,
            "selection_fitness": champion_evaluation.fitness,
            "selection_world_seeds": list(selection_seeds),
        }
        manifest["benchmark_best"] = (
            None if benchmark_best_state is None else benchmark_best_state["row"]
        )
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            "utf-8",
        )
        if progress is not None:
            progress(
                f"Generation {generation} complete: best selection fitness "
                f"{champion_evaluation.fitness:.6g}; benchmark "
                f"{'performed' if benchmark_performed else 'not scheduled'}"
            )

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        "utf-8",
    )
    return destination.resolve()


def run_distributional_evolution(
    config: DistributionalEvolutionConfig,
    *,
    fork_from: Path | None = None,
    run_directory: Path | None = None,
    cli_arguments: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = print,
) -> Path:
    if not isinstance(config, DistributionalEvolutionConfig):
        raise TypeError("config must be a DistributionalEvolutionConfig")
    created_at = datetime.now(UTC)
    if fork_from is None:
        population = initialize_distributional_population(config)
        fork_provenance = None
        population_initialization = _fresh_population_initialization()
    else:
        population, fork_provenance = _fork_population(fork_from, config)
        population_initialization = {
            "population_origin": "forked-explicit-genomes",
            "source_population_initialization": fork_provenance[
                "source_population_initialization"
            ],
            "explicit_genome_parameters_override_initializer": True,
        }
    benchmark_seeds = fixed_benchmark_world_seeds(config)
    destination = (
        Path(run_directory)
        if run_directory is not None
        else config.output_root
        / DISTRIBUTIONAL_EVOLUTION_ID
        / _run_identifier(config, created_at, fork_provenance)
    )
    destination.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(
        config,
        destination,
        created_at,
        benchmark_seeds,
        fork_provenance,
        cli_arguments,
        population_initialization,
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        "utf-8",
    )
    return _run_loop(
        destination=destination,
        config=config,
        manifest=manifest,
        next_generation=0,
        population=population,
        generation_rows=[],
        candidate_rows=[],
        benchmark_rows=[],
        benchmark_history=[],
        benchmark_best_state=None,
        current_selection_champion=None,
        benchmark_seeds=benchmark_seeds,
        fork_provenance=fork_provenance,
        population_initialization=population_initialization,
        progress=progress,
    )


def resume_distributional_evolution(
    checkpoint_path: Path,
    *,
    generation_count: int,
    device: str | None = None,
    cli_arguments: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = print,
) -> Path:
    resolved = Path(checkpoint_path).resolve()
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    if state.get("evolution_id") != DISTRIBUTIONAL_EVOLUTION_ID:
        raise ValueError("checkpoint is not Distributional Evolution v2 state")
    next_generation = int(state["next_generation"])
    if generation_count <= next_generation:
        raise ValueError("resume target must exceed completed generations")
    config = DistributionalEvolutionConfig.from_state(
        state["configuration"]
    ).with_resume_target(generation_count=generation_count, device=device)
    destination = resolved.parent
    generation_rows = list(state["generation_rows"])
    benchmark_rows = list(state["benchmark_rows"])
    benchmark_history = list(state["benchmark_history"])
    benchmark_best_state = state["benchmark_best_state"]
    population_initialization = _saved_population_initialization(state)
    prior_final = next_generation - 1
    if prior_final % config.benchmark_interval != 0:
        benchmark_rows = [
            row for row in benchmark_rows if row["generation"] != prior_final
        ]
        benchmark_history = [
            item
            for item in benchmark_history
            if item["row"]["generation"] != prior_final
        ]
        benchmark_best_state = _rebuild_benchmark_best(
            benchmark_history,
            benchmark_rows,
        )
        previous_row = generation_rows[prior_final]
        previous_row["benchmark_performed"] = False
        previous_row["benchmark_mean_lifetime"] = None
        previous_row["benchmark_median_lifetime"] = None
        previous_row["benchmark_best_so_far_mean"] = (
            None
            if benchmark_best_state is None
            else benchmark_best_state["row"]["mean_lifetime"]
        )
        prior_checkpoint = (
            destination
            / "champion"
            / f"benchmark-generation-{prior_final:03d}.pt"
        )
        if prior_checkpoint.is_file():
            prior_checkpoint.unlink()
        if benchmark_best_state is not None:
            best_candidate = _deserialize_candidate(
                benchmark_best_state["candidate"]
            )
            _save_distributional_champion(
                destination / "champion" / "best.pt",
                best_candidate,
                config,
                tuple(state["benchmark_world_seeds"]),
                benchmark_best_state["row"],
                champion_kind="benchmark-best",
                population_initialization=population_initialization,
            )
    manifest = json.loads((destination / "manifest.json").read_text("utf-8"))
    manifest.setdefault("resume_history", []).append(
        {
            "resumed_at_utc": datetime.now(UTC).isoformat(),
            "from_generation": next_generation,
            "target_generation_count": generation_count,
            "device": config.device,
            "cli_arguments": (
                list(cli_arguments) if cli_arguments is not None else None
            ),
        }
    )
    manifest["status"] = "running"
    manifest["completed_at_utc"] = None
    manifest["device"] = config.device
    manifest["full_evolution_configuration"] = config.state()
    manifest.setdefault("population_initialization", population_initialization)
    return _run_loop(
        destination=destination,
        config=config,
        manifest=manifest,
        next_generation=next_generation,
        population=tuple(
            _deserialize_candidate(value) for value in state["population"]
        ),
        generation_rows=generation_rows,
        candidate_rows=list(state["candidate_rows"]),
        benchmark_rows=benchmark_rows,
        benchmark_history=benchmark_history,
        benchmark_best_state=benchmark_best_state,
        current_selection_champion=state["current_selection_champion"],
        benchmark_seeds=tuple(state["benchmark_world_seeds"]),
        fork_provenance=state["fork_provenance"],
        population_initialization=population_initialization,
        progress=progress,
    )
