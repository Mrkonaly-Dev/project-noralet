"""Headless mutation-only Evolution Bootstrap v1 execution and persistence."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

import torch

from noralet.evolution.config import (
    EVOLUTION_ID,
    EVOLUTION_SCHEMA_VERSION,
    EvolutionConfig,
)
from noralet.evolution.evaluation import (
    CandidateEvaluation,
    EvolutionCandidate,
    evaluate_candidate,
    evaluate_generation,
)
from noralet.evolution.genome import BaseBrainGenome
from noralet.evolution.selection import (
    create_next_generation,
    initialize_generation_zero,
    ranked_candidates,
)


GENERATION_COLUMNS = (
    "generation",
    "best_training_fitness",
    "mean_training_fitness",
    "median_training_fitness",
    "worst_training_fitness",
    "best_candidate_id",
    "validation_fitness",
    "boundary_death_fraction",
    "energy_depletion_death_fraction",
    "natural_death_fraction",
)

CANDIDATE_COLUMNS = (
    "generation",
    "candidate_id",
    "parent_id",
    "source",
    "elite_copied",
    "mutation_sigma",
    "training_fitness",
    "validation_fitness",
    "mean_lifetime",
    "median_lifetime",
    "boundary_death_count",
    "energy_death_count",
    "natural_death_count",
    "total_individuals_evaluated",
    "mean_consumed_energy",
)


def _git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _provenance() -> dict[str, Any]:
    dirty = _git_value("status", "--porcelain")
    cuda_available = torch.cuda.is_available()
    return {
        "git_commit_sha": _git_value("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
    }


def _run_identifier(config: EvolutionConfig, created_at: datetime) -> str:
    encoded = json.dumps(
        config.scientific_configuration(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:10]
    return f"{created_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{fingerprint}"


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _serialize_candidate(candidate: EvolutionCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "genome": candidate.genome.state(),
        "parent_id": candidate.parent_id,
        "source": candidate.source,
        "elite_copied": candidate.elite_copied,
        "mutation_sigma": candidate.mutation_sigma,
    }


def _deserialize_candidate(state: dict[str, Any]) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=state["candidate_id"],
        genome=BaseBrainGenome.from_state(state["genome"]),
        parent_id=state["parent_id"],
        source=state["source"],
        elite_copied=state["elite_copied"],
        mutation_sigma=state["mutation_sigma"],
    )


def _champion_payload(
    candidate: EvolutionCandidate,
    *,
    generation: int,
    training_fitness: float,
    validation_fitness: float,
    config: EvolutionConfig,
) -> dict[str, Any]:
    return {
        "evolution_id": EVOLUTION_ID,
        "schema_version": EVOLUTION_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "generation": generation,
        "training_fitness": training_fitness,
        "validation_fitness_at_selection_generation": validation_fitness,
        "genome": candidate.genome.state(),
        "configuration": config.state(),
        "learning_mode": "full-current-brain",
        "initial_body_energy": config.initial_body_energy,
    }


def save_champion(
    path: Path,
    candidate: EvolutionCandidate,
    *,
    generation: int,
    training_fitness: float,
    validation_fitness: float,
    config: EvolutionConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _champion_payload(
            candidate,
            generation=generation,
            training_fitness=training_fitness,
            validation_fitness=validation_fitness,
            config=config,
        ),
        path,
    )


def load_champion(
    path: Path,
) -> tuple[BaseBrainGenome, dict[str, Any]]:
    """Load a generated best.pt or result directory with safe tensor-only mode."""

    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "champion" / "best.pt"
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if payload.get("evolution_id") != EVOLUTION_ID:
        raise ValueError("checkpoint is not an Evolution Bootstrap v1 champion")
    genome = BaseBrainGenome.from_state(payload["genome"])
    metadata = {key: value for key, value in payload.items() if key != "genome"}
    metadata["checkpoint_path"] = str(resolved.resolve())
    return genome, metadata


def _write_summary(
    destination: Path,
    config: EvolutionConfig,
    generation_rows: list[dict[str, Any]],
    best_state: dict[str, Any] | None,
) -> None:
    completed = len(generation_rows)
    lines = [
        "# Evolution Bootstrap v1 Summary",
        "",
        f"- Evolution ID: `{EVOLUTION_ID}`",
        f"- Result directory: `{destination}`",
        f"- Generations completed: {completed} / {config.generation_count}",
        f"- Population size: {config.population_size}",
        f"- Training worlds per candidate: {len(config.training_world_seeds)}",
        f"- Validation worlds: {len(config.validation_world_seeds)}",
        f"- Noralets per world: {config.noralets_per_world}",
        f"- Maximum ticks: {config.max_ticks}",
        f"- Initial stored Energy: {config.initial_body_energy:g} eU",
        "- Fitness: mean observed lifetime ticks truncated at the evaluation cap",
        "- Interpretation: viability proxy for evolutionary bootstrap",
        "- Lifetime learning: predictive and homeostatic learning enabled",
        "",
        "## Progression",
        "",
        "| Generation | Best train | Mean train | Validation |",
        "|---:|---:|---:|---:|",
    ]
    for row in generation_rows:
        lines.append(
            f"| {row['generation']} | {row['best_training_fitness']:.6g} | "
            f"{row['mean_training_fitness']:.6g} | "
            f"{row['validation_fitness']:.6g} |"
        )
    lines.extend(("", "## Champion", ""))
    if best_state is None:
        lines.append("No completed generation yet.")
    else:
        lines.extend(
            (
                f"- Candidate: `{best_state['candidate']['candidate_id']}`",
                f"- Generation: {best_state['generation']}",
                f"- Training fitness: {best_state['training_fitness']:.6g}",
                f"- Validation fitness at selection generation: "
                f"{best_state['validation_fitness']:.6g}",
            )
        )
    lines.extend(
        (
            "",
            "## Caveats",
            "",
            "Observed lifetimes at the cap are truncated/right-censored. Validation "
            "never affects selection. This bootstrap records a viability proxy, not "
            "true reproductive fitness or a consciousness measure.",
            "",
        )
    )
    (destination / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    destination: Path,
    generation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    config: EvolutionConfig,
    best_state: dict[str, Any] | None,
) -> None:
    _write_csv(destination / "generations.csv", GENERATION_COLUMNS, generation_rows)
    _write_csv(destination / "candidates.csv", CANDIDATE_COLUMNS, candidate_rows)
    _write_summary(destination, config, generation_rows, best_state)


def _checkpoint_state(
    *,
    config: EvolutionConfig,
    next_generation: int,
    population: tuple[EvolutionCandidate, ...],
    generation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    best_state: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "evolution_id": EVOLUTION_ID,
        "schema_version": EVOLUTION_SCHEMA_VERSION,
        "configuration": config.state(),
        "next_generation": next_generation,
        "population": [_serialize_candidate(value) for value in population],
        "generation_rows": generation_rows,
        "candidate_rows": candidate_rows,
        "best_state": best_state,
        "rng_state": {
            "scheme": "stateless SHA-256 domain-separated seeds",
            "initial_seed": config.initial_seed,
            "continuation_requires_no ambient RNG state": True,
        },
    }


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "evolution-state.pt"
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    if state.get("evolution_id") != EVOLUTION_ID:
        raise ValueError("checkpoint is not Evolution Bootstrap v1 state")
    state["checkpoint_path"] = resolved.resolve()
    return state


def _initial_manifest(
    config: EvolutionConfig,
    *,
    destination: Path,
    created_at: datetime,
    cli_arguments: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "evolution_id": EVOLUTION_ID,
        "schema_version": EVOLUTION_SCHEMA_VERSION,
        "run_id": destination.name,
        "created_at_utc": created_at.isoformat(),
        "completed_at_utc": None,
        "status": "running",
        "result_directory": str(destination),
        "cli_arguments": list(cli_arguments) if cli_arguments is not None else None,
        "full_evolution_configuration": config.state(),
        **_provenance(),
        **config.scientific_configuration(),
        "seed_derivation": {
            "algorithm": "SHA-256",
            "domain": "project-noralet:evolution-001:seed:v1",
            "uses_python_hash": False,
            "roles": [
                "generation-0 independent BaseBrain initialization",
                "parent selection",
                "per-child Gaussian mutation",
            ],
        },
        "inheritance_rules": {
            "genome": "all named BaseBrain prototype parameters",
            "adult_learned_state_inherited": False,
            "target_encoder_evolved_independently": False,
            "death_reward_or_terminal_update": False,
            "respawn": False,
        },
    }


def _run_loop(
    *,
    destination: Path,
    config: EvolutionConfig,
    manifest: dict[str, Any],
    next_generation: int,
    population: tuple[EvolutionCandidate, ...],
    generation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    best_state: dict[str, Any] | None,
    progress: Callable[[str], None] | None,
) -> Path:
    if next_generation > config.generation_count:
        raise ValueError("resume target precedes the checkpoint generation")
    champion_directory = destination / "champion"
    champion_directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"Evolution directory: {destination}")

    for generation in range(next_generation, config.generation_count):
        evaluations = evaluate_generation(
            generation,
            population,
            config,
            progress=progress,
        )
        ranked = ranked_candidates(population, evaluations)
        generation_best, best_evaluation = ranked[0]

        # Selection is completed from training metrics before validation runs.
        next_population = create_next_generation(
            generation,
            population,
            evaluations,
            config,
        )
        validation = evaluate_candidate(
            generation_best,
            config,
            world_seeds=config.validation_world_seeds,
        )

        validation_by_id = {generation_best.candidate_id: validation.fitness}
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
                    "training_fitness": evaluation.fitness,
                    "validation_fitness": validation_by_id.get(candidate.candidate_id),
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

        fitnesses = [value.fitness for value in evaluations]
        total_individuals = sum(value.total_individuals for value in evaluations)
        generation_rows.append(
            {
                "generation": generation,
                "best_training_fitness": max(fitnesses),
                "mean_training_fitness": math.fsum(fitnesses) / len(fitnesses),
                "median_training_fitness": float(statistics.median(fitnesses)),
                "worst_training_fitness": min(fitnesses),
                "best_candidate_id": generation_best.candidate_id,
                "validation_fitness": validation.fitness,
                "boundary_death_fraction": (
                    sum(value.boundary_death_count for value in evaluations)
                    / total_individuals
                ),
                "energy_depletion_death_fraction": (
                    sum(value.energy_death_count for value in evaluations)
                    / total_individuals
                ),
                "natural_death_fraction": (
                    sum(value.natural_death_count for value in evaluations)
                    / total_individuals
                ),
            }
        )

        if (
            best_state is None
            or best_evaluation.fitness > best_state["training_fitness"]
        ):
            best_state = {
                "candidate": _serialize_candidate(generation_best),
                "generation": generation,
                "training_fitness": best_evaluation.fitness,
                "validation_fitness": validation.fitness,
            }
        assert best_state is not None
        overall_best = _deserialize_candidate(best_state["candidate"])
        save_champion(
            champion_directory / "best.pt",
            overall_best,
            generation=best_state["generation"],
            training_fitness=best_state["training_fitness"],
            validation_fitness=best_state["validation_fitness"],
            config=config,
        )
        if (
            generation == 0
            or generation % config.champion_checkpoint_interval == 0
            or generation == config.generation_count - 1
        ):
            save_champion(
                champion_directory / f"generation-{generation:03d}.pt",
                overall_best,
                generation=best_state["generation"],
                training_fitness=best_state["training_fitness"],
                validation_fitness=best_state["validation_fitness"],
                config=config,
            )

        population = next_population
        checkpoint = _checkpoint_state(
            config=config,
            next_generation=generation + 1,
            population=population,
            generation_rows=generation_rows,
            candidate_rows=candidate_rows,
            best_state=best_state,
        )
        _save_checkpoint(destination / "evolution-state.pt", checkpoint)
        _write_outputs(
            destination,
            generation_rows,
            candidate_rows,
            config,
            best_state,
        )
        manifest["generations_completed"] = generation + 1
        manifest["generation_count_target"] = config.generation_count
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if progress is not None:
            progress(
                f"Generation {generation} complete: best training fitness "
                f"{best_evaluation.fitness:.6g}; validation "
                f"{validation.fitness:.6g}"
            )

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    manifest["generations_completed"] = len(generation_rows)
    manifest["generation_count_target"] = config.generation_count
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_outputs(destination, generation_rows, candidate_rows, config, best_state)
    return destination


def run_evolution(
    config: EvolutionConfig,
    *,
    cli_arguments: Sequence[str] | None = None,
    run_directory: Path | None = None,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Start a fresh deterministic evolution run."""

    if not isinstance(config, EvolutionConfig):
        raise TypeError("config must be an EvolutionConfig")
    created_at = datetime.now(UTC)
    destination = (
        Path(run_directory)
        if run_directory is not None
        else config.output_root / EVOLUTION_ID / _run_identifier(config, created_at)
    )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"evolution result directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = _initial_manifest(
        config,
        destination=destination,
        created_at=created_at,
        cli_arguments=cli_arguments,
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return _run_loop(
        destination=destination,
        config=config,
        manifest=manifest,
        next_generation=0,
        population=initialize_generation_zero(config),
        generation_rows=[],
        candidate_rows=[],
        best_state=None,
        progress=progress,
    )


def resume_evolution(
    checkpoint_path: Path,
    *,
    generation_count: int,
    device: str | None = None,
    cli_arguments: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Continue from the exact saved next population using stateless RNG seeds."""

    state = _load_checkpoint(checkpoint_path)
    saved_config = EvolutionConfig.from_state(state["configuration"])
    config = saved_config.with_resume_target(
        generation_count=generation_count,
        device=device,
    )
    next_generation = int(state["next_generation"])
    if generation_count < next_generation:
        raise ValueError(
            f"generation target {generation_count} precedes completed generation "
            f"count {next_generation}"
        )
    destination = Path(state["checkpoint_path"]).parent
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["completed_at_utc"] = None
    manifest["generation_count_target"] = generation_count
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
    return _run_loop(
        destination=destination,
        config=config,
        manifest=manifest,
        next_generation=next_generation,
        population=tuple(
            _deserialize_candidate(value) for value in state["population"]
        ),
        generation_rows=list(state["generation_rows"]),
        candidate_rows=list(state["candidate_rows"]),
        best_state=state["best_state"],
        progress=progress,
    )
