"""Validated Evolution Bootstrap CLI construction for the Qt launcher."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch

from noralet.evolution.config import EVOLUTION_ID
from noralet.ui.research_launcher import ProcessInvocation


PILOT_PRESET_VALUES = {
    "generations": 5,
    "population_size": 8,
    "elite_count": 2,
    "parent_pool_size": 4,
    "training_worlds": 2,
    "validation_worlds": 2,
    "noralets_per_world": 4,
    "maximum_ticks": 1_000,
    "mutation_sigma": 0.02,
    "initial_energy": 10.0,
}

STANDARD_PRESET_VALUES = {
    "generations": 50,
    "population_size": 32,
    "elite_count": 4,
    "parent_pool_size": 8,
    "training_worlds": 4,
    "validation_worlds": 4,
    "noralets_per_world": 6,
    "maximum_ticks": 2_000,
    "mutation_sigma": 0.02,
    "initial_energy": 10.0,
}


@dataclass(frozen=True, slots=True)
class EvolutionLaunchSetup:
    generations: int = 50
    device: str = "cuda"
    population_size: int = 32
    elite_count: int = 4
    parent_pool_size: int = 8
    mutation_sigma: float = 0.02
    training_worlds: int = 4
    validation_worlds: int = 4
    noralets_per_world: int = 6
    maximum_ticks: int = 2_000
    initial_energy: float = 10.0
    initial_seed: int = 1
    output_root: Path = Path("evolution-results")

    def __post_init__(self) -> None:
        for name in (
            "generations",
            "population_size",
            "elite_count",
            "parent_pool_size",
            "training_worlds",
            "validation_worlds",
            "noralets_per_world",
            "maximum_ticks",
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
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        device = self.device.strip().lower()
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        for name in ("mutation_sigma", "initial_energy"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "output_root", Path(self.output_root))


@dataclass(frozen=True, slots=True)
class EvolutionWorkloadEstimate:
    training_lives_per_generation: int
    validation_lives_per_generation: int
    maximum_training_ticks_per_generation: int


@dataclass(frozen=True, slots=True)
class EvolutionResumeMetadata:
    checkpoint_path: Path
    result_directory: Path
    run_id: str
    completed_generations: int
    original_generation_target: int
    population_size: int
    elite_count: int
    parent_pool_size: int
    mutation_sigma: float
    training_worlds: int
    validation_worlds: int
    noralets_per_world: int
    maximum_ticks: int
    initial_energy: float
    initial_seed: int
    device: str
    best_candidate_id: str | None
    best_generation: int | None
    best_training_fitness: float | None
    best_validation_fitness: float | None


def estimate_evolution_workload(
    setup: EvolutionLaunchSetup,
) -> EvolutionWorkloadEstimate:
    """Calculate observer-only maximum counts; never enter evolution logic."""

    if not isinstance(setup, EvolutionLaunchSetup):
        raise TypeError("setup must be an EvolutionLaunchSetup")
    training_lives = (
        setup.population_size
        * setup.training_worlds
        * setup.noralets_per_world
    )
    validation_lives = setup.validation_worlds * setup.noralets_per_world
    return EvolutionWorkloadEstimate(
        training_lives_per_generation=training_lives,
        validation_lives_per_generation=validation_lives,
        maximum_training_ticks_per_generation=(
            training_lives * setup.maximum_ticks
        ),
    )


def load_evolution_resume_metadata(
    checkpoint_path: Path,
) -> EvolutionResumeMetadata:
    """Read locked checkpoint metadata without performing any resume work."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"evolution checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("evolution_id") != EVOLUTION_ID:
        raise ValueError("checkpoint is not Evolution Bootstrap v1 state")
    configuration = state.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint configuration is missing")
    completed = state.get("next_generation")
    if type(completed) is not int or completed < 0:
        raise ValueError("checkpoint completed-generation count is invalid")
    best = state.get("best_state")
    best_candidate_id = None
    best_generation = None
    best_training = None
    best_validation = None
    if best is not None:
        candidate = best.get("candidate", {})
        best_candidate_id = candidate.get("candidate_id")
        best_generation = best.get("generation")
        best_training = best.get("training_fitness")
        best_validation = best.get("validation_fitness")
    return EvolutionResumeMetadata(
        checkpoint_path=path,
        result_directory=path.parent,
        run_id=path.parent.name,
        completed_generations=completed,
        original_generation_target=int(configuration["generation_count"]),
        population_size=int(configuration["population_size"]),
        elite_count=int(configuration["elite_count"]),
        parent_pool_size=int(configuration["parent_pool_size"]),
        mutation_sigma=float(configuration["mutation_sigma"]),
        training_worlds=len(configuration["training_world_seeds"]),
        validation_worlds=len(configuration["validation_world_seeds"]),
        noralets_per_world=int(configuration["noralets_per_world"]),
        maximum_ticks=int(configuration["max_ticks"]),
        initial_energy=float(configuration["initial_body_energy"]),
        initial_seed=int(configuration["initial_seed"]),
        device=str(configuration["device"]),
        best_candidate_id=best_candidate_id,
        best_generation=best_generation,
        best_training_fitness=(
            None if best_training is None else float(best_training)
        ),
        best_validation_fitness=(
            None if best_validation is None else float(best_validation)
        ),
    )


def build_evolution_resume_invocation(
    metadata: EvolutionResumeMetadata,
    *,
    target_generation: int,
    device_override: str | None = None,
    python_executable: str | None = None,
    working_directory: Path | None = None,
) -> ProcessInvocation:
    """Construct only the generation/device overrides supported by resume."""

    if not isinstance(metadata, EvolutionResumeMetadata):
        raise TypeError("metadata must be EvolutionResumeMetadata")
    if type(target_generation) is not int:
        raise TypeError("target_generation must be an integer")
    if target_generation <= metadata.completed_generations:
        raise ValueError(
            "continue-to generation must exceed completed generations "
            f"({metadata.completed_generations})"
        )
    if device_override is not None:
        if not isinstance(device_override, str):
            raise TypeError("device_override must be a string or None")
        device_override = device_override.strip().lower()
        if device_override not in ("cpu", "cuda", "auto"):
            raise ValueError("device_override must be cpu, cuda, auto, or None")
    executable = sys.executable if python_executable is None else python_executable
    directory = Path.cwd() if working_directory is None else Path(working_directory)
    arguments = [
        "-u",
        "-m",
        "noralet",
        "evolution",
        "basebrain-bootstrap",
        "--generations",
        str(target_generation),
        "--resume",
        str(metadata.checkpoint_path),
    ]
    if device_override is not None:
        arguments.extend(("--device", device_override))
    return ProcessInvocation(
        program=executable,
        arguments=tuple(arguments),
        working_directory=directory,
    )


def build_evolution_invocation(
    setup: EvolutionLaunchSetup,
    *,
    python_executable: str | None = None,
    working_directory: Path | None = None,
) -> ProcessInvocation:
    if not isinstance(setup, EvolutionLaunchSetup):
        raise TypeError("setup must be an EvolutionLaunchSetup")
    executable = sys.executable if python_executable is None else python_executable
    directory = Path.cwd() if working_directory is None else Path(working_directory)
    return ProcessInvocation(
        program=executable,
        arguments=(
            "-u",
            "-m",
            "noralet",
            "evolution",
            "basebrain-bootstrap",
            "--generations",
            str(setup.generations),
            "--device",
            setup.device,
            "--population-size",
            str(setup.population_size),
            "--elite-count",
            str(setup.elite_count),
            "--parent-pool-size",
            str(setup.parent_pool_size),
            "--mutation-sigma",
            str(setup.mutation_sigma),
            "--training-worlds",
            str(setup.training_worlds),
            "--validation-worlds",
            str(setup.validation_worlds),
            "--noralets-per-world",
            str(setup.noralets_per_world),
            "--max-ticks",
            str(setup.maximum_ticks),
            "--initial-energy",
            str(setup.initial_energy),
            "--seed",
            str(setup.initial_seed),
            "--output-root",
            str(setup.output_root),
        ),
        working_directory=directory,
    )


def evolution_directory_from_line(
    line: str,
    working_directory: Path,
) -> Path | None:
    stripped = line.strip()
    prefixes = ("Evolution directory:", "Evolution outputs:")
    prefix = next((value for value in prefixes if stripped.startswith(value)), None)
    if prefix is None:
        return None
    raw_path = stripped[len(prefix) :].strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(working_directory) / path
    return path.resolve()
