"""Validated Evolution Bootstrap CLI construction for the Qt launcher."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

from noralet.ui.research_launcher import ProcessInvocation


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
