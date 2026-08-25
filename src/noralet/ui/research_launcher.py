"""Validated command construction for the existing Research 001 CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from noralet.research.config import (
    EXPERIMENT_ID,
    PROTOCOL_CONDITIONS,
    LearningCondition,
)


@dataclass(frozen=True, slots=True)
class ResearchLaunchSetup:
    seeds: int = 10
    maximum_ticks: int = 5_000
    sample_every_ticks: int = 10
    population: int = 6
    device: str = "cuda"
    conditions: tuple[LearningCondition, ...] = PROTOCOL_CONDITIONS

    def __post_init__(self) -> None:
        for name in (
            "seeds",
            "maximum_ticks",
            "sample_every_ticks",
            "population",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seeds < 2:
            raise ValueError("Research 001 requires at least two replicate seeds")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        device = self.device.strip().lower()
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if not isinstance(self.conditions, tuple) or not self.conditions:
            raise TypeError("conditions must be a non-empty tuple")
        if not all(
            isinstance(condition, LearningCondition)
            for condition in self.conditions
        ):
            raise TypeError("every condition must be a LearningCondition")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be unique")
        canonical = tuple(
            condition
            for condition in PROTOCOL_CONDITIONS
            if condition in self.conditions
        )
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "conditions", canonical)


@dataclass(frozen=True, slots=True)
class ProcessInvocation:
    program: str
    arguments: tuple[str, ...]
    working_directory: Path


def build_research_invocation(
    setup: ResearchLaunchSetup,
    *,
    python_executable: str | None = None,
    working_directory: Path | None = None,
) -> ProcessInvocation:
    """Construct the authoritative existing CLI call without executing it."""

    if not isinstance(setup, ResearchLaunchSetup):
        raise TypeError("setup must be a ResearchLaunchSetup")
    executable = sys.executable if python_executable is None else python_executable
    if not isinstance(executable, str) or not executable:
        raise ValueError("python_executable must be a non-empty string")
    directory = Path.cwd() if working_directory is None else Path(working_directory)
    return ProcessInvocation(
        program=executable,
        arguments=(
            "-u",
            "-m",
            "noralet",
            "research",
            "baseline-lifetime-adaptation",
            "--seeds",
            str(setup.seeds),
            "--max-ticks",
            str(setup.maximum_ticks),
            "--sample-every",
            str(setup.sample_every_ticks),
            "--population",
            str(setup.population),
            "--device",
            setup.device,
            "--conditions",
            ",".join(condition.value for condition in setup.conditions),
        ),
        working_directory=directory,
    )


def result_directory_from_line(line: str, working_directory: Path) -> Path | None:
    prefix = "Research outputs:"
    stripped = line.strip()
    if not stripped.startswith(prefix):
        return None
    raw_path = stripped[len(prefix) :].strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(working_directory) / path
    return path.resolve()


def latest_partial_result_directory(
    working_directory: Path,
    *,
    started_after: float,
) -> Path | None:
    root = Path(working_directory) / "research-results" / EXPERIMENT_ID
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.stat().st_mtime >= started_after - 1.0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)
