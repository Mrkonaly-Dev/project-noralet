"""Headless execution harness for Research Iteration 001."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Sequence

import torch

from noralet.brain import AutonomousSimulationRunner
from noralet.research.config import (
    CONDITION_DEFINITIONS,
    EXPERIMENT_ID,
    EXPERIMENT_SCHEMA_VERSION,
    PREDEFINED_HYPOTHESES,
    BaselineExperimentConfig,
    LearningCondition,
    SeedMapping,
    build_run_components,
)
from noralet.research.metrics import RUN_SUMMARY_COLUMNS, ResearchRunObserver
from noralet.research.output import ResearchOutputWriter


class ResearchBatchExecutionError(RuntimeError):
    """Raised after a batch with technical failures has written all outputs."""

    def __init__(self, result_directory: Path, failures: tuple[str, ...]) -> None:
        self.result_directory = Path(result_directory)
        self.failures = failures
        super().__init__(
            f"{len(failures)} research run(s) failed; results: {result_directory}"
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


def _git_provenance() -> dict[str, Any]:
    status = _git_value("status", "--porcelain")
    return {
        "commit_sha": _git_value("rev-parse", "HEAD"),
        "worktree_dirty": None if status is None else bool(status),
    }


def _torch_provenance() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _run_id(config: BaselineExperimentConfig, started_at: datetime) -> str:
    scientific_json = json.dumps(
        config.scientific_configuration(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(scientific_json).hexdigest()[:10]
    return f"{started_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{fingerprint}"


def build_manifest(
    config: BaselineExperimentConfig,
    *,
    run_id: str,
    started_at: datetime,
    cli_arguments: Sequence[str] | None,
) -> dict[str, Any]:
    """Build the immutable, pre-run protocol and provenance record."""

    scientific = config.scientific_configuration()
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": started_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "status": "running",
        "completed_at_utc": None,
        "cli_arguments": list(cli_arguments) if cli_arguments is not None else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_provenance(),
        "torch": _torch_provenance(),
        "device": config.device,
        "conditions": scientific["conditions"],
        "condition_definitions": {
            condition.value: CONDITION_DEFINITIONS[condition.value]
            for condition in config.conditions
        },
        "seed_derivation": {
            "algorithm": "SHA-256",
            "domain": "project-noralet:research-001:seed:v1",
            "note": "Does not use Python hash().",
        },
        "seed_mappings": scientific["seed_mappings"],
        "replicate_seeds": scientific["replicate_seeds"],
        "max_ticks": config.max_ticks,
        "sample_every_ticks": config.sample_every_ticks,
        "initial_population": config.initial_population,
        "planned_run_count": config.total_runs,
        "baseline_configuration": scientific["baseline_configuration"],
        "predefined_hypotheses": list(PREDEFINED_HYPOTHESES),
        "measurement_protocol": {
            "prediction_loss_windows": (
                "first and last 100 successful predictive updates per individual"
            ),
            "timeseries_sampling": (
                "tick 0, each configured cadence tick, and a non-cadence final "
                "tick for surviving individuals"
            ),
            "lifetime_handling": (
                "survivors at max_ticks are right-censored; death ticks are observed"
            ),
            "memory_policy": (
                "timeseries rows stream to CSV; only bounded windows, immutable "
                "birth parameter snapshots, and scalar accumulators remain in memory"
            ),
        },
        "interpretive_limits": [
            "pilot baseline",
            "right-censored lifetimes may be present",
            "correlation does not establish mechanism",
            "limited random-seed count",
            "no intelligence or consciousness claim",
            "no automatic architecture recommendation",
        ],
    }


def _failure_row(
    config: BaselineExperimentConfig,
    condition: LearningCondition,
    seeds: SeedMapping,
    *,
    elapsed: float,
    error: BaseException,
) -> dict[str, Any]:
    row = {column: None for column in RUN_SUMMARY_COLUMNS}
    row.update(
        {
            "condition": condition.value,
            "replicate_seed": seeds.replicate_seed,
            "simulation_seed": seeds.simulation_seed,
            "base_brain_seed": seeds.base_brain_seed,
            "status": "technical-failure",
            "technical_error": f"{type(error).__name__}: {error}",
            "start_tick": 0,
            "max_ticks": config.max_ticks,
            "runtime_seconds": elapsed,
            "device": config.device,
            "initial_population": config.initial_population,
        }
    )
    return row


def run_baseline_experiment(
    config: BaselineExperimentConfig,
    *,
    cli_arguments: Sequence[str] | None = None,
    run_directory: Path | None = None,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Run all requested conditions sequentially and persist the complete batch."""

    if not isinstance(config, BaselineExperimentConfig):
        raise TypeError("config must be a BaselineExperimentConfig")
    started_at = datetime.now(UTC)
    identifier = _run_id(config, started_at)
    destination = (
        Path(run_directory)
        if run_directory is not None
        else config.output_root / EXPERIMENT_ID / identifier
    )
    writer = ResearchOutputWriter(destination)
    manifest = build_manifest(
        config,
        run_id=destination.name,
        started_at=started_at,
        cli_arguments=cli_arguments,
    )
    writer.write_manifest(manifest)
    failures: list[str] = []

    try:
        run_number = 0
        for seeds in config.seed_mappings:
            for condition in config.conditions:
                run_number += 1
                label = (
                    f"[{run_number}/{config.total_runs}] {condition.value}, "
                    f"replicate seed {seeds.replicate_seed}"
                )
                if progress is not None:
                    progress(f"Starting {label}")
                run_started = time.perf_counter()
                observer: ResearchRunObserver | None = None
                try:
                    simulation, base_brain = build_run_components(
                        config,
                        condition,
                        seeds,
                    )
                    runner = AutonomousSimulationRunner(simulation, base_brain)
                    observer = ResearchRunObserver(
                        runner,
                        condition,
                        seeds,
                        sample_every_ticks=config.sample_every_ticks,
                        timeseries_sink=writer.write_timeseries,
                    )
                    while (
                        runner.simulation.state.tick < config.max_ticks
                        and runner.brain_ids
                    ):
                        observer.observe(runner.step())
                        current_tick = runner.simulation.state.tick
                        if (
                            progress is not None
                            and current_tick % 1_000 == 0
                            and current_tick < config.max_ticks
                        ):
                            progress(
                                f"Progress {label}: tick {current_tick}/"
                                f"{config.max_ticks}"
                            )
                    elapsed = time.perf_counter() - run_started
                    writer.append_noralets(observer.finish(max_ticks=config.max_ticks))
                    writer.append_run(
                        observer.run_summary(
                            max_ticks=config.max_ticks,
                            runtime_seconds=elapsed,
                        )
                    )
                    if progress is not None:
                        progress(
                            f"Completed {label} at tick "
                            f"{runner.simulation.state.tick} in {elapsed:.3f}s"
                        )
                except Exception as error:  # record the whole planned matrix
                    elapsed = time.perf_counter() - run_started
                    failure = f"{label}: {type(error).__name__}: {error}"
                    failures.append(failure)
                    if observer is not None:
                        writer.append_noralets(
                            observer.finish(max_ticks=config.max_ticks)
                        )
                        writer.append_run(
                            observer.run_summary(
                                max_ticks=config.max_ticks,
                                runtime_seconds=elapsed,
                                status="technical-failure",
                                technical_error=f"{type(error).__name__}: {error}",
                            )
                        )
                    else:
                        writer.append_run(
                            _failure_row(
                                config,
                                condition,
                                seeds,
                                elapsed=elapsed,
                                error=error,
                            )
                        )
                    if progress is not None:
                        progress(f"Failed {failure}")

        completed_at = datetime.now(UTC)
        manifest["status"] = "completed-with-failures" if failures else "completed"
        manifest["completed_at_utc"] = completed_at.isoformat()
        manifest["completed_run_count"] = sum(
            row["status"] == "completed" for row in writer.run_rows
        )
        manifest["technical_failure_count"] = len(failures)
        manifest["technical_failures"] = failures
        writer.write_manifest(manifest)
        writer.finalize(
            max_ticks=config.max_ticks,
            conditions=config.conditions,
            manifest=manifest,
        )
    except BaseException:
        writer.close_after_failure()
        raise

    if failures:
        raise ResearchBatchExecutionError(destination, tuple(failures))
    return destination
