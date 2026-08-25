"""Small observer-only audit of saved Evolution Bootstrap v1 genomes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import torch

from noralet.evolution.config import EvolutionConfig
from noralet.evolution.engine import load_champion
from noralet.evolution.evaluation import (
    CandidateEvaluation,
    EvolutionCandidate,
    evaluate_candidate,
)
from noralet.evolution.genome import BaseBrainGenome


AUDIT_ID = "002-evolution-generalization-audit"
AUDIT_SCHEMA_VERSION = "1.0"
AUDIT_SEED = 20_260_825
AUDIT_SEED_DOMAIN = (
    b"project-noralet:research-002:evolution-generalization-audit:seed:v1\0"
)
QUALITATIVE_PROBE_SEEDS = (94_476,)
REQUIRED_CHECKPOINTS = (
    "generation-000.pt",
    "generation-005.pt",
    "generation-010.pt",
    "generation-014.pt",
    "best.pt",
)

GENOME_SUMMARY_COLUMNS = (
    "checkpoint_label",
    "checkpoint_path",
    "checkpoint_sha256",
    "candidate_id",
    "source_generation",
    "world_count",
    "total_individuals",
    "mean_observed_lifetime",
    "median_observed_lifetime",
    "world_mean_stddev",
    "world_mean_minimum",
    "world_mean_maximum",
    "world_mean_spread",
    "boundary_death_count",
    "boundary_death_fraction",
    "energy_depletion_death_count",
    "energy_depletion_death_fraction",
    "natural_death_count",
    "natural_death_fraction",
    "total_consumed_energy",
    "mean_consumed_energy",
    "survivor_count_at_max_ticks",
)

WORLD_RESULT_COLUMNS = (
    "checkpoint_label",
    "candidate_id",
    "source_generation",
    "world_seed",
    "noralets",
    "max_ticks",
    "world_ticks_executed",
    "mean_observed_lifetime",
    "median_observed_lifetime",
    "minimum_observed_lifetime",
    "maximum_observed_lifetime",
    "boundary_death_count",
    "boundary_death_fraction",
    "energy_depletion_death_count",
    "energy_depletion_death_fraction",
    "natural_death_count",
    "natural_death_fraction",
    "total_consumed_energy",
    "mean_consumed_energy",
    "survivor_count_at_max_ticks",
)


@dataclass(frozen=True, slots=True)
class EvolutionAuditConfig:
    evolution_result: Path
    output_root: Path = Path("research-results")
    audit_seed: int = AUDIT_SEED
    unseen_world_count: int = 8
    benchmark_world_count: int = 4
    noralets_per_world: int = 4
    generalization_max_ticks: int = 1_000
    performance_max_ticks: int = 250
    timing_repetitions: int = 3
    timing_warmups: int = 1
    generalization_device: str = "auto"
    performance_devices: tuple[str, ...] = ("cpu", "cuda")

    def __post_init__(self) -> None:
        object.__setattr__(self, "evolution_result", Path(self.evolution_result))
        object.__setattr__(self, "output_root", Path(self.output_root))
        for name in (
            "unseen_world_count",
            "benchmark_world_count",
            "noralets_per_world",
            "generalization_max_ticks",
            "performance_max_ticks",
            "timing_repetitions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.timing_warmups) is not int or self.timing_warmups < 0:
            raise ValueError("timing_warmups must be a non-negative integer")
        if type(self.audit_seed) is not int:
            raise TypeError("audit_seed must be an integer")
        device = self.generalization_device.strip().lower()
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("generalization_device must be cpu, cuda, or auto")
        object.__setattr__(self, "generalization_device", device)
        if (
            not isinstance(self.performance_devices, tuple)
            or not self.performance_devices
            or any(value not in ("cpu", "cuda") for value in self.performance_devices)
            or len(set(self.performance_devices)) != len(self.performance_devices)
        ):
            raise ValueError("performance_devices must be unique cpu/cuda values")


def derive_audit_world_seeds(
    audit_seed: int,
    *,
    role: str,
    count: int,
    excluded: Sequence[int] = (),
) -> tuple[int, ...]:
    """Derive a deterministic audit bank without hand-picked observations."""

    if type(audit_seed) is not int:
        raise TypeError("audit_seed must be an integer")
    if not isinstance(role, str) or not role:
        raise TypeError("role must be a non-empty string")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    excluded_set = set(excluded)
    seeds: list[int] = []
    index = 0
    while len(seeds) < count:
        digest = hashlib.sha256()
        digest.update(AUDIT_SEED_DOMAIN)
        digest.update(str(audit_seed).encode("ascii"))
        digest.update(b"\0")
        digest.update(role.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(index).encode("ascii"))
        value = int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)
        index += 1
        if value in excluded_set or value in seeds:
            continue
        seeds.append(value)
    return tuple(seeds)


def locate_audit_checkpoints(evolution_result: Path) -> tuple[Path, ...]:
    result = Path(evolution_result).resolve()
    champion_directory = result / "champion"
    paths = tuple(champion_directory / name for name in REQUIRED_CHECKPOINTS)
    missing = tuple(path.name for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "required evolution checkpoints are missing: " + ", ".join(missing)
        )
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the audit but is unavailable")
    return requested


def _audit_evolution_config(
    checkpoint_metadata: dict[str, Any],
    *,
    device: str,
    noralets_per_world: int,
    max_ticks: int,
) -> EvolutionConfig:
    saved = EvolutionConfig.from_state(checkpoint_metadata["configuration"])
    return replace(
        saved,
        device=device,
        noralets_per_world=noralets_per_world,
        max_ticks=max_ticks,
    )


def _candidate(
    label: str,
    genome: BaseBrainGenome,
    metadata: dict[str, Any],
) -> EvolutionCandidate:
    return EvolutionCandidate(
        candidate_id=f"audit-{label}-{metadata['candidate_id']}",
        genome=genome,
        parent_id=None,
        source="saved-inherited-checkpoint",
        elite_copied=False,
        mutation_sigma=0.0,
    )


def _world_row(
    *,
    checkpoint_label: str,
    metadata: dict[str, Any],
    world_seed: int,
    evaluation: CandidateEvaluation,
    max_ticks: int,
) -> dict[str, Any]:
    individuals = evaluation.total_individuals
    deaths = (
        evaluation.boundary_death_count
        + evaluation.energy_death_count
        + evaluation.natural_death_count
    )
    return {
        "checkpoint_label": checkpoint_label,
        "candidate_id": metadata["candidate_id"],
        "source_generation": metadata["generation"],
        "world_seed": world_seed,
        "noralets": individuals,
        "max_ticks": max_ticks,
        "world_ticks_executed": max(evaluation.lifetimes),
        "mean_observed_lifetime": evaluation.fitness,
        "median_observed_lifetime": evaluation.median_lifetime,
        "minimum_observed_lifetime": min(evaluation.lifetimes),
        "maximum_observed_lifetime": max(evaluation.lifetimes),
        "boundary_death_count": evaluation.boundary_death_count,
        "boundary_death_fraction": evaluation.boundary_death_count / individuals,
        "energy_depletion_death_count": evaluation.energy_death_count,
        "energy_depletion_death_fraction": (
            evaluation.energy_death_count / individuals
        ),
        "natural_death_count": evaluation.natural_death_count,
        "natural_death_fraction": evaluation.natural_death_count / individuals,
        "total_consumed_energy": evaluation.consumed_energy,
        "mean_consumed_energy": evaluation.consumed_energy / individuals,
        "survivor_count_at_max_ticks": individuals - deaths,
    }


def evaluate_saved_checkpoint(
    checkpoint_path: Path,
    *,
    world_seeds: tuple[int, ...],
    device: str,
    noralets_per_world: int,
    max_ticks: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one immutable genome as independent fresh lives per world."""

    path = Path(checkpoint_path).resolve()
    genome, metadata = load_champion(path)
    label = path.stem
    config = _audit_evolution_config(
        metadata,
        device=device,
        noralets_per_world=noralets_per_world,
        max_ticks=max_ticks,
    )
    candidate = _candidate(label, genome, metadata)
    world_rows: list[dict[str, Any]] = []
    evaluations: list[CandidateEvaluation] = []
    for index, seed in enumerate(world_seeds, start=1):
        if progress is not None:
            progress(f"{label} world [{index}/{len(world_seeds)}] seed {seed}")
        evaluation = evaluate_candidate(candidate, config, world_seeds=(seed,))
        evaluations.append(evaluation)
        world_rows.append(
            _world_row(
                checkpoint_label=label,
                metadata=metadata,
                world_seed=seed,
                evaluation=evaluation,
                max_ticks=max_ticks,
            )
        )

    lifetimes = tuple(
        lifetime
        for evaluation in evaluations
        for lifetime in evaluation.lifetimes
    )
    world_means = tuple(row["mean_observed_lifetime"] for row in world_rows)
    boundary = sum(value.boundary_death_count for value in evaluations)
    energy = sum(value.energy_death_count for value in evaluations)
    natural = sum(value.natural_death_count for value in evaluations)
    consumed = math.fsum(value.consumed_energy for value in evaluations)
    total = len(lifetimes)
    summary = {
        "checkpoint_label": label,
        "checkpoint_path": str(path),
        "checkpoint_sha256": _file_sha256(path),
        "candidate_id": metadata["candidate_id"],
        "source_generation": metadata["generation"],
        "world_count": len(world_seeds),
        "total_individuals": total,
        "mean_observed_lifetime": math.fsum(lifetimes) / total,
        "median_observed_lifetime": float(statistics.median(lifetimes)),
        "world_mean_stddev": float(statistics.pstdev(world_means)),
        "world_mean_minimum": min(world_means),
        "world_mean_maximum": max(world_means),
        "world_mean_spread": max(world_means) - min(world_means),
        "boundary_death_count": boundary,
        "boundary_death_fraction": boundary / total,
        "energy_depletion_death_count": energy,
        "energy_depletion_death_fraction": energy / total,
        "natural_death_count": natural,
        "natural_death_fraction": natural / total,
        "total_consumed_energy": consumed,
        "mean_consumed_energy": consumed / total,
        "survivor_count_at_max_ticks": total - boundary - energy - natural,
    }
    identity = {
        "checkpoint_label": label,
        "path": str(path),
        "sha256": summary["checkpoint_sha256"],
        "candidate_id": metadata["candidate_id"],
        "source_generation": metadata["generation"],
        "training_fitness": metadata["training_fitness"],
        "validation_fitness_at_selection_generation": metadata[
            "validation_fitness_at_selection_generation"
        ],
    }
    return summary, world_rows, identity


def benchmark_saved_checkpoint(
    checkpoint_path: Path,
    *,
    world_seeds: tuple[int, ...],
    device: str,
    noralets_per_world: int,
    max_ticks: int,
    repetitions: int,
    warmups: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Measure total wall time around an identical fixed micro-workload."""

    resolved = _resolved_device(device)
    genome, metadata = load_champion(Path(checkpoint_path))
    config = _audit_evolution_config(
        metadata,
        device=resolved,
        noralets_per_world=noralets_per_world,
        max_ticks=max_ticks,
    )
    candidate = _candidate(Path(checkpoint_path).stem, genome, metadata)
    measurements: list[dict[str, Any]] = []
    for index in range(warmups + repetitions):
        measured = index >= warmups
        label = "measurement" if measured else "warm-up"
        ordinal = index - warmups + 1 if measured else index + 1
        if progress is not None:
            progress(f"Performance {resolved} {label} {ordinal}")
        if resolved == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        evaluation = evaluate_candidate(
            candidate,
            config,
            world_seeds=world_seeds,
        )
        if resolved == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if measured:
            world_ticks = sum(
                max(evaluation.lifetimes[offset : offset + noralets_per_world])
                for offset in range(
                    0,
                    len(evaluation.lifetimes),
                    noralets_per_world,
                )
            )
            activations = sum(evaluation.lifetimes)
            measurements.append(
                {
                    "repetition": ordinal,
                    "wall_clock_seconds": elapsed,
                    "world_ticks_executed": world_ticks,
                    "noralet_activations_or_lived_transitions": activations,
                    "effective_world_ticks_per_second": world_ticks / elapsed,
                }
            )
    seconds = tuple(value["wall_clock_seconds"] for value in measurements)
    ticks_per_second = tuple(
        value["effective_world_ticks_per_second"] for value in measurements
    )
    return {
        "device": resolved,
        "warmup_repetitions": warmups,
        "measured_repetitions": repetitions,
        "measurements": measurements,
        "mean_wall_clock_seconds": math.fsum(seconds) / len(seconds),
        "median_wall_clock_seconds": float(statistics.median(seconds)),
        "mean_effective_world_ticks_per_second": (
            math.fsum(ticks_per_second) / len(ticks_per_second)
        ),
        "world_ticks_executed_per_repetition": measurements[0][
            "world_ticks_executed"
        ],
        "noralet_activations_or_lived_transitions_per_repetition": measurements[
            0
        ]["noralet_activations_or_lived_transitions"],
    }


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
    status = _git_value("status", "--porcelain")
    cuda_available = torch.cuda.is_available()
    return {
        "git": {
            "commit_sha": _git_value("rev-parse", "HEAD"),
            "worktree_dirty": None if status is None else bool(status),
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
    }


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _decision_gate(
    summaries: list[dict[str, Any]],
    performance: dict[str, Any],
) -> dict[str, str]:
    by_label = {value["checkpoint_label"]: value for value in summaries}
    early = by_label["generation-005"]
    later = by_label["generation-014"]
    if (
        later["mean_observed_lifetime"] > early["mean_observed_lifetime"]
        and later["median_observed_lifetime"] >= early["median_observed_lifetime"]
    ):
        generalization = "later genomes clearly better"
        specialization = "weakened"
    elif (
        early["mean_observed_lifetime"] > later["mean_observed_lifetime"]
        and early["median_observed_lifetime"] >= later["median_observed_lifetime"]
    ):
        generalization = "earlier genome better"
        specialization = "strengthened"
    else:
        generalization = "unclear"
        specialization = "unresolved"
    speedup = performance.get("cuda_speedup_vs_cpu")
    if speedup is None:
        cuda = "roughly neutral"
    elif speedup >= 1.2:
        cuda = "clearly beneficial"
    elif speedup <= 0.8:
        cuda = "slower"
    else:
        cuda = "roughly neutral"
    return {
        "generalization_signal": generalization,
        "fixed_world_specialization_concern": specialization,
        "cuda_effectiveness": cuda,
    }


def _write_summary(
    path: Path,
    summaries: list[dict[str, Any]],
    performance: dict[str, Any],
    gate: dict[str, str],
) -> None:
    lines = [
        "# Fast Evolution Generalization + Performance Audit",
        "",
        "## Generalization",
        "",
        "| Genome | unseen mean | median | boundary % | energy % | consumed Energy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['checkpoint_label']} | "
            f"{row['mean_observed_lifetime']:.3f} | "
            f"{row['median_observed_lifetime']:.3f} | "
            f"{100.0 * row['boundary_death_fraction']:.1f} | "
            f"{100.0 * row['energy_depletion_death_fraction']:.1f} | "
            f"{row['mean_consumed_energy']:.3f} |"
        )
    lines.extend(("", "## World variance", ""))
    for row in summaries:
        lines.append(
            f"- {row['checkpoint_label']}: world-mean SD "
            f"{row['world_mean_stddev']:.3f}; range "
            f"{row['world_mean_minimum']:.3f}–"
            f"{row['world_mean_maximum']:.3f}."
        )
    cpu = performance.get("devices", {}).get("cpu")
    cuda = performance.get("devices", {}).get("cuda")
    lines.extend(("", "## CPU vs CUDA", ""))
    lines.append(
        f"- CPU mean runtime: {cpu['mean_wall_clock_seconds']:.6f} s"
        if cpu is not None
        else "- CPU mean runtime: not measured"
    )
    lines.append(
        f"- CUDA mean runtime: {cuda['mean_wall_clock_seconds']:.6f} s"
        if cuda is not None
        else "- CUDA mean runtime: not measured"
    )
    speedup = performance.get("cuda_speedup_vs_cpu")
    lines.append(
        f"- CUDA speedup: {speedup:.3f}×"
        if speedup is not None
        else "- CUDA speedup: not available"
    )
    best = max(summaries, key=lambda value: value["mean_observed_lifetime"])
    early = next(
        value for value in summaries if value["checkpoint_label"] == "generation-005"
    )
    later = next(
        value for value in summaries if value["checkpoint_label"] == "generation-014"
    )
    lines.extend(
        (
            "",
            "## Observations",
            "",
            f"- Highest unseen mean: {best['checkpoint_label']} "
            f"({best['mean_observed_lifetime']:.3f}).",
            f"- generation-005 unseen mean versus generation-014: "
            f"{early['mean_observed_lifetime']:.3f} versus "
            f"{later['mean_observed_lifetime']:.3f}.",
            "- Wall-clock timing is non-deterministic; GPU utilization was not "
            "inferred from it.",
            "- No mutation, selection, continued evolution, or adult-weight "
            "inheritance occurred.",
            "",
            "## Decision gate",
            "",
            "Generalization signal:",
            f"- {gate['generalization_signal']}",
            "",
            "Fixed-world specialization concern:",
            f"- {gate['fixed_world_specialization_concern']}",
            "",
            "CUDA effectiveness:",
            f"- {gate['cuda_effectiveness']}",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evolution_audit(
    config: EvolutionAuditConfig,
    *,
    cli_arguments: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Run the fixed fast audit without changing any saved scientific state."""

    if not isinstance(config, EvolutionAuditConfig):
        raise TypeError("config must be an EvolutionAuditConfig")
    checkpoints = locate_audit_checkpoints(config.evolution_result)
    source_manifest_path = config.evolution_result.resolve() / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text("utf-8"))
    if source_manifest.get("status") != "completed":
        raise ValueError("evolution audit requires a completed source result")
    known_seeds = tuple(source_manifest["training_world_seeds"]) + tuple(
        source_manifest["validation_world_seeds"]
    )
    excluded = known_seeds + QUALITATIVE_PROBE_SEEDS
    unseen_seeds = derive_audit_world_seeds(
        config.audit_seed,
        role="unseen-generalization",
        count=config.unseen_world_count,
        excluded=excluded,
    )
    benchmark_seeds = derive_audit_world_seeds(
        config.audit_seed,
        role="cpu-cuda-throughput",
        count=config.benchmark_world_count,
        excluded=excluded + unseen_seeds,
    )
    generalization_device = _resolved_device(config.generalization_device)
    started_at = datetime.now(UTC)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "source": str(config.evolution_result.resolve()),
                "audit_seed": config.audit_seed,
                "unseen": unseen_seeds,
                "benchmark": benchmark_seeds,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{fingerprint}"
    destination = config.output_root / AUDIT_ID / run_id
    destination.mkdir(parents=True, exist_ok=False)
    if progress is not None:
        progress(f"Audit result directory: {destination.resolve()}")

    summaries: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        if progress is not None:
            progress(f"Generalization genome [{index}/{len(checkpoints)}] {checkpoint.name}")
        summary, rows, identity = evaluate_saved_checkpoint(
            checkpoint,
            world_seeds=unseen_seeds,
            device=generalization_device,
            noralets_per_world=config.noralets_per_world,
            max_ticks=config.generalization_max_ticks,
            progress=progress,
        )
        summaries.append(summary)
        world_rows.extend(rows)
        identities.append(identity)

    benchmark_checkpoint = checkpoints[1]
    device_results: dict[str, Any] = {}
    for device in config.performance_devices:
        device_results[device] = benchmark_saved_checkpoint(
            benchmark_checkpoint,
            world_seeds=benchmark_seeds,
            device=device,
            noralets_per_world=config.noralets_per_world,
            max_ticks=config.performance_max_ticks,
            repetitions=config.timing_repetitions,
            warmups=config.timing_warmups,
            progress=progress,
        )
    cpu = device_results.get("cpu")
    cuda = device_results.get("cuda")
    performance = {
        "wall_clock_timings_are_nondeterministic": True,
        "gpu_utilization_measured": False,
        "coarse_phase_timing": (
            "skipped: existing runner combines activation, predictive learning, "
            "homeostatic update, and world orchestration inside one causal step"
        ),
        "checkpoint_path": str(benchmark_checkpoint.resolve()),
        "checkpoint_label": benchmark_checkpoint.stem,
        "world_seeds": list(benchmark_seeds),
        "world_count": len(benchmark_seeds),
        "noralets_per_world": config.noralets_per_world,
        "max_ticks": config.performance_max_ticks,
        "devices": device_results,
        "cuda_speedup_vs_cpu": (
            None
            if cpu is None or cuda is None
            else cpu["mean_wall_clock_seconds"] / cuda["mean_wall_clock_seconds"]
        ),
    }
    gate = _decision_gate(summaries, performance)
    manifest = {
        "audit_id": AUDIT_ID,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed",
        "created_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "result_directory": str(destination.resolve()),
        "source_evolution_result": str(config.evolution_result.resolve()),
        "source_evolution_run_id": source_manifest["run_id"],
        "cli_arguments": list(cli_arguments) if cli_arguments is not None else None,
        **_provenance(),
        "audit_seed": config.audit_seed,
        "audit_seed_domain": AUDIT_SEED_DOMAIN.rstrip(b"\0").decode("utf-8"),
        "known_training_world_seeds": source_manifest["training_world_seeds"],
        "known_validation_world_seeds": source_manifest["validation_world_seeds"],
        "excluded_qualitative_probe_seeds": list(QUALITATIVE_PROBE_SEEDS),
        "unseen_world_seeds": list(unseen_seeds),
        "benchmark_world_seeds": list(benchmark_seeds),
        "checkpoint_identities": identities,
        "generalization_protocol": {
            "genomes": len(checkpoints),
            "worlds_per_genome": len(unseen_seeds),
            "noralets_per_world": config.noralets_per_world,
            "max_ticks": config.generalization_max_ticks,
            "device": generalization_device,
            "learning_mode": "full-current-brain",
            "fresh_life_per_world": True,
            "adult_learned_weights_disposable": True,
            "selection": False,
            "mutation": False,
            "continued_evolution": False,
        },
        "performance_protocol": {
            "checkpoint": benchmark_checkpoint.name,
            "worlds": len(benchmark_seeds),
            "noralets_per_world": config.noralets_per_world,
            "max_ticks": config.performance_max_ticks,
            "warmups": config.timing_warmups,
            "measured_repetitions_per_device": config.timing_repetitions,
            "devices": list(config.performance_devices),
        },
        "decision_rules": {
            "generalization": (
                "generation-014 mean must exceed generation-005 and its median "
                "must be no lower; inverse comparison yields earlier-genome signal"
            ),
            "cuda": "speedup >=1.2 beneficial; <=0.8 slower; otherwise neutral",
        },
        "decision_gate": gate,
    }
    _write_csv(destination / "genome-summary.csv", GENOME_SUMMARY_COLUMNS, summaries)
    _write_csv(destination / "world-results.csv", WORLD_RESULT_COLUMNS, world_rows)
    (destination / "performance.json").write_text(
        json.dumps(performance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary(destination / "summary.md", summaries, performance, gate)
    return destination.resolve()
