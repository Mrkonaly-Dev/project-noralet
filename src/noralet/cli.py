"""Headless-first command-line interface with a lazily loaded desktop UI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from noralet.simulation import Simulation, SimulationConfig


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


_RESEARCH_CONDITIONS = (
    "no-learning",
    "predictive-only",
    "full-current-brain",
    "homeostatic-only",
)


def _conditions(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = tuple(item for item in parsed if item not in _RESEARCH_CONDITIONS)
    if not parsed or invalid or len(parsed) != len(set(parsed)):
        allowed = ", ".join(_RESEARCH_CONDITIONS)
        raise argparse.ArgumentTypeError(
            f"must be a unique comma-separated subset of: {allowed}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the Project Noralet command-line parser."""

    parser = argparse.ArgumentParser(prog="noralet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a headless simulation")
    run_parser.add_argument(
        "--ticks",
        required=True,
        type=_non_negative_int,
        help="number of ticks to execute",
    )
    run_parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="explicit master random seed",
    )

    subparsers.add_parser("ui", help="open the Renderer / Observer desktop UI")

    research_parser = subparsers.add_parser(
        "research",
        help="run a headless research protocol",
    )
    research_subparsers = research_parser.add_subparsers(
        dest="research_experiment",
        required=True,
    )
    baseline = research_subparsers.add_parser(
        "baseline-lifetime-adaptation",
        help="run Research 001's controlled lifetime-learning batch",
    )
    baseline.add_argument(
        "--seeds",
        type=_positive_int,
        default=10,
        help="number of deterministic replicate seeds (minimum: 2; default: 10)",
    )
    baseline.add_argument(
        "--max-ticks",
        type=_positive_int,
        default=5_000,
        help="per-run stopping tick (default: 5000)",
    )
    baseline.add_argument(
        "--sample-every",
        type=_positive_int,
        default=10,
        help="timeseries cadence in ticks (default: 10)",
    )
    baseline.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cuda",
        help="PyTorch execution device (default: cuda)",
    )
    baseline.add_argument(
        "--population",
        type=_positive_int,
        default=6,
        help="initial Noralet count per run (default: 6)",
    )
    baseline.add_argument(
        "--conditions",
        type=_conditions,
        default=_RESEARCH_CONDITIONS,
        metavar="CONDITION,...",
        help="comma-separated condition subset (default: all four)",
    )
    baseline.add_argument(
        "--output-root",
        type=Path,
        default=Path("research-results"),
        help="generated-result root (default: research-results)",
    )
    audit = research_subparsers.add_parser(
        "evolution-audit",
        help="audit saved evolved genomes on unseen worlds and CPU/CUDA timing",
    )
    audit.add_argument(
        "--evolution-result",
        type=Path,
        required=True,
        help="completed Evolution Bootstrap v1 result directory",
    )
    audit.add_argument(
        "--audit-seed",
        type=int,
        default=20_260_825,
        help="deterministic audit seed (default: 20260825)",
    )
    audit.add_argument(
        "--generalization-device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="device for unseen-world evaluation (default: auto)",
    )
    audit.add_argument(
        "--output-root",
        type=Path,
        default=Path("research-results"),
        help="generated-result root (default: research-results)",
    )
    initialization_audit = research_subparsers.add_parser(
        "basebrain-initialization-audit",
        help="audit neutral action priors of fresh BaseBrains without worlds",
    )
    initialization_audit.add_argument(
        "--samples",
        type=_positive_int,
        default=100,
        help="number of independently initialized BaseBrains (default: 100)",
    )
    initialization_audit.add_argument(
        "--seed",
        type=int,
        default=1,
        help="deterministic audit seed (default: 1)",
    )
    initialization_audit.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="activation device (default: cpu)",
    )

    evolution_parser = subparsers.add_parser(
        "evolution",
        help="evolve inherited BaseBrain initializations",
    )
    evolution_subparsers = evolution_parser.add_subparsers(
        dest="evolution_protocol",
        required=True,
    )
    bootstrap = evolution_subparsers.add_parser(
        "basebrain-bootstrap",
        help="run mutation-only Evolution Bootstrap v1",
    )
    bootstrap.add_argument("--generations", type=_positive_int, default=50)
    bootstrap.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=None,
        help="PyTorch device (new-run default: cuda; resume default: saved device)",
    )
    bootstrap.add_argument("--resume", type=Path, default=None)
    bootstrap.add_argument(
        "--output-root",
        type=Path,
        default=Path("evolution-results"),
    )
    bootstrap.add_argument("--population-size", type=_positive_int, default=32)
    bootstrap.add_argument("--elite-count", type=_positive_int, default=None)
    bootstrap.add_argument("--parent-pool-size", type=_positive_int, default=None)
    bootstrap.add_argument("--mutation-sigma", type=_positive_float, default=0.02)
    bootstrap.add_argument("--training-worlds", type=_positive_int, default=4)
    bootstrap.add_argument("--validation-worlds", type=_positive_int, default=4)
    bootstrap.add_argument("--noralets-per-world", type=_positive_int, default=6)
    bootstrap.add_argument("--max-ticks", type=_positive_int, default=2_000)
    bootstrap.add_argument("--initial-energy", type=_positive_float, default=10.0)
    bootstrap.add_argument("--seed", type=int, default=1)
    bootstrap.add_argument("--checkpoint-every", type=_positive_int, default=5)

    distributional = evolution_subparsers.add_parser(
        "distributional",
        help="run Distributional Evolution v2",
    )
    distributional.add_argument("--generations", type=_positive_int, default=20)
    distributional.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=None,
        help="PyTorch device (new-run default: cpu; resume default: saved device)",
    )
    source = distributional.add_mutually_exclusive_group()
    source.add_argument("--fork-from", type=Path, default=None)
    source.add_argument("--resume", type=Path, default=None)
    distributional.add_argument(
        "--output-root",
        type=Path,
        default=Path("evolution-results"),
    )
    distributional.add_argument("--population-size", type=_positive_int, default=8)
    distributional.add_argument("--elite-count", type=_positive_int, default=2)
    distributional.add_argument("--parent-pool-size", type=_positive_int, default=4)
    distributional.add_argument("--mutation-sigma", type=_positive_float, default=0.02)
    distributional.add_argument("--selection-worlds", type=_positive_int, default=4)
    distributional.add_argument("--benchmark-worlds", type=_positive_int, default=8)
    distributional.add_argument("--benchmark-every", type=_positive_int, default=5)
    distributional.add_argument("--noralets-per-world", type=_positive_int, default=4)
    distributional.add_argument("--max-ticks", type=_positive_int, default=1_000)
    distributional.add_argument("--initial-energy", type=_positive_float, default=10.0)
    distributional.add_argument("--seed", type=int, default=1)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested headless command and return its exit status."""

    parser = build_parser()
    effective_argv = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    args = parser.parse_args(effective_argv)

    if args.command == "run":
        simulation = Simulation(SimulationConfig(master_seed=args.seed))
        for _ in range(args.ticks):
            simulation.step()

        print(
            f"Completed {args.ticks} tick(s); "
            f"final tick: {simulation.state.tick}; seed: {args.seed}"
        )
        return 0

    if args.command == "ui":
        from noralet.ui.app import run_ui

        return run_ui(["noralet"])

    if args.command == "research":
        if args.research_experiment == "basebrain-initialization-audit":
            from noralet.research.initialization_audit import (
                run_initialization_audit,
            )

            result = run_initialization_audit(
                sample_count=args.samples,
                audit_seed=args.seed,
                device=args.device,
            )
            print(json.dumps(result.state(), indent=2, sort_keys=True))
            return 0
        if args.research_experiment == "baseline-lifetime-adaptation":
            from noralet.research import (
                BaselineExperimentConfig,
                LearningCondition,
                ResearchBatchExecutionError,
                run_baseline_experiment,
            )

            if args.seeds < 2:
                parser.error("--seeds must be at least 2 for this protocol")
            config = BaselineExperimentConfig(
                replicate_seeds=tuple(range(1, args.seeds + 1)),
                max_ticks=args.max_ticks,
                sample_every_ticks=args.sample_every,
                initial_population=args.population,
                device=args.device,
                conditions=tuple(
                    LearningCondition(value) for value in args.conditions
                ),
                output_root=args.output_root,
            )
            try:
                result_directory = run_baseline_experiment(
                    config,
                    cli_arguments=effective_argv,
                )
            except ResearchBatchExecutionError as error:
                print(str(error), file=sys.stderr)
                return 1
        elif args.research_experiment == "evolution-audit":
            from noralet.research.evolution_audit import (
                EvolutionAuditConfig,
                run_evolution_audit,
            )

            config = EvolutionAuditConfig(
                evolution_result=args.evolution_result,
                output_root=args.output_root,
                audit_seed=args.audit_seed,
                generalization_device=args.generalization_device,
            )
            try:
                result_directory = run_evolution_audit(
                    config,
                    cli_arguments=effective_argv,
                    progress=lambda line: print(line, flush=True),
                )
            except Exception as error:
                print(
                    f"Evolution audit failed: {type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                return 1
        else:
            raise AssertionError(
                f"Unhandled research experiment: {args.research_experiment}"
            )
        print(f"Research outputs: {result_directory}")
        return 0

    if args.command == "evolution":
        if args.evolution_protocol == "distributional":
            from noralet.evolution.distributional import (
                DistributionalEvolutionConfig,
                resume_distributional_evolution,
                run_distributional_evolution,
            )

            try:
                if args.resume is not None:
                    result_directory = resume_distributional_evolution(
                        args.resume,
                        generation_count=args.generations,
                        device=args.device,
                        cli_arguments=effective_argv,
                    )
                else:
                    config = DistributionalEvolutionConfig(
                        generation_count=args.generations,
                        device="cpu" if args.device is None else args.device,
                        population_size=args.population_size,
                        elite_count=args.elite_count,
                        parent_pool_size=args.parent_pool_size,
                        mutation_sigma=args.mutation_sigma,
                        selection_world_count=args.selection_worlds,
                        benchmark_world_count=args.benchmark_worlds,
                        benchmark_interval=args.benchmark_every,
                        noralets_per_world=args.noralets_per_world,
                        max_ticks=args.max_ticks,
                        initial_body_energy=args.initial_energy,
                        initial_seed=args.seed,
                        output_root=args.output_root,
                    )
                    result_directory = run_distributional_evolution(
                        config,
                        fork_from=args.fork_from,
                        cli_arguments=effective_argv,
                    )
            except Exception as error:
                print(
                    f"Distributional evolution failed: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                return 1
            print(f"Evolution outputs: {result_directory}")
            return 0

        from noralet.evolution import (
            EvolutionConfig,
            fixed_world_seeds,
            resume_evolution,
            run_evolution,
        )

        try:
            if args.resume is not None:
                result_directory = resume_evolution(
                    args.resume,
                    generation_count=args.generations,
                    device=args.device,
                    cli_arguments=effective_argv,
                )
            else:
                elite_count = (
                    min(4, max(1, args.population_size - 1))
                    if args.elite_count is None
                    else args.elite_count
                )
                parent_pool_size = (
                    min(8, args.population_size)
                    if args.parent_pool_size is None
                    else args.parent_pool_size
                )
                config = EvolutionConfig(
                    generation_count=args.generations,
                    device="cuda" if args.device is None else args.device,
                    population_size=args.population_size,
                    elite_count=elite_count,
                    parent_pool_size=parent_pool_size,
                    mutation_sigma=args.mutation_sigma,
                    training_world_seeds=fixed_world_seeds(
                        "training",
                        args.training_worlds,
                    ),
                    validation_world_seeds=fixed_world_seeds(
                        "validation",
                        args.validation_worlds,
                    ),
                    noralets_per_world=args.noralets_per_world,
                    max_ticks=args.max_ticks,
                    initial_body_energy=args.initial_energy,
                    initial_seed=args.seed,
                    champion_checkpoint_interval=args.checkpoint_every,
                    output_root=args.output_root,
                )
                result_directory = run_evolution(
                    config,
                    cli_arguments=effective_argv,
                )
        except Exception as error:
            print(f"Evolution failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(f"Evolution outputs: {result_directory}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")
