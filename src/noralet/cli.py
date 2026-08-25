"""Dependency-free headless command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
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

    if args.command == "research":
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
            conditions=tuple(LearningCondition(value) for value in args.conditions),
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
        print(f"Research outputs: {result_directory}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")
