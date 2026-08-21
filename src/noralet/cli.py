"""Dependency-free headless command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from noralet.simulation import Simulation, SimulationConfig


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested headless command and return its exit status."""

    args = build_parser().parse_args(argv)

    if args.command == "run":
        simulation = Simulation(SimulationConfig(master_seed=args.seed))
        for _ in range(args.ticks):
            simulation.step()

        print(
            f"Completed {args.ticks} tick(s); "
            f"final tick: {simulation.state.tick}; seed: {args.seed}"
        )
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")
