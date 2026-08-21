"""Tests for stable, independent deterministic random streams."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

from noralet.simulation import DeterministicRandomStreams, Simulation, SimulationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


class DeterministicRandomStreamsTests(unittest.TestCase):
    def test_same_master_seed_and_name_reproduce_the_same_sequence(self) -> None:
        first = DeterministicRandomStreams(master_seed=9876)
        second = DeterministicRandomStreams(master_seed=9876)

        first_sequence = [first.stream("world").random() for _ in range(8)]
        second_sequence = [second.stream("world").random() for _ in range(8)]

        self.assertEqual(first_sequence, second_sequence)

    def test_different_master_seeds_produce_different_sequences(self) -> None:
        first = DeterministicRandomStreams(master_seed=1)
        second = DeterministicRandomStreams(master_seed=2)

        first_sequence = [first.stream("world").random() for _ in range(8)]
        second_sequence = [second.stream("world").random() for _ in range(8)]

        self.assertNotEqual(first_sequence, second_sequence)

    def test_consuming_one_stream_does_not_advance_another(self) -> None:
        with_extra_draws = DeterministicRandomStreams(master_seed=42)
        without_extra_draws = DeterministicRandomStreams(master_seed=42)

        for _ in range(100):
            with_extra_draws.stream("world").random()

        isolated_sequence = [
            with_extra_draws.stream("mortality").random() for _ in range(8)
        ]
        reference_sequence = [
            without_extra_draws.stream("mortality").random() for _ in range(8)
        ]

        self.assertEqual(isolated_sequence, reference_sequence)

    def test_simulation_owns_streams_derived_from_its_config(self) -> None:
        first = Simulation(SimulationConfig(master_seed=321))
        second = Simulation(SimulationConfig(master_seed=321))

        first_draws = [first.random_streams.stream("world").random() for _ in range(4)]
        second_draws = [second.random_streams.stream("world").random() for _ in range(4)]

        self.assertEqual(first_draws, second_draws)

    def test_seed_derivation_is_independent_of_python_hash_randomisation(self) -> None:
        script = (
            "from noralet.simulation import DeterministicRandomStreams; "
            "r = DeterministicRandomStreams(77).stream('noralet:5:exploration'); "
            "print(','.join(repr(r.random()) for _ in range(5)))"
        )

        outputs: list[str] = []
        for hash_seed in ("1", "999"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONPATH"] = str(SOURCE_ROOT)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(completed.stdout.strip())

        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()

