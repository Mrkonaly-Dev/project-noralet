"""Smoke tests for the headless module entry point."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


class HeadlessCliTests(unittest.TestCase):
    def test_cli_runs_a_finite_number_of_ticks(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "noralet",
                "run",
                "--ticks",
                "5",
                "--seed",
                "12345",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.strip(),
            "Completed 5 tick(s); final tick: 5; seed: 12345",
        )
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()

