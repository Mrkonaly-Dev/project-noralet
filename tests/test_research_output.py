"""Batch reproducibility, cadence and stable output-schema tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from noralet.research.baseline import run_baseline_experiment
from noralet.research.config import (
    PREDEFINED_HYPOTHESES,
    BaselineExperimentConfig,
    LearningCondition,
)
from noralet.research.metrics import (
    NORALET_SUMMARY_COLUMNS,
    RUN_SUMMARY_COLUMNS,
    TIMESERIES_COLUMNS,
)
from noralet.research.output import aggregate_results


EXPECTED_FILES = {
    "manifest.json",
    "run-summary.csv",
    "noralet-summary.csv",
    "timeseries.csv",
    "aggregate-summary.json",
    "summary.md",
}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scientific_run_rows(path: Path) -> list[dict[str, str]]:
    rows = _csv_rows(path)
    for row in rows:
        del row["runtime_seconds"]
    return rows


class ResearchOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = BaselineExperimentConfig(
            replicate_seeds=(7, 11),
            max_ticks=3,
            sample_every_ticks=2,
            initial_population=2,
            device="cpu",
            conditions=(
                LearningCondition.NO_LEARNING,
                LearningCondition.FULL_CURRENT_BRAIN,
            ),
            output_root=self.root,
        )
        self.first = run_baseline_experiment(
            self.config,
            cli_arguments=("research", "baseline-lifetime-adaptation", "--test"),
            run_directory=self.root / "first",
            progress=None,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_all_required_outputs_and_stable_schemas_are_written(self) -> None:
        self.assertEqual(
            {path.name for path in self.first.iterdir()},
            EXPECTED_FILES,
        )
        manifest = json.loads((self.first / "manifest.json").read_text("utf-8"))
        aggregate = json.loads(
            (self.first / "aggregate-summary.json").read_text("utf-8")
        )
        run_rows = _csv_rows(self.first / "run-summary.csv")
        noralet_rows = _csv_rows(self.first / "noralet-summary.csv")
        timeseries_rows = _csv_rows(self.first / "timeseries.csv")

        self.assertEqual(tuple(run_rows[0]), RUN_SUMMARY_COLUMNS)
        self.assertEqual(tuple(noralet_rows[0]), NORALET_SUMMARY_COLUMNS)
        self.assertEqual(tuple(timeseries_rows[0]), TIMESERIES_COLUMNS)
        self.assertEqual(len(run_rows), 4)
        self.assertEqual(len(noralet_rows), 8)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["planned_run_count"], 4)
        self.assertEqual(manifest["predefined_hypotheses"], list(PREDEFINED_HYPOTHESES))
        self.assertEqual(
            set(aggregate["conditions"]),
            {"no-learning", "full-current-brain"},
        )
        self.assertIn(
            "no intelligence or consciousness claim",
            (self.first / "summary.md").read_text("utf-8").lower(),
        )

    def test_cadence_and_right_censoring_are_explicit(self) -> None:
        timeseries = _csv_rows(self.first / "timeseries.csv")
        lifetimes = _csv_rows(self.first / "noralet-summary.csv")

        grouped_ticks: dict[tuple[str, str, str], set[int]] = {}
        for row in timeseries:
            key = (row["condition"], row["replicate_seed"], row["noralet_id"])
            grouped_ticks.setdefault(key, set()).add(int(row["tick"]))
            self.assertIn(row["condition"], ("no-learning", "full-current-brain"))
            self.assertNotEqual(row["physiological_condition"], "")
        self.assertTrue(grouped_ticks)
        self.assertTrue(all(ticks == {0, 2, 3} for ticks in grouped_ticks.values()))
        self.assertTrue(all(row["death_occurred"] == "False" for row in lifetimes))
        self.assertTrue(all(row["right_censored"] == "True" for row in lifetimes))
        self.assertTrue(all(row["death_cause"] == "" for row in lifetimes))

    def test_same_arguments_reproduce_all_scientific_values(self) -> None:
        second = run_baseline_experiment(
            self.config,
            cli_arguments=("research", "baseline-lifetime-adaptation", "--test"),
            run_directory=self.root / "second",
            progress=None,
        )

        self.assertEqual(
            _scientific_run_rows(self.first / "run-summary.csv"),
            _scientific_run_rows(second / "run-summary.csv"),
        )
        for filename in (
            "noralet-summary.csv",
            "timeseries.csv",
            "aggregate-summary.json",
        ):
            self.assertEqual(
                (self.first / filename).read_text("utf-8"),
                (second / filename).read_text("utf-8"),
            )
        first_manifest = json.loads(
            (self.first / "manifest.json").read_text("utf-8")
        )
        second_manifest = json.loads(
            (second / "manifest.json").read_text("utf-8")
        )
        for key in ("run_id", "created_at_utc", "started_at_utc", "completed_at_utc"):
            first_manifest.pop(key)
            second_manifest.pop(key)
        self.assertEqual(first_manifest, second_manifest)

    def test_survival_checkpoint_excludes_a_death_on_that_tick(self) -> None:
        run_rows = [
            {
                "condition": "no-learning",
                "replicate_seed": 1,
                "status": "completed",
                "extinct": True,
            }
        ]
        common = {
            "condition": "no-learning",
            "replicate_seed": 1,
            "death_cause": None,
            "mean_energy_distress": 0.0,
            "mean_condition_distress": 0.0,
            "consume_attempt_count": 0,
            "successful_consumption_count": 0,
            "total_energy_consumed": 0.0,
            "signal_emission_count": 0,
            "signal_LEFT_count": 0,
            "signal_RIGHT_count": 0,
            "total_absolute_distance_travelled": 0.0,
            "mean_requested_acceleration": 0.0,
            "mean_absolute_requested_acceleration": 0.0,
            "prediction_loss_initial_window_mean": None,
            "prediction_loss_final_window_mean": None,
            "mean_homeostatic_drive": None,
            "mean_absolute_modulation": None,
            "mean_homeostatic_update_norm": None,
            "online_encoder_parameter_drift_norm": 0.0,
            "GRU_parameter_drift_norm": 0.0,
            "predictor_parameter_drift_norm": None,
            "action_head_parameter_drift_norm": 0.0,
        }
        noralets = [
            {
                **common,
                "observed_lifetime_ticks": 1_000,
                "death_occurred": True,
                "death_cause": "natural",
            },
            {
                **common,
                "observed_lifetime_ticks": 1_000,
                "death_occurred": False,
            },
        ]

        aggregate = aggregate_results(
            run_rows,
            noralets,
            max_ticks=1_000,
            conditions=(LearningCondition.NO_LEARNING,),
        )

        self.assertEqual(
            aggregate["conditions"]["no-learning"]["survival_fractions"]["1000"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
