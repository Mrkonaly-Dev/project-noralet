"""Stable machine-readable and factual human-readable research output."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from noralet.research.config import EXPERIMENT_ID, LearningCondition
from noralet.research.metrics import (
    NORALET_SUMMARY_COLUMNS,
    RUN_SUMMARY_COLUMNS,
    TIMESERIES_COLUMNS,
)


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _numeric(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) is not None]


def aggregate_results(
    run_rows: list[dict[str, Any]],
    noralet_rows: list[dict[str, Any]],
    *,
    max_ticks: int,
    conditions: tuple[LearningCondition, ...],
) -> dict[str, Any]:
    """Return descriptive condition aggregates without inferential claims."""

    thresholds = tuple(value for value in (1_000, 2_500, 5_000) if value <= max_ticks)
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "lifetime_summary_label": "observed/censored lifetime summary",
        "survival_thresholds": list(thresholds),
        "conditions": {},
    }
    for condition in conditions:
        condition_runs = [
            row for row in run_rows if row["condition"] == condition.value
        ]
        completed_runs = [
            row for row in condition_runs if row["status"] == "completed"
        ]
        completed_keys = {
            (row["condition"], int(row["replicate_seed"]))
            for row in completed_runs
        }
        individuals = [
            row
            for row in noralet_rows
            if row["condition"] == condition.value
            and (row["condition"], int(row["replicate_seed"])) in completed_keys
        ]
        lifetimes = _numeric(individuals, "observed_lifetime_ticks")
        death_causes = {
            cause: sum(row.get("death_cause") == cause for row in individuals)
            for cause in ("world_boundary", "energy_depletion", "natural")
        }
        survival_fractions = {
            str(threshold): (
                sum(
                    (
                        float(row["observed_lifetime_ticks"]) > threshold
                        or (
                            float(row["observed_lifetime_ticks"]) == threshold
                            and not bool(row["death_occurred"])
                        )
                    )
                    for row in individuals
                )
                / len(individuals)
                if individuals
                else None
            )
            for threshold in thresholds
        }
        result["conditions"][condition.value] = {
            "run_count": len(condition_runs),
            "completed_run_count": len(completed_runs),
            "technical_failure_count": len(condition_runs) - len(completed_runs),
            "total_noralets_observed": len(individuals),
            "extinction_fraction": (
                sum(bool(row["extinct"]) for row in completed_runs)
                / len(completed_runs)
                if completed_runs
                else None
            ),
            "survival_fractions": survival_fractions,
            "death_cause_counts": death_causes,
            "observed_censored_lifetime_mean": _mean(lifetimes),
            "observed_censored_lifetime_median": (
                median(lifetimes) if lifetimes else None
            ),
            "mean_energy_distress": _mean(
                _numeric(individuals, "mean_energy_distress")
            ),
            "mean_condition_distress": _mean(
                _numeric(individuals, "mean_condition_distress")
            ),
            "consume_attempt_count": sum(
                int(row["consume_attempt_count"]) for row in individuals
            ),
            "successful_consumption_count": sum(
                int(row["successful_consumption_count"]) for row in individuals
            ),
            "total_energy_consumed": math.fsum(
                float(row["total_energy_consumed"]) for row in individuals
            ),
            "signal_emission_count": sum(
                int(row["signal_emission_count"]) for row in individuals
            ),
            "signal_LEFT_count": sum(
                int(row["signal_LEFT_count"]) for row in individuals
            ),
            "signal_RIGHT_count": sum(
                int(row["signal_RIGHT_count"]) for row in individuals
            ),
            "mean_total_absolute_distance_travelled": _mean(
                _numeric(individuals, "total_absolute_distance_travelled")
            ),
            "mean_requested_acceleration": _mean(
                _numeric(individuals, "mean_requested_acceleration")
            ),
            "mean_absolute_requested_acceleration": _mean(
                _numeric(individuals, "mean_absolute_requested_acceleration")
            ),
            "prediction_loss_initial_window_mean": _mean(
                _numeric(individuals, "prediction_loss_initial_window_mean")
            ),
            "prediction_loss_final_window_mean": _mean(
                _numeric(individuals, "prediction_loss_final_window_mean")
            ),
            "mean_homeostatic_drive": _mean(
                _numeric(individuals, "mean_homeostatic_drive")
            ),
            "mean_absolute_modulation": _mean(
                _numeric(individuals, "mean_absolute_modulation")
            ),
            "mean_homeostatic_update_norm": _mean(
                _numeric(individuals, "mean_homeostatic_update_norm")
            ),
            "online_encoder_parameter_drift_norm_mean": _mean(
                _numeric(individuals, "online_encoder_parameter_drift_norm")
            ),
            "GRU_parameter_drift_norm_mean": _mean(
                _numeric(individuals, "GRU_parameter_drift_norm")
            ),
            "predictor_parameter_drift_norm_mean": _mean(
                _numeric(individuals, "predictor_parameter_drift_norm")
            ),
            "action_head_parameter_drift_norm_mean": _mean(
                _numeric(individuals, "action_head_parameter_drift_norm")
            ),
        }
    return result


class ResearchOutputWriter:
    """Stream timeseries rows while retaining only compact summary tables."""

    def __init__(self, run_directory: Path) -> None:
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.run_rows: list[dict[str, Any]] = []
        self.noralet_rows: list[dict[str, Any]] = []
        self._timeseries_file = (self.run_directory / "timeseries.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._timeseries_writer = csv.DictWriter(
            self._timeseries_file,
            fieldnames=TIMESERIES_COLUMNS,
            extrasaction="raise",
        )
        self._timeseries_writer.writeheader()

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self._write_json("manifest.json", manifest)

    def write_timeseries(self, row: dict[str, Any]) -> None:
        self._timeseries_writer.writerow(row)

    def append_run(self, row: dict[str, Any]) -> None:
        self.run_rows.append({column: row.get(column) for column in RUN_SUMMARY_COLUMNS})

    def append_noralets(self, rows: tuple[dict[str, Any], ...]) -> None:
        self.noralet_rows.extend(rows)

    def finalize(
        self,
        *,
        max_ticks: int,
        conditions: tuple[LearningCondition, ...],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        self._timeseries_file.flush()
        self._timeseries_file.close()
        self._write_csv("run-summary.csv", RUN_SUMMARY_COLUMNS, self.run_rows)
        self._write_csv(
            "noralet-summary.csv",
            NORALET_SUMMARY_COLUMNS,
            self.noralet_rows,
        )
        aggregate = aggregate_results(
            self.run_rows,
            self.noralet_rows,
            max_ticks=max_ticks,
            conditions=conditions,
        )
        self._write_json("aggregate-summary.json", aggregate)
        (self.run_directory / "summary.md").write_text(
            self._summary_markdown(manifest, aggregate),
            encoding="utf-8",
        )
        return aggregate

    def close_after_failure(self) -> None:
        if not self._timeseries_file.closed:
            self._timeseries_file.close()

    def _write_csv(
        self,
        filename: str,
        columns: tuple[str, ...],
        rows: list[dict[str, Any]],
    ) -> None:
        with (self.run_directory / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, filename: str, value: dict[str, Any]) -> None:
        (self.run_directory / filename).write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _summary_markdown(
        self,
        manifest: dict[str, Any],
        aggregate: dict[str, Any],
    ) -> str:
        completed = sum(row["status"] == "completed" for row in self.run_rows)
        failures = len(self.run_rows) - completed
        extinctions = sum(
            row["status"] == "completed" and bool(row["extinct"])
            for row in self.run_rows
        )
        survivors = sum(
            int(row["survivors_at_end"])
            for row in self.run_rows
            if row["status"] == "completed"
        )
        resolved_devices = sorted(
            {
                str(row["device"])
                for row in self.run_rows
                if row["status"] == "completed"
            }
        )
        lines = [
            "# Baseline Lifetime Adaptation — Factual Batch Summary",
            "",
            "## Experiment configuration",
            "",
            f"- Experiment: `{EXPERIMENT_ID}`",
            f"- Conditions: {', '.join(manifest['conditions'])}",
            f"- Replicate seeds: {len(manifest['seed_mappings'])}",
            f"- Maximum ticks: {manifest['max_ticks']}",
            f"- Sample cadence: every {manifest['sample_every_ticks']} ticks",
            f"- Initial population per run: {manifest['initial_population']}",
            f"- Requested device: `{manifest['device']}`",
            f"- Resolved run device(s): {', '.join(resolved_devices) or 'none'}",
            f"- PyTorch: `{manifest['torch']['torch_version']}`",
            f"- CUDA available: {manifest['torch']['cuda_available']}",
            f"- CUDA runtime: `{manifest['torch']['torch_cuda_version']}`",
            f"- GPU: `{manifest['torch']['cuda_device_name']}`",
            "",
            "## Completion",
            "",
            f"- Runs completed: {completed}/{len(self.run_rows)}",
            f"- Technical failures: {failures}",
            f"- Extinct completed runs: {extinctions}",
            f"- Survivors at configured stopping points: {survivors}",
            "",
            "## Condition summaries",
            "",
            "| Condition | Runs | Extinction fraction | Observed/censored lifetime mean | Mean energy distress |",
            "|---|---:|---:|---:|---:|",
        ]
        for condition in manifest["conditions"]:
            values = aggregate["conditions"][condition]
            lines.append(
                "| "
                + condition
                + f" | {values['completed_run_count']}"
                + f" | {self._format(values['extinction_fraction'])}"
                + f" | {self._format(values['observed_censored_lifetime_mean'])}"
                + f" | {self._format(values['mean_energy_distress'])} |"
            )
        lines.extend(
            [
                "",
                "## Predictive learning",
                "",
                "| Condition | Initial-window loss | Final-window loss |",
                "|---|---:|---:|",
            ]
        )
        for condition in manifest["conditions"]:
            values = aggregate["conditions"][condition]
            lines.append(
                f"| {condition}"
                f" | {self._format(values['prediction_loss_initial_window_mean'])}"
                f" | {self._format(values['prediction_loss_final_window_mean'])} |"
            )
        lines.extend(
            [
                "",
                "Disabled components use null values in machine-readable outputs.",
                "",
                "## Homeostatic plasticity",
                "",
                "| Condition | Mean drive | Mean absolute modulation | Mean update norm | Action-head drift |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for condition in manifest["conditions"]:
            values = aggregate["conditions"][condition]
            lines.append(
                f"| {condition}"
                f" | {self._format(values['mean_homeostatic_drive'])}"
                f" | {self._format(values['mean_absolute_modulation'])}"
                f" | {self._format(values['mean_homeostatic_update_norm'])}"
                f" | {self._format(values['action_head_parameter_drift_norm_mean'])} |"
            )
        lines.extend(
            [
                "",
                "Statistics are descriptive and do not assume ecological benefit.",
                "",
                "## Survival and physiology",
                "",
                "| Condition | Survival fractions | Death causes (boundary/energy/natural) | Mean condition distress |",
                "|---|---|---|---:|",
            ]
        )
        for condition in manifest["conditions"]:
            values = aggregate["conditions"][condition]
            survival = ", ".join(
                f"{tick}: {self._format(fraction)}"
                for tick, fraction in values["survival_fractions"].items()
            ) or "n/a below first checkpoint"
            causes = values["death_cause_counts"]
            lines.append(
                f"| {condition} | {survival}"
                f" | {causes['world_boundary']}/{causes['energy_depletion']}/{causes['natural']}"
                f" | {self._format(values['mean_condition_distress'])} |"
            )
        lines.extend(
            [
                "",
                "Lifetimes include right-censored survivors and are explicitly labelled "
                "as observed/censored summaries.",
                "",
                "## Behaviour",
                "",
                "| Condition | Consume attempts | Positive transfers | Energy consumed | Signals L/R | Mean distance |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in manifest["conditions"]:
            values = aggregate["conditions"][condition]
            lines.append(
                f"| {condition} | {values['consume_attempt_count']}"
                f" | {values['successful_consumption_count']}"
                f" | {self._format(values['total_energy_consumed'])}"
                f" | {values['signal_LEFT_count']}/{values['signal_RIGHT_count']}"
                f" | {self._format(values['mean_total_absolute_distance_travelled'])} |"
            )
        lines.extend(
            [
                "",
                "Signal identifiers are observer engine identifiers; no semantic meaning is inferred.",
                "",
                "## Parameter development",
                "",
                "| Condition | Encoder drift | GRU drift | Predictor drift | Action-head drift |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for condition in manifest["conditions"]:
            values = aggregate["conditions"][condition]
            lines.append(
                f"| {condition}"
                f" | {self._format(values['online_encoder_parameter_drift_norm_mean'])}"
                f" | {self._format(values['GRU_parameter_drift_norm_mean'])}"
                f" | {self._format(values['predictor_parameter_drift_norm_mean'])}"
                f" | {self._format(values['action_head_parameter_drift_norm_mean'])} |"
            )
        lines.extend(
            [
                "",
                "Values are final L2 distances from immutable inherited birth snapshots; complete tensors are not serialized.",
                "",
                "## Important caveats",
                "",
                "- This is a pilot baseline.",
                "- Right-censored lifetimes are present where applicable.",
                "- Correlation does not establish mechanism.",
                "- The random-seed count is limited.",
                "- No intelligence or consciousness claim follows from these results.",
                "- This report makes no automatic architecture recommendation.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _format(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
