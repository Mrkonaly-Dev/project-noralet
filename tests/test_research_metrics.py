"""Observer purity and metric correctness tests for Research 001."""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from noralet.brain import AutonomousSimulationRunner
from noralet.noralets.body import NoraletBodyState
from noralet.research.config import (
    BaselineExperimentConfig,
    LearningCondition,
    build_run_components,
)
from noralet.research.metrics import (
    _HomeostaticAccumulator,
    _PredictionAccumulator,
    ResearchRunObserver,
    parameter_drift_norm,
)
from noralet.simulation.events import EnergyConsumed, NoraletMoved, SignalEmitted
from noralet.simulation.runtime import Simulation


def _config(condition: LearningCondition, *, max_ticks: int = 5):
    return BaselineExperimentConfig(
        replicate_seeds=(3, 4),
        max_ticks=max_ticks,
        sample_every_ticks=2,
        initial_population=2,
        device="cpu",
        conditions=(condition,),
    )


def _parameter_snapshots(runner: AutonomousSimulationRunner):
    return {
        identity: runner.brain_for(identity).parameter_snapshot()
        for identity in runner.brain_ids
    }


class ResearchMetricTests(unittest.TestCase):
    def test_observation_does_not_change_any_causal_result(self) -> None:
        config = _config(LearningCondition.FULL_CURRENT_BRAIN, max_ticks=6)
        seeds = config.seed_mappings[0]
        control_simulation, control_base = build_run_components(
            config,
            LearningCondition.FULL_CURRENT_BRAIN,
            seeds,
        )
        observed_simulation, observed_base = build_run_components(
            config,
            LearningCondition.FULL_CURRENT_BRAIN,
            seeds,
        )
        control = AutonomousSimulationRunner(control_simulation, control_base)
        observed = AutonomousSimulationRunner(observed_simulation, observed_base)
        rows: list[dict[str, object]] = []
        observer = ResearchRunObserver(
            observed,
            LearningCondition.FULL_CURRENT_BRAIN,
            seeds,
            sample_every_ticks=2,
            timeseries_sink=rows.append,
        )

        for _ in range(6):
            control_result = control.step()
            observed_result = observed.step()
            observer.observe(observed_result)
            self.assertEqual(observed_result, control_result)
            self.assertEqual(observed.simulation.state, control.simulation.state)

        observer.finish(max_ticks=6)
        self.assertEqual(_parameter_snapshots(observed).keys(), _parameter_snapshots(control).keys())
        for identity in observed.brain_ids:
            for actual, expected in zip(
                observed.brain_for(identity).parameter_snapshot(),
                control.brain_for(identity).parameter_snapshot(),
                strict=True,
            ):
                self.assertTrue(torch.equal(actual, expected))
            stream_name = observed.action_stream_name(identity)
            self.assertEqual(
                observed.simulation.random_streams.stream(stream_name).getstate(),
                control.simulation.random_streams.stream(stream_name).getstate(),
            )
            self.assertTrue(
                torch.equal(
                    observed.brain_for(identity).hidden_state,
                    control.brain_for(identity).hidden_state,
                )
            )

    def test_action_event_and_movement_metrics_match_authoritative_results(self) -> None:
        config = _config(LearningCondition.NO_LEARNING)
        seeds = config.seed_mappings[0]
        simulation, base = build_run_components(
            config,
            LearningCondition.NO_LEARNING,
            seeds,
        )
        runner = AutonomousSimulationRunner(simulation, base)
        observer = ResearchRunObserver(
            runner,
            LearningCondition.NO_LEARNING,
            seeds,
            sample_every_ticks=2,
            timeseries_sink=lambda row: None,
        )
        expected: dict[int, dict[str, float | int]] = {
            identity: {
                "distance": 0.0,
                "attempts": 0,
                "ingestions": 0,
                "energy": 0.0,
                "signals": 0,
                "left": 0,
                "right": 0,
            }
            for identity in runner.brain_ids
        }
        for _ in range(5):
            result = runner.step()
            for identity, intent in result.action_intents:
                expected[identity]["attempts"] += int(intent.consume)
            for event in result.tick_result.events:
                if isinstance(event, NoraletMoved):
                    expected[event.noralet_id]["distance"] += abs(
                        event.position_after - event.position_before
                    )
                elif isinstance(event, EnergyConsumed):
                    expected[event.noralet_id]["ingestions"] += 1
                    expected[event.noralet_id]["energy"] += event.energy_transferred
                elif isinstance(event, SignalEmitted):
                    expected[event.noralet_id]["signals"] += 1
                    expected[event.noralet_id][event.emission_direction.value] += 1
            observer.observe(result)

        summaries = {
            row["noralet_id"]: row for row in observer.finish(max_ticks=5)
        }
        for identity, row in summaries.items():
            values = expected[identity]
            self.assertAlmostEqual(
                row["total_absolute_distance_travelled"], values["distance"]
            )
            self.assertEqual(row["consume_attempt_count"], values["attempts"])
            self.assertEqual(
                row["successful_consumption_count"], values["ingestions"]
            )
            self.assertAlmostEqual(row["total_energy_consumed"], values["energy"])
            self.assertEqual(row["signal_emission_count"], values["signals"])
            self.assertEqual(row["signal_LEFT_count"], values["left"])
            self.assertEqual(row["signal_RIGHT_count"], values["right"])
            self.assertEqual(row["observed_lifetime_ticks"], 5)
            self.assertFalse(row["death_occurred"])
            self.assertTrue(row["right_censored"])

    def test_prediction_windows_and_homeostatic_sign_counts_are_exact(self) -> None:
        prediction = _PredictionAccumulator()
        for value in range(150):
            prediction.add(float(value))
        summary = prediction.summary()
        self.assertEqual(summary["predictive_update_count"], 150)
        self.assertEqual(summary["prediction_loss_initial_window_mean"], 49.5)
        self.assertEqual(summary["prediction_loss_final_window_mean"], 99.5)
        self.assertEqual(summary["prediction_loss_min"], 0.0)
        self.assertEqual(summary["prediction_loss_max"], 149.0)

        homeostatic = _HomeostaticAccumulator()
        for modulation in (0.2, -0.4, 0.0):
            homeostatic.add(
                drive_after=0.5,
                modulation=modulation,
                eligibility_norm=2.0,
                update_norm=0.25,
            )
        homeostatic_summary = homeostatic.summary()
        self.assertEqual(homeostatic_summary["positive_modulation_count"], 1)
        self.assertEqual(homeostatic_summary["negative_modulation_count"], 1)
        self.assertEqual(homeostatic_summary["neutral_modulation_count"], 1)
        self.assertAlmostEqual(
            homeostatic_summary["mean_absolute_modulation"], 0.2
        )

    def test_parameter_drift_is_global_l2_and_disabled_is_null(self) -> None:
        layer = nn.Linear(1, 1)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
        birth = tuple(parameter.detach().clone() for parameter in layer.parameters())
        with torch.no_grad():
            layer.weight.fill_(3.0)
            layer.bias.fill_(4.0)

        self.assertEqual(parameter_drift_norm(layer, birth), 5.0)
        self.assertIsNone(parameter_drift_norm(None, None))

    def test_controlled_boundary_extinction_stops_without_later_samples(self) -> None:
        config = _config(LearningCondition.NO_LEARNING, max_ticks=10)
        seeds = config.seed_mappings[0]
        template, base = build_run_components(
            config,
            LearningCondition.NO_LEARNING,
            seeds,
        )
        fatal = Simulation(
            template.config,
            initial_bodies=(
                NoraletBodyState(
                    noralet_id=1,
                    position=99.9,
                    velocity=1.0,
                    energy=60.0,
                    age_ticks=0,
                    condition=1.0,
                    perceptual_signature=(0.5, -0.5),
                ),
            ),
        )
        runner = AutonomousSimulationRunner(fatal, base)
        sampled: list[dict[str, object]] = []
        observer = ResearchRunObserver(
            runner,
            LearningCondition.NO_LEARNING,
            seeds,
            sample_every_ticks=2,
            timeseries_sink=sampled.append,
        )

        observer.observe(runner.step())
        rows = observer.finish(max_ticks=10)
        run = observer.run_summary(max_ticks=10, runtime_seconds=0.0)

        self.assertEqual(runner.simulation.state.tick, 1)
        self.assertTrue(run["extinct"])
        self.assertEqual(run["boundary_death_count"], 1)
        self.assertEqual(rows[0]["death_cause"], "world_boundary")
        self.assertEqual(rows[0]["observed_lifetime_ticks"], 1)
        self.assertFalse(rows[0]["right_censored"])
        self.assertEqual({row["tick"] for row in sampled}, {0})


if __name__ == "__main__":
    unittest.main()
