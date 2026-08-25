"""Offscreen Qt observer purity, controls, inspector and process tests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PySide6.QtCore import QProcess
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from noralet.brain import AutonomousSimulationRunner
from noralet.noralets.body import NoraletBodyState
from noralet.noralets.signals import SignalDirection, SignalType
from noralet.research.config import (
    EXPERIMENT_ID,
    LearningCondition,
    build_baseline_components,
)
from noralet.simulation.runtime import Simulation
from noralet.ui.app import NoraletMainWindow
from noralet.ui.canvas import SignalGlyph
from noralet.ui.research_launcher import (
    ResearchLaunchSetup,
    build_research_invocation,
)
from noralet.ui.session import LiveRunSetup, LiveSession, create_live_session
from noralet.world.signals import ActiveSignal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def _qt_application() -> QApplication:
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    assert isinstance(application, QApplication)
    return application


def _runner_snapshot(runner: AutonomousSimulationRunner) -> dict[str, object]:
    return {
        "state": runner.simulation.state,
        "parameters": {
            identity: runner.brain_for(identity).parameter_snapshot()
            for identity in runner.brain_ids
        },
        "hidden": {
            identity: runner.brain_for(identity).hidden_state
            for identity in runner.brain_ids
        },
        "rng": {
            identity: runner.simulation.random_streams.stream(
                runner.action_stream_name(identity)
            ).getstate()
            for identity in runner.brain_ids
        },
    }


def _assert_snapshot_equal(
    case: unittest.TestCase,
    actual: dict[str, object],
    expected: dict[str, object],
) -> None:
    case.assertEqual(actual["state"], expected["state"])
    case.assertEqual(actual["parameters"].keys(), expected["parameters"].keys())
    for identity in actual["parameters"]:
        for left, right in zip(
            actual["parameters"][identity],
            expected["parameters"][identity],
            strict=True,
        ):
            case.assertTrue(torch.equal(left, right))
        case.assertTrue(
            torch.equal(
                actual["hidden"][identity],
                expected["hidden"][identity],
            )
        )
        case.assertEqual(actual["rng"][identity], expected["rng"][identity])


class QtObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = _qt_application()

    def setUp(self) -> None:
        self.window = NoraletMainWindow()
        self.live = self.window.live
        self.live.device_combo.setCurrentIndex(
            self.live.device_combo.findData("cpu")
        )

    def tearDown(self) -> None:
        self.window.close()
        self.application.processEvents()

    def test_window_construction_does_not_create_or_tick_a_run(self) -> None:
        self.assertIsNone(self.live.session)
        self.assertFalse(self.live.timer_active)
        self.assertEqual(self.live.tick_label.text(), "Tick: 0")

    def test_pause_does_not_advance_simulation(self) -> None:
        self.assertTrue(self.live.reset_run(show_errors=False))
        self.live.set_speed_mode("1x")
        self.live.start()
        deadline = time.monotonic() + 3.0
        while self.live.session.tick == 0 and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.01)
        self.assertGreater(self.live.session.tick, 0)

        self.live.pause()
        state = self.live.session.runner.simulation.state
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.005)

        self.assertIs(self.live.session.runner.simulation.state, state)

    def test_step_advances_exactly_one_tick_and_each_living_brain_once(self) -> None:
        self.assertTrue(self.live.reset_run(show_errors=False))
        before = {
            identity: self.live.session.runner.brain_for(identity).activation_count
            for identity in self.live.session.runner.brain_ids
        }

        self.live.step_once()

        self.assertEqual(self.live.session.tick, 1)
        for identity, count in before.items():
            self.assertEqual(
                self.live.session.runner.brain_for(identity).activation_count,
                count + 1,
            )

    def test_repaint_and_selection_are_causally_pure(self) -> None:
        self.assertTrue(self.live.reset_run(show_errors=False))
        self.live.canvas.resize(900, 480)
        before = _runner_snapshot(self.live.session.runner)
        pixmap = QPixmap(900, 480)

        for selection in (1, 2, None, 1, None):
            self.live.canvas.select_noralet(selection)
            self.live.canvas.render(pixmap)
            self.live.canvas.update()
            self.application.processEvents()

        after = _runner_snapshot(self.live.session.runner)
        _assert_snapshot_equal(self, after, before)

    def test_single_and_batched_ui_ticks_have_identical_history(self) -> None:
        setup = LiveRunSetup(
            simulation_seed=41,
            population=2,
            device="cpu",
            maximum_ticks=20,
            condition=LearningCondition.FULL_CURRENT_BRAIN,
        )
        first = NoraletMainWindow()
        second = NoraletMainWindow()
        try:
            first.live.session = create_live_session(setup)
            first.live.canvas.set_session(first.live.session)
            second.live.session = create_live_session(setup)
            second.live.canvas.set_session(second.live.session)

            single_results = tuple(first.live.advance_burst(1)[0] for _ in range(6))
            batched_results = second.live.advance_burst(6)

            self.assertEqual(single_results, batched_results)
            _assert_snapshot_equal(
                self,
                _runner_snapshot(first.live.session.runner),
                _runner_snapshot(second.live.session.runner),
            )
        finally:
            first.close()
            second.close()

    def test_reset_recreates_the_exact_initial_run(self) -> None:
        self.live.seed_edit.setText("222")
        self.live.population_spin.setValue(3)
        self.assertTrue(self.live.reset_run(show_errors=False))
        initial = _runner_snapshot(self.live.session.runner)
        self.live.advance_burst(4)

        self.assertTrue(self.live.reset_run(show_errors=False))

        _assert_snapshot_equal(
            self,
            _runner_snapshot(self.live.session.runner),
            initial,
        )

    def test_inspector_matches_body_experience_and_learning_metrics(self) -> None:
        self.assertTrue(self.live.reset_run(show_errors=False))
        self.live.canvas.select_noralet(1)
        self.live.step_once()
        body = self.live.session.runner.simulation.state.body(1)
        experience = self.live.session.runner.simulation.experience_for(1)
        predictive = self.live.session.latest_learning[1]
        homeostatic = self.live.session.latest_homeostatic[1]

        self.assertEqual(self.live.inspector.value_for("noralet_id"), "1")
        self.assertEqual(
            float(self.live.inspector.value_for("position")),
            float(f"{body.position:.6g}"),
        )
        self.assertEqual(
            float(self.live.inspector.value_for("energy_distress")),
            float(f"{experience.interoception.energy_distress:.6g}"),
        )
        self.assertEqual(
            float(self.live.inspector.value_for("prediction_loss")),
            float(f"{predictive.prediction_loss:.6g}"),
        )
        self.assertEqual(
            float(self.live.inspector.value_for("homeostatic_update_norm")),
            float(f"{homeostatic.applied_update_norm:.6g}"),
        )

    def test_dead_selected_noralet_is_cleared_without_experience_lookup(self) -> None:
        setup = LiveRunSetup(
            simulation_seed=7,
            population=1,
            device="cpu",
            maximum_ticks=10,
            condition=LearningCondition.NO_LEARNING,
        )
        template, base = build_baseline_components(
            initial_population=1,
            device="cpu",
            condition=setup.condition,
            simulation_seed=setup.simulation_seed,
            base_brain_seed=setup.base_brain_seed,
        )
        fatal_simulation = Simulation(
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
        self.live.session = LiveSession(
            setup,
            AutonomousSimulationRunner(fatal_simulation, base),
        )
        self.live.canvas.set_session(self.live.session)
        self.live.canvas.select_noralet(1)

        self.live.advance_burst(1)

        self.assertTrue(self.live.session.is_extinct)
        self.assertIsNone(self.live.canvas.selected_id)
        self.assertEqual(self.live.inspector.value_for("noralet_id"), "—")

    def test_coordinate_mapping_covers_boundaries_center_and_resize(self) -> None:
        self.assertTrue(self.live.reset_run(show_errors=False))
        canvas = self.live.canvas
        config = self.live.session.runner.simulation.config
        canvas.resize(800, 400)
        left = canvas.world_to_canvas(config.left_boundary)
        center = canvas.world_to_canvas(0.0)
        right = canvas.world_to_canvas(config.right_boundary)
        self.assertAlmostEqual(left, 42.0)
        self.assertAlmostEqual(center, 400.0)
        self.assertAlmostEqual(right, 758.0)

        canvas.resize(1_000, 400)
        self.assertAlmostEqual(canvas.world_to_canvas(0.0), 500.0)

    def test_signal_glyph_copies_direction_origin_without_mutating_state(self) -> None:
        signal = ActiveSignal(
            sender_noralet_id=1,
            signal_type=SignalType.C,
            origin=-12.5,
            emission_direction=SignalDirection.LEFT,
        )

        glyph = SignalGlyph.from_active(signal)

        self.assertEqual(glyph, SignalGlyph("C", -12.5, -1))
        self.assertEqual(signal.origin, -12.5)
        self.assertIs(signal.emission_direction, SignalDirection.LEFT)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_full_current_brain_live_session_is_finite(self) -> None:
        session = create_live_session(
            LiveRunSetup(
                simulation_seed=19,
                population=2,
                device="cuda",
                maximum_ticks=3,
                condition=LearningCondition.FULL_CURRENT_BRAIN,
            )
        )

        results = session.step_many(3)

        self.assertEqual(len(results), 3)
        for identity in session.runner.brain_ids:
            brain = session.runner.brain_for(identity)
            self.assertEqual(str(brain.device), "cuda")
            self.assertTrue(torch.isfinite(brain.hidden_state).all().item())
            self.assertTrue(
                all(
                    torch.isfinite(parameter).all().item()
                    for parameter in brain.parameter_snapshot()
                )
            )


class ResearchUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = _qt_application()

    def test_command_contains_all_validated_research_fields(self) -> None:
        setup = ResearchLaunchSetup(
            seeds=4,
            maximum_ticks=123,
            sample_every_ticks=7,
            population=5,
            device="cpu",
            conditions=(
                LearningCondition.HOMEOSTATIC_ONLY,
                LearningCondition.NO_LEARNING,
            ),
        )

        invocation = build_research_invocation(
            setup,
            python_executable="python-test",
            working_directory=PROJECT_ROOT,
        )
        command = " ".join(invocation.arguments)

        self.assertEqual(invocation.program, "python-test")
        self.assertIn("-m noralet research baseline-lifetime-adaptation", command)
        self.assertIn("--seeds 4", command)
        self.assertIn("--max-ticks 123", command)
        self.assertIn("--sample-every 7", command)
        self.assertIn("--population 5", command)
        self.assertIn("--device cpu", command)
        self.assertIn("--conditions no-learning,homeostatic-only", command)

    def test_tiny_qprocess_research_batch_completes_and_discovers_result(self) -> None:
        window = NoraletMainWindow()
        panel = window.research
        panel.seeds_spin.setValue(2)
        panel.maximum_ticks_spin.setValue(2)
        panel.sample_every_spin.setValue(1)
        panel.population_spin.setValue(2)
        panel.device_combo.setCurrentIndex(panel.device_combo.findData("cpu"))
        for condition, checkbox in panel.condition_checks.items():
            checkbox.setChecked(condition is LearningCondition.NO_LEARNING)
        try:
            self.assertTrue(panel.start_experiment(show_errors=False))
            deadline = time.monotonic() + 45.0
            while (
                panel.process is not None
                and panel.process.state() != QProcess.ProcessState.NotRunning
                and time.monotonic() < deadline
            ):
                self.application.processEvents()
                time.sleep(0.01)
            self.application.processEvents()

            self.assertIsNotNone(panel.process)
            self.assertEqual(panel.process.state(), QProcess.ProcessState.NotRunning)
            self.assertEqual(panel.process.exitCode(), 0, panel.output.toPlainText())
            self.assertIsNotNone(panel.result_directory)
            self.assertTrue(panel.result_directory.is_dir())
            manifest = json.loads(
                (panel.result_directory / "manifest.json").read_text("utf-8")
            )
            self.assertEqual(manifest["experiment_id"], EXPERIMENT_ID)
            self.assertEqual(manifest["status"], "completed")
            self.assertIn("Research batch completed successfully", panel.progress_label.text())
        finally:
            window.close()

    def test_headless_cli_import_does_not_eagerly_load_qt(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import noralet.cli; "
                    "print(any(name.startswith('PySide6') for name in sys.modules))"
                ),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")

    def test_open_result_folder_uses_qt_desktop_services(self) -> None:
        window = NoraletMainWindow()
        with tempfile.TemporaryDirectory() as temporary:
            window.research.result_directory = Path(temporary)
            with patch(
                "noralet.ui.app.QDesktopServices.openUrl",
                return_value=True,
            ) as open_url:
                window.research.open_result_folder()

            open_url.assert_called_once()
            self.assertEqual(
                Path(open_url.call_args.args[0].toLocalFile()).resolve(),
                Path(temporary).resolve(),
            )
        window.close()


if __name__ == "__main__":
    unittest.main()
