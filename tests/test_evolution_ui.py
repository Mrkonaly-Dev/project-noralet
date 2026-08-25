"""Offscreen Evolution QProcess launcher and Watch Champion integration."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication

from noralet.ui.app import NoraletMainWindow
from noralet.ui.evolution_launcher import (
    EvolutionLaunchSetup,
    build_evolution_invocation,
    estimate_evolution_workload,
)


class EvolutionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_pilot_preset_populates_exact_scientific_values(self) -> None:
        window = NoraletMainWindow()
        try:
            panel = window.evolution
            panel.device_combo.setCurrentIndex(panel.device_combo.findData("cpu"))
            panel.apply_preset("Pilot")
            setup = panel.current_setup()
            self.assertEqual(panel.preset_name, "Pilot")
            self.assertEqual(
                (
                    setup.generations,
                    setup.population_size,
                    setup.elite_count,
                    setup.parent_pool_size,
                    setup.training_worlds,
                    setup.validation_worlds,
                    setup.noralets_per_world,
                    setup.maximum_ticks,
                    setup.mutation_sigma,
                    setup.initial_energy,
                    setup.device,
                ),
                (5, 8, 2, 4, 2, 2, 4, 1_000, 0.02, 10.0, "cpu"),
            )
            self.assertEqual(
                panel.preset_description.text(),
                "Pilot — fast exploratory run",
            )
        finally:
            window.close()

    def test_standard_preset_restores_iteration_13_defaults(self) -> None:
        window = NoraletMainWindow()
        try:
            panel = window.evolution
            panel.population_spin.setValue(7)
            self.assertEqual(panel.preset_name, "Custom")
            panel.apply_preset("Standard")
            setup = panel.current_setup()
            self.assertEqual(panel.preset_name, "Standard")
            self.assertEqual(
                (
                    setup.generations,
                    setup.population_size,
                    setup.elite_count,
                    setup.parent_pool_size,
                    setup.training_worlds,
                    setup.validation_worlds,
                    setup.noralets_per_world,
                    setup.maximum_ticks,
                    setup.mutation_sigma,
                    setup.initial_energy,
                ),
                (50, 32, 4, 8, 4, 4, 6, 2_000, 0.02, 10.0),
            )
            self.assertEqual(
                panel.preset_description.text(),
                "Standard — full default protocol",
            )
        finally:
            window.close()

    def test_custom_values_construct_existing_cli_flags(self) -> None:
        setup = EvolutionLaunchSetup(
            generations=7,
            device="cpu",
            population_size=9,
            elite_count=2,
            parent_pool_size=5,
            mutation_sigma=0.03,
            training_worlds=3,
            validation_worlds=2,
            noralets_per_world=4,
            maximum_ticks=321,
            initial_energy=11.5,
            initial_seed=12,
            output_root=Path("custom-evolution-root"),
        )
        arguments = build_evolution_invocation(setup).arguments
        expected = {
            "--generations": "7",
            "--device": "cpu",
            "--population-size": "9",
            "--elite-count": "2",
            "--parent-pool-size": "5",
            "--mutation-sigma": "0.03",
            "--training-worlds": "3",
            "--validation-worlds": "2",
            "--noralets-per-world": "4",
            "--max-ticks": "321",
            "--initial-energy": "11.5",
            "--seed": "12",
            "--output-root": "custom-evolution-root",
        }
        for flag, value in expected.items():
            index = arguments.index(flag)
            self.assertEqual(arguments[index + 1], value)

    def test_invalid_elite_parent_population_relationships_are_rejected(self) -> None:
        invalid = (
            {"population_size": 4, "elite_count": 3, "parent_pool_size": 2},
            {"population_size": 4, "elite_count": 1, "parent_pool_size": 5},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                EvolutionLaunchSetup(**values)

        window = NoraletMainWindow()
        try:
            panel = window.evolution
            panel.population_spin.setValue(4)
            panel.parent_pool_spin.setValue(5)
            self.assertFalse(panel.start_evolution(show_errors=False))
            self.assertIsNone(panel.process)
            self.assertIn("Invalid setup", panel.progress_label.text())
        finally:
            window.close()

    def test_workload_estimate_is_exact_and_observer_only(self) -> None:
        setup = EvolutionLaunchSetup(
            generations=5,
            population_size=8,
            elite_count=2,
            parent_pool_size=4,
            training_worlds=2,
            validation_worlds=2,
            noralets_per_world=4,
            maximum_ticks=1_000,
        )
        estimate = estimate_evolution_workload(setup)
        self.assertEqual(estimate.training_lives_per_generation, 64)
        self.assertEqual(estimate.validation_lives_per_generation, 8)
        self.assertEqual(estimate.maximum_training_ticks_per_generation, 64_000)

    def test_editing_preset_field_only_switches_to_custom(self) -> None:
        window = NoraletMainWindow()
        try:
            panel = window.evolution
            self.assertEqual(panel.preset_name, "Pilot")
            self.assertIsNone(panel.process)
            self.assertIsNone(window.live.session)
            panel.population_spin.setValue(9)
            self.application.processEvents()
            self.assertEqual(panel.preset_name, "Custom")
            self.assertIsNone(panel.process)
            self.assertIsNone(window.live.session)
            self.assertIn("72", panel.workload_label.text())
        finally:
            window.close()

    def test_tiny_process_is_detected_and_champion_can_be_watched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            window = NoraletMainWindow()
            panel = window.evolution
            setup = EvolutionLaunchSetup(
                generations=1,
                device="cpu",
                population_size=2,
                elite_count=1,
                parent_pool_size=2,
                training_worlds=1,
                validation_worlds=1,
                noralets_per_world=2,
                maximum_ticks=2,
                output_root=Path(temporary),
            )
            try:
                self.assertTrue(panel.start_evolution(setup, show_errors=False))
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
                self.assertEqual(panel.process.exitCode(), 0, panel.output.toPlainText())
                self.assertIsNotNone(panel.result_directory)
                self.assertTrue((panel.result_directory / "champion" / "best.pt").is_file())
                self.assertIn("Generation 0 complete", panel.output.toPlainText())
                self.assertTrue(panel.watch_button.isEnabled())

                panel.watch_button.click()
                self.application.processEvents()
                self.assertIsNotNone(window.live.session)
                self.assertEqual(window.live.session.tick, 0)
                self.assertIn("champion", window.live.status_label.text())
                window.live.step_once()
                self.assertEqual(window.live.session.tick, 1)
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
