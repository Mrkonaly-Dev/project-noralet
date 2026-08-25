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
from noralet.ui.evolution_launcher import EvolutionLaunchSetup


class EvolutionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

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
