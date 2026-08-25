"""Offscreen Evolution QProcess launcher and Watch Champion integration."""

from __future__ import annotations

import os
import csv
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication

from noralet.evolution import EvolutionConfig
from noralet.evolution.distributional import (
    DISTRIBUTIONAL_EVOLUTION_ID,
    DistributionalEvolutionConfig,
    run_distributional_evolution,
)
from noralet.evolution.engine import load_champion, run_evolution
from noralet.ui.app import NoraletMainWindow
from noralet.ui.evolution_launcher import (
    DistributionalEvolutionLaunchSetup,
    EvolutionLaunchSetup,
    build_distributional_evolution_invocation,
    build_evolution_invocation,
    build_evolution_resume_invocation,
    estimate_evolution_workload,
    load_evolution_fork_metadata,
    load_evolution_resume_metadata,
)


def _resume_fixture_config(output_root: Path) -> EvolutionConfig:
    return EvolutionConfig(
        generation_count=1,
        device="cpu",
        population_size=2,
        elite_count=1,
        parent_pool_size=2,
        mutation_sigma=0.031,
        training_world_seeds=(701, 702),
        validation_world_seeds=(801,),
        noralets_per_world=2,
        max_ticks=2,
        initial_body_energy=12.5,
        initial_seed=73,
        output_root=output_root,
    )


def _create_resume_fixture(root: Path, name: str = "resume-source") -> Path:
    return run_evolution(
        _resume_fixture_config(root),
        run_directory=root / name,
        progress=None,
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
            self.assertIn("144", panel.workload_label.text())
        finally:
            window.close()

    def test_distributional_v2_is_normal_cpu_protocol_with_exact_defaults(self) -> None:
        window = NoraletMainWindow()
        try:
            panel = window.evolution
            setup = panel.current_distributional_setup()
            self.assertEqual(panel.protocol_combo.currentData(), "v2")
            self.assertEqual(setup, DistributionalEvolutionLaunchSetup())
            self.assertEqual(setup.device, "cpu")
            self.assertIn("DISTRIBUTIONAL", panel.page_title.text())
            self.assertIn(
                "Selection worlds",
                panel._advanced_label_widgets[panel.training_worlds_spin].text(),
            )
            arguments = build_distributional_evolution_invocation(setup).arguments
            self.assertEqual(arguments[4], "distributional")
            expected = {
                "--generations": "20",
                "--device": "cpu",
                "--population-size": "8",
                "--elite-count": "2",
                "--parent-pool-size": "4",
                "--mutation-sigma": "0.02",
                "--selection-worlds": "4",
                "--benchmark-worlds": "8",
                "--benchmark-every": "5",
                "--noralets-per-world": "4",
                "--max-ticks": "1000",
                "--initial-energy": "10.0",
            }
            for flag, value in expected.items():
                self.assertEqual(arguments[arguments.index(flag) + 1], value)
        finally:
            window.close()

    def test_resume_checkpoint_populates_locked_scientific_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = _create_resume_fixture(Path(temporary))
            checkpoint = result / "evolution-state.pt"
            expected_champion = load_champion(result)[1]
            metadata = load_evolution_resume_metadata(checkpoint)
            self.assertEqual(metadata.checkpoint_path, checkpoint.resolve())
            self.assertEqual(metadata.result_directory, result.resolve())
            self.assertEqual(metadata.run_id, result.name)
            self.assertEqual(metadata.completed_generations, 1)
            self.assertEqual(
                (
                    metadata.population_size,
                    metadata.elite_count,
                    metadata.parent_pool_size,
                    metadata.mutation_sigma,
                    metadata.training_worlds,
                    metadata.validation_worlds,
                    metadata.noralets_per_world,
                    metadata.maximum_ticks,
                    metadata.initial_energy,
                    metadata.initial_seed,
                    metadata.device,
                ),
                (2, 1, 2, 0.031, 2, 1, 2, 2, 12.5, 73, "cpu"),
            )
            self.assertEqual(
                metadata.best_candidate_id,
                expected_champion["candidate_id"],
            )
            self.assertEqual(
                metadata.best_generation,
                expected_champion["generation"],
            )

            window = NoraletMainWindow()
            try:
                panel = window.evolution
                self.assertTrue(
                    panel.load_resume_checkpoint(checkpoint, show_errors=False)
                )
                self.assertFalse(panel.setup_group.isEnabled())
                self.assertFalse(panel.resume_group.isHidden())
                self.assertEqual(panel.continue_to_spin.value(), 2)
                shown = panel.resume_metadata_label.text()
                for expected in (
                    result.name,
                    str(result.resolve()),
                    "Generations already completed: 1",
                    "Population size: 2",
                    "Elite count: 1",
                    "Parent pool: 2",
                    "Mutation sigma: 0.031",
                    "Training / validation worlds: 2 / 1",
                    "Noralets / world: 2",
                    "Max ticks: 2",
                    "Initial Energy: 12.5 eU",
                    "Original evolution seed: 73",
                    "Saved device: cpu",
                    str(expected_champion["candidate_id"]),
                ):
                    self.assertIn(expected, shown)
            finally:
                window.close()

    def test_v1_fork_ui_shows_lineage_and_builds_new_v2_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = _create_resume_fixture(Path(temporary), "fork-source")
            checkpoint = (result / "evolution-state.pt").resolve()
            source_before = checkpoint.read_bytes()
            metadata = load_evolution_fork_metadata(checkpoint)
            self.assertEqual(metadata.source_run_id, result.name)
            self.assertEqual(metadata.completed_generations, 1)
            self.assertEqual(metadata.population_size, 2)

            window = NoraletMainWindow()
            try:
                panel = window.evolution
                panel.apply_preset("Standard")
                self.assertEqual(panel.protocol_combo.currentData(), "v1")
                self.assertTrue(
                    panel.load_fork_checkpoint(checkpoint, show_errors=False)
                )
                self.assertEqual(panel.protocol_combo.currentData(), "v2")
                self.assertEqual(panel.training_worlds_spin.value(), 4)
                self.assertEqual(panel.validation_worlds_spin.value(), 8)
                self.assertEqual(panel.benchmark_interval_spin.value(), 5)
                self.assertFalse(panel.population_spin.isEnabled())
                shown = panel.fork_metadata_label.text()
                for expected in (
                    "FORK V1 → V2",
                    result.name,
                    str(checkpoint),
                    "Source completed generation: 1",
                    "Copied final population: 2 genomes",
                    "new generation 0",
                ):
                    self.assertIn(expected, shown)
                invocation = build_distributional_evolution_invocation(
                    panel.current_distributional_setup()
                )
                self.assertEqual(
                    invocation.arguments[-2:],
                    ("--fork-from", str(checkpoint)),
                )
                self.assertEqual(checkpoint.read_bytes(), source_before)
            finally:
                window.close()

    def test_distributional_resume_is_protocol_aware_and_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_distributional_evolution(
                DistributionalEvolutionConfig(
                    generation_count=1,
                    device="cpu",
                    population_size=2,
                    elite_count=1,
                    parent_pool_size=2,
                    selection_world_count=2,
                    benchmark_world_count=3,
                    benchmark_interval=5,
                    noralets_per_world=2,
                    max_ticks=2,
                    mutation_sigma=0.041,
                    initial_body_energy=13.0,
                    initial_seed=91,
                    output_root=root,
                ),
                run_directory=root / "v2-resume-source",
                progress=None,
            )
            checkpoint = (result / "evolution-state.pt").resolve()
            metadata = load_evolution_resume_metadata(checkpoint)
            self.assertEqual(metadata.evolution_id, DISTRIBUTIONAL_EVOLUTION_ID)
            self.assertEqual(
                (
                    metadata.selection_worlds,
                    metadata.benchmark_worlds,
                    metadata.benchmark_interval,
                    metadata.completed_generations,
                ),
                (2, 3, 5, 1),
            )
            invocation = build_evolution_resume_invocation(
                metadata,
                target_generation=3,
            )
            self.assertEqual(invocation.arguments[4], "distributional")
            self.assertEqual(
                invocation.arguments[-4:],
                ("--generations", "3", "--resume", str(checkpoint)),
            )
            for prohibited in (
                "--population-size",
                "--selection-worlds",
                "--benchmark-worlds",
                "--benchmark-every",
                "--mutation-sigma",
                "--seed",
            ):
                self.assertNotIn(prohibited, invocation.arguments)

            window = NoraletMainWindow()
            try:
                panel = window.evolution
                self.assertTrue(
                    panel.load_resume_checkpoint(checkpoint, show_errors=False)
                )
                shown = panel.resume_metadata_label.text()
                self.assertIn(DISTRIBUTIONAL_EVOLUTION_ID, shown)
                self.assertIn("Selection worlds / generation: 2", shown)
                self.assertIn("Fixed benchmark worlds: 3", shown)
                self.assertIn("Benchmark every: 5 generations", shown)
                self.assertIn("Benchmark-best champion", shown)
                panel.population_spin.setValue(99)
                panel.continue_to_spin.setValue(2)
                self.assertTrue(panel.start_resume(show_errors=False))
                self.assertNotIn("--population-size", panel._invocation.arguments)
                deadline = time.monotonic() + 45.0
                while (
                    panel.process is not None
                    and panel.process.state() != QProcess.ProcessState.NotRunning
                    and time.monotonic() < deadline
                ):
                    self.application.processEvents()
                    time.sleep(0.01)
                self.application.processEvents()
                self.assertEqual(panel.process.exitCode(), 0, panel.output.toPlainText())
                self.assertEqual(panel.resume_metadata.completed_generations, 2)
                panel.watch_button.click()
                self.application.processEvents()
                self.assertIn(
                    "BENCHMARK-BEST CHAMPION WATCH",
                    window.live.setup_group.title(),
                )
                champion_metadata = load_champion(result)[1]
                shown_watch = window.live.champion_metadata_label.text()
                for expected in (
                    str(champion_metadata["candidate_id"]),
                    str(champion_metadata["generation"]),
                    "Champion role: benchmark-best",
                    "Benchmark mean lifetime",
                    "Benchmark median lifetime",
                    "fresh learned life",
                    "Current simulation: seed 1",
                ):
                    self.assertIn(expected, shown_watch)
            finally:
                window.close()

    def test_resume_command_is_exact_and_excludes_scientific_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = _create_resume_fixture(Path(temporary))
            checkpoint = (result / "evolution-state.pt").resolve()
            metadata = load_evolution_resume_metadata(checkpoint)
            invocation = build_evolution_resume_invocation(
                metadata,
                target_generation=4,
            )
            self.assertEqual(
                invocation.arguments[-4:],
                ("--generations", "4", "--resume", str(checkpoint)),
            )
            for prohibited in (
                "--population-size",
                "--elite-count",
                "--parent-pool-size",
                "--mutation-sigma",
                "--training-worlds",
                "--validation-worlds",
                "--noralets-per-world",
                "--max-ticks",
                "--initial-energy",
                "--seed",
                "--output-root",
            ):
                self.assertNotIn(prohibited, invocation.arguments)
            overridden = build_evolution_resume_invocation(
                metadata,
                target_generation=4,
                device_override="cpu",
            )
            self.assertEqual(overridden.arguments[-2:], ("--device", "cpu"))

    def test_resume_rejects_target_not_above_completed_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = _create_resume_fixture(Path(temporary))
            metadata = load_evolution_resume_metadata(
                result / "evolution-state.pt"
            )
            with self.assertRaisesRegex(ValueError, "must exceed completed"):
                build_evolution_resume_invocation(
                    metadata,
                    target_generation=1,
                )

            window = NoraletMainWindow()
            try:
                panel = window.evolution
                self.assertTrue(
                    panel.load_resume_checkpoint(
                        result / "evolution-state.pt",
                        show_errors=False,
                    )
                )
                panel.continue_to_spin.setValue(1)
                self.assertFalse(panel.start_resume(show_errors=False))
                self.assertIsNone(panel.process)
                self.assertIn("must exceed completed", panel.progress_label.text())
            finally:
                window.close()

    def test_resumed_evolution_qprocess_uses_locked_checkpoint_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = _create_resume_fixture(Path(temporary))
            checkpoint = (result / "evolution-state.pt").resolve()
            window = NoraletMainWindow()
            panel = window.evolution
            try:
                self.assertTrue(
                    panel.load_resume_checkpoint(checkpoint, show_errors=False)
                )
                panel.population_spin.setValue(99)
                panel.mutation_sigma_spin.setValue(7.0)
                panel.continue_to_spin.setValue(2)
                self.assertTrue(panel.start_resume(show_errors=False))
                self.assertIsNotNone(panel._invocation)
                arguments = panel._invocation.arguments
                self.assertIn(str(checkpoint), arguments)
                self.assertEqual(
                    arguments[arguments.index("--generations") + 1],
                    "2",
                )
                self.assertNotIn("--population-size", arguments)
                self.assertNotIn("--mutation-sigma", arguments)

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
                self.assertEqual(panel.result_directory, result.resolve())
                self.assertIsNotNone(panel.resume_metadata)
                self.assertEqual(panel.resume_metadata.completed_generations, 2)
                self.assertTrue(panel.continue_button.isEnabled())
                with (result / "generations.csv").open(
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row["generation"] for row in rows], ["0", "1"])
                self.assertFalse(panel.setup_group.isEnabled())
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
                self.assertTrue(panel.continue_button.isEnabled())
                panel.continue_button.click()
                self.application.processEvents()
                self.assertIsNotNone(panel.resume_metadata)
                self.assertEqual(
                    panel.resume_metadata.checkpoint_path,
                    (panel.result_directory / "evolution-state.pt").resolve(),
                )
                self.assertFalse(panel.setup_group.isEnabled())

                _, champion_metadata = load_champion(panel.result_directory)
                panel.watch_button.click()
                self.application.processEvents()
                self.assertIsNotNone(window.live.session)
                self.assertEqual(window.live.session.tick, 0)
                self.assertIn("champion", window.live.status_label.text())
                self.assertIn("CHAMPION WATCH", window.live.setup_group.title())
                self.assertNotIn("Baseline", window.live.setup_group.title())
                self.assertFalse(window.live.seed_edit.isEnabled())
                self.assertTrue(window.live.seed_edit.isHidden())
                self.assertEqual(
                    window.live.reset_button.text(),
                    "Return to baseline setup",
                )
                shown = window.live.champion_metadata_label.text()
                for expected in (
                    str(champion_metadata["candidate_id"]),
                    str(champion_metadata["generation"]),
                    f"Birth Energy: {champion_metadata['initial_body_energy']:g} eU",
                    panel.result_directory.name,
                    "fresh learned life",
                    f"Source evolution device: {champion_metadata['configuration']['device']}",
                    f"Watch device: {window.live.session.setup.device}",
                    "seed 1",
                    "population 6",
                ):
                    self.assertIn(expected, shown)
                window.live.step_once()
                self.assertEqual(window.live.session.tick, 1)
                window.live.reset_button.click()
                self.application.processEvents()
                self.assertIsNone(window.live.session)
                self.assertEqual(
                    window.live.setup_group.title(),
                    "Baseline world · run setup",
                )
                self.assertFalse(window.live.seed_edit.isHidden())
                self.assertTrue(window.live.seed_edit.isEnabled())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
