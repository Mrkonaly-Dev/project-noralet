"""Evolution outputs, resume, CLI, CUDA and champion-watch tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from noralet.evolution import EvolutionConfig
from noralet.evolution.engine import load_champion, resume_evolution, run_evolution
from noralet.evolution.watch import create_champion_live_session
from noralet.research.config import LearningCondition
from noralet.ui.session import LiveRunSetup


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def engine_config(output_root: Path, *, device: str = "cpu", generations: int = 1) -> EvolutionConfig:
    return EvolutionConfig(
        generation_count=generations,
        device=device,
        population_size=2,
        elite_count=1,
        parent_pool_size=2,
        mutation_sigma=0.02,
        training_world_seeds=(501,),
        validation_world_seeds=(601,),
        noralets_per_world=2,
        max_ticks=2,
        initial_seed=9,
        champion_checkpoint_interval=5,
        output_root=output_root,
    )


class EvolutionEngineTests(unittest.TestCase):
    def test_output_schema_and_resume_continue_later_generations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            result = run_evolution(
                engine_config(Path(temporary)),
                run_directory=run_directory,
                progress=None,
            )
            required = (
                "manifest.json",
                "generations.csv",
                "candidates.csv",
                "evolution-state.pt",
                "summary.md",
                "champion/best.pt",
                "champion/generation-000.pt",
            )
            self.assertTrue(all((result / path).is_file() for path in required))
            manifest = json.loads((result / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["evolution_id"], "001-basebrain-bootstrap")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["initial_body_energy"], 10.0)
            self.assertEqual(manifest["learning_mode"], "full-current-brain")

            resumed = resume_evolution(
                result / "evolution-state.pt",
                generation_count=2,
                device="cpu",
                progress=None,
            )
            self.assertEqual(resumed, result)
            with (result / "generations.csv").open(newline="", encoding="utf-8") as handle:
                generations = list(csv.DictReader(handle))
            with (result / "candidates.csv").open(newline="", encoding="utf-8") as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual([row["generation"] for row in generations], ["0", "1"])
            self.assertEqual(len(candidates), 4)
            state = torch.load(
                result / "evolution-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(state["next_generation"], 2)

            uninterrupted = run_evolution(
                engine_config(Path(temporary), generations=2),
                run_directory=Path(temporary) / "uninterrupted",
                progress=None,
            )
            self.assertEqual(
                (result / "generations.csv").read_text("utf-8"),
                (uninterrupted / "generations.csv").read_text("utf-8"),
            )
            self.assertEqual(
                (result / "candidates.csv").read_text("utf-8"),
                (uninterrupted / "candidates.csv").read_text("utf-8"),
            )

    def test_champion_watch_is_fresh_full_learning_run_from_saved_genome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_evolution(
                engine_config(Path(temporary)),
                run_directory=Path(temporary) / "run",
                progress=None,
            )
            genome, _ = load_champion(result)
            setup = LiveRunSetup(
                simulation_seed=777,
                population=2,
                device="cpu",
                maximum_ticks=3,
                condition=LearningCondition.FULL_CURRENT_BRAIN,
            )
            session, metadata = create_champion_live_session(result, setup)
            self.assertEqual(metadata["learning_mode"], "full-current-brain")
            self.assertEqual(
                tuple(body.energy for body in session.runner.simulation.state.bodies),
                (10.0, 10.0),
            )
            self.assertTrue(
                all(
                    torch.equal(
                        expected,
                        dict(
                            session.runner.brain_for(1).model.named_parameters()
                        )[name].detach().cpu(),
                    )
                    for name, expected in genome.state().items()
                )
            )
            result_tick = session.step()
            self.assertIsNotNone(result_tick)
            self.assertTrue(result_tick.learning_results)
            self.assertTrue(result_tick.homeostatic_learning_results)

    def test_tiny_two_generation_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                (
                    sys.executable,
                    "-m",
                    "noralet",
                    "evolution",
                    "basebrain-bootstrap",
                    "--generations",
                    "2",
                    "--device",
                    "cpu",
                    "--population-size",
                    "2",
                    "--elite-count",
                    "1",
                    "--parent-pool-size",
                    "2",
                    "--training-worlds",
                    "1",
                    "--validation-worlds",
                    "1",
                    "--noralets-per-world",
                    "2",
                    "--max-ticks",
                    "2",
                    "--output-root",
                    temporary,
                ),
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Generation 1 complete", completed.stdout)
            self.assertIn("Evolution outputs:", completed.stdout)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_tiny_cuda_evolution_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_evolution(
                engine_config(Path(temporary), device="cuda"),
                run_directory=Path(temporary) / "cuda-run",
                progress=None,
            )
            self.assertTrue((result / "champion" / "best.pt").is_file())
            manifest = json.loads((result / "manifest.json").read_text("utf-8"))
            self.assertTrue(manifest["cuda_available"])
            self.assertEqual(manifest["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
