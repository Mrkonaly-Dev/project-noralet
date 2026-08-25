"""Evolution v2 distributional selection, benchmark, fork and resume tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from noralet.evolution import EvolutionConfig
from noralet.evolution.distributional import (
    BENCHMARK_COLUMNS,
    CANDIDATE_COLUMNS,
    DISTRIBUTIONAL_EVOLUTION_ID,
    GENERATION_COLUMNS,
    DistributionalEvolutionConfig,
    _genome_sha256,
    create_distributional_next_generation,
    evaluate_distributional_generation,
    fixed_benchmark_world_seeds,
    initialize_distributional_population,
    resume_distributional_evolution,
    run_distributional_evolution,
    selection_world_seeds,
)
from noralet.evolution.engine import load_champion, run_evolution
from noralet.evolution.genome import BaseBrainGenome
from noralet.evolution.evaluation import CandidateEvaluation
from noralet.evolution.watch import create_champion_live_session
from noralet.research.config import LearningCondition
from noralet.ui.session import LiveRunSetup


def _v2_config(
    root: Path,
    *,
    generations: int = 2,
    device: str = "cpu",
    benchmark_interval: int = 5,
) -> DistributionalEvolutionConfig:
    return DistributionalEvolutionConfig(
        generation_count=generations,
        device=device,
        population_size=4,
        elite_count=2,
        parent_pool_size=3,
        mutation_sigma=0.02,
        selection_world_count=2,
        benchmark_world_count=2,
        benchmark_interval=benchmark_interval,
        noralets_per_world=2,
        max_ticks=2,
        initial_seed=17,
        output_root=root,
    )


def _v1_config(root: Path) -> EvolutionConfig:
    return EvolutionConfig(
        generation_count=1,
        device="cpu",
        population_size=4,
        elite_count=2,
        parent_pool_size=3,
        mutation_sigma=0.02,
        training_world_seeds=(101,),
        validation_world_seeds=(202,),
        noralets_per_world=2,
        max_ticks=2,
        initial_seed=11,
        output_root=root,
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DistributionalEvolutionTests(unittest.TestCase):
    def test_defaults_and_fair_disjoint_generation_worlds(self) -> None:
        defaults = DistributionalEvolutionConfig()
        self.assertEqual(
            (
                defaults.device,
                defaults.population_size,
                defaults.elite_count,
                defaults.parent_pool_size,
                defaults.selection_world_count,
                defaults.benchmark_world_count,
                defaults.benchmark_interval,
                defaults.noralets_per_world,
                defaults.max_ticks,
            ),
            ("cpu", 8, 2, 4, 4, 8, 5, 4, 1_000),
        )
        with tempfile.TemporaryDirectory() as temporary:
            config = _v2_config(Path(temporary), generations=1)
            generation_zero = selection_world_seeds(config, 0)
            generation_one = selection_world_seeds(config, 1)
            benchmarks = fixed_benchmark_world_seeds(config)
            self.assertNotEqual(generation_zero, generation_one)
            self.assertFalse(set(generation_zero) & set(generation_one))
            self.assertFalse(set(generation_zero) & set(benchmarks))
            self.assertFalse(set(generation_one) & set(benchmarks))
            candidates = initialize_distributional_population(config)
            evaluations, observed = evaluate_distributional_generation(
                0,
                candidates,
                config,
            )
            self.assertEqual(observed, generation_zero)
            self.assertTrue(
                all(value.world_seeds == generation_zero for value in evaluations)
            )

    def test_benchmark_values_cannot_change_parent_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _v2_config(Path(temporary), generations=1)
            candidates = initialize_distributional_population(config)
            evaluations, _ = evaluate_distributional_generation(0, candidates, config)
            first = create_distributional_next_generation(
                0,
                candidates,
                evaluations,
                config,
            )
            changed_benchmark = CandidateEvaluation(
                candidate_id=candidates[-1].candidate_id,
                world_seeds=fixed_benchmark_world_seeds(config),
                lifetimes=(999, 999, 999, 999),
                boundary_death_count=0,
                energy_death_count=0,
                natural_death_count=0,
                consumed_energy=0.0,
            )
            self.assertGreater(
                changed_benchmark.fitness,
                max(value.fitness for value in evaluations),
            )
            second = create_distributional_next_generation(
                0,
                candidates,
                evaluations,
                config,
            )
            self.assertEqual(
                tuple(value.parent_id for value in first),
                tuple(value.parent_id for value in second),
            )
            self.assertTrue(
                all(
                    left.genome.exactly_equals(right.genome)
                    for left, right in zip(first, second, strict=True)
                )
            )

    def test_benchmark_schedule_and_best_pt_use_benchmark_not_raw_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _v2_config(root, generations=2, benchmark_interval=5)

            def benchmark(candidate, ignored_config, ignored_seeds):
                del ignored_config, ignored_seeds
                value = 20 if candidate.candidate_id.startswith("g000") else 5
                return (
                    CandidateEvaluation(
                        candidate_id=candidate.candidate_id,
                        world_seeds=(1, 3),
                        lifetimes=(value, value, value, value),
                        boundary_death_count=0,
                        energy_death_count=0,
                        natural_death_count=0,
                        consumed_energy=0.0,
                    ),
                    0.0,
                )

            with patch(
                "noralet.evolution.distributional._benchmark_evaluation",
                side_effect=benchmark,
            ):
                result = run_distributional_evolution(
                    config,
                    run_directory=root / "run",
                    progress=None,
                )
            with (result / "generations.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                generations = list(csv.DictReader(handle))
            with (result / "benchmarks.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                benchmarks = list(csv.DictReader(handle))
            self.assertEqual(tuple(generations[0]), GENERATION_COLUMNS)
            self.assertEqual(tuple(benchmarks[0]), BENCHMARK_COLUMNS)
            self.assertEqual([row["generation"] for row in benchmarks], ["0", "1"])
            self.assertEqual(benchmarks[0]["mean_lifetime"], "20.0")
            self.assertEqual(benchmarks[1]["mean_lifetime"], "5.0")
            _, best = load_champion(result)
            self.assertEqual(best["champion_kind"], "benchmark-best")
            self.assertEqual(best["generation"], 0)
            self.assertEqual(best["benchmark_mean_lifetime"], 20.0)
            manifest = json.loads((result / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["current_selection_champion"]["generation"], 1)
            self.assertEqual(manifest["benchmark_best"]["generation"], 0)

    def test_interval_schedule_includes_regular_and_final_generations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_distributional_evolution(
                _v2_config(Path(temporary), generations=4, benchmark_interval=2),
                progress=None,
            )
            with (result / "benchmarks.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["generation"] for row in rows], ["0", "2", "3"])

    def test_v1_fork_copies_population_into_new_v2_generation_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1 = run_evolution(
                _v1_config(root),
                run_directory=root / "v1",
                progress=None,
            )
            checkpoint = v1 / "evolution-state.pt"
            before_hash = _hash(checkpoint)
            source_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            source_hashes = tuple(
                _genome_sha256(
                    BaseBrainGenome.from_state(value["genome"])
                )
                for value in source_state["population"]
            )
            v2 = run_distributional_evolution(
                _v2_config(root, generations=1),
                fork_from=checkpoint,
                run_directory=root / "v2",
                progress=None,
            )
            self.assertEqual(_hash(checkpoint), before_hash)
            manifest = json.loads((v2 / "manifest.json").read_text("utf-8"))
            provenance = manifest["fork_provenance"]
            self.assertEqual(provenance["source_completed_generation"], 1)
            self.assertEqual(provenance["v2_start_generation"], 0)
            self.assertEqual(
                tuple(
                    value["genome_sha256"]
                    for value in provenance["source_candidate_identities"]
                ),
                source_hashes,
            )
            with (v2 / "candidates.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual(tuple(candidates[0]), CANDIDATE_COLUMNS)
            self.assertTrue(all(row["generation"] == "0" for row in candidates))
            self.assertTrue(
                all(row["source"] == "forked-v1-population" for row in candidates)
            )

    def test_v1_cannot_resume_as_v2_and_v2_resume_matches_uninterrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1 = run_evolution(
                _v1_config(root),
                run_directory=root / "v1",
                progress=None,
            )
            with self.assertRaisesRegex(ValueError, "not Distributional"):
                resume_distributional_evolution(
                    v1 / "evolution-state.pt",
                    generation_count=2,
                    progress=None,
                )

            interrupted = run_distributional_evolution(
                _v2_config(root, generations=2),
                run_directory=root / "interrupted",
                progress=None,
            )
            resume_distributional_evolution(
                interrupted / "evolution-state.pt",
                generation_count=4,
                progress=None,
            )
            uninterrupted = run_distributional_evolution(
                _v2_config(root, generations=4),
                run_directory=root / "uninterrupted",
                progress=None,
            )
            for name in ("generations.csv", "candidates.csv", "benchmarks.csv"):
                self.assertEqual(
                    (interrupted / name).read_text("utf-8"),
                    (uninterrupted / name).read_text("utf-8"),
                    name,
                )
            resumed_state = torch.load(
                interrupted / "evolution-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            direct_state = torch.load(
                uninterrupted / "evolution-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(resumed_state["next_generation"], 4)
            self.assertTrue(
                all(
                    _genome_sha256(_candidate_genome(left))
                    == _genome_sha256(_candidate_genome(right))
                    for left, right in zip(
                        resumed_state["population"],
                        direct_state["population"],
                        strict=True,
                    )
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_explicit_cuda_and_benchmark_best_watch_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_distributional_evolution(
                _v2_config(Path(temporary), generations=1, device="cuda"),
                progress=None,
            )
            genome, metadata = load_champion(result)
            self.assertEqual(metadata["evolution_id"], DISTRIBUTIONAL_EVOLUTION_ID)
            self.assertEqual(metadata["champion_kind"], "benchmark-best")
            setup = LiveRunSetup(
                simulation_seed=777,
                population=2,
                device="cuda",
                maximum_ticks=2,
                condition=LearningCondition.FULL_CURRENT_BRAIN,
            )
            session, watched = create_champion_live_session(result, setup)
            self.assertEqual(watched["benchmark_mean_lifetime"], metadata[
                "benchmark_mean_lifetime"
            ])
            self.assertTrue(
                all(
                    torch.equal(
                        expected,
                        dict(session.runner.brain_for(1).model.named_parameters())[
                            name
                        ].detach().cpu(),
                    )
                    for name, expected in genome.state().items()
                )
            )
            self.assertIsNotNone(session.step())


def _candidate_genome(state: dict[str, object]):
    return BaseBrainGenome.from_state(state["genome"])


if __name__ == "__main__":
    unittest.main()
