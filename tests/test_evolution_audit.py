"""Focused saved-genome generalization and throughput audit tests."""

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
from noralet.evolution.engine import load_champion, save_champion
from noralet.evolution.selection import initialize_generation_zero
from noralet.research.evolution_audit import (
    AUDIT_ID,
    GENOME_SUMMARY_COLUMNS,
    QUALITATIVE_PROBE_SEEDS,
    REQUIRED_CHECKPOINTS,
    WORLD_RESULT_COLUMNS,
    EvolutionAuditConfig,
    benchmark_saved_checkpoint,
    derive_audit_world_seeds,
    locate_audit_checkpoints,
    run_evolution_audit,
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage(root: Path) -> Path:
    result = root / "lineage"
    champion = result / "champion"
    config = EvolutionConfig(
        generation_count=15,
        device="cpu",
        population_size=2,
        elite_count=1,
        parent_pool_size=2,
        training_world_seeds=(1101, 2203),
        validation_world_seeds=(5501, 6607),
        noralets_per_world=2,
        max_ticks=2,
        initial_seed=7,
        output_root=root,
    )
    candidate = initialize_generation_zero(config)[0]
    generations = (0, 5, 10, 14, 14)
    for name, generation in zip(REQUIRED_CHECKPOINTS, generations, strict=True):
        save_champion(
            champion / name,
            candidate,
            generation=generation,
            training_fitness=10.0 + generation,
            validation_fitness=9.0 + generation,
            config=config,
        )
    (result / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "test-lineage",
                "status": "completed",
                "training_world_seeds": [1101, 2203],
                "validation_world_seeds": [5501, 6607],
            }
        ),
        encoding="utf-8",
    )
    return result


class EvolutionAuditTests(unittest.TestCase):
    def test_saved_genomes_load_and_audit_seeds_are_new(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lineage = _lineage(Path(temporary))
            checkpoints = locate_audit_checkpoints(lineage)
            self.assertEqual(tuple(path.name for path in checkpoints), REQUIRED_CHECKPOINTS)
            for checkpoint in checkpoints:
                genome, metadata = load_champion(checkpoint)
                self.assertTrue(genome.state())
                self.assertEqual(metadata["checkpoint_path"], str(checkpoint.resolve()))

            excluded = (1101, 2203, 5501, 6607) + QUALITATIVE_PROBE_SEEDS
            first = derive_audit_world_seeds(
                20_260_825,
                role="unseen-generalization",
                count=8,
                excluded=excluded,
            )
            second = derive_audit_world_seeds(
                20_260_825,
                role="unseen-generalization",
                count=8,
                excluded=excluded,
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), len(set(first)))
            self.assertFalse(set(first) & set(excluded))

    def test_output_schemas_shared_seeds_and_no_evolutionary_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lineage = _lineage(root)
            checkpoints = locate_audit_checkpoints(lineage)
            before = {path.name: _file_hash(path) for path in checkpoints}
            config = EvolutionAuditConfig(
                evolution_result=lineage,
                output_root=root / "research-results",
                unseen_world_count=2,
                benchmark_world_count=1,
                noralets_per_world=2,
                generalization_max_ticks=2,
                performance_max_ticks=2,
                timing_repetitions=1,
                timing_warmups=0,
                generalization_device="cpu",
                performance_devices=("cpu",),
            )
            with (
                patch("noralet.evolution.genome.mutate_genome") as mutation,
                patch("noralet.evolution.selection.create_next_generation") as selection,
            ):
                result = run_evolution_audit(config)
            mutation.assert_not_called()
            selection.assert_not_called()
            self.assertEqual(
                before,
                {path.name: _file_hash(path) for path in checkpoints},
            )

            required = (
                "manifest.json",
                "genome-summary.csv",
                "world-results.csv",
                "performance.json",
                "summary.md",
            )
            self.assertTrue(all((result / name).is_file() for name in required))
            with (result / "genome-summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                summaries = list(csv.DictReader(handle))
                self.assertEqual(tuple(summaries[0]), GENOME_SUMMARY_COLUMNS)
            with (result / "world-results.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                worlds = list(csv.DictReader(handle))
                self.assertEqual(tuple(worlds[0]), WORLD_RESULT_COLUMNS)
            self.assertEqual(len(summaries), 5)
            self.assertEqual(len(worlds), 10)
            seeds_by_genome = {
                label: tuple(
                    int(row["world_seed"])
                    for row in worlds
                    if row["checkpoint_label"] == label
                )
                for label in {row["checkpoint_label"] for row in worlds}
            }
            self.assertEqual(len(set(seeds_by_genome.values())), 1)

            manifest = json.loads((result / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["audit_id"], AUDIT_ID)
            self.assertFalse(manifest["generalization_protocol"]["selection"])
            self.assertFalse(manifest["generalization_protocol"]["mutation"])
            self.assertFalse(
                manifest["generalization_protocol"]["continued_evolution"]
            )
            self.assertFalse(
                set(manifest["unseen_world_seeds"])
                & {1101, 2203, 5501, 6607, *QUALITATIVE_PROBE_SEEDS}
            )
            summary = (result / "summary.md").read_text("utf-8")
            for heading in (
                "## Generalization",
                "## World variance",
                "## CPU vs CUDA",
                "## Observations",
                "## Decision gate",
            ):
                self.assertIn(heading, summary)

    def test_cpu_timing_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = locate_audit_checkpoints(_lineage(Path(temporary)))[1]
            result = benchmark_saved_checkpoint(
                checkpoint,
                world_seeds=(123_456,),
                device="cpu",
                noralets_per_world=2,
                max_ticks=2,
                repetitions=2,
                warmups=1,
            )
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(len(result["measurements"]), 2)
            self.assertGreater(result["mean_wall_clock_seconds"], 0.0)
            self.assertGreater(result["mean_effective_world_ticks_per_second"], 0.0)
            self.assertEqual(
                len(
                    {
                        row["noralet_activations_or_lived_transitions"]
                        for row in result["measurements"]
                    }
                ),
                1,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_timing_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = locate_audit_checkpoints(_lineage(Path(temporary)))[1]
            result = benchmark_saved_checkpoint(
                checkpoint,
                world_seeds=(123_456,),
                device="cuda",
                noralets_per_world=2,
                max_ticks=2,
                repetitions=1,
                warmups=1,
            )
            self.assertEqual(result["device"], "cuda")
            self.assertEqual(len(result["measurements"]), 1)
            self.assertGreater(result["measurements"][0]["wall_clock_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
