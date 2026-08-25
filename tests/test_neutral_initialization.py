"""Neutral BaseBrain initialization, provenance and compatibility tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from brain_test_support import homeostatic_config, sample_experience
from noralet import (
    BASE_BRAIN_INITIALIZATION_VERSION,
    BaseBrain,
    BaseBrainInitializationConfig,
)
from noralet.evolution.config import EVOLUTION_ID, EvolutionConfig
from noralet.evolution.distributional import (
    DISTRIBUTIONAL_EVOLUTION_ID,
    DistributionalEvolutionConfig,
    run_distributional_evolution,
)
from noralet.evolution.engine import load_champion, run_evolution
from noralet.evolution.genome import BaseBrainGenome, mutate_genome
from noralet.evolution.watch import create_champion_live_session
from noralet.research.config import (
    LearningCondition,
    _actuator_config,
    _brain_config,
    _experience_config,
    _signal_config,
    predictive_learning_config,
)
from noralet.research.initialization_audit import run_initialization_audit
from noralet.ui.session import LiveRunSetup


def _base_brain(
    seed: int,
    *,
    initialization: BaseBrainInitializationConfig | None = None,
    lifetime_learning: bool = False,
) -> BaseBrain:
    config = _brain_config(seed, "cpu")
    if initialization is not None:
        config = replace(config, initialization=initialization)
    return BaseBrain(
        config,
        _experience_config(),
        _signal_config(),
        _actuator_config(),
        predictive_learning_config() if lifetime_learning else None,
        homeostatic_config() if lifetime_learning else None,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _SequenceRandom:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


class NeutralInitializationTests(unittest.TestCase):
    def test_configuration_is_small_versioned_and_strict(self) -> None:
        config = BaseBrainInitializationConfig()
        self.assertEqual(config.version, BASE_BRAIN_INITIALIZATION_VERSION)
        self.assertEqual(
            (
                config.acceleration_output_weight_scale,
                config.initial_consume_probability,
                config.initial_signal_probability,
            ),
            (0.01, 0.05, 0.05),
        )
        for field_name in (
            "initial_consume_probability",
            "initial_signal_probability",
        ):
            for value in (0.0, 1.0, float("nan"), True):
                with self.subTest(field=field_name, value=value), self.assertRaises(
                    (TypeError, ValueError)
                ):
                    replace(config, **{field_name: value})
        with self.assertRaises(ValueError):
            replace(config, acceleration_output_weight_scale=0.0)
        with self.assertRaises(ValueError):
            replace(config, version="legacy")

    def test_action_head_biases_define_exact_content_free_baselines(self) -> None:
        initialization = BaseBrainInitializationConfig(
            acceleration_output_weight_scale=0.007,
            initial_consume_probability=0.08,
            initial_signal_probability=0.03,
        )
        model = _base_brain(31, initialization=initialization).prototype_model

        self.assertTrue(torch.equal(model.acceleration_head.bias, torch.zeros(1)))
        self.assertLessEqual(
            float(model.acceleration_head.weight.detach().abs().max()),
            initialization.acceleration_output_weight_scale,
        )
        self.assertLess(float(model.acceleration_head.weight.detach().min()), 0.0)
        self.assertGreater(float(model.acceleration_head.weight.detach().max()), 0.0)
        self.assertAlmostEqual(
            float(torch.sigmoid(model.consume_head.bias).item()),
            initialization.initial_consume_probability,
            places=6,
        )
        probabilities = torch.softmax(model.signal_head.bias.detach(), dim=0)
        self.assertAlmostEqual(
            float(probabilities[0]),
            1.0 - initialization.initial_signal_probability,
            places=6,
        )
        self.assertAlmostEqual(
            float(probabilities[1:].sum()),
            initialization.initial_signal_probability,
            places=6,
        )
        self.assertTrue(
            torch.equal(
                model.signal_head.bias[1:],
                model.signal_head.bias[1].expand(8),
            )
        )
        self.assertAlmostEqual(sum((1.0 / 9.0,) * 8), 8.0 / 9.0)

    def test_recurrent_core_is_zero_bias_and_gatewise_stable(self) -> None:
        recurrent = _base_brain(44).prototype_model.recurrent_core
        self.assertTrue(
            torch.equal(recurrent.bias_ih, torch.zeros_like(recurrent.bias_ih))
        )
        self.assertTrue(
            torch.equal(recurrent.bias_hh, torch.zeros_like(recurrent.bias_hh))
        )
        identity = torch.eye(recurrent.hidden_size)
        for gate in recurrent.weight_hh.chunk(3, dim=0):
            self.assertTrue(
                torch.allclose(gate.T @ gate, identity, atol=1e-5, rtol=1e-5)
            )

    def test_many_seed_acceleration_weights_are_symmetric_and_diverse(self) -> None:
        values = torch.cat(
            tuple(
                _base_brain(seed).prototype_model.acceleration_head.weight.flatten()
                for seed in range(100, 200)
            )
        ).detach()
        scale = BaseBrainInitializationConfig().acceleration_output_weight_scale
        self.assertLess(abs(float(values.mean())), scale / 10.0)
        self.assertGreater(float((values > 0).float().mean()), 0.4)
        self.assertLess(float((values > 0).float().mean()), 0.6)
        self.assertGreater(float(values.std(unbiased=False)), scale / 3.0)

    def test_seed_reproducibility_and_inherited_diversity_remain(self) -> None:
        first = _base_brain(101).parameter_snapshot()
        repeated = _base_brain(101).parameter_snapshot()
        different = _base_brain(102).parameter_snapshot()
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(first, repeated, strict=True))
        )
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(first, different, strict=True))
        )

    def test_initialization_audit_is_neutral_without_world_or_learning(self) -> None:
        result = run_initialization_audit(sample_count=100, audit_seed=1)
        self.assertLess(abs(result.acceleration_mean), 0.001)
        self.assertLess(result.acceleration_standard_deviation, 0.005)
        self.assertGreater(result.acceleration_positive_fraction, 0.35)
        self.assertGreater(result.acceleration_negative_fraction, 0.35)
        self.assertAlmostEqual(result.consume_activation_probability, 0.05, delta=0.01)
        self.assertAlmostEqual(result.signal_emission_probability, 0.05, delta=0.01)
        self.assertAlmostEqual(result.signal_none_probability, 0.95, delta=0.01)
        self.assertEqual(len(result.conditional_emission_probabilities), 8)
        self.assertTrue(
            all(
                abs(value - 0.125) < 0.01
                for value in result.conditional_emission_probabilities.values()
            )
        )

    def test_lifetime_plasticity_can_move_every_action_head_tensor(self) -> None:
        brain = _base_brain(61, lifetime_learning=True).spawn(
            action_random_source=_SequenceRandom((0.8, 0.01, 0.97))
        )
        before = {
            name: value.detach().clone()
            for name, value in brain.model.named_parameters()
            if name.startswith(("acceleration_head.", "consume_head.", "signal_head."))
        }
        brain.act(sample_experience(energy_distress=0.8))
        result = brain.apply_homeostatic_update(
            sample_experience(energy_distress=0.2)
        )
        after = dict(brain.model.named_parameters())
        self.assertGreater(result.applied_update_norm, 0.0)
        self.assertTrue(
            all(not torch.equal(value, after[name]) for name, value in before.items())
        )

    def test_evolutionary_mutation_can_move_every_neutral_action_parameter(self) -> None:
        genome = BaseBrainGenome.from_base_brain(_base_brain(71, lifetime_learning=True))
        mutated = mutate_genome(genome, sigma=0.02, seed=99)
        before = genome.state()
        after = mutated.state()
        action_names = tuple(
            name
            for name in before
            if name.startswith(("acceleration_head.", "consume_head.", "signal_head."))
        )
        self.assertEqual(len(action_names), 6)
        self.assertTrue(
            all(not torch.equal(before[name], after[name]) for name in action_names)
        )

    def test_explicit_historical_v1_and_v2_genomes_load_without_reinitialization(self) -> None:
        legacy = BaseBrainGenome.from_base_brain(_base_brain(81, lifetime_learning=True))
        legacy_state = {
            name: value + (index + 1) * 0.001
            for index, (name, value) in enumerate(legacy.state().items())
        }
        setup = LiveRunSetup(
            simulation_seed=3,
            population=2,
            device="cpu",
            maximum_ticks=2,
            condition=LearningCondition.FULL_CURRENT_BRAIN,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for evolution_id in (EVOLUTION_ID, DISTRIBUTIONAL_EVOLUTION_ID):
                with self.subTest(evolution_id=evolution_id):
                    result = root / evolution_id
                    champion = result / "champion" / "best.pt"
                    champion.parent.mkdir(parents=True)
                    torch.save(
                        {
                            "evolution_id": evolution_id,
                            "candidate_id": "historical-candidate",
                            "generation": 7,
                            "genome": legacy_state,
                            "initial_body_energy": 10.0,
                            "configuration": {"device": "cpu"},
                        },
                        champion,
                    )
                    before_hash = _sha256(champion)
                    loaded, _ = load_champion(result)
                    session, _ = create_champion_live_session(result, setup)
                    self.assertEqual(_sha256(champion), before_hash)
                    self.assertTrue(
                        loaded.exactly_equals(BaseBrainGenome.from_state(legacy_state))
                    )
                    parameters = dict(
                        session.runner.brain_for(1).model.named_parameters()
                    )
                    self.assertTrue(
                        all(
                            torch.equal(value, parameters[name].detach().cpu())
                            for name, value in legacy_state.items()
                        )
                    )

    def test_future_v1_and_v2_artifacts_record_neutral_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1 = run_evolution(
                EvolutionConfig(
                    generation_count=1,
                    device="cpu",
                    population_size=2,
                    elite_count=1,
                    parent_pool_size=2,
                    training_world_seeds=(101,),
                    validation_world_seeds=(202,),
                    noralets_per_world=2,
                    max_ticks=1,
                    output_root=root,
                ),
                run_directory=root / "v1",
                progress=None,
            )
            v2 = run_distributional_evolution(
                DistributionalEvolutionConfig(
                    generation_count=1,
                    device="cpu",
                    population_size=2,
                    elite_count=1,
                    parent_pool_size=2,
                    selection_world_count=1,
                    benchmark_world_count=1,
                    noralets_per_world=2,
                    max_ticks=1,
                    output_root=root,
                ),
                run_directory=root / "v2",
                progress=None,
            )
            for result in (v1, v2):
                with self.subTest(result=result.name):
                    manifest = json.loads((result / "manifest.json").read_text("utf-8"))
                    state = torch.load(
                        result / "evolution-state.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    _, champion = load_champion(result)
                    for record in (manifest, state, champion):
                        initialization = record["population_initialization"]
                        self.assertEqual(
                            initialization["base_brain_initialization"]["version"],
                            BASE_BRAIN_INITIALIZATION_VERSION,
                        )


if __name__ == "__main__":
    unittest.main()
