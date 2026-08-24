"""Validation and information-boundary tests for Iteration 6 values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import math
import unittest

from experience_test_support import experience_config, experience_simulation
from noralet import (
    ExternalPercept,
    Interoception,
    NoraletExperience,
    NoraletExperienceConfig,
    SensorimotorFeedback,
)
from noralet.simulation import NoraletBodyState, Simulation, SimulationConfig
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config


class ExperienceConfigurationTests(unittest.TestCase):
    def test_configuration_is_canonical_and_immutable(self) -> None:
        config = NoraletExperienceConfig(
            vision_radius=12,
            consumable_base_appearance=[0.7, -0.1],  # type: ignore[arg-type]
            noralet_base_appearance=[0.1, 0.8],  # type: ignore[arg-type]
            boundary_base_appearance=[-0.4, 0.2],  # type: ignore[arg-type]
            signature_length=3,
            energy_distress_exponent=2,
            condition_distress_exponent=1.5,
            motor_effort_scale=2,
            ingestion_sensation_scale=5,
            exertion_sensation_scale=3,
        )

        self.assertEqual(config.vision_radius, 12.0)
        self.assertEqual(config.consumable_base_appearance, (0.7, -0.1))
        self.assertEqual(config.base_pattern_length, 2)
        self.assertEqual(config.appearance_length, 5)
        with self.assertRaises(FrozenInstanceError):
            config.vision_radius = 2  # type: ignore[misc]

    def test_vision_radius_must_be_positive_and_finite(self) -> None:
        valid = experience_config()
        for value in (0, -1, math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(valid, vision_radius=value)
        with self.assertRaises(TypeError):
            replace(valid, vision_radius=True)

    def test_base_patterns_require_equal_nonzero_distinguishable_vectors(self) -> None:
        valid = experience_config()
        invalid_changes = (
            {"consumable_base_appearance": ()},
            {"noralet_base_appearance": (0.1,)},
            {
                "boundary_base_appearance": valid.consumable_base_appearance,
            },
            {"boundary_base_appearance": (math.nan, 0.2)},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(valid, **changes)
        for malformed in ({1.0, 2.0}, "pattern"):
            with self.subTest(malformed=malformed), self.assertRaises(TypeError):
                replace(
                    valid,
                    consumable_base_appearance=malformed,  # type: ignore[arg-type]
                )

    def test_signature_length_must_be_a_positive_integer(self) -> None:
        valid = experience_config()
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(valid, signature_length=value)
        for value in (True, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                replace(valid, signature_length=value)  # type: ignore[arg-type]

    def test_distress_exponents_enforce_their_distinct_bounds(self) -> None:
        valid = experience_config()
        for value in (1.0, 0.0, -1.0, math.inf, math.nan):
            with self.subTest(energy=value), self.assertRaises(ValueError):
                replace(valid, energy_distress_exponent=value)
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(condition=value), self.assertRaises(ValueError):
                replace(valid, condition_distress_exponent=value)

    def test_all_sensation_scales_must_be_positive_and_finite(self) -> None:
        valid = experience_config()
        for field_name in (
            "motor_effort_scale",
            "ingestion_sensation_scale",
            "exertion_sensation_scale",
        ):
            for value in (0.0, -1.0, math.inf, math.nan):
                with (
                    self.subTest(field=field_name, value=value),
                    self.assertRaises(ValueError),
                ):
                    replace(valid, **{field_name: value})

    def test_experience_requires_energy_ecology_and_physiology(self) -> None:
        config = experience_config()
        with self.assertRaises(ValueError):
            SimulationConfig(master_seed=1, noralet_experience=config)
        with self.assertRaises(ValueError):
            noralet_energy_simulation(experience=config)
        simulation = noralet_energy_simulation(
            experience=config,
            physiology=physiology_config(),
        )
        self.assertEqual(simulation.experiences_for_all(), ())


class SignatureAndExperienceValueTests(unittest.TestCase):
    def test_signature_is_finite_canonical_immutable_body_state(self) -> None:
        supplied = [0.25, -0.75]
        body = NoraletBodyState(1, 0, perceptual_signature=supplied)
        supplied[0] = 99

        self.assertEqual(body.perceptual_signature, (0.25, -0.75))
        with self.assertRaises(FrozenInstanceError):
            body.perceptual_signature = ()  # type: ignore[misc]
        for invalid in ((math.nan,), (math.inf,)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                NoraletBodyState(1, 0, perceptual_signature=invalid)
        with self.assertRaises(TypeError):
            NoraletBodyState(1, 0, perceptual_signature={1.0})  # type: ignore[arg-type]

    def test_enabled_simulation_requires_exact_signature_length(self) -> None:
        for signature in ((), (1.0,), (1.0, 2.0, 3.0)):
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                experience_simulation(
                    bodies=(
                        NoraletBodyState(
                            1,
                            0,
                            energy=10,
                            perceptual_signature=signature,
                        ),
                    )
                )

    def test_disabled_compatibility_accepts_default_empty_signature(self) -> None:
        simulation = Simulation(
            SimulationConfig(master_seed=1),
            initial_bodies=(NoraletBodyState(1, 0),),
        )
        simulation.step()

        self.assertEqual(simulation.state.body(1).perceptual_signature, ())
        with self.assertRaises(RuntimeError):
            simulation.experience_for(1)
        with self.assertRaises(RuntimeError):
            simulation.experiences_for_all()

    def test_signature_persists_exactly_across_life_transitions(self) -> None:
        signature = (0.125, -0.625)
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(
                    1,
                    0,
                    energy=50,
                    perceptual_signature=signature,
                ),
            )
        )
        for _ in range(5):
            simulation.step()
            self.assertEqual(
                simulation.state.body(1).perceptual_signature,
                signature,
            )

    def test_brain_facing_structures_have_only_the_intended_fields(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(NoraletExperience)),
            ("external_percepts", "interoception", "sensorimotor_feedback"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(ExternalPercept)),
            ("appearance_pattern", "direction_signal", "proximity_signal"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Interoception)),
            ("energy_distress", "condition_distress", "energetic_exertion"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(SensorimotorFeedback)),
            (
                "motor_direction",
                "motor_effort",
                "consume_activation",
                "ingestion_signal",
            ),
        )

    def test_experience_values_are_deeply_immutable(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(
                    1,
                    0,
                    energy=50,
                    perceptual_signature=(0.1, 0.2),
                ),
                NoraletBodyState(
                    2,
                    1,
                    energy=50,
                    perceptual_signature=(0.3, 0.4),
                ),
            )
        )
        experience = simulation.experience_for(1)

        self.assertIsInstance(experience.external_percepts, tuple)
        self.assertIsInstance(experience.external_percepts[0].appearance_pattern, tuple)
        with self.assertRaises(FrozenInstanceError):
            experience.interoception.energy_distress = 0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            experience.external_percepts[0].direction_signal = 0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
