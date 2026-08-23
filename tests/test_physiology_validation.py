"""Validation tests for objective age, condition and physiology config."""

from __future__ import annotations

import dataclasses
import math
import unittest

from noralet.simulation import (
    NoraletBodyState,
    NoraletPhysiologyConfig,
    SimulationConfig,
)
from physiology_test_support import physiology_config


class PhysiologyValidationTests(unittest.TestCase):
    def test_body_has_explicit_valid_age_and_condition(self) -> None:
        body = NoraletBodyState(1, 0, age_ticks=12_345, condition=0.625)
        compatible_default = NoraletBodyState(2, 0)

        self.assertEqual(body.age_ticks, 12_345)
        self.assertEqual(body.condition, 0.625)
        self.assertEqual(compatible_default.age_ticks, 0)
        self.assertEqual(compatible_default.condition, 1.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            body.condition = 0.7  # type: ignore[misc]

    def test_body_rejects_invalid_age(self) -> None:
        with self.assertRaises(ValueError):
            NoraletBodyState(1, 0, age_ticks=-1)
        for value in (True, 1.0, "1"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    NoraletBodyState(1, 0, age_ticks=value)  # type: ignore[arg-type]

    def test_body_rejects_invalid_condition(self) -> None:
        for value in (-0.001, 1.001, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    NoraletBodyState(1, 0, condition=value)
        with self.assertRaises(TypeError):
            NoraletBodyState(1, 0, condition=True)

    def test_physiology_config_is_canonical_and_immutable(self) -> None:
        config = physiology_config()

        self.assertIsInstance(config, NoraletPhysiologyConfig)
        self.assertEqual(config.mortality_age_scale, 1_000.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.base_mortality_hazard = 1.0  # type: ignore[misc]

    def test_every_physiology_parameter_must_be_finite(self) -> None:
        valid = dataclasses.asdict(physiology_config())
        for name in valid:
            values = dict(valid)
            values[name] = math.nan
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    NoraletPhysiologyConfig(**values)

    def test_condition_configuration_bounds_are_enforced(self) -> None:
        invalid = (
            {"threshold": 0},
            {"threshold": 1.01},
            {"baseline_loss": -0.1},
            {"deprivation_scale": -0.1},
            {"deprivation_exponent": 0.99},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    physiology_config(**values)

    def test_mortality_configuration_bounds_are_enforced(self) -> None:
        invalid = (
            {"base_hazard": -0.1},
            {"age_scale": 0},
            {"age_exponent": 1},
            {"condition_hazard_scale": -0.1},
            {"condition_exponent": 0.99},
            {"age_hazard_scale": -0.1},
            {"interaction_hazard_scale": -0.1},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    physiology_config(**values)

    def test_active_physiology_requires_noralet_energy(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an active"):
            SimulationConfig(1, noralet_physiology=physiology_config())
        with self.assertRaises(TypeError):
            SimulationConfig(1, noralet_physiology=object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
