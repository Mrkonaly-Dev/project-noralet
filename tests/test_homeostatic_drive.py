"""Homeostatic-drive configuration and bounded-modulation tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import math
import unittest

from brain_test_support import homeostatic_config
from noralet import (
    Interoception,
    NoraletHomeostaticPlasticityConfig,
    homeostatic_drive,
    homeostatic_modulation,
)


class HomeostaticConfigurationTests(unittest.TestCase):
    def test_configuration_is_focused_immutable_and_canonical(self) -> None:
        config = homeostatic_config()

        self.assertEqual(
            tuple(field.name for field in fields(config)),
            (
                "energy_distress_weight",
                "condition_distress_weight",
                "homeostatic_modulation_scale",
                "eligibility_decay",
                "action_learning_rate",
                "max_homeostatic_update_norm",
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            config.eligibility_decay = 0.2  # type: ignore[misc]

    def test_distress_weights_are_non_negative_and_not_both_zero(self) -> None:
        for name in ("energy_distress_weight", "condition_distress_weight"):
            for value in (-1, math.inf, -math.inf, math.nan, True):
                values = dict(
                    energy_distress_weight=1.0,
                    condition_distress_weight=1.0,
                    homeostatic_modulation_scale=0.2,
                    eligibility_decay=0.8,
                    action_learning_rate=0.05,
                    max_homeostatic_update_norm=2.0,
                )
                values[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        NoraletHomeostaticPlasticityConfig(**values)
        with self.assertRaisesRegex(ValueError, "at least one"):
            homeostatic_config(
                energy_distress_weight=0,
                condition_distress_weight=0,
            )

    def test_positive_controls_and_half_open_decay_are_validated(self) -> None:
        positive_names = (
            "homeostatic_modulation_scale",
            "action_learning_rate",
            "max_homeostatic_update_norm",
        )
        for name in positive_names:
            for value in (0, -1, math.inf, math.nan, True):
                values = dict(
                    energy_distress_weight=1.0,
                    condition_distress_weight=1.0,
                    homeostatic_modulation_scale=0.2,
                    eligibility_decay=0.8,
                    action_learning_rate=0.05,
                    max_homeostatic_update_norm=2.0,
                )
                values[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        NoraletHomeostaticPlasticityConfig(**values)
        for decay in (-0.1, 1.0, 1.1, math.inf, math.nan, True):
            with self.subTest(decay=decay):
                with self.assertRaises((TypeError, ValueError)):
                    homeostatic_config(eligibility_decay=decay)
        self.assertEqual(homeostatic_config(eligibility_decay=0).eligibility_decay, 0)


class HomeostaticDriveTests(unittest.TestCase):
    @staticmethod
    def interoception(energy: float, condition: float) -> Interoception:
        return Interoception(energy, condition, 0.999)

    def test_weighted_drive_is_exact_and_bounded(self) -> None:
        config = homeostatic_config(
            energy_distress_weight=3,
            condition_distress_weight=1,
        )

        drive = homeostatic_drive(self.interoception(0.8, 0.2), config)

        self.assertEqual(drive, (3 * 0.8 + 1 * 0.2) / 4)
        self.assertGreaterEqual(drive, 0.0)
        self.assertLessEqual(drive, 1.0)

    def test_zero_distress_produces_zero_drive(self) -> None:
        self.assertEqual(
            homeostatic_drive(self.interoception(0, 0), homeostatic_config()),
            0.0,
        )

    def test_higher_distress_produces_higher_drive(self) -> None:
        config = homeostatic_config()
        low = homeostatic_drive(self.interoception(0.1, 0.2), config)
        high = homeostatic_drive(self.interoception(0.7, 0.9), config)

        self.assertGreater(high, low)

    def test_exertion_does_not_enter_drive(self) -> None:
        config = homeostatic_config()
        low_exertion = homeostatic_drive(Interoception(0.4, 0.2, 0), config)
        high_exertion = homeostatic_drive(
            Interoception(0.4, 0.2, math.nextafter(1.0, 0.0)),
            config,
        )

        self.assertEqual(low_exertion, high_exertion)

    def test_drive_api_requires_only_brain_facing_interoception(self) -> None:
        import inspect

        self.assertEqual(
            tuple(inspect.signature(homeostatic_drive).parameters),
            ("interoception", "config"),
        )
        self.assertNotIn("Body", inspect.getsource(homeostatic_drive))
        self.assertNotIn("Energy", inspect.getsource(homeostatic_drive))


class HomeostaticModulationTests(unittest.TestCase):
    def test_improvement_worsening_and_neutral_polarity(self) -> None:
        config = homeostatic_config()

        self.assertGreater(homeostatic_modulation(0.8, 0.2, config), 0.0)
        self.assertLess(homeostatic_modulation(0.2, 0.8, config), 0.0)
        self.assertEqual(homeostatic_modulation(0.4, 0.4, config), 0.0)

    def test_tanh_modulation_is_bounded_and_scale_sensitive(self) -> None:
        narrow = homeostatic_config(homeostatic_modulation_scale=1e-6)
        modulation = homeostatic_modulation(1.0, 0.0, narrow)

        self.assertGreaterEqual(modulation, 0.0)
        self.assertLess(modulation, 1.0)


if __name__ == "__main__":
    unittest.main()
