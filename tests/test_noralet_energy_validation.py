"""Validation and initial-accounting tests for Noralet Energy."""

from __future__ import annotations

import dataclasses
import math
import unittest

from energy_test_support import ecology_config
from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    EnvironmentalEnergyPool,
    NoraletBodyState,
    NoraletEnergyConfig,
    RegionDefinition,
    RegionKind,
    Simulation,
    SimulationConfig,
)
from noralet_energy_test_support import noralet_energy_simulation


class NoraletEnergyValidationTests(unittest.TestCase):
    def test_body_energy_is_finite_non_negative_and_immutable(self) -> None:
        body = NoraletBodyState(1, 0, energy=4.5)

        self.assertEqual(body.energy, 4.5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            body.energy = 5.0  # type: ignore[misc]
        for value in (-1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    NoraletBodyState(1, 0, energy=value)

    def test_consume_intent_is_explicit_boolean_and_immutable(self) -> None:
        inactive = ActionIntent()
        active = ActionIntent(consume=True)

        self.assertFalse(inactive.consume)
        self.assertTrue(active.consume)
        with self.assertRaises(TypeError):
            ActionIntent(consume=1)  # type: ignore[arg-type]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            active.consume = False  # type: ignore[misc]

    def test_noralet_energy_config_accepts_only_valid_finite_values(self) -> None:
        config = NoraletEnergyConfig(10, 0, 0, 1)

        self.assertEqual(config.energy_capacity, 10.0)
        invalid = (
            (0, 0, 0, 1),
            (-1, 0, 0, 1),
            (math.inf, 0, 0, 1),
            (10, -1, 0, 1),
            (10, math.nan, 0, 1),
            (10, 0, -1, 1),
            (10, 0, math.inf, 1),
            (10, 0, 0, 0),
            (10, 0, 0, -1),
            (10, 0, 0, math.nan),
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    NoraletEnergyConfig(*values)

    def test_active_noralet_energy_requires_energy_ecology(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an active"):
            SimulationConfig(
                master_seed=1,
                noralet_energy=NoraletEnergyConfig(10, 1, 1, 1),
            )

    def test_minimum_spacing_must_be_finite_and_non_negative(self) -> None:
        region = RegionDefinition("all", -10, 10, RegionKind.INFERTILE)
        pool = EnvironmentalEnergyPool("all", 0)
        for value in (-1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ecology_config(
                        (region,),
                        (pool,),
                        minimum_spacing=value,
                    )

    def test_energy_enabled_spacing_must_strictly_exceed_twice_radius(self) -> None:
        region = RegionDefinition("all", -10, 10, RegionKind.INFERTILE)
        pool = EnvironmentalEnergyPool("all", 0)
        energy = NoraletEnergyConfig(10, 0, 0, 1)
        for spacing in (1.9, 2.0):
            with self.subTest(spacing=spacing):
                ecology = ecology_config(
                    (region,),
                    (pool,),
                    minimum_spacing=spacing,
                )
                with self.assertRaisesRegex(ValueError, "twice consume_radius"):
                    SimulationConfig(1, -10, 10, ecology, energy)

        valid = ecology_config(
            (region,),
            (pool,),
            minimum_spacing=2.000001,
        )
        SimulationConfig(1, -10, 10, valid, energy)

    def test_positive_body_energy_requires_energy_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a NoraletEnergyConfig"):
            Simulation(
                SimulationConfig(1),
                initial_bodies=(NoraletBodyState(1, 0, energy=1),),
            )

    def test_initial_body_energy_cannot_exceed_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds energy_capacity"):
            noralet_energy_simulation(
                bodies=(NoraletBodyState(1, 0, energy=10.01),),
                energy_capacity=10,
            )

    def test_initial_point_spacing_is_global_and_order_independent(self) -> None:
        regions = (
            RegionDefinition("left", -10, 0, RegionKind.INFERTILE),
            RegionDefinition("right", 0, 10, RegionKind.INFERTILE),
        )
        pools = (
            EnvironmentalEnergyPool("left", 0),
            EnvironmentalEnergyPool("right", 0),
        )
        too_close = (
            ConsumableEnergyPoint(8, 0.9, 2),
            ConsumableEnergyPoint(2, -0.9, 2),
        )

        for points in (too_close, tuple(reversed(too_close))):
            with self.subTest(points=points):
                with self.assertRaisesRegex(ValueError, "minimum_energy_point_spacing"):
                    noralet_energy_simulation(
                        regions=regions,
                        pools=pools,
                        points=points,
                        minimum_spacing=3,
                    )

    def test_points_at_exact_minimum_spacing_are_valid(self) -> None:
        simulation = noralet_energy_simulation(
            points=(
                ConsumableEnergyPoint(1, -1.5, 2),
                ConsumableEnergyPoint(2, 1.5, 2),
            ),
            minimum_spacing=3,
        )

        self.assertEqual(len(simulation.state.energy_points), 2)

    def test_initial_totals_include_all_three_energy_forms(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(
                NoraletBodyState(1, -5, energy=3),
                NoraletBodyState(2, 5, energy=7),
            ),
            points=(ConsumableEnergyPoint(4, 0, 6),),
            pools=(EnvironmentalEnergyPool("all", 20),),
        )

        totals = simulation.state.energy_totals
        self.assertEqual(totals.environmental_energy, 20.0)
        self.assertEqual(totals.consumable_energy, 6.0)
        self.assertEqual(totals.noralet_energy, 10.0)
        self.assertEqual(totals.total_energy, 36.0)
        self.assertEqual(simulation.initial_total_energy, 36.0)


if __name__ == "__main__":
    unittest.main()
