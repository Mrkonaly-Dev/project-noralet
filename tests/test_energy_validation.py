"""Validation tests for regions and closed-energy values."""

from __future__ import annotations

import dataclasses
import math
import unittest

from energy_test_support import DEFAULT_PROBABILITIES, ecology_config
from noralet.simulation import (
    ConsumableEnergyPoint,
    EnergyEcologyConfig,
    EnvironmentalEnergyPool,
    FormationProbabilities,
    RegionDefinition,
    RegionKind,
    Simulation,
    SimulationConfig,
)


def region(
    region_id: str,
    left: float,
    right: float,
    kind: RegionKind = RegionKind.SPARSE,
) -> RegionDefinition:
    return RegionDefinition(region_id, left, right, kind)


def pool(region_id: str, energy: float = 0.0) -> EnvironmentalEnergyPool:
    return EnvironmentalEnergyPool(region_id, energy)


class RegionAndEnergyValidationTests(unittest.TestCase):
    def test_valid_partition_is_canonical_and_immutable(self) -> None:
        right = region("right", 0, 10, RegionKind.FERTILE)
        left = region("left", -10, 0, RegionKind.INFERTILE)
        ecology = ecology_config(
            (right, left),
            (pool("right", 7), pool("left", 3)),
        )
        config = SimulationConfig(5, -10, 10, ecology)

        self.assertEqual(tuple(item.region_id for item in ecology.regions), ("left", "right"))
        self.assertEqual(
            tuple(item.region_id for item in ecology.initial_environmental_energy),
            ("left", "right"),
        )
        self.assertIs(config.energy_ecology, ecology)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            left.left = -20.0  # type: ignore[misc]

    def test_partition_gap_is_rejected(self) -> None:
        ecology = ecology_config(
            (region("left", -10, -1), region("right", 0, 10)),
            (pool("left"), pool("right")),
        )

        with self.assertRaisesRegex(ValueError, "gap"):
            SimulationConfig(1, -10, 10, ecology)

    def test_partition_overlap_is_rejected(self) -> None:
        ecology = ecology_config(
            (region("left", -10, 1), region("right", 0, 10)),
            (pool("left"), pool("right")),
        )

        with self.assertRaisesRegex(ValueError, "overlap"):
            SimulationConfig(1, -10, 10, ecology)

    def test_region_outside_world_is_rejected(self) -> None:
        ecology = ecology_config(
            (region("all", -11, 10),),
            (pool("all"),),
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            SimulationConfig(1, -10, 10, ecology)

    def test_incomplete_world_coverage_is_rejected(self) -> None:
        for bounds in ((-9, 10), (-10, 9)):
            with self.subTest(bounds=bounds):
                ecology = ecology_config(
                    (region("partial", *bounds),),
                    (pool("partial"),),
                )
                with self.assertRaisesRegex(ValueError, "complete"):
                    SimulationConfig(1, -10, 10, ecology)

    def test_shared_coordinate_belongs_to_right_region_and_final_right_is_included(self) -> None:
        ecology = ecology_config(
            (region("left", -10, 0), region("right", 0, 10)),
            (pool("left"), pool("right")),
        )

        self.assertEqual(ecology.region_for(-10).region_id, "left")
        self.assertEqual(ecology.region_for(-0.0001).region_id, "left")
        self.assertEqual(ecology.region_for(0).region_id, "right")
        self.assertEqual(ecology.region_for(10).region_id, "right")

    def test_duplicate_region_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "region identities"):
            ecology_config(
                (region("same", -10, 0), region("same", 0, 10)),
                (pool("same"),),
            )

    def test_pools_must_be_unique_and_match_regions_exactly(self) -> None:
        regions = (region("left", -10, 0), region("right", 0, 10))

        with self.assertRaisesRegex(ValueError, "pool region identities"):
            ecology_config(regions, (pool("left"), pool("left")))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ecology_config(regions, (pool("left"),))

    def test_probability_bounds_and_fertility_hierarchy_are_enforced(self) -> None:
        invalid = (
            (-0.1, 0.5, 1.0),
            (0.0, 0.5, 1.1),
            (0.0, math.nan, 1.0),
            (0.0, 0.5, math.inf),
            (0.0, 0.5, 0.5),
            (0.6, 0.5, 1.0),
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    FormationProbabilities(*values)

        self.assertEqual(DEFAULT_PROBABILITIES.for_kind(RegionKind.FERTILE), 1.0)
        self.assertEqual(DEFAULT_PROBABILITIES.for_kind(RegionKind.SPARSE), 0.5)
        self.assertEqual(DEFAULT_PROBABILITIES.for_kind(RegionKind.INFERTILE), 0.0)

    def test_invalid_formation_range_decay_and_threshold_are_rejected(self) -> None:
        regions = (region("all", -1, 1),)
        pools = (pool("all"),)
        invalid_options = (
            {"formation_min": 0.0},
            {"formation_min": 3.0, "formation_max": 2.0},
            {"formation_max": math.inf},
            {"decay_rate": -0.01},
            {"decay_rate": 1.01},
            {"decay_rate": math.nan},
            {"removal_threshold": -0.01},
            {"removal_threshold": 2.0},
            {"removal_threshold": math.inf},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    ecology_config(regions, pools, **options)

    def test_environmental_energy_must_be_finite_and_non_negative(self) -> None:
        for value in (-1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EnvironmentalEnergyPool("region", value)

    def test_consumable_point_requires_valid_identity_position_and_positive_energy(self) -> None:
        invalid = (
            {"point_id": -1, "position": 0, "energy": 1},
            {"point_id": 1, "position": math.nan, "energy": 1},
            {"point_id": 1, "position": math.inf, "energy": 1},
            {"point_id": 1, "position": 0, "energy": 0},
            {"point_id": 1, "position": 0, "energy": -1},
            {"point_id": 1, "position": 0, "energy": math.inf},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ConsumableEnergyPoint(**values)

    def test_duplicate_initial_point_id_is_rejected(self) -> None:
        ecology = ecology_config(
            (region("all", -1, 1),),
            (pool("all"),),
        )
        points = (
            ConsumableEnergyPoint(4, -0.5, 1),
            ConsumableEnergyPoint(4, 0.5, 2),
        )

        with self.assertRaisesRegex(ValueError, "identities must be unique"):
            Simulation(SimulationConfig(1, -1, 1, ecology), initial_energy_points=points)

    def test_initial_points_outside_world_are_rejected(self) -> None:
        ecology = ecology_config(
            (region("all", -1, 1),),
            (pool("all"),),
        )
        config = SimulationConfig(1, -1, 1, ecology)

        for position in (-1.01, 1.01):
            with self.subTest(position=position):
                with self.assertRaisesRegex(ValueError, "outside"):
                    Simulation(
                        config,
                        initial_energy_points=(ConsumableEnergyPoint(0, position, 1),),
                    )

    def test_initial_points_at_world_and_shared_boundaries_are_valid(self) -> None:
        ecology = ecology_config(
            (region("left", -1, 0), region("right", 0, 1)),
            (pool("left"), pool("right")),
        )
        simulation = Simulation(
            SimulationConfig(1, -1, 1, ecology),
            initial_energy_points=(
                ConsumableEnergyPoint(2, 1, 1),
                ConsumableEnergyPoint(1, 0, 1),
                ConsumableEnergyPoint(0, -1, 1),
            ),
        )

        self.assertEqual(
            tuple(point.point_id for point in simulation.state.energy_points),
            (0, 1, 2),
        )
        self.assertEqual(ecology.region_for(0).region_id, "right")

    def test_points_require_an_explicit_energy_ecology(self) -> None:
        with self.assertRaisesRegex(ValueError, "require an EnergyEcologyConfig"):
            Simulation(
                SimulationConfig(1),
                initial_energy_points=(ConsumableEnergyPoint(0, 0, 1),),
            )


if __name__ == "__main__":
    unittest.main()
