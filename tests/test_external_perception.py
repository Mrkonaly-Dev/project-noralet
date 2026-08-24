"""External sensory-field tests for Iteration 6."""

from __future__ import annotations

import unittest

from experience_test_support import experience_config, experience_simulation
from noralet.simulation import (
    ConsumableEnergyPoint,
    EnvironmentalEnergyPool,
    NoraletBodyState,
    RegionDefinition,
    RegionKind,
)
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config


class ExternalPerceptionTests(unittest.TestCase):
    def test_own_body_is_not_an_external_percept(self) -> None:
        config = experience_config(vision_radius=3)
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0.7, 0.9)),
            ),
        )

        self.assertEqual(simulation.experience_for(1).external_percepts, ())

    def test_every_visible_object_uses_left_to_right_spatial_order(self) -> None:
        config = experience_config(vision_radius=5)
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(10, 0, energy=50, perceptual_signature=(1, 0)),
                NoraletBodyState(3, -4, energy=20, perceptual_signature=(0.3, 0.4)),
                NoraletBodyState(4, 0, energy=20, perceptual_signature=(0.5, 0.6)),
                NoraletBodyState(5, 6, energy=20, perceptual_signature=(0.7, 0.8)),
            ),
            points=(
                ConsumableEnergyPoint(8, -5, 2),
                ConsumableEnergyPoint(2, 1, 2),
            ),
        )

        percepts = simulation.experience_for(10).external_percepts

        self.assertEqual(len(percepts), 4)
        self.assertEqual(
            tuple(percept.appearance_pattern for percept in percepts),
            (
                (*config.consumable_base_appearance, 0.0, 0.0),
                (*config.noralet_base_appearance, 0.3, 0.4),
                (*config.noralet_base_appearance, 0.5, 0.6),
                (*config.consumable_base_appearance, 0.0, 0.0),
            ),
        )
        self.assertEqual(
            tuple(percept.direction_signal for percept in percepts),
            (-1.0, -1.0, 0.0, 1.0),
        )
        for observed, expected in zip(
            (percept.proximity_signal for percept in percepts),
            (0.0, 0.2, 1.0, 0.8),
        ):
            self.assertAlmostEqual(observed, expected)

    def test_visibility_radius_is_inclusive_and_excludes_just_outside(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
                NoraletBodyState(2, -5, energy=50, perceptual_signature=(1, 0)),
                NoraletBodyState(
                    3,
                    5.0000001,
                    energy=50,
                    perceptual_signature=(0, 1),
                ),
            )
        )

        percepts = simulation.experience_for(1).external_percepts

        self.assertEqual(len(percepts), 1)
        self.assertEqual(percepts[0].direction_signal, -1.0)
        self.assertEqual(percepts[0].proximity_signal, 0.0)

    def test_boundary_at_radius_and_at_body_position_has_forced_direction(self) -> None:
        config = experience_config(vision_radius=5)
        near_left = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, -5, energy=50, perceptual_signature=(0, 0)),
            ),
        ).experience_for(1)
        on_right = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 10, energy=50, perceptual_signature=(0, 0)),
            ),
        ).experience_for(1)

        self.assertEqual(len(near_left.external_percepts), 1)
        self.assertEqual(near_left.external_percepts[0].direction_signal, -1.0)
        self.assertEqual(near_left.external_percepts[0].proximity_signal, 0.0)
        self.assertEqual(len(on_right.external_percepts), 1)
        self.assertEqual(on_right.external_percepts[0].direction_signal, 1.0)
        self.assertEqual(on_right.external_percepts[0].proximity_signal, 1.0)

    def test_boundaries_are_absent_until_they_enter_vision_radius(self) -> None:
        body = NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0))
        outside = experience_simulation(
            experience=experience_config(vision_radius=9.999),
            bodies=(body,),
        )
        inclusive = experience_simulation(
            experience=experience_config(vision_radius=10),
            bodies=(body,),
        )

        self.assertEqual(outside.experience_for(1).external_percepts, ())
        self.assertEqual(len(inclusive.experience_for(1).external_percepts), 2)

    def test_same_position_tie_break_is_stable_but_not_exposed(self) -> None:
        config = experience_config(vision_radius=1)
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(9, -10, energy=50, perceptual_signature=(0, 0)),
                NoraletBodyState(2, -10, energy=50, perceptual_signature=(0.2, 0.3)),
            ),
            points=(ConsumableEnergyPoint(7, -10, 1),),
        )

        percepts = simulation.experience_for(9).external_percepts

        self.assertEqual(
            tuple(percept.appearance_pattern for percept in percepts),
            (
                (*config.boundary_base_appearance, 0.0, 0.0),
                (*config.consumable_base_appearance, 0.0, 0.0),
                (*config.noralet_base_appearance, 0.2, 0.3),
            ),
        )
        self.assertTrue(
            all(percept.direction_signal in (-1.0, 0.0) for percept in percepts)
        )

    def test_external_collection_has_no_artificial_object_cap(self) -> None:
        config = experience_config(vision_radius=10)
        points = tuple(
            ConsumableEnergyPoint(index, position, 1)
            for index, position in enumerate((-9, -6, -3, 0, 3, 6, 9))
        )
        other_bodies = tuple(
            NoraletBodyState(
                index,
                position,
                energy=10,
                perceptual_signature=(index / 10, -index / 10),
            )
            for index, position in enumerate((-8, -4, -2, 2, 4, 8), start=2)
        )
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
                *other_bodies,
            ),
            points=points,
        )

        percepts = simulation.experience_for(1).external_percepts

        self.assertEqual(len(percepts), 15)
        self.assertTrue(
            all(
                len(percept.appearance_pattern) == config.appearance_length
                for percept in percepts
            )
        )

    def test_objective_other_body_facts_remain_private(self) -> None:
        focal = NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0))
        first = experience_simulation(
            bodies=(
                focal,
                NoraletBodyState(
                    2,
                    2,
                    velocity=-3,
                    energy=99,
                    age_ticks=1,
                    condition=0.95,
                    perceptual_signature=(0.25, 0.75),
                ),
            )
        )
        second = experience_simulation(
            bodies=(
                focal,
                NoraletBodyState(
                    99,
                    2,
                    velocity=8,
                    energy=1,
                    age_ticks=50_000,
                    condition=0.1,
                    perceptual_signature=(0.25, 0.75),
                ),
            )
        )

        self.assertEqual(first.experience_for(1), second.experience_for(1))

    def test_signature_changes_appearance_without_exposing_simulation_id(self) -> None:
        config = experience_config()
        first = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
                NoraletBodyState(2, 1, energy=50, perceptual_signature=(0.2, 0.8)),
            ),
        ).experience_for(1).external_percepts[0]
        second = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
                NoraletBodyState(999, 1, energy=50, perceptual_signature=(0.6, 0.4)),
            ),
        ).experience_for(1).external_percepts[0]

        self.assertEqual(
            first.appearance_pattern[: config.base_pattern_length],
            config.noralet_base_appearance,
        )
        self.assertEqual(first.appearance_pattern[-2:], (0.2, 0.8))
        self.assertEqual(second.appearance_pattern[-2:], (0.6, 0.4))
        self.assertFalse(hasattr(first, "noralet_id"))
        self.assertFalse(hasattr(first, "object_type"))

    def test_different_individuals_share_base_but_keep_distinct_signatures(self) -> None:
        config = experience_config()
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
                NoraletBodyState(2, -1, energy=50, perceptual_signature=(0.2, 0.8)),
                NoraletBodyState(3, 1, energy=50, perceptual_signature=(0.6, 0.4)),
            ),
        )

        appearances = tuple(
            percept.appearance_pattern
            for percept in simulation.experience_for(1).external_percepts
        )

        self.assertEqual(
            tuple(item[: config.base_pattern_length] for item in appearances),
            (config.noralet_base_appearance, config.noralet_base_appearance),
        )
        self.assertEqual(tuple(item[-2:] for item in appearances), ((0.2, 0.8), (0.6, 0.4)))

    def test_motion_changes_spatial_sensation_without_exposing_velocity(self) -> None:
        config = experience_config(vision_radius=5)
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
                NoraletBodyState(
                    2,
                    2,
                    velocity=1,
                    energy=50,
                    perceptual_signature=(0.4, 0.6),
                ),
            ),
        )

        before = simulation.experience_for(1).external_percepts[0]
        simulation.step()
        after = simulation.experience_for(1).external_percepts[0]

        self.assertEqual(before.appearance_pattern, after.appearance_pattern)
        self.assertGreater(before.proximity_signal, after.proximity_signal)
        self.assertFalse(hasattr(before, "velocity"))
        self.assertFalse(hasattr(before, "distance"))

    def test_individual_appearance_stays_stable_across_positions_and_ticks(self) -> None:
        config = experience_config(vision_radius=10)
        signature = (0.125, -0.625)
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
                NoraletBodyState(
                    2,
                    -3,
                    velocity=1,
                    energy=50,
                    perceptual_signature=signature,
                ),
            ),
        )
        appearances = []

        for _ in range(4):
            appearances.append(
                next(
                    percept.appearance_pattern
                    for percept in simulation.experience_for(1).external_percepts
                    if percept.appearance_pattern[: config.base_pattern_length]
                    == config.noralet_base_appearance
                )
            )
            simulation.step()

        self.assertEqual(
            appearances,
            [(*config.noralet_base_appearance, *signature)] * 4,
        )

    def test_region_structure_and_environmental_pools_are_not_perceived(self) -> None:
        body = NoraletBodyState(
            1,
            0,
            energy=50,
            perceptual_signature=(0.1, 0.2),
        )
        config = experience_config(vision_radius=3)
        one_region = experience_simulation(
            bodies=(body,),
            experience=config,
        )
        split_regions = (
            RegionDefinition("left", -10, 0, RegionKind.FERTILE),
            RegionDefinition("right", 0, 10, RegionKind.SPARSE),
        )
        split = noralet_energy_simulation(
            bodies=(body,),
            regions=split_regions,
            pools=(
                EnvironmentalEnergyPool("left", 20),
                EnvironmentalEnergyPool("right", 30),
            ),
            physiology=physiology_config(baseline_loss=0),
            experience=config,
        )

        self.assertEqual(one_region.experience_for(1), split.experience_for(1))
        self.assertEqual(one_region.experience_for(1).external_percepts, ())

    def test_dead_noralet_disappears_from_external_perception(self) -> None:
        config = experience_config(vision_radius=10)
        simulation = experience_simulation(
            experience=config,
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0, 0)),
                NoraletBodyState(
                    2,
                    9,
                    velocity=2,
                    energy=50,
                    perceptual_signature=(1, 1),
                ),
            ),
        )
        self.assertIn(
            (*config.noralet_base_appearance, 1.0, 1.0),
            tuple(
                percept.appearance_pattern
                for percept in simulation.experience_for(1).external_percepts
            ),
        )

        simulation.step()

        self.assertNotIn(
            (*config.noralet_base_appearance, 1.0, 1.0),
            tuple(
                percept.appearance_pattern
                for percept in simulation.experience_for(1).external_percepts
            ),
        )
        with self.assertRaises(KeyError):
            simulation.experience_for(2)


if __name__ == "__main__":
    unittest.main()
