"""Validation tests for external acceleration intentions."""

from __future__ import annotations

import dataclasses
import math
import unittest

from noralet.simulation import (
    ActionIntent,
    NoraletBodyState,
    Simulation,
    SimulationConfig,
)


class ActionIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = Simulation(
            SimulationConfig(
                master_seed=1,
                left_boundary=-1,
                right_boundary=1,
            ),
            initial_bodies=(
                NoraletBodyState(noralet_id=1, position=0, velocity=0),
            ),
        )

    def test_intent_is_finite_and_immutable(self) -> None:
        intent = ActionIntent(acceleration=1)

        self.assertEqual(intent.acceleration, 1.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            intent.acceleration = 2.0  # type: ignore[misc]

        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ActionIntent(acceleration=value)

    def test_unknown_target_is_rejected_before_state_changes(self) -> None:
        state_before = self.simulation.state

        with self.assertRaisesRegex(ValueError, "non-living"):
            self.simulation.step({999: ActionIntent(acceleration=1)})

        self.assertIs(self.simulation.state, state_before)
        self.assertEqual(self.simulation.state.tick, 0)

    def test_targeting_a_dead_noralet_is_rejected(self) -> None:
        self.simulation.step({1: ActionIntent(acceleration=2)})
        self.assertEqual(self.simulation.state.bodies, ())

        with self.assertRaisesRegex(ValueError, "non-living"):
            self.simulation.step({1: ActionIntent(acceleration=0)})

        self.assertEqual(self.simulation.state.tick, 1)

    def test_non_mapping_intent_collection_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.simulation.step([ActionIntent(acceleration=1)])  # type: ignore[arg-type]

    def test_non_int_target_or_non_intent_value_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.simulation.step({"1": ActionIntent(1)})  # type: ignore[dict-item]
        with self.assertRaises(TypeError):
            self.simulation.step({1: 1.0})  # type: ignore[dict-item]


if __name__ == "__main__":
    unittest.main()

