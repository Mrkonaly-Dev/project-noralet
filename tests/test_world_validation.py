"""Validation tests for finite worlds and initial living bodies."""

from __future__ import annotations

import dataclasses
import math
import unittest

from noralet.simulation import NoraletBodyState, Simulation, SimulationConfig


class WorldValidationTests(unittest.TestCase):
    def test_valid_boundaries_and_initial_bodies_are_accepted(self) -> None:
        config = SimulationConfig(
            master_seed=10,
            left_boundary=-20,
            right_boundary=30,
        )
        simulation = Simulation(
            config,
            initial_bodies=(
                NoraletBodyState(noralet_id=1, position=-10, velocity=0),
                NoraletBodyState(noralet_id=2, position=15, velocity=-0.02),
            ),
        )

        self.assertEqual(config.left_boundary, -20.0)
        self.assertEqual(config.right_boundary, 30.0)
        self.assertEqual(len(simulation.state.bodies), 2)

    def test_equal_or_reversed_boundaries_are_rejected(self) -> None:
        for left, right in ((0, 0), (1, 0)):
            with self.subTest(left=left, right=right):
                with self.assertRaises(ValueError):
                    SimulationConfig(
                        master_seed=10,
                        left_boundary=left,
                        right_boundary=right,
                    )

    def test_non_finite_boundaries_are_rejected(self) -> None:
        cases = (
            (math.nan, 1.0),
            (-1.0, math.nan),
            (-math.inf, 1.0),
            (-1.0, math.inf),
        )
        for left, right in cases:
            with self.subTest(left=left, right=right):
                with self.assertRaises(ValueError):
                    SimulationConfig(
                        master_seed=10,
                        left_boundary=left,
                        right_boundary=right,
                    )

    def test_body_state_is_finite_and_immutable(self) -> None:
        body = NoraletBodyState(noralet_id=1, position=2, velocity=-0.5)

        self.assertEqual(body.position, 2.0)
        self.assertEqual(body.velocity, -0.5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            body.position = 3.0  # type: ignore[misc]

    def test_non_finite_position_or_velocity_is_rejected(self) -> None:
        invalid_values = (math.nan, math.inf, -math.inf)
        for value in invalid_values:
            with self.subTest(field="position", value=value):
                with self.assertRaises(ValueError):
                    NoraletBodyState(noralet_id=1, position=value, velocity=0)
            with self.subTest(field="velocity", value=value):
                with self.assertRaises(ValueError):
                    NoraletBodyState(noralet_id=1, position=0, velocity=value)

    def test_duplicate_noralet_identities_are_rejected(self) -> None:
        bodies = (
            NoraletBodyState(noralet_id=7, position=-1),
            NoraletBodyState(noralet_id=7, position=1),
        )

        with self.assertRaisesRegex(ValueError, "identities must be unique"):
            Simulation(SimulationConfig(master_seed=10), initial_bodies=bodies)

    def test_initial_bodies_outside_the_world_are_rejected(self) -> None:
        config = SimulationConfig(
            master_seed=10,
            left_boundary=-10,
            right_boundary=10,
        )
        for position in (-10.01, 10.01):
            with self.subTest(position=position):
                with self.assertRaisesRegex(ValueError, "outside"):
                    Simulation(
                        config,
                        initial_bodies=(
                            NoraletBodyState(noralet_id=1, position=position),
                        ),
                    )

    def test_initial_bodies_at_exact_boundaries_are_valid(self) -> None:
        config = SimulationConfig(
            master_seed=10,
            left_boundary=-10,
            right_boundary=10,
        )
        simulation = Simulation(
            config,
            initial_bodies=(
                NoraletBodyState(noralet_id=2, position=10),
                NoraletBodyState(noralet_id=1, position=-10),
            ),
        )

        self.assertEqual(
            tuple(body.noralet_id for body in simulation.state.bodies),
            (1, 2),
        )


if __name__ == "__main__":
    unittest.main()

