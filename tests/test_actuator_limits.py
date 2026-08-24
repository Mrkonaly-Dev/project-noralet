"""Physical acceleration-limit validation and compatibility tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import unittest

from brain_test_support import actuator_config, brain_body
from noralet import (
    ActionIntent,
    NoraletActuatorConfig,
    NoraletAccelerated,
    NoraletBodyState,
    Simulation,
    SimulationConfig,
)
from noralet_energy_test_support import noralet_energy_simulation


class ActuatorConfigurationTests(unittest.TestCase):
    def test_actuator_limit_is_positive_finite_and_immutable(self) -> None:
        config = NoraletActuatorConfig(2)
        self.assertEqual(config.max_acceleration, 2.0)
        with self.assertRaises(FrozenInstanceError):
            config.max_acceleration = 3  # type: ignore[misc]
        for value in (0, -1, math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NoraletActuatorConfig(value)
        with self.assertRaises(TypeError):
            NoraletActuatorConfig(True)

    def test_simulation_config_rejects_non_actuator_value(self) -> None:
        with self.assertRaises(TypeError):
            SimulationConfig(
                master_seed=1,
                noralet_actuators=object(),  # type: ignore[arg-type]
            )


class PhysicalActuatorTests(unittest.TestCase):
    def test_positive_manual_acceleration_saturates_in_legacy_mode(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                master_seed=1,
                noralet_actuators=actuator_config(2),
            ),
            initial_bodies=(NoraletBodyState(1, 0),),
        )

        simulation.step({1: ActionIntent(acceleration=50)})

        self.assertEqual(simulation.state.body(1).velocity, 2)
        self.assertEqual(simulation.state.body(1).position, 2)

    def test_negative_manual_acceleration_saturates_symmetrically(self) -> None:
        simulation = Simulation(
            SimulationConfig(
                master_seed=1,
                noralet_actuators=actuator_config(2),
            ),
            initial_bodies=(NoraletBodyState(1, 0),),
        )

        simulation.step({1: ActionIntent(acceleration=-50)})

        self.assertEqual(simulation.state.body(1).velocity, -2)
        self.assertEqual(simulation.state.body(1).position, -2)

    def test_body_saturation_occurs_before_energy_affordability(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(brain_body(1, 0, energy=10),),
            acceleration_cost=3,
            actuators=actuator_config(2),
        )

        simulation.step({1: ActionIntent(acceleration=50)})

        self.assertEqual(simulation.state.body(1).velocity, 2)
        self.assertEqual(simulation.state.body(1).energy, 4)

    def test_energy_affordability_still_reduces_body_limited_request(self) -> None:
        simulation = noralet_energy_simulation(
            bodies=(brain_body(1, 0, energy=3),),
            acceleration_cost=3,
            actuators=actuator_config(2),
        )

        result = simulation.step({1: ActionIntent(acceleration=50)})

        acceleration_events = [
            event
            for event in result.events
            if isinstance(event, NoraletAccelerated)
        ]
        self.assertEqual(acceleration_events[0].acceleration, 1)

    def test_absent_actuator_preserves_unbounded_manual_compatibility(self) -> None:
        simulation = Simulation(
            SimulationConfig(master_seed=1),
            initial_bodies=(NoraletBodyState(1, 0),),
        )

        simulation.step({1: ActionIntent(acceleration=7)})

        self.assertEqual(simulation.state.body(1).velocity, 7)
        self.assertEqual(simulation.state.body(1).position, 7)


if __name__ == "__main__":
    unittest.main()
