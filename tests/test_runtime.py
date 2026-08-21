"""Tests for the authoritative empty-world simulation runtime."""

from __future__ import annotations

import dataclasses
import unittest

from noralet.simulation import (
    Simulation,
    SimulationConfig,
    TickAdvanced,
    TickResult,
    WorldState,
)


class SimulationRuntimeTests(unittest.TestCase):
    def test_initial_state_starts_at_tick_zero(self) -> None:
        simulation = Simulation(SimulationConfig(master_seed=17))

        self.assertEqual(simulation.state, WorldState(tick=0))

    def test_step_advances_one_authoritative_tick(self) -> None:
        simulation = Simulation(SimulationConfig(master_seed=17))

        results = [simulation.step() for _ in range(3)]

        self.assertEqual(simulation.state.tick, 3)
        self.assertEqual(
            [(result.tick_before, result.tick_after) for result in results],
            [(0, 1), (1, 2), (2, 3)],
        )

    def test_step_creates_a_new_state_without_mutating_the_old_state(self) -> None:
        simulation = Simulation(SimulationConfig(master_seed=17))
        state_before = simulation.state

        simulation.step()

        self.assertEqual(state_before.tick, 0)
        self.assertEqual(simulation.state.tick, 1)
        self.assertIsNot(state_before, simulation.state)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state_before.tick = 99  # type: ignore[misc]

    def test_tick_result_contains_one_structured_tick_event(self) -> None:
        simulation = Simulation(SimulationConfig(master_seed=17))

        result = simulation.step()

        expected_event = TickAdvanced(tick_before=0, tick_after=1)
        self.assertEqual(
            result,
            TickResult(tick_before=0, tick_after=1, events=(expected_event,)),
        )
        self.assertIsInstance(result.events[0], TickAdvanced)

    def test_observer_result_is_immutable(self) -> None:
        result = Simulation(SimulationConfig(master_seed=17)).step()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.tick_after = 10  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.events[0].tick_after = 10  # type: ignore[misc]

    def test_identical_simulations_produce_identical_histories(self) -> None:
        config = SimulationConfig(master_seed=12345)
        first = Simulation(config)
        second = Simulation(config)

        first_history = [first.step() for _ in range(10)]
        second_history = [second.step() for _ in range(10)]

        self.assertEqual(first_history, second_history)
        self.assertEqual(first.state, second.state)

    def test_configuration_is_immutable(self) -> None:
        config = SimulationConfig(master_seed=17)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.master_seed = 18  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

