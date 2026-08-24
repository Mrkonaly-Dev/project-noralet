"""Read purity, stable batching and determinism tests for Iteration 6."""

from __future__ import annotations

import unittest

from experience_test_support import experience_config, experience_simulation
from noralet.simulation import (
    ActionIntent,
    ConsumableEnergyPoint,
    NoraletBodyState,
)
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config


class ExperienceApiTests(unittest.TestCase):
    def test_repeated_reads_are_equivalent_and_do_not_mutate_state_or_rng(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(1, 0, energy=50, perceptual_signature=(0.1, 0.2)),
            )
        )
        state_before = simulation.state
        probe = simulation.random_streams.stream("experience-read-probe")
        rng_before = probe.getstate()

        first = simulation.experience_for(1)
        second = simulation.experience_for(1)
        all_experiences = simulation.experiences_for_all()

        self.assertEqual(first, second)
        self.assertEqual(all_experiences, (first,))
        self.assertIs(simulation.state, state_before)
        self.assertEqual(probe.getstate(), rng_before)

    def test_all_living_experiences_follow_stable_identity_order(self) -> None:
        bodies = (
            NoraletBodyState(3, 0, energy=25, perceptual_signature=(0.3, 0.3)),
            NoraletBodyState(1, -8, energy=100, perceptual_signature=(0.1, 0.1)),
            NoraletBodyState(2, 8, energy=50, perceptual_signature=(0.2, 0.2)),
        )
        simulation = experience_simulation(bodies=bodies)

        experiences = simulation.experiences_for_all()

        self.assertEqual(
            experiences,
            tuple(simulation.experience_for(identity) for identity in (1, 2, 3)),
        )
        self.assertEqual(
            tuple(item.interoception.energy_distress for item in experiences),
            (0.0, 0.25, 0.5625),
        )

    def test_api_rejects_non_integer_and_unknown_or_dead_routing_identity(self) -> None:
        simulation = experience_simulation(
            bodies=(
                NoraletBodyState(
                    1,
                    9,
                    velocity=2,
                    energy=50,
                    perceptual_signature=(0, 0),
                ),
            )
        )
        with self.assertRaises(TypeError):
            simulation.experience_for(True)
        with self.assertRaises(KeyError):
            simulation.experience_for(999)

        simulation.step()

        self.assertEqual(simulation.experiences_for_all(), ())
        with self.assertRaises(KeyError):
            simulation.experience_for(1)

    def test_experience_reads_cannot_change_the_next_state_or_events(self) -> None:
        bodies = (
            NoraletBodyState(2, 2, energy=50, perceptual_signature=(0.2, 0.2)),
            NoraletBodyState(1, -2, energy=50, perceptual_signature=(0.1, 0.1)),
        )
        read_run = experience_simulation(bodies=bodies, acceleration_cost=1, seed=44)
        control_run = experience_simulation(bodies=bodies, acceleration_cost=1, seed=44)
        actions = {
            1: ActionIntent(acceleration=0.25, consume=True),
            2: ActionIntent(acceleration=-0.5),
        }

        for _ in range(4):
            for _ in range(7):
                read_run.experiences_for_all()
                read_run.experience_for(1)
            read_result = read_run.step(actions)
            control_result = control_run.step(actions)
            self.assertEqual(read_result, control_result)
            self.assertEqual(read_run.state, control_run.state)


class ExperienceDeterminismTests(unittest.TestCase):
    def test_body_and_point_insertion_order_cannot_change_external_order(self) -> None:
        bodies = (
            NoraletBodyState(3, 2, energy=30, perceptual_signature=(0.3, -0.3)),
            NoraletBodyState(1, 0, energy=70, perceptual_signature=(0.1, -0.1)),
            NoraletBodyState(2, -2, energy=50, perceptual_signature=(0.2, -0.2)),
        )
        points = (
            ConsumableEnergyPoint(8, 5, 1),
            ConsumableEnergyPoint(4, -5, 1),
        )
        first = experience_simulation(bodies=bodies, points=points)
        second = experience_simulation(
            bodies=tuple(reversed(bodies)),
            points=tuple(reversed(points)),
        )

        self.assertEqual(first.experience_for(1), second.experience_for(1))

    def test_initial_body_insertion_order_cannot_change_experience(self) -> None:
        bodies = (
            NoraletBodyState(3, 3, energy=30, perceptual_signature=(0.3, -0.3)),
            NoraletBodyState(1, -3, energy=70, perceptual_signature=(0.1, -0.1)),
            NoraletBodyState(2, 0, energy=50, perceptual_signature=(0.2, -0.2)),
        )
        first = experience_simulation(bodies=bodies, seed=91)
        second = experience_simulation(bodies=tuple(reversed(bodies)), seed=91)

        self.assertEqual(first.state, second.state)
        self.assertEqual(first.experiences_for_all(), second.experiences_for_all())

    def test_same_seed_state_and_actions_reproduce_experience_history(self) -> None:
        bodies = (
            NoraletBodyState(1, -3, energy=60, perceptual_signature=(0.1, 0.9)),
            NoraletBodyState(2, 3, energy=60, perceptual_signature=(0.9, 0.1)),
        )
        first = experience_simulation(
            bodies=bodies,
            existence_cost=0.1,
            acceleration_cost=0.2,
            seed=2026,
        )
        second = experience_simulation(
            bodies=bodies,
            existence_cost=0.1,
            acceleration_cost=0.2,
            seed=2026,
        )
        history_first = []
        history_second = []

        for tick in range(6):
            actions = {
                1: ActionIntent(acceleration=0.1 if tick % 2 == 0 else -0.1),
                2: ActionIntent(acceleration=-0.05, consume=tick % 3 == 0),
            }
            first.step(actions)
            second.step(actions)
            history_first.append(first.experiences_for_all())
            history_second.append(second.experiences_for_all())

        self.assertEqual(history_first, history_second)
        self.assertEqual(first.state, second.state)

    def test_enabling_and_reading_experience_does_not_shift_world_dynamics(self) -> None:
        physiology = physiology_config(
            baseline_loss=0.0001,
            base_hazard=0.0001,
        )
        bodies = (
            NoraletBodyState(1, -2, energy=80, perceptual_signature=(0.1, 0.2)),
            NoraletBodyState(2, 2, energy=80, perceptual_signature=(0.3, 0.4)),
        )
        enabled = noralet_energy_simulation(
            bodies=bodies,
            physiology=physiology,
            experience=experience_config(),
            existence_cost=0.1,
            acceleration_cost=0.2,
            seed=12345,
        )
        disabled = noralet_energy_simulation(
            bodies=bodies,
            physiology=physiology,
            existence_cost=0.1,
            acceleration_cost=0.2,
            seed=12345,
        )
        actions = {
            1: ActionIntent(acceleration=0.05),
            2: ActionIntent(acceleration=-0.05),
        }

        for _ in range(20):
            enabled.experiences_for_all()
            enabled_result = enabled.step(actions)
            disabled_result = disabled.step(actions)
            self.assertEqual(enabled_result, disabled_result)
            self.assertEqual(enabled.state, disabled.state)


if __name__ == "__main__":
    unittest.main()
