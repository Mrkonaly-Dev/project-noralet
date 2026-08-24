"""Determinism, read purity and compatibility tests for Iteration 7."""

from __future__ import annotations

import unittest

from noralet import (
    ActionIntent,
    NoraletBodyState,
    SignalDirection,
    SignalType,
)
from signal_test_support import emission, signal_config, signal_simulation


def body(identity: int, position: float, energy: float = 50.0) -> NoraletBodyState:
    return NoraletBodyState(
        identity,
        position,
        energy=energy,
        perceptual_signature=(identity / 100, -identity / 100),
    )


class SignalReadPurityTests(unittest.TestCase):
    def test_repeated_reads_do_not_mutate_active_signals_state_tick_energy_or_rng(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(energy_cost=0),
            bodies=(body(1, -2), body(2, 0)),
        )
        simulation.step({1: ActionIntent(signal_emission=emission())})
        state_before = simulation.state
        totals_before = simulation.state.energy_totals
        probe = simulation.random_streams.stream("signal-perception-probe")
        rng_before = probe.getstate()

        experiences = tuple(simulation.experience_for(2) for _ in range(20))

        self.assertTrue(all(item == experiences[0] for item in experiences))
        self.assertIs(simulation.state, state_before)
        self.assertEqual(simulation.state.tick, 1)
        self.assertEqual(simulation.state.energy_totals, totals_before)
        self.assertEqual(probe.getstate(), rng_before)
        self.assertEqual(len(simulation.state.active_signals), 1)

    def test_experience_reads_cannot_change_following_events_or_signal_expiry(self) -> None:
        bodies = (body(1, -2), body(2, 0))
        read_run = signal_simulation(bodies=bodies, seed=77)
        control_run = signal_simulation(bodies=bodies, seed=77)
        first_action = {1: ActionIntent(signal_emission=emission())}
        self.assertEqual(read_run.step(first_action), control_run.step(first_action))

        for _ in range(20):
            read_run.experiences_for_all()
        read_result = read_run.step()
        control_result = control_run.step()

        self.assertEqual(read_result, control_result)
        self.assertEqual(read_run.state, control_run.state)
        self.assertEqual(read_run.state.active_signals, ())

    def test_signal_execution_and_perception_do_not_advance_unrelated_rng(self) -> None:
        bodies = (body(1, -1), body(2, 1))
        emitting = signal_simulation(bodies=bodies, seed=123)
        quiet = signal_simulation(bodies=bodies, seed=123)
        emitting_probe = emitting.random_streams.stream("unrelated-probe")
        quiet_probe = quiet.random_streams.stream("unrelated-probe")

        emitting.step({1: ActionIntent(signal_emission=emission())})
        quiet.step()
        for _ in range(10):
            emitting.experience_for(2)

        self.assertEqual(emitting_probe.getstate(), quiet_probe.getstate())
        for identity in (1, 2):
            stream_name = f"mortality:noralet:{len(str(identity))}:{identity}"
            self.assertEqual(
                emitting.random_streams.stream(stream_name).getstate(),
                quiet.random_streams.stream(stream_name).getstate(),
            )


class SignalDeterminismTests(unittest.TestCase):
    def test_long_signal_run_preserves_three_form_energy_total(self) -> None:
        bodies = (body(1, -4, 100), body(2, 0, 100), body(3, 4, 100))
        simulation = signal_simulation(
            signal=signal_config(energy_cost=0.5),
            bodies=bodies,
            existence_cost=0.1,
            acceleration_cost=0.2,
            seed=606,
        )
        initial_total = simulation.initial_total_energy

        for tick in range(100):
            actions = {
                1: ActionIntent(
                    acceleration=0.02 if tick % 2 == 0 else -0.02,
                    signal_emission=(
                        emission(SignalType.A) if tick % 2 == 0 else None
                    ),
                ),
                2: ActionIntent(
                    signal_emission=(
                        emission(SignalType.B, SignalDirection.LEFT)
                        if tick % 3 == 0
                        else None
                    )
                ),
                3: ActionIntent(
                    signal_emission=(
                        emission(SignalType.D, SignalDirection.LEFT)
                        if tick % 4 == 0
                        else None
                    )
                ),
            }
            simulation.step(actions)
            simulation.audit_energy_conservation()

        self.assertAlmostEqual(
            simulation.state.energy_totals.total_energy,
            initial_total,
            delta=simulation.ENERGY_CONSERVATION_ABS_TOLERANCE,
        )

    def test_body_and_action_insertion_order_cannot_change_signal_history(self) -> None:
        bodies = (body(3, 0), body(1, -3), body(2, 3))
        first = signal_simulation(
            signal=signal_config(energy_cost=1),
            bodies=bodies,
            seed=404,
        )
        second = signal_simulation(
            signal=signal_config(energy_cost=1),
            bodies=tuple(reversed(bodies)),
            seed=404,
        )
        first_actions = {
            1: ActionIntent(signal_emission=emission(SignalType.A)),
            2: ActionIntent(
                signal_emission=emission(SignalType.D, SignalDirection.LEFT)
            ),
        }
        second_actions = {
            2: first_actions[2],
            1: first_actions[1],
        }

        first_result = first.step(first_actions)
        second_result = second.step(second_actions)

        self.assertEqual(first_result, second_result)
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.experiences_for_all(), second.experiences_for_all())

    def test_same_seed_state_and_actions_reproduce_complete_signal_history(self) -> None:
        bodies = (body(1, -4, 80), body(2, 0, 80), body(3, 4, 80))
        first = signal_simulation(bodies=bodies, seed=9876)
        second = signal_simulation(bodies=bodies, seed=9876)
        first_history = []
        second_history = []

        for tick in range(12):
            actions = {
                1: ActionIntent(
                    acceleration=0.02 if tick % 2 == 0 else -0.02,
                    signal_emission=(
                        emission(SignalType(chr(tick % 4 + 65)))
                        if tick % 3 == 0
                        else None
                    ),
                ),
                3: ActionIntent(
                    signal_emission=(
                        emission(SignalType.C, SignalDirection.LEFT)
                        if tick % 2 == 0
                        else None
                    )
                ),
            }
            first_result = first.step(actions)
            second_result = second.step(actions)
            first_history.append(
                (first.state, first_result, first.experiences_for_all())
            )
            second_history.append(
                (second.state, second_result, second.experiences_for_all())
            )

        self.assertEqual(first_history, second_history)

    def test_signal_perception_equality_requires_no_seed_or_rng_draw(self) -> None:
        bodies = (body(1, -2), body(2, 0))
        first = signal_simulation(bodies=bodies, seed=1)
        second = signal_simulation(bodies=bodies, seed=999)
        action = {1: ActionIntent(signal_emission=emission(SignalType.B))}

        first.step(action)
        second.step(action)

        self.assertEqual(
            first.experience_for(2).signal_percepts,
            second.experience_for(2).signal_percepts,
        )


if __name__ == "__main__":
    unittest.main()
