"""Directional, bounded and identity-free signal perception tests."""

from __future__ import annotations

from dataclasses import fields
import unittest

from noralet import (
    ActionIntent,
    NoraletBodyState,
    SignalDirection,
    SignalPercept,
    SignalType,
)
from signal_test_support import emission, signal_config, signal_simulation


def body(
    identity: int,
    position: float,
    *,
    velocity: float = 0.0,
) -> NoraletBodyState:
    return NoraletBodyState(
        identity,
        position,
        velocity=velocity,
        energy=50,
        perceptual_signature=(identity / 100, -identity / 100),
    )


class DirectionAndRangeTests(unittest.TestCase):
    def test_right_directed_signal_reaches_only_receivers_on_right(self) -> None:
        simulation = signal_simulation(
            bodies=(body(1, 0), body(2, -2), body(3, 2)),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertEqual(simulation.experience_for(2).signal_percepts, ())
        received = simulation.experience_for(3).signal_percepts
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].direction_signal, -1)

    def test_left_directed_signal_reaches_only_receivers_on_left(self) -> None:
        simulation = signal_simulation(
            bodies=(body(1, 0), body(2, -2), body(3, 2)),
        )

        simulation.step(
            {
                1: ActionIntent(
                    signal_emission=emission(
                        SignalType.A,
                        SignalDirection.LEFT,
                    )
                )
            }
        )

        left_received = simulation.experience_for(2).signal_percepts
        self.assertEqual(len(left_received), 1)
        self.assertEqual(left_received[0].direction_signal, 1)
        self.assertEqual(simulation.experience_for(3).signal_percepts, ())

    def test_colocated_non_sender_receives_either_emission_direction(self) -> None:
        for direction in SignalDirection:
            with self.subTest(direction=direction):
                simulation = signal_simulation(
                    bodies=(body(1, 0), body(2, 0)),
                )
                simulation.step(
                    {
                        1: ActionIntent(
                            signal_emission=emission(SignalType.B, direction)
                        )
                    }
                )

                percept = simulation.experience_for(2).signal_percepts[0]
                self.assertEqual(percept.direction_signal, 0)
                self.assertEqual(percept.strength_signal, 1)

    def test_signal_range_is_inclusive_and_excludes_just_outside(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(radius=5),
            bodies=(
                body(1, 0),
                body(2, 2),
                body(3, 5),
                body(4, 5.000001),
            ),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertEqual(len(simulation.experience_for(2).signal_percepts), 1)
        self.assertEqual(len(simulation.experience_for(3).signal_percepts), 1)
        self.assertEqual(simulation.experience_for(4).signal_percepts, ())

    def test_strength_maps_same_midpoint_and_radius_without_distance_field(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(radius=5),
            bodies=(
                body(1, 0),
                body(2, 0),
                body(3, 2.5),
                body(4, 5),
            ),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertEqual(
            (
                simulation.experience_for(2).signal_percepts[0].strength_signal,
                simulation.experience_for(3).signal_percepts[0].strength_signal,
                simulation.experience_for(4).signal_percepts[0].strength_signal,
            ),
            (1, 0.5, 0),
        )
        percept = simulation.experience_for(3).signal_percepts[0]
        self.assertFalse(hasattr(percept, "distance"))
        self.assertFalse(hasattr(percept, "signal_radius"))

    def test_reception_uses_receivers_current_published_position(self) -> None:
        simulation = signal_simulation(
            bodies=(body(1, 0), body(2, -1, velocity=2)),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertEqual(simulation.state.body(2).position, 1)
        self.assertEqual(len(simulation.experience_for(2).signal_percepts), 1)


class SignalSensoryBoundaryTests(unittest.TestCase):
    def test_sender_never_receives_own_signal_as_incoming_perception(self) -> None:
        simulation = signal_simulation(bodies=(body(1, 0),))

        simulation.step({1: ActionIntent(signal_emission=emission())})

        experience = simulation.experience_for(1)
        self.assertEqual(experience.signal_percepts, ())
        self.assertEqual(
            experience.sensorimotor_feedback.signal_emission_activation,
            1,
        )

    def test_each_engine_type_becomes_only_its_configured_sensory_pattern(self) -> None:
        config = signal_config(energy_cost=0)
        for signal_type in SignalType:
            with self.subTest(signal_type=signal_type):
                simulation = signal_simulation(
                    signal=config,
                    bodies=(body(1, -1), body(2, 1)),
                )
                simulation.step(
                    {
                        1: ActionIntent(
                            signal_emission=emission(signal_type)
                        )
                    }
                )
                percept = simulation.experience_for(2).signal_percepts[0]

                self.assertEqual(
                    percept.signal_pattern,
                    config.pattern_for(signal_type),
                )
                self.assertFalse(hasattr(percept, "signal_type"))

    def test_all_simultaneous_eligible_signals_are_separate_percepts(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(energy_cost=0),
            bodies=(
                body(10, 0),
                body(1, -1),
                body(2, -2),
                body(3, 1),
                body(4, 2),
            ),
        )
        actions = {
            1: ActionIntent(signal_emission=emission(SignalType.A)),
            2: ActionIntent(signal_emission=emission(SignalType.B)),
            3: ActionIntent(
                signal_emission=emission(SignalType.C, SignalDirection.LEFT)
            ),
            4: ActionIntent(
                signal_emission=emission(SignalType.D, SignalDirection.LEFT)
            ),
        }

        simulation.step(actions)
        percepts = simulation.experience_for(10).signal_percepts

        self.assertEqual(len(percepts), 4)
        self.assertEqual(
            {percept.signal_pattern for percept in percepts},
            {
                simulation.config.noralet_signals.signal_pattern_a,
                simulation.config.noralet_signals.signal_pattern_b,
                simulation.config.noralet_signals.signal_pattern_c,
                simulation.config.noralet_signals.signal_pattern_d,
            },
        )

    def test_physically_identical_simultaneous_signals_preserve_multiplicity(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(energy_cost=0),
            bodies=(body(10, 0), body(1, -2), body(2, -2)),
        )

        simulation.step(
            {
                1: ActionIntent(signal_emission=emission(SignalType.C)),
                2: ActionIntent(signal_emission=emission(SignalType.C)),
            }
        )
        percepts = simulation.experience_for(10).signal_percepts

        self.assertEqual(len(percepts), 2)
        self.assertEqual(percepts[0], percepts[1])

    def test_signal_can_be_received_while_sender_is_outside_visual_radius(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(radius=8, energy_cost=0),
            vision_radius=5,
            bodies=(body(1, -7), body(2, 0)),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})
        experience = simulation.experience_for(2)

        self.assertEqual(experience.external_percepts, ())
        self.assertEqual(len(experience.signal_percepts), 1)

    def test_visible_noralets_are_not_bound_to_signal_sender(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(energy_cost=0),
            bodies=(body(1, -2), body(2, -1), body(3, 0)),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})
        experience = simulation.experience_for(3)
        percept = experience.signal_percepts[0]

        self.assertEqual(len(experience.external_percepts), 2)
        self.assertEqual(
            tuple(field.name for field in fields(SignalPercept)),
            ("signal_pattern", "direction_signal", "strength_signal"),
        )
        for hidden_name in (
            "sender_id",
            "sender_noralet_id",
            "sender_signature",
            "signal_type",
            "origin",
            "emission_direction",
            "emission_tick",
        ):
            self.assertFalse(hasattr(percept, hidden_name))

    def test_sender_identity_cannot_affect_visible_content_order(self) -> None:
        config = signal_config(energy_cost=0)
        first = signal_simulation(
            signal=config,
            bodies=(body(10, 0), body(1, -2), body(2, -1)),
        )
        second = signal_simulation(
            signal=config,
            bodies=(body(10, 0), body(2, -2), body(1, -1)),
        )
        first.step(
            {
                1: ActionIntent(signal_emission=emission(SignalType.A)),
                2: ActionIntent(signal_emission=emission(SignalType.B)),
            }
        )
        second.step(
            {
                2: ActionIntent(signal_emission=emission(SignalType.A)),
                1: ActionIntent(signal_emission=emission(SignalType.B)),
            }
        )

        self.assertEqual(
            first.experience_for(10).signal_percepts,
            second.experience_for(10).signal_percepts,
        )


if __name__ == "__main__":
    unittest.main()
