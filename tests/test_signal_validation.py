"""Domain, configuration and dependency validation for Iteration 7."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import math
import unittest

from experience_test_support import experience_simulation
from noralet import (
    ActionIntent,
    ActiveSignal,
    NoraletBodyState,
    NoraletSignalConfig,
    SignalDirection,
    SignalEmissionIntent,
    SignalPercept,
    SignalType,
    Simulation,
    SimulationConfig,
    WorldState,
)
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config
from signal_test_support import emission, signal_config, signal_simulation


class SignalDomainValidationTests(unittest.TestCase):
    def test_exactly_four_meaningless_engine_signal_types_exist(self) -> None:
        self.assertEqual(tuple(SignalType), tuple(SignalType.__members__.values()))
        self.assertEqual(tuple(item.value for item in SignalType), ("A", "B", "C", "D"))
        self.assertEqual(tuple(SignalDirection), (SignalDirection.LEFT, SignalDirection.RIGHT))

    def test_emission_intent_requires_one_valid_type_and_direction(self) -> None:
        intent = SignalEmissionIntent(SignalType.A, SignalDirection.LEFT)
        self.assertEqual(intent.signal_type, SignalType.A)
        self.assertEqual(intent.direction, SignalDirection.LEFT)
        with self.assertRaises(FrozenInstanceError):
            intent.signal_type = SignalType.B  # type: ignore[misc]
        with self.assertRaises(TypeError):
            SignalEmissionIntent("A", SignalDirection.LEFT)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SignalEmissionIntent(SignalType.A, "left")  # type: ignore[arg-type]

    def test_action_can_hold_none_or_exactly_one_emission_by_construction(self) -> None:
        empty = ActionIntent()
        requested = ActionIntent(signal_emission=emission())

        self.assertIsNone(empty.signal_emission)
        self.assertIsInstance(requested.signal_emission, SignalEmissionIntent)
        self.assertEqual(
            tuple(field.name for field in fields(ActionIntent)),
            ("acceleration", "consume", "signal_emission"),
        )
        with self.assertRaises(TypeError):
            ActionIntent(signal_emission=[emission(), emission()])  # type: ignore[arg-type]

    def test_active_signal_is_validated_immutable_objective_state(self) -> None:
        signal = ActiveSignal(7, SignalType.C, 1, SignalDirection.RIGHT)
        self.assertEqual(signal.origin, 1.0)
        self.assertEqual(
            tuple(field.name for field in fields(ActiveSignal)),
            (
                "sender_noralet_id",
                "signal_type",
                "origin",
                "emission_direction",
            ),
        )
        self.assertFalse(hasattr(signal, "energy"))
        with self.assertRaises(FrozenInstanceError):
            signal.origin = 2  # type: ignore[misc]
        with self.assertRaises(TypeError):
            ActiveSignal(True, SignalType.A, 0, SignalDirection.LEFT)
        with self.assertRaises(TypeError):
            ActiveSignal(1, "A", 0, SignalDirection.LEFT)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ActiveSignal(1, SignalType.A, 0, "left")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ActiveSignal(1, SignalType.A, math.nan, SignalDirection.LEFT)

    def test_world_state_canonicalizes_signals_and_rejects_two_per_sender(self) -> None:
        first = ActiveSignal(1, SignalType.A, -1, SignalDirection.RIGHT)
        second = ActiveSignal(2, SignalType.B, 1, SignalDirection.LEFT)
        state = WorldState(active_signals=(second, first))

        self.assertEqual(state.active_signals, (first, second))
        with self.assertRaises(ValueError):
            WorldState(
                active_signals=(
                    first,
                    ActiveSignal(1, SignalType.D, 2, SignalDirection.LEFT),
                )
            )

    def test_signal_percept_has_only_sensory_fields(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(SignalPercept)),
            ("signal_pattern", "direction_signal", "strength_signal"),
        )


class SignalConfigurationValidationTests(unittest.TestCase):
    def test_configuration_canonicalizes_patterns_and_is_immutable(self) -> None:
        config = NoraletSignalConfig(
            signal_radius=8,
            signal_energy_cost=2,
            signal_pattern_a=[1, 0],  # type: ignore[arg-type]
            signal_pattern_b=[0, 1],  # type: ignore[arg-type]
            signal_pattern_c=[-1, 0],  # type: ignore[arg-type]
            signal_pattern_d=[0, -1],  # type: ignore[arg-type]
        )

        self.assertEqual(config.signal_radius, 8.0)
        self.assertEqual(config.signal_energy_cost, 2.0)
        self.assertEqual(config.signal_pattern_a, (1.0, 0.0))
        self.assertEqual(config.signal_pattern_length, 2)
        self.assertEqual(config.pattern_for(SignalType.D), (0.0, -1.0))
        with self.assertRaises(FrozenInstanceError):
            config.signal_radius = 2  # type: ignore[misc]

    def test_radius_must_be_positive_and_finite(self) -> None:
        valid = signal_config()
        for value in (0, -1, math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(valid, signal_radius=value)
        with self.assertRaises(TypeError):
            replace(valid, signal_radius=True)

    def test_energy_cost_must_be_non_negative_and_finite(self) -> None:
        valid = signal_config()
        self.assertEqual(replace(valid, signal_energy_cost=0).signal_energy_cost, 0)
        for value in (-1, math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(valid, signal_energy_cost=value)
        with self.assertRaises(TypeError):
            replace(valid, signal_energy_cost=True)

    def test_four_patterns_must_be_finite_equal_nonempty_and_pairwise_distinct(self) -> None:
        valid = signal_config()
        invalid_changes = (
            {"signal_pattern_a": ()},
            {"signal_pattern_b": (1.0,)},
            {"signal_pattern_c": valid.signal_pattern_a},
            {"signal_pattern_d": (math.nan, 0.0, 1.0)},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(valid, **changes)
        with self.assertRaises(TypeError):
            replace(valid, signal_pattern_a={1.0, 2.0})  # type: ignore[arg-type]

    def test_signal_system_requires_energy_ecology_and_experience(self) -> None:
        config = signal_config()
        with self.assertRaises(ValueError):
            SimulationConfig(master_seed=1, noralet_signals=config)
        with self.assertRaises(ValueError):
            noralet_energy_simulation(
                physiology=physiology_config(),
                signals=config,
            )

        simulation = signal_simulation(signal=config)
        self.assertEqual(simulation.state.active_signals, ())

    def test_signal_request_is_rejected_when_system_is_disabled(self) -> None:
        body = NoraletBodyState(
            1,
            0,
            energy=10,
            perceptual_signature=(0, 0),
        )
        simulation = experience_simulation(bodies=(body,))
        state_before = simulation.state

        with self.assertRaises(ValueError):
            simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertIs(simulation.state, state_before)

    def test_prior_action_without_signal_remains_compatible(self) -> None:
        simulation = Simulation(
            SimulationConfig(master_seed=1),
            initial_bodies=(NoraletBodyState(1, 0),),
        )
        simulation.step({1: ActionIntent(acceleration=1, consume=False)})

        self.assertEqual(simulation.state.body(1).position, 1)
        self.assertEqual(simulation.state.active_signals, ())


if __name__ == "__main__":
    unittest.main()
