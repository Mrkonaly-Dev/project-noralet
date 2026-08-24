"""Physical execution, cost, lifecycle and feedback tests for signals."""

from __future__ import annotations

import math
import unittest

from noralet import (
    ActionIntent,
    ActiveSignal,
    EnvironmentalEnergyPool,
    NoraletAccelerated,
    NoraletBodyState,
    NoraletDeathCause,
    NoraletDied,
    NoraletEnergyExpenditureReason,
    NoraletEnergySpent,
    NoraletMoved,
    RegionDefinition,
    RegionKind,
    SignalDirection,
    SignalEmitted,
    SignalType,
    TickAdvanced,
)
from signal_test_support import emission, signal_config, signal_simulation


def body(
    identity: int,
    position: float,
    *,
    energy: float = 50.0,
    velocity: float = 0.0,
) -> NoraletBodyState:
    return NoraletBodyState(
        identity,
        position,
        velocity=velocity,
        energy=energy,
        perceptual_signature=(identity / 100, -identity / 100),
    )


class SignalExecutionTests(unittest.TestCase):
    def test_successful_emission_transfers_cost_and_creates_state_and_events(self) -> None:
        config = signal_config(energy_cost=2)
        simulation = signal_simulation(
            signal=config,
            bodies=(body(1, 0, energy=10),),
        )
        initial_total = simulation.state.energy_totals.total_energy

        result = simulation.step(
            {1: ActionIntent(signal_emission=emission(SignalType.B))}
        )

        self.assertEqual(simulation.state.body(1).energy, 8)
        self.assertEqual(simulation.state.environmental_energy_for("all"), 2)
        self.assertEqual(
            simulation.state.active_signals,
            (ActiveSignal(1, SignalType.B, 0, SignalDirection.RIGHT),),
        )
        self.assertIn(
            NoraletEnergySpent(
                1,
                "all",
                NoraletEnergyExpenditureReason.SIGNAL,
                2,
                0,
                1,
            ),
            result.events,
        )
        self.assertIn(
            SignalEmitted(1, SignalType.B, SignalDirection.RIGHT, 0, 0, 1),
            result.events,
        )
        feedback = simulation.experience_for(1).sensorimotor_feedback
        self.assertEqual(feedback.signal_emission_activation, 1)
        self.assertEqual(feedback.signal_emission_pattern, config.signal_pattern_b)
        self.assertEqual(feedback.signal_emission_direction, 1)
        self.assertEqual(simulation.experience_for(1).signal_percepts, ())
        self.assertEqual(simulation.state.energy_totals.total_energy, initial_total)

    def test_unaffordable_request_has_no_partial_effect_or_execution_feedback(self) -> None:
        config = signal_config(energy_cost=5)
        simulation = signal_simulation(
            signal=config,
            bodies=(body(1, 0, energy=3),),
        )

        result = simulation.step(
            {1: ActionIntent(signal_emission=emission(SignalType.C))}
        )
        feedback = simulation.experience_for(1).sensorimotor_feedback

        self.assertEqual(simulation.state.body(1).energy, 3)
        self.assertEqual(simulation.state.active_signals, ())
        self.assertFalse(any(isinstance(event, SignalEmitted) for event in result.events))
        self.assertFalse(
            any(
                isinstance(event, NoraletEnergySpent)
                and event.reason is NoraletEnergyExpenditureReason.SIGNAL
                for event in result.events
            )
        )
        self.assertEqual(feedback.signal_emission_activation, 0)
        self.assertEqual(feedback.signal_emission_pattern, (0, 0, 0))
        self.assertEqual(feedback.signal_emission_direction, 0)

    def test_zero_cost_signal_executes_without_energy_transfer(self) -> None:
        config = signal_config(energy_cost=0)
        simulation = signal_simulation(
            signal=config,
            bodies=(body(1, 0, energy=10),),
        )

        result = simulation.step(
            {
                1: ActionIntent(
                    signal_emission=emission(
                        SignalType.D,
                        SignalDirection.LEFT,
                    )
                )
            }
        )

        self.assertEqual(simulation.state.body(1).energy, 10)
        self.assertEqual(simulation.state.environmental_energy_for("all"), 0)
        self.assertEqual(len(simulation.state.active_signals), 1)
        self.assertTrue(any(isinstance(event, SignalEmitted) for event in result.events))
        self.assertFalse(any(isinstance(event, NoraletEnergySpent) for event in result.events))

    def test_acceleration_affordability_is_resolved_before_signal(self) -> None:
        config = signal_config(energy_cost=2)
        simulation = signal_simulation(
            signal=config,
            acceleration_cost=1,
            bodies=(body(1, 0, energy=4),),
        )

        result = simulation.step(
            {
                1: ActionIntent(
                    acceleration=3,
                    signal_emission=emission(),
                )
            }
        )

        self.assertEqual(simulation.state.body(1).velocity, 3)
        self.assertEqual(simulation.state.body(1).energy, 1)
        self.assertEqual(simulation.state.active_signals, ())
        self.assertIn(NoraletAccelerated(1, 3, 0, 1), result.events)
        self.assertEqual(
            simulation.experience_for(1)
            .sensorimotor_feedback.signal_emission_activation,
            0,
        )

    def test_signal_origin_is_tick_start_position_while_sender_moves(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(energy_cost=0),
            bodies=(body(1, 0, energy=10, velocity=2),),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertEqual(simulation.state.body(1).position, 2)
        self.assertEqual(simulation.state.active_signals[0].origin, 0)

    def test_event_order_places_signal_phase_before_motion(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(energy_cost=2),
            existence_cost=1,
            acceleration_cost=1,
            bodies=(body(1, 0, energy=10),),
        )

        result = simulation.step(
            {1: ActionIntent(acceleration=1, signal_emission=emission())}
        )

        self.assertEqual(
            tuple(
                event.reason if isinstance(event, NoraletEnergySpent) else type(event)
                for event in result.events
            ),
            (
                NoraletEnergyExpenditureReason.EXISTENCE,
                NoraletEnergyExpenditureReason.ACCELERATION,
                NoraletEnergyExpenditureReason.SIGNAL,
                SignalEmitted,
                NoraletAccelerated,
                NoraletMoved,
                TickAdvanced,
            ),
        )

    def test_signal_cost_returns_to_tick_start_region_before_crossing(self) -> None:
        regions = (
            RegionDefinition("left", -10, 0, RegionKind.INFERTILE),
            RegionDefinition("right", 0, 10, RegionKind.INFERTILE),
        )
        simulation = signal_simulation(
            signal=signal_config(energy_cost=2),
            bodies=(body(1, -1, energy=10, velocity=2),),
            regions=regions,
            pools=(
                EnvironmentalEnergyPool("left", 0),
                EnvironmentalEnergyPool("right", 0),
            ),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertEqual(simulation.state.body(1).position, 1)
        self.assertEqual(simulation.state.environmental_energy_for("left"), 2)
        self.assertEqual(simulation.state.environmental_energy_for("right"), 0)

    def test_signal_cost_is_included_in_general_exertion(self) -> None:
        config = signal_config(energy_cost=2)
        emitting = signal_simulation(signal=config, bodies=(body(1, 0),))
        quiet = signal_simulation(signal=config, bodies=(body(1, 0),))

        emitting.step({1: ActionIntent(signal_emission=emission())})
        quiet.step()
        emitting_exertion = emitting.experience_for(1).interoception.energetic_exertion
        quiet_exertion = quiet.experience_for(1).interoception.energetic_exertion

        self.assertAlmostEqual(emitting_exertion, 1 - math.exp(-2 / 3))
        self.assertEqual(quiet_exertion, 0)
        self.assertGreater(emitting_exertion, quiet_exertion)


class SignalLifetimeAndDeathTests(unittest.TestCase):
    def test_emission_appears_at_t_plus_one_and_expires_at_t_plus_two(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(energy_cost=0),
            bodies=(body(1, -1), body(2, 1)),
        )
        experience_at_zero = simulation.experience_for(2)

        simulation.step({1: ActionIntent(signal_emission=emission())})
        experience_at_one = simulation.experience_for(2)
        self.assertEqual(len(simulation.state.active_signals), 1)
        simulation.step()
        experience_at_two = simulation.experience_for(2)

        self.assertEqual(experience_at_zero.signal_percepts, ())
        self.assertEqual(len(experience_at_one.signal_percepts), 1)
        self.assertEqual(experience_at_two.signal_percepts, ())
        self.assertEqual(simulation.state.active_signals, ())

    def test_new_emission_replaces_prior_active_signal(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(energy_cost=0),
            bodies=(body(1, 0), body(2, 2)),
        )
        simulation.step(
            {1: ActionIntent(signal_emission=emission(SignalType.A))}
        )

        simulation.step(
            {1: ActionIntent(signal_emission=emission(SignalType.D))}
        )

        self.assertEqual(len(simulation.state.active_signals), 1)
        self.assertEqual(simulation.state.active_signals[0].signal_type, SignalType.D)

    def test_signal_spending_last_energy_emits_then_causes_depletion_death(self) -> None:
        config = signal_config(energy_cost=2)
        simulation = signal_simulation(
            signal=config,
            bodies=(body(1, 0, energy=2), body(2, 1, energy=10)),
        )
        initial_total = simulation.state.energy_totals.total_energy

        result = simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertEqual(tuple(item.noralet_id for item in simulation.state.bodies), (2,))
        self.assertEqual(len(simulation.state.active_signals), 1)
        self.assertIn(
            NoraletDied(1, NoraletDeathCause.ENERGY_DEPLETION, 0, 0, 1),
            result.events,
        )
        self.assertEqual(len(simulation.experience_for(2).signal_percepts), 1)
        with self.assertRaises(KeyError):
            simulation.experience_for(1)
        self.assertEqual(simulation.state.energy_totals.total_energy, initial_total)

    def test_boundary_dead_sender_leaves_final_signal_for_survivor(self) -> None:
        simulation = signal_simulation(
            signal=signal_config(radius=4, energy_cost=2),
            bodies=(
                body(1, 9, energy=10, velocity=2),
                body(2, 10, energy=10),
            ),
        )

        simulation.step({1: ActionIntent(signal_emission=emission())})

        self.assertEqual(tuple(item.noralet_id for item in simulation.state.bodies), (2,))
        self.assertEqual(simulation.state.active_signals[0].origin, 9)
        self.assertEqual(len(simulation.experience_for(2).signal_percepts), 1)
        with self.assertRaises(KeyError):
            simulation.experience_for(1)


if __name__ == "__main__":
    unittest.main()
