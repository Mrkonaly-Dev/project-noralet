"""Shared compact neural and autonomous constructors for Iteration 8 tests."""

from __future__ import annotations

from noralet import (
    AutonomousSimulationRunner,
    BaseBrain,
    ExternalPercept,
    Interoception,
    NoraletActuatorConfig,
    NoraletBodyState,
    NoraletBrainConfig,
    NoraletExperience,
    SensorimotorFeedback,
    SignalPercept,
)
from signal_test_support import signal_config, signal_simulation


def brain_config(
    *,
    seed: int = 808,
    device: str = "cpu",
    exploration_std: float = 0.2,
    external_embedding: int = 4,
    signal_embedding: int = 4,
    interoception_embedding: int = 3,
    sensorimotor_embedding: int = 4,
    experience_embedding: int = 6,
    hidden_size: int = 7,
) -> NoraletBrainConfig:
    return NoraletBrainConfig(
        base_brain_seed=seed,
        external_percept_embedding_size=external_embedding,
        signal_percept_embedding_size=signal_embedding,
        interoception_embedding_size=interoception_embedding,
        sensorimotor_embedding_size=sensorimotor_embedding,
        experience_embedding_size=experience_embedding,
        hidden_size=hidden_size,
        acceleration_exploration_std=exploration_std,
        device=device,
    )


def actuator_config(max_acceleration: float = 0.25) -> NoraletActuatorConfig:
    return NoraletActuatorConfig(max_acceleration=max_acceleration)


def brain_body(
    identity: int,
    position: float,
    *,
    energy: float = 50.0,
    velocity: float = 0.0,
) -> NoraletBodyState:
    return NoraletBodyState(
        noralet_id=identity,
        position=position,
        velocity=velocity,
        energy=energy,
        perceptual_signature=(identity / 100.0, -identity / 100.0),
    )


def sample_experience(
    *,
    external_percepts: tuple[ExternalPercept, ...] = (),
    signal_percepts: tuple[SignalPercept, ...] = (),
    energy_distress: float = 0.25,
    condition_distress: float = 0.1,
    energetic_exertion: float = 0.2,
    motor_direction: float = 0.0,
) -> NoraletExperience:
    return NoraletExperience(
        external_percepts=external_percepts,
        signal_percepts=signal_percepts,
        interoception=Interoception(
            energy_distress=energy_distress,
            condition_distress=condition_distress,
            energetic_exertion=energetic_exertion,
        ),
        sensorimotor_feedback=SensorimotorFeedback(
            motor_direction=motor_direction,
            motor_effort=0.15,
            consume_activation=0.0,
            ingestion_signal=0.0,
            signal_emission_activation=0.0,
            signal_emission_pattern=(0.0, 0.0, 0.0),
            signal_emission_direction=0.0,
        ),
    )


def external_percept(
    *,
    direction: float = 1.0,
    proximity: float = 0.5,
) -> ExternalPercept:
    return ExternalPercept(
        appearance_pattern=(0.72, -0.11, 0.0, 0.0),
        direction_signal=direction,
        proximity_signal=proximity,
    )


def signal_percept(
    *,
    direction: float = -1.0,
    strength: float = 0.5,
) -> SignalPercept:
    return SignalPercept(
        signal_pattern=(0.91, -0.13, 0.27),
        direction_signal=direction,
        strength_signal=strength,
    )


def autonomous_setup(
    *,
    bodies: tuple[NoraletBodyState, ...] | None = None,
    brain: NoraletBrainConfig | None = None,
    actuator: NoraletActuatorConfig | None = None,
    simulation_seed: int = 1234,
    signal_energy_cost: float = 0.0,
    existence_cost: float = 0.0,
    acceleration_cost: float = 0.0,
) -> tuple[AutonomousSimulationRunner, BaseBrain]:
    body_values = (
        (brain_body(1, -2.0), brain_body(2, 2.0))
        if bodies is None
        else bodies
    )
    brain_values = brain or brain_config()
    actuator_values = actuator or actuator_config()
    simulation = signal_simulation(
        bodies=body_values,
        signal=signal_config(energy_cost=signal_energy_cost),
        actuator=actuator_values,
        existence_cost=existence_cost,
        acceleration_cost=acceleration_cost,
        seed=simulation_seed,
    )
    assert simulation.config.noralet_experience is not None
    assert simulation.config.noralet_signals is not None
    base = BaseBrain(
        brain_values,
        simulation.config.noralet_experience,
        simulation.config.noralet_signals,
        actuator_values,
    )
    return AutonomousSimulationRunner(simulation, base), base
