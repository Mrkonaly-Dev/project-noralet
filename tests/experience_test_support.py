"""Shared explicit constructors for Iteration 6 experience tests."""

from __future__ import annotations

from noralet.simulation import (
    ConsumableEnergyPoint,
    NoraletBodyState,
    NoraletExperienceConfig,
    NoraletSignalConfig,
    Simulation,
)
from noralet_energy_test_support import noralet_energy_simulation
from physiology_test_support import physiology_config


def experience_config(
    *,
    vision_radius: float = 5.0,
    signature_length: int = 2,
    energy_exponent: float = 2.0,
    condition_exponent: float = 1.5,
    motor_scale: float = 2.0,
    ingestion_scale: float = 5.0,
    exertion_scale: float = 3.0,
) -> NoraletExperienceConfig:
    """Build one fully explicit immutable experience configuration."""

    return NoraletExperienceConfig(
        vision_radius=vision_radius,
        consumable_base_appearance=(0.72, -0.11),
        noralet_base_appearance=(0.09, 0.83),
        boundary_base_appearance=(-0.44, 0.17),
        signature_length=signature_length,
        energy_distress_exponent=energy_exponent,
        condition_distress_exponent=condition_exponent,
        motor_effort_scale=motor_scale,
        ingestion_sensation_scale=ingestion_scale,
        exertion_sensation_scale=exertion_scale,
    )


def experience_simulation(
    *,
    bodies: tuple[NoraletBodyState, ...] = (),
    points: tuple[ConsumableEnergyPoint, ...] = (),
    experience: NoraletExperienceConfig | None = None,
    energy_capacity: float = 100.0,
    existence_cost: float = 0.0,
    acceleration_cost: float = 0.0,
    consume_radius: float = 1.0,
    seed: int = 20260824,
    signals: NoraletSignalConfig | None = None,
) -> Simulation:
    """Construct an experience-enabled simulation with neutral physiology."""

    return noralet_energy_simulation(
        bodies=bodies,
        points=points,
        energy_capacity=energy_capacity,
        existence_cost=existence_cost,
        acceleration_cost=acceleration_cost,
        consume_radius=consume_radius,
        physiology=physiology_config(baseline_loss=0.0),
        experience=experience or experience_config(),
        signals=signals,
        seed=seed,
    )
