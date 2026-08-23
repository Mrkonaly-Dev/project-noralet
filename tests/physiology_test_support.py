"""Shared explicit constructors for Iteration 5 physiology tests."""

from __future__ import annotations

from noralet.simulation import NoraletPhysiologyConfig


def physiology_config(
    *,
    threshold: float = 0.5,
    baseline_loss: float = 0.001,
    deprivation_scale: float = 0.01,
    deprivation_exponent: float = 2.0,
    base_hazard: float = 0.0,
    age_scale: float = 1_000.0,
    age_exponent: float = 2.0,
    condition_hazard_scale: float = 0.0,
    condition_exponent: float = 2.0,
    age_hazard_scale: float = 0.0,
    interaction_hazard_scale: float = 0.0,
) -> NoraletPhysiologyConfig:
    """Build one fully explicit immutable physiology configuration."""

    return NoraletPhysiologyConfig(
        low_energy_condition_threshold_ratio=threshold,
        baseline_condition_loss_per_tick=baseline_loss,
        deprivation_condition_loss_scale=deprivation_scale,
        deprivation_exponent=deprivation_exponent,
        base_mortality_hazard=base_hazard,
        mortality_age_scale=age_scale,
        mortality_age_exponent=age_exponent,
        condition_hazard_scale=condition_hazard_scale,
        mortality_condition_exponent=condition_exponent,
        age_hazard_scale=age_hazard_scale,
        interaction_hazard_scale=interaction_hazard_scale,
    )
