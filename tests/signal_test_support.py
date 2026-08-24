"""Shared explicit constructors for Iteration 7 signal tests."""

from __future__ import annotations

from noralet.simulation import (
    EnvironmentalEnergyPool,
    NoraletBodyState,
    NoraletPhysiologyConfig,
    NoraletSignalConfig,
    RegionDefinition,
    SignalDirection,
    SignalEmissionIntent,
    SignalType,
    Simulation,
)
from experience_test_support import experience_config
from noralet_energy_test_support import (
    DEFAULT_POOLS,
    DEFAULT_REGIONS,
    noralet_energy_simulation,
)
from physiology_test_support import physiology_config


def signal_config(
    *,
    radius: float = 8.0,
    energy_cost: float = 2.0,
) -> NoraletSignalConfig:
    """Build one explicit four-channel signal configuration."""

    return NoraletSignalConfig(
        signal_radius=radius,
        signal_energy_cost=energy_cost,
        signal_pattern_a=(0.91, -0.13, 0.27),
        signal_pattern_b=(-0.22, 0.84, 0.31),
        signal_pattern_c=(0.18, 0.36, -0.77),
        signal_pattern_d=(-0.63, -0.24, 0.52),
    )


def emission(
    signal_type: SignalType = SignalType.A,
    direction: SignalDirection = SignalDirection.RIGHT,
) -> SignalEmissionIntent:
    """Build one exact single-channel directional emission request."""

    return SignalEmissionIntent(signal_type=signal_type, direction=direction)


def signal_simulation(
    *,
    bodies: tuple[NoraletBodyState, ...] = (),
    signal: NoraletSignalConfig | None = None,
    energy_capacity: float = 100.0,
    existence_cost: float = 0.0,
    acceleration_cost: float = 0.0,
    regions: tuple[RegionDefinition, ...] = DEFAULT_REGIONS,
    pools: tuple[EnvironmentalEnergyPool, ...] = DEFAULT_POOLS,
    physiology: NoraletPhysiologyConfig | None = None,
    vision_radius: float = 5.0,
    seed: int = 20260825,
) -> Simulation:
    """Construct a signal-enabled simulation with neutral physiology."""

    return noralet_energy_simulation(
        bodies=bodies,
        energy_capacity=energy_capacity,
        existence_cost=existence_cost,
        acceleration_cost=acceleration_cost,
        regions=regions,
        pools=pools,
        physiology=physiology or physiology_config(baseline_loss=0.0),
        experience=experience_config(vision_radius=vision_radius),
        signals=signal or signal_config(),
        seed=seed,
    )
