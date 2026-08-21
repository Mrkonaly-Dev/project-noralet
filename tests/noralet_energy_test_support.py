"""Shared explicit constructors for Iteration 4 tests."""

from __future__ import annotations

from energy_test_support import ecology_config
from noralet.simulation import (
    ConsumableEnergyPoint,
    EnvironmentalEnergyPool,
    FormationProbabilities,
    NoraletBodyState,
    NoraletEnergyConfig,
    RegionDefinition,
    RegionKind,
    Simulation,
    SimulationConfig,
)


DEFAULT_REGIONS = (
    RegionDefinition("all", -10, 10, RegionKind.INFERTILE),
)
DEFAULT_POOLS = (EnvironmentalEnergyPool("all", 0),)
DEFAULT_PROBABILITIES = FormationProbabilities(0.0, 0.5, 1.0)


def noralet_energy_simulation(
    *,
    bodies: tuple[NoraletBodyState, ...] = (),
    points: tuple[ConsumableEnergyPoint, ...] = (),
    regions: tuple[RegionDefinition, ...] = DEFAULT_REGIONS,
    pools: tuple[EnvironmentalEnergyPool, ...] = DEFAULT_POOLS,
    energy_capacity: float = 100.0,
    existence_cost: float = 0.0,
    acceleration_cost: float = 0.0,
    consume_radius: float = 1.0,
    minimum_spacing: float = 3.0,
    formation_min: float = 2.0,
    formation_max: float = 2.0,
    decay_rate: float = 0.0,
    removal_threshold: float = 0.0,
    probabilities: FormationProbabilities = DEFAULT_PROBABILITIES,
    seed: int = 20260821,
) -> Simulation:
    """Construct one fully explicit energy-enabled simulation."""

    ecology = ecology_config(
        regions,
        pools,
        probabilities=probabilities,
        formation_min=formation_min,
        formation_max=formation_max,
        decay_rate=decay_rate,
        removal_threshold=removal_threshold,
        minimum_spacing=minimum_spacing,
    )
    energy = NoraletEnergyConfig(
        energy_capacity=energy_capacity,
        existence_energy_cost_per_tick=existence_cost,
        acceleration_energy_cost_per_unit=acceleration_cost,
        consume_radius=consume_radius,
    )
    ordered_regions = tuple(sorted(regions, key=lambda region: region.left))
    return Simulation(
        SimulationConfig(
            master_seed=seed,
            left_boundary=ordered_regions[0].left,
            right_boundary=ordered_regions[-1].right,
            energy_ecology=ecology,
            noralet_energy=energy,
        ),
        initial_bodies=bodies,
        initial_energy_points=points,
    )
