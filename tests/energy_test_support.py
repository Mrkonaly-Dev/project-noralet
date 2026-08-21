"""Shared constructors for focused Iteration 3 tests."""

from __future__ import annotations

from collections.abc import Iterable

from noralet.simulation import (
    EnergyEcologyConfig,
    EnvironmentalEnergyPool,
    FormationProbabilities,
    RegionDefinition,
)


DEFAULT_PROBABILITIES = FormationProbabilities(
    infertile=0.0,
    sparse=0.5,
    fertile=1.0,
)


def ecology_config(
    regions: Iterable[RegionDefinition],
    pools: Iterable[EnvironmentalEnergyPool],
    *,
    probabilities: FormationProbabilities = DEFAULT_PROBABILITIES,
    formation_min: float = 2.0,
    formation_max: float = 4.0,
    decay_rate: float = 0.25,
    removal_threshold: float = 0.1,
) -> EnergyEcologyConfig:
    """Build a compact explicit energy ecology for tests."""

    return EnergyEcologyConfig(
        regions=tuple(regions),
        initial_environmental_energy=tuple(pools),
        formation_probabilities=probabilities,
        formation_energy_min=formation_min,
        formation_energy_max=formation_max,
        decay_rate=decay_rate,
        point_removal_threshold=removal_threshold,
    )
