"""Immutable closed-energy ecology values and configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math

from noralet.world.regions import RegionDefinition, RegionKind


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class EnvironmentalEnergyPool:
    """Region-local Environmental Energy measured in eU."""

    region_id: str
    energy: float

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str):
            raise TypeError("region_id must be a string")
        if not self.region_id:
            raise ValueError("region_id cannot be empty")

        energy = _finite_float("environmental energy", self.energy)
        if energy < 0.0:
            raise ValueError("environmental energy cannot be negative")
        object.__setattr__(self, "energy", energy)


@dataclass(frozen=True, slots=True)
class ConsumableEnergyPoint:
    """A stationary discrete point containing positive Consumable Energy."""

    point_id: int
    position: float
    energy: float

    def __post_init__(self) -> None:
        if type(self.point_id) is not int:
            raise TypeError("point_id must be an integer")
        if self.point_id < 0:
            raise ValueError("point_id cannot be negative")

        position = _finite_float("energy-point position", self.position)
        energy = _finite_float("consumable energy", self.energy)
        if energy <= 0.0:
            raise ValueError("consumable energy must be positive")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "energy", energy)


@dataclass(frozen=True, slots=True)
class EnergyTotals:
    """Observer-safe totals for all implemented fundamental energy forms."""

    environmental_energy: float
    consumable_energy: float
    noralet_energy: float = 0.0

    @property
    def total_energy(self) -> float:
        return math.fsum(
            (
                self.environmental_energy,
                self.consumable_energy,
                self.noralet_energy,
            )
        )


@dataclass(frozen=True, slots=True)
class FormationProbabilities:
    """Per-tick formation probabilities ordered by fertility semantics."""

    infertile: float
    sparse: float
    fertile: float

    def __post_init__(self) -> None:
        for name in ("infertile", "sparse", "fertile"):
            probability = _finite_float(name, getattr(self, name))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} formation probability must be in [0, 1]")
            object.__setattr__(self, name, probability)

        if not self.fertile > self.sparse > self.infertile:
            raise ValueError(
                "formation probabilities must satisfy fertile > sparse > infertile"
            )

    def for_kind(self, kind: RegionKind) -> float:
        if kind is RegionKind.INFERTILE:
            return self.infertile
        if kind is RegionKind.SPARSE:
            return self.sparse
        if kind is RegionKind.FERTILE:
            return self.fertile
        raise TypeError("kind must be a RegionKind")


@dataclass(frozen=True, slots=True)
class EnergyEcologyConfig:
    """Configuration required for region-local energy transfer mechanics."""

    regions: tuple[RegionDefinition, ...]
    initial_environmental_energy: tuple[EnvironmentalEnergyPool, ...]
    formation_probabilities: FormationProbabilities
    formation_energy_min: float
    formation_energy_max: float
    decay_rate: float
    point_removal_threshold: float
    minimum_energy_point_spacing: float = 0.0

    def __post_init__(self) -> None:
        try:
            regions = tuple(self.regions)
            pools = tuple(self.initial_environmental_energy)
        except TypeError as error:
            raise TypeError("regions and environmental pools must be iterable") from error

        if not regions:
            raise ValueError("energy ecology requires at least one region")
        if not all(isinstance(region, RegionDefinition) for region in regions):
            raise TypeError("every region must be a RegionDefinition")
        if not all(isinstance(pool, EnvironmentalEnergyPool) for pool in pools):
            raise TypeError("every environmental pool must be an EnvironmentalEnergyPool")
        if not isinstance(self.formation_probabilities, FormationProbabilities):
            raise TypeError("formation_probabilities must be FormationProbabilities")

        ordered_regions = tuple(
            sorted(regions, key=lambda region: (region.left, region.right, region.region_id))
        )
        region_ids = tuple(region.region_id for region in ordered_regions)
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region identities must be unique")

        pool_ids = tuple(pool.region_id for pool in pools)
        if len(pool_ids) != len(set(pool_ids)):
            raise ValueError("environmental pool region identities must be unique")
        if set(pool_ids) != set(region_ids):
            raise ValueError("every configured region requires exactly one energy pool")
        pool_by_id = {pool.region_id: pool for pool in pools}
        ordered_pools = tuple(pool_by_id[region_id] for region_id in region_ids)

        formation_min = _finite_float(
            "formation_energy_min", self.formation_energy_min
        )
        formation_max = _finite_float(
            "formation_energy_max", self.formation_energy_max
        )
        if formation_min <= 0.0 or formation_min > formation_max:
            raise ValueError("formation energy must satisfy 0 < min <= max")

        decay_rate = _finite_float("decay_rate", self.decay_rate)
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay_rate must be in [0, 1]")

        threshold = _finite_float(
            "point_removal_threshold", self.point_removal_threshold
        )
        if threshold < 0.0:
            raise ValueError("point_removal_threshold cannot be negative")
        if threshold >= formation_min:
            raise ValueError(
                "point_removal_threshold must be below formation_energy_min"
            )

        minimum_spacing = _finite_float(
            "minimum_energy_point_spacing",
            self.minimum_energy_point_spacing,
        )
        if minimum_spacing < 0.0:
            raise ValueError("minimum_energy_point_spacing cannot be negative")

        object.__setattr__(self, "regions", ordered_regions)
        object.__setattr__(self, "initial_environmental_energy", ordered_pools)
        object.__setattr__(self, "formation_energy_min", formation_min)
        object.__setattr__(self, "formation_energy_max", formation_max)
        object.__setattr__(self, "decay_rate", decay_rate)
        object.__setattr__(self, "point_removal_threshold", threshold)
        object.__setattr__(
            self,
            "minimum_energy_point_spacing",
            minimum_spacing,
        )

    def validate_world_partition(
        self,
        left_boundary: float,
        right_boundary: float,
    ) -> None:
        """Require an exact contiguous partition of the traversable world."""

        first = self.regions[0]
        last = self.regions[-1]
        for region in self.regions:
            if region.left < left_boundary or region.right > right_boundary:
                raise ValueError("region extends outside the traversable world")

        if first.left != left_boundary or last.right != right_boundary:
            raise ValueError("regions must cover the complete traversable world")

        for previous, current in zip(self.regions, self.regions[1:]):
            if previous.right < current.left:
                raise ValueError("region partition contains a gap")
            if previous.right > current.left:
                raise ValueError("region partition contains an overlap")

    def region_for(self, position: float) -> RegionDefinition:
        """Resolve ownership using [left, right), with final-right inclusive."""

        for index, region in enumerate(self.regions):
            is_final = index == len(self.regions) - 1
            if region.left <= position < region.right:
                return region
            if is_final and position == region.right:
                return region
        raise ValueError(f"position {position!r} is outside the region partition")


class EnergyConservationError(RuntimeError):
    """Raised before publication when the closed energy total changes."""
