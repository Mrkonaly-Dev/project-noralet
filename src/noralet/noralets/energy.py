"""Immutable configuration for energy-enabled Noralet bodies."""

from dataclasses import dataclass
import math


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class NoraletEnergyConfig:
    """Physical energy capacity, costs and consume reach shared by Noralets."""

    energy_capacity: float
    existence_energy_cost_per_tick: float
    acceleration_energy_cost_per_unit: float
    consume_radius: float

    def __post_init__(self) -> None:
        capacity = _finite_float("energy_capacity", self.energy_capacity)
        existence_cost = _finite_float(
            "existence_energy_cost_per_tick",
            self.existence_energy_cost_per_tick,
        )
        acceleration_cost = _finite_float(
            "acceleration_energy_cost_per_unit",
            self.acceleration_energy_cost_per_unit,
        )
        consume_radius = _finite_float("consume_radius", self.consume_radius)

        if capacity <= 0.0:
            raise ValueError("energy_capacity must be positive")
        if existence_cost < 0.0:
            raise ValueError("existence_energy_cost_per_tick cannot be negative")
        if acceleration_cost < 0.0:
            raise ValueError("acceleration_energy_cost_per_unit cannot be negative")
        if consume_radius <= 0.0:
            raise ValueError("consume_radius must be positive")

        object.__setattr__(self, "energy_capacity", capacity)
        object.__setattr__(
            self,
            "existence_energy_cost_per_tick",
            existence_cost,
        )
        object.__setattr__(
            self,
            "acceleration_energy_cost_per_unit",
            acceleration_cost,
        )
        object.__setattr__(self, "consume_radius", consume_radius)
