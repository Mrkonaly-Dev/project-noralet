"""Objective world state for one simulation tick."""

from dataclasses import dataclass
import math

from noralet.noralets.body import NoraletBodyState
from noralet.world.energy import (
    ConsumableEnergyPoint,
    EnergyTotals,
    EnvironmentalEnergyPool,
)


@dataclass(frozen=True, slots=True)
class WorldState:
    """Immutable objective bodies and ecology state at one tick."""

    tick: int = 0
    bodies: tuple[NoraletBodyState, ...] = ()
    environmental_energy: tuple[EnvironmentalEnergyPool, ...] = ()
    energy_points: tuple[ConsumableEnergyPoint, ...] = ()
    next_energy_point_id: int = 0

    def __post_init__(self) -> None:
        if type(self.tick) is not int:
            raise TypeError("tick must be an integer")
        if self.tick < 0:
            raise ValueError("tick cannot be negative")
        if not isinstance(self.bodies, tuple):
            raise TypeError("bodies must be an immutable tuple")
        if not all(isinstance(body, NoraletBodyState) for body in self.bodies):
            raise TypeError("every body must be a NoraletBodyState")

        ordered_bodies = tuple(sorted(self.bodies, key=lambda body: body.noralet_id))
        identities = tuple(body.noralet_id for body in ordered_bodies)
        if len(identities) != len(set(identities)):
            raise ValueError("Noralet identities must be unique")

        if not isinstance(self.environmental_energy, tuple):
            raise TypeError("environmental_energy must be an immutable tuple")
        if not all(
            isinstance(pool, EnvironmentalEnergyPool)
            for pool in self.environmental_energy
        ):
            raise TypeError("every energy pool must be an EnvironmentalEnergyPool")
        ordered_pools = tuple(
            sorted(self.environmental_energy, key=lambda pool: pool.region_id)
        )
        pool_ids = tuple(pool.region_id for pool in ordered_pools)
        if len(pool_ids) != len(set(pool_ids)):
            raise ValueError("environmental pool region identities must be unique")

        if not isinstance(self.energy_points, tuple):
            raise TypeError("energy_points must be an immutable tuple")
        if not all(
            isinstance(point, ConsumableEnergyPoint) for point in self.energy_points
        ):
            raise TypeError("every energy point must be a ConsumableEnergyPoint")
        ordered_points = tuple(sorted(self.energy_points, key=lambda point: point.point_id))
        point_ids = tuple(point.point_id for point in ordered_points)
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("energy-point identities must be unique")

        if type(self.next_energy_point_id) is not int:
            raise TypeError("next_energy_point_id must be an integer")
        if self.next_energy_point_id < 0:
            raise ValueError("next_energy_point_id cannot be negative")
        if point_ids and self.next_energy_point_id <= max(point_ids):
            raise ValueError("next_energy_point_id must exceed existing point IDs")

        object.__setattr__(self, "bodies", ordered_bodies)
        object.__setattr__(self, "environmental_energy", ordered_pools)
        object.__setattr__(self, "energy_points", ordered_points)

    def body(self, noralet_id: int) -> NoraletBodyState:
        """Return one living body, or raise ``KeyError`` if it is absent."""

        for body in self.bodies:
            if body.noralet_id == noralet_id:
                return body
        raise KeyError(noralet_id)

    def environmental_energy_for(self, region_id: str) -> float:
        """Return one region-local Environmental Energy pool."""

        for pool in self.environmental_energy:
            if pool.region_id == region_id:
                return pool.energy
        raise KeyError(region_id)

    def energy_point(self, point_id: int) -> ConsumableEnergyPoint:
        """Return one existing Consumable Energy point."""

        for point in self.energy_points:
            if point.point_id == point_id:
                return point
        raise KeyError(point_id)

    @property
    def energy_totals(self) -> EnergyTotals:
        """Return observer-safe totals calculated with careful summation."""

        return EnergyTotals(
            environmental_energy=math.fsum(
                pool.energy for pool in self.environmental_energy
            ),
            consumable_energy=math.fsum(point.energy for point in self.energy_points),
            noralet_energy=math.fsum(body.energy for body in self.bodies),
        )
