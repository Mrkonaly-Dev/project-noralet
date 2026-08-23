"""Configuration for a simulation run."""

from dataclasses import dataclass
import math

from noralet.noralets.energy import NoraletEnergyConfig
from noralet.noralets.physiology import NoraletPhysiologyConfig
from noralet.world.energy import EnergyEcologyConfig


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Immutable settings required by the current simulation runtime."""

    master_seed: int
    left_boundary: float = -100.0
    right_boundary: float = 100.0
    energy_ecology: EnergyEcologyConfig | None = None
    noralet_energy: NoraletEnergyConfig | None = None
    noralet_physiology: NoraletPhysiologyConfig | None = None

    def __post_init__(self) -> None:
        if type(self.master_seed) is not int:
            raise TypeError("master_seed must be an integer")

        left_boundary = self._finite_float("left_boundary", self.left_boundary)
        right_boundary = self._finite_float("right_boundary", self.right_boundary)
        if left_boundary >= right_boundary:
            raise ValueError("left_boundary must be less than right_boundary")

        object.__setattr__(self, "left_boundary", left_boundary)
        object.__setattr__(self, "right_boundary", right_boundary)

        if self.energy_ecology is not None:
            if not isinstance(self.energy_ecology, EnergyEcologyConfig):
                raise TypeError("energy_ecology must be an EnergyEcologyConfig")
            self.energy_ecology.validate_world_partition(
                left_boundary,
                right_boundary,
            )

        if self.noralet_energy is not None:
            if not isinstance(self.noralet_energy, NoraletEnergyConfig):
                raise TypeError("noralet_energy must be a NoraletEnergyConfig")
            if self.energy_ecology is None:
                raise ValueError(
                    "Noralet Energy requires an active EnergyEcologyConfig"
                )
            if (
                self.energy_ecology.minimum_energy_point_spacing
                <= 2.0 * self.noralet_energy.consume_radius
            ):
                raise ValueError(
                    "minimum_energy_point_spacing must be greater than "
                    "twice consume_radius"
                )

        if self.noralet_physiology is not None:
            if not isinstance(self.noralet_physiology, NoraletPhysiologyConfig):
                raise TypeError(
                    "noralet_physiology must be a NoraletPhysiologyConfig"
                )
            if self.noralet_energy is None:
                raise ValueError(
                    "Noralet physiology requires an active NoraletEnergyConfig"
                )

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        return converted
