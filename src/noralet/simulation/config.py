"""Configuration for a simulation run."""

from dataclasses import dataclass
import math

from noralet.world.energy import EnergyEcologyConfig


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Immutable settings required by the current simulation runtime."""

    master_seed: int
    left_boundary: float = -100.0
    right_boundary: float = 100.0
    energy_ecology: EnergyEcologyConfig | None = None

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

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        return converted
