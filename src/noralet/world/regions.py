"""Immutable spatial region definitions."""

from dataclasses import dataclass
from enum import StrEnum
import math


class RegionKind(StrEnum):
    """Observer-side region classifications."""

    INFERTILE = "infertile"
    SPARSE = "sparse"
    FERTILE = "fertile"


@dataclass(frozen=True, slots=True)
class RegionDefinition:
    """Stable identity, interval and fertility kind for one region."""

    region_id: str
    left: float
    right: float
    kind: RegionKind

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str):
            raise TypeError("region_id must be a string")
        if not self.region_id:
            raise ValueError("region_id cannot be empty")
        if not isinstance(self.kind, RegionKind):
            raise TypeError("kind must be a RegionKind")

        left = self._finite_float("left", self.left)
        right = self._finite_float("right", self.right)
        if left >= right:
            raise ValueError("region left must be less than region right")

        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"region {name} must be a real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"region {name} must be finite")
        return converted

