"""Immutable intentions proposed for Noralet bodies."""

from dataclasses import dataclass
import math

from noralet.noralets.signals import SignalEmissionIntent


@dataclass(frozen=True, slots=True)
class ActionIntent:
    """Request physical actions for one lockstep transition."""

    acceleration: float = 0.0
    consume: bool = False
    signal_emission: SignalEmissionIntent | None = None

    def __post_init__(self) -> None:
        if isinstance(self.acceleration, bool) or not isinstance(
            self.acceleration, (int, float)
        ):
            raise TypeError("acceleration must be a real number")

        acceleration = float(self.acceleration)
        if not math.isfinite(acceleration):
            raise ValueError("acceleration must be finite")
        if type(self.consume) is not bool:
            raise TypeError("consume must be a boolean")
        if self.signal_emission is not None and not isinstance(
            self.signal_emission,
            SignalEmissionIntent,
        ):
            raise TypeError(
                "signal_emission must be a SignalEmissionIntent or None"
            )
        object.__setattr__(self, "acceleration", acceleration)
