"""Immutable objective state for a transient physical signal."""

from dataclasses import dataclass
import math

from noralet.noralets.signals import SignalDirection, SignalType


@dataclass(frozen=True, slots=True)
class ActiveSignal:
    """One executed emission present for one published world moment."""

    sender_noralet_id: int
    signal_type: SignalType
    origin: float
    emission_direction: SignalDirection

    def __post_init__(self) -> None:
        if type(self.sender_noralet_id) is not int:
            raise TypeError("sender_noralet_id must be an integer")
        if not isinstance(self.signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType")
        if isinstance(self.origin, bool) or not isinstance(
            self.origin,
            (int, float),
        ):
            raise TypeError("origin must be a real number")
        origin = float(self.origin)
        if not math.isfinite(origin):
            raise ValueError("origin must be finite")
        if not isinstance(self.emission_direction, SignalDirection):
            raise TypeError("emission_direction must be a SignalDirection")
        object.__setattr__(self, "origin", origin)
