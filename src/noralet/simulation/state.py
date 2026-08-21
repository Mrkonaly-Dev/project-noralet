"""Objective physical state for one simulation tick."""

from dataclasses import dataclass

from noralet.noralets.body import NoraletBodyState


@dataclass(frozen=True, slots=True)
class WorldState:
    """Immutable objective state containing the living bodies at one tick."""

    tick: int = 0
    bodies: tuple[NoraletBodyState, ...] = ()

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

        object.__setattr__(self, "bodies", ordered_bodies)

    def body(self, noralet_id: int) -> NoraletBodyState:
        """Return one living body, or raise ``KeyError`` if it is absent."""

        for body in self.bodies:
            if body.noralet_id == noralet_id:
                return body
        raise KeyError(noralet_id)
