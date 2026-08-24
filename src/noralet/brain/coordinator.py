"""Lockstep orchestration of independent autonomous Noralet brains."""

from __future__ import annotations

from dataclasses import dataclass

from noralet.brain.base import BaseBrain
from noralet.brain.runtime import NoraletBrain
from noralet.noralets.actions import ActionIntent
from noralet.simulation.runtime import Simulation
from noralet.simulation.tick import TickResult


@dataclass(frozen=True, slots=True)
class AutonomousTickResult:
    """Observer-facing neural intentions and the resolved physical transition."""

    action_intents: tuple[tuple[int, ActionIntent], ...]
    tick_result: TickResult

    def __post_init__(self) -> None:
        if not isinstance(self.action_intents, tuple):
            raise TypeError("action_intents must be an immutable tuple")
        identities: list[int] = []
        for item in self.action_intents:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "each routed action must be an (ID, ActionIntent) tuple"
                )
            noralet_id, intent = item
            if type(noralet_id) is not int:
                raise TypeError("routed action IDs must be integers")
            if not isinstance(intent, ActionIntent):
                raise TypeError("every routed action must contain an ActionIntent")
            identities.append(noralet_id)
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise ValueError("routed actions must have unique canonical identities")
        if not isinstance(self.tick_result, TickResult):
            raise TypeError("tick_result must be a TickResult")

    def action_for(self, noralet_id: int) -> ActionIntent:
        if type(noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")
        for routed_id, action in self.action_intents:
            if routed_id == noralet_id:
                return action
        raise KeyError(noralet_id)


class AutonomousSimulationRunner:
    """Activate every living brain once, then resolve all actions together."""

    def __init__(self, simulation: Simulation, base_brain: BaseBrain) -> None:
        if not isinstance(simulation, Simulation):
            raise TypeError("simulation must be a Simulation")
        if not isinstance(base_brain, BaseBrain):
            raise TypeError("base_brain must be a BaseBrain")
        config = simulation.config
        if (
            config.noralet_experience is None
            or config.noralet_signals is None
            or config.noralet_actuators is None
        ):
            raise ValueError(
                "autonomous control requires Experience, signals and actuators"
            )
        if not base_brain.matches_simulation_configs(
            config.noralet_experience,
            config.noralet_signals,
            config.noralet_actuators,
        ):
            raise ValueError(
                "BaseBrain sensory/actuator interface does not match Simulation"
            )

        self._simulation = simulation
        self._base_brain = base_brain
        self._brains = {
            routed.noralet_id: base_brain.spawn(
                action_random_source=simulation.random_streams.stream(
                    self.action_stream_name(routed.noralet_id)
                )
            )
            for routed in simulation.routed_experiences_for_all()
        }

    @property
    def simulation(self) -> Simulation:
        return self._simulation

    @property
    def brain_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._brains))

    def brain_for(self, noralet_id: int) -> NoraletBrain:
        if type(noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")
        try:
            return self._brains[noralet_id]
        except KeyError:
            raise KeyError(noralet_id) from None

    def step(self) -> AutonomousTickResult:
        """Perform exactly one complete autonomous lockstep world tick."""

        routed_experiences = self._simulation.routed_experiences_for_all()
        living_ids = {routed.noralet_id for routed in routed_experiences}
        for dead_id in tuple(set(self._brains) - living_ids):
            del self._brains[dead_id]
        missing_ids = living_ids - set(self._brains)
        if missing_ids:
            missing = ", ".join(str(value) for value in sorted(missing_ids))
            raise RuntimeError(
                f"living Noralet(s) lack an autonomous brain: {missing}"
            )

        actions: list[tuple[int, ActionIntent]] = []
        for routed in routed_experiences:
            action = self._brains[routed.noralet_id].act(routed.experience)
            actions.append((routed.noralet_id, action))

        tick_result = self._simulation.step(dict(actions))
        surviving_ids = {
            routed.noralet_id
            for routed in self._simulation.routed_experiences_for_all()
        }
        for dead_id in tuple(set(self._brains) - surviving_ids):
            del self._brains[dead_id]

        return AutonomousTickResult(
            action_intents=tuple(actions),
            tick_result=tick_result,
        )

    @staticmethod
    def action_stream_name(noralet_id: int) -> str:
        if type(noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")
        stable_id = str(noralet_id)
        return f"brain:action:noralet:{len(stable_id)}:{stable_id}"
