"""Lockstep orchestration of independent autonomous Noralet brains."""

from __future__ import annotations

from dataclasses import dataclass
import math

from noralet.brain.base import BaseBrain
from noralet.brain.runtime import NoraletBrain
from noralet.noralets.actions import ActionIntent
from noralet.simulation.runtime import Simulation
from noralet.simulation.tick import TickResult


@dataclass(frozen=True, slots=True)
class NoraletLearningResult:
    """Observer-only metrics for one successful lived-transition update."""

    noralet_id: int
    prediction_loss: float
    gradient_norm: float

    def __post_init__(self) -> None:
        if type(self.noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")
        for name in ("prediction_loss", "gradient_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            if converted < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, converted)


@dataclass(frozen=True, slots=True)
class AutonomousTickResult:
    """Observer-facing neural intentions and the resolved physical transition."""

    action_intents: tuple[tuple[int, ActionIntent], ...]
    tick_result: TickResult
    learning_results: tuple[NoraletLearningResult, ...] = ()

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
        if not isinstance(self.learning_results, tuple):
            raise TypeError("learning_results must be an immutable tuple")
        learning_ids: list[int] = []
        for result in self.learning_results:
            if not isinstance(result, NoraletLearningResult):
                raise TypeError(
                    "every learning result must be a NoraletLearningResult"
                )
            learning_ids.append(result.noralet_id)
        if learning_ids != sorted(learning_ids) or len(learning_ids) != len(
            set(learning_ids)
        ):
            raise ValueError(
                "learning results must have unique canonical identities"
            )

    def action_for(self, noralet_id: int) -> ActionIntent:
        if type(noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")
        for routed_id, action in self.action_intents:
            if routed_id == noralet_id:
                return action
        raise KeyError(noralet_id)

    def learning_for(self, noralet_id: int) -> NoraletLearningResult:
        if type(noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")
        for result in self.learning_results:
            if result.noralet_id == noralet_id:
                return result
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
            self._brains[dead_id].discard_pending_transition()
            del self._brains[dead_id]
        missing_ids = living_ids - set(self._brains)
        if missing_ids:
            missing = ", ".join(str(value) for value in sorted(missing_ids))
            raise RuntimeError(
                f"living Noralet(s) lack an autonomous brain: {missing}"
            )

        actions: list[tuple[int, ActionIntent]] = []
        try:
            for routed in routed_experiences:
                action = self._brains[routed.noralet_id].act(
                    routed.experience
                )
                actions.append((routed.noralet_id, action))
        except Exception:
            for brain in self._brains.values():
                brain.discard_pending_transition()
            raise

        try:
            tick_result = self._simulation.step(dict(actions))
        except Exception:
            for brain in self._brains.values():
                brain.discard_pending_transition()
            raise

        next_experiences = self._simulation.routed_experiences_for_all()
        surviving_ids = {routed.noralet_id for routed in next_experiences}
        for dead_id in tuple(set(self._brains) - surviving_ids):
            self._brains[dead_id].discard_pending_transition()
            del self._brains[dead_id]

        learning_results: list[NoraletLearningResult] = []
        try:
            for routed in next_experiences:
                brain = self._brains[routed.noralet_id]
                if not brain.learning_enabled:
                    continue
                result = brain.learn(routed.experience)
                learning_results.append(
                    NoraletLearningResult(
                        noralet_id=routed.noralet_id,
                        prediction_loss=result.prediction_loss,
                        gradient_norm=result.gradient_norm,
                    )
                )
        except Exception as error:
            for brain in self._brains.values():
                brain.discard_pending_transition()
            identity = routed.noralet_id
            raise RuntimeError(
                f"predictive learning failed for Noralet {identity}"
            ) from error

        return AutonomousTickResult(
            action_intents=tuple(actions),
            tick_result=tick_result,
            learning_results=tuple(learning_results),
        )

    @staticmethod
    def action_stream_name(noralet_id: int) -> str:
        if type(noralet_id) is not int:
            raise TypeError("noralet_id must be an integer")
        stable_id = str(noralet_id)
        return f"brain:action:noralet:{len(stable_id)}:{stable_id}"
