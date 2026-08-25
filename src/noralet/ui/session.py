"""Qt-independent live observer session over the authoritative runtime."""

from __future__ import annotations

from dataclasses import dataclass

from noralet.brain import (
    AutonomousSimulationRunner,
    AutonomousTickResult,
    NoraletHomeostaticLearningResult,
    NoraletLearningResult,
)
from noralet.research.config import (
    LearningCondition,
    build_baseline_components,
    seed_mapping,
)
from noralet.simulation.events import NoraletAccelerated


@dataclass(frozen=True, slots=True)
class LiveRunSetup:
    """User-selected pre-run controls for one UI-owned baseline world."""

    simulation_seed: int = 1
    population: int = 6
    device: str = "auto"
    maximum_ticks: int = 5_000
    condition: LearningCondition = LearningCondition.FULL_CURRENT_BRAIN

    def __post_init__(self) -> None:
        if type(self.simulation_seed) is not int:
            raise TypeError("simulation_seed must be an integer")
        for name in ("population", "maximum_ticks"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        device = self.device.strip().lower()
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if not isinstance(self.condition, LearningCondition):
            raise TypeError("condition must be a LearningCondition")
        object.__setattr__(self, "device", device)

    @property
    def base_brain_seed(self) -> int:
        """Derive an independent inherited-brain seed from the displayed seed."""

        return seed_mapping(self.simulation_seed).base_brain_seed


class LiveSession:
    """Own one live runner and compact observer-only latest-tick state."""

    def __init__(
        self,
        setup: LiveRunSetup,
        runner: AutonomousSimulationRunner,
    ) -> None:
        if not isinstance(setup, LiveRunSetup):
            raise TypeError("setup must be a LiveRunSetup")
        if not isinstance(runner, AutonomousSimulationRunner):
            raise TypeError("runner must be an AutonomousSimulationRunner")
        self.setup = setup
        self.runner = runner
        self.latest_result: AutonomousTickResult | None = None
        self.latest_learning: dict[int, NoraletLearningResult] = {}
        self.latest_homeostatic: dict[int, NoraletHomeostaticLearningResult] = {}
        self.latest_applied_acceleration: dict[int, float] = {
            identity: 0.0 for identity in runner.brain_ids
        }

    @property
    def tick(self) -> int:
        return self.runner.simulation.state.tick

    @property
    def is_extinct(self) -> bool:
        return not self.runner.brain_ids

    @property
    def reached_maximum_ticks(self) -> bool:
        return self.tick >= self.setup.maximum_ticks

    @property
    def can_step(self) -> bool:
        return not self.is_extinct and not self.reached_maximum_ticks

    @property
    def completion_message(self) -> str | None:
        if self.is_extinct:
            return f"Population extinct at tick {self.tick}"
        if self.reached_maximum_ticks:
            return f"Completed at maximum tick {self.tick}"
        return None

    def step(self) -> AutonomousTickResult | None:
        """Advance exactly one authoritative autonomous tick when still active."""

        if not self.can_step:
            return None
        living_before = self.runner.brain_ids
        result = self.runner.step()
        self.latest_result = result
        self.latest_learning = {
            item.noralet_id: item for item in result.learning_results
        }
        self.latest_homeostatic = {
            item.noralet_id: item
            for item in result.homeostatic_learning_results
        }
        accelerations = {identity: 0.0 for identity in living_before}
        for event in result.tick_result.events:
            if isinstance(event, NoraletAccelerated):
                accelerations[event.noralet_id] = event.acceleration
        self.latest_applied_acceleration = accelerations
        return result

    def step_many(self, count: int) -> tuple[AutonomousTickResult, ...]:
        """Execute a bounded sequential burst without skipping simulation ticks."""

        if type(count) is not int:
            raise TypeError("count must be an integer")
        if count <= 0:
            raise ValueError("count must be positive")
        results: list[AutonomousTickResult] = []
        for _ in range(count):
            result = self.step()
            if result is None:
                break
            results.append(result)
        return tuple(results)


def create_live_session(setup: LiveRunSetup) -> LiveSession:
    """Create a fresh UI-owned runner from the shared baseline factory."""

    if not isinstance(setup, LiveRunSetup):
        raise TypeError("setup must be a LiveRunSetup")
    simulation, base_brain = build_baseline_components(
        initial_population=setup.population,
        device=setup.device,
        condition=setup.condition,
        simulation_seed=setup.simulation_seed,
        base_brain_seed=setup.base_brain_seed,
    )
    return LiveSession(
        setup,
        AutonomousSimulationRunner(simulation, base_brain),
    )
