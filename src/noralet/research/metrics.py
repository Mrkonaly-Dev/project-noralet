"""Bounded-memory, observer-only measurements for Research Iteration 001."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Callable, Any

import torch
from torch import Tensor, nn

from noralet.brain.coordinator import AutonomousSimulationRunner, AutonomousTickResult
from noralet.brain.runtime import NoraletBrain
from noralet.noralets.signals import SignalDirection
from noralet.research.config import LearningCondition, SeedMapping
from noralet.simulation.events import (
    EnergyConsumed,
    NoraletDied,
    NoraletMoved,
    SignalEmitted,
)


PREDICTION_WINDOW_SIZE = 100
NEAR_ZERO_ACCELERATION = 1e-12

RUN_SUMMARY_COLUMNS = (
    "condition",
    "replicate_seed",
    "simulation_seed",
    "base_brain_seed",
    "status",
    "technical_error",
    "start_tick",
    "final_tick",
    "max_ticks",
    "extinct",
    "survivors_at_end",
    "runtime_seconds",
    "device",
    "initial_population",
    "death_count",
    "boundary_death_count",
    "energy_depletion_death_count",
    "natural_death_count",
)

NORALET_SUMMARY_COLUMNS = (
    "condition",
    "replicate_seed",
    "noralet_id",
    "initial_tick",
    "final_observed_tick",
    "death_occurred",
    "death_cause",
    "right_censored",
    "observed_lifetime_ticks",
    "initial_energy",
    "final_observed_energy",
    "minimum_energy",
    "mean_sampled_energy",
    "initial_condition",
    "final_observed_condition",
    "minimum_condition",
    "mean_energy_distress",
    "maximum_energy_distress",
    "mean_condition_distress",
    "maximum_condition_distress",
    "total_absolute_distance_travelled",
    "mean_absolute_velocity",
    "consume_attempt_count",
    "successful_consumption_count",
    "total_energy_consumed",
    "signal_emission_count",
    "signal_A_count",
    "signal_B_count",
    "signal_C_count",
    "signal_D_count",
    "signal_LEFT_count",
    "signal_RIGHT_count",
    "received_signal_percept_count",
    "mean_requested_acceleration",
    "mean_absolute_requested_acceleration",
    "positive_acceleration_count",
    "negative_acceleration_count",
    "near_zero_acceleration_count",
    "consume_selected_count",
    "consume_not_selected_count",
    "signal_NONE_count",
    "signal_A_LEFT_count",
    "signal_A_RIGHT_count",
    "signal_B_LEFT_count",
    "signal_B_RIGHT_count",
    "signal_C_LEFT_count",
    "signal_C_RIGHT_count",
    "signal_D_LEFT_count",
    "signal_D_RIGHT_count",
    "predictive_update_count",
    "prediction_loss_mean",
    "prediction_loss_initial_window_mean",
    "prediction_loss_final_window_mean",
    "prediction_loss_min",
    "prediction_loss_max",
    "homeostatic_update_count",
    "mean_homeostatic_drive",
    "mean_modulation",
    "mean_absolute_modulation",
    "positive_modulation_count",
    "negative_modulation_count",
    "neutral_modulation_count",
    "mean_eligibility_norm",
    "maximum_eligibility_norm",
    "mean_homeostatic_update_norm",
    "maximum_homeostatic_update_norm",
    "online_encoder_parameter_drift_norm",
    "GRU_parameter_drift_norm",
    "predictor_parameter_drift_norm",
    "action_head_parameter_drift_norm",
)

TIMESERIES_COLUMNS = (
    "condition",
    "replicate_seed",
    "tick",
    "noralet_id",
    "alive",
    "position",
    "velocity",
    "stored_energy",
    "physiological_condition",
    "energy_distress",
    "condition_distress",
    "energetic_exertion",
    "prediction_loss",
    "gradient_norm",
    "homeostatic_drive_before",
    "homeostatic_drive_after",
    "homeostatic_modulation",
    "eligibility_norm",
    "homeostatic_update_norm",
    "hidden_state_norm",
    "cumulative_absolute_displacement",
    "consume_attempt_count",
    "successful_consumption_count",
    "signal_emission_count",
    "received_signal_percept_count",
)


def parameter_drift_norm(
    module: nn.Module | None,
    birth_parameters: tuple[Tensor, ...] | None,
) -> float | None:
    """Return an observer-only global L2 distance from immutable birth values."""

    if module is None or birth_parameters is None:
        return None
    current = tuple(module.parameters())
    if len(current) != len(birth_parameters):
        raise ValueError("birth snapshot does not match current module")
    squared = math.fsum(
        float(
            torch.sum(
                (
                    parameter.detach().cpu().double()
                    - birth.detach().cpu().double()
                ).square()
            ).item()
        )
        for parameter, birth in zip(current, birth_parameters, strict=True)
    )
    drift = math.sqrt(squared)
    if not math.isfinite(drift):
        raise FloatingPointError("parameter drift norm is non-finite")
    return drift


def _snapshot(module: nn.Module | None) -> tuple[Tensor, ...] | None:
    if module is None:
        return None
    return tuple(parameter.detach().cpu().clone() for parameter in module.parameters())


@dataclass(slots=True)
class _PredictionAccumulator:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    first_window: list[float] = field(default_factory=list)
    last_window: deque[float] = field(
        default_factory=lambda: deque(maxlen=PREDICTION_WINDOW_SIZE)
    )

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.first_window) < PREDICTION_WINDOW_SIZE:
            self.first_window.append(value)
        self.last_window.append(value)

    def summary(self) -> dict[str, float | int | None]:
        return {
            "predictive_update_count": self.count,
            "prediction_loss_mean": self.total / self.count if self.count else None,
            "prediction_loss_initial_window_mean": (
                math.fsum(self.first_window) / len(self.first_window)
                if self.first_window
                else None
            ),
            "prediction_loss_final_window_mean": (
                math.fsum(self.last_window) / len(self.last_window)
                if self.last_window
                else None
            ),
            "prediction_loss_min": self.minimum,
            "prediction_loss_max": self.maximum,
        }


@dataclass(slots=True)
class _HomeostaticAccumulator:
    count: int = 0
    drive_total: float = 0.0
    modulation_total: float = 0.0
    absolute_modulation_total: float = 0.0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    eligibility_total: float = 0.0
    eligibility_maximum: float | None = None
    update_total: float = 0.0
    update_maximum: float | None = None

    def add(
        self,
        *,
        drive_after: float,
        modulation: float,
        eligibility_norm: float,
        update_norm: float,
    ) -> None:
        self.count += 1
        self.drive_total += drive_after
        self.modulation_total += modulation
        self.absolute_modulation_total += abs(modulation)
        if modulation > 0.0:
            self.positive += 1
        elif modulation < 0.0:
            self.negative += 1
        else:
            self.neutral += 1
        self.eligibility_total += eligibility_norm
        self.eligibility_maximum = (
            eligibility_norm
            if self.eligibility_maximum is None
            else max(self.eligibility_maximum, eligibility_norm)
        )
        self.update_total += update_norm
        self.update_maximum = (
            update_norm
            if self.update_maximum is None
            else max(self.update_maximum, update_norm)
        )

    def summary(self) -> dict[str, float | int | None]:
        return {
            "homeostatic_update_count": self.count,
            "mean_homeostatic_drive": (
                self.drive_total / self.count if self.count else None
            ),
            "mean_modulation": (
                self.modulation_total / self.count if self.count else None
            ),
            "mean_absolute_modulation": (
                self.absolute_modulation_total / self.count if self.count else None
            ),
            "positive_modulation_count": self.positive,
            "negative_modulation_count": self.negative,
            "neutral_modulation_count": self.neutral,
            "mean_eligibility_norm": (
                self.eligibility_total / self.count if self.count else None
            ),
            "maximum_eligibility_norm": self.eligibility_maximum,
            "mean_homeostatic_update_norm": (
                self.update_total / self.count if self.count else None
            ),
            "maximum_homeostatic_update_norm": self.update_maximum,
        }


@dataclass(slots=True)
class _LifetimeAccumulator:
    noralet_id: int
    initial_tick: int
    initial_energy: float
    initial_condition: float
    final_observed_tick: int
    final_observed_energy: float
    final_observed_condition: float
    minimum_energy: float
    minimum_condition: float
    energy_distress_total: float = 0.0
    energy_distress_count: int = 0
    maximum_energy_distress: float = 0.0
    condition_distress_total: float = 0.0
    condition_distress_count: int = 0
    maximum_condition_distress: float = 0.0
    sampled_energy_total: float = 0.0
    sampled_energy_count: int = 0
    total_absolute_distance: float = 0.0
    absolute_velocity_total: float = 0.0
    velocity_observation_count: int = 0
    consume_attempt_count: int = 0
    successful_consumption_count: int = 0
    total_energy_consumed: float = 0.0
    signal_emission_count: int = 0
    signal_type_counts: dict[str, int] = field(
        default_factory=lambda: {value: 0 for value in "ABCD"}
    )
    signal_direction_counts: dict[str, int] = field(
        default_factory=lambda: {value: 0 for value in ("LEFT", "RIGHT")}
    )
    received_signal_percept_count: int = 0
    acceleration_total: float = 0.0
    absolute_acceleration_total: float = 0.0
    action_count: int = 0
    positive_acceleration_count: int = 0
    negative_acceleration_count: int = 0
    near_zero_acceleration_count: int = 0
    consume_selected_count: int = 0
    consume_not_selected_count: int = 0
    signal_selection_counts: dict[str, int] = field(
        default_factory=lambda: {
            "NONE": 0,
            "A_LEFT": 0,
            "A_RIGHT": 0,
            "B_LEFT": 0,
            "B_RIGHT": 0,
            "C_LEFT": 0,
            "C_RIGHT": 0,
            "D_LEFT": 0,
            "D_RIGHT": 0,
        }
    )
    death_cause: str | None = None
    prediction: _PredictionAccumulator = field(default_factory=_PredictionAccumulator)
    homeostatic: _HomeostaticAccumulator = field(
        default_factory=_HomeostaticAccumulator
    )

    def observe_published(self, *, tick: int, body: Any, experience: Any) -> None:
        self.final_observed_tick = tick
        self.final_observed_energy = body.energy
        self.final_observed_condition = body.condition
        self.minimum_energy = min(self.minimum_energy, body.energy)
        self.minimum_condition = min(self.minimum_condition, body.condition)
        self.absolute_velocity_total += abs(body.velocity)
        self.velocity_observation_count += 1
        energy_distress = experience.interoception.energy_distress
        condition_distress = experience.interoception.condition_distress
        self.energy_distress_total += energy_distress
        self.energy_distress_count += 1
        self.maximum_energy_distress = max(
            self.maximum_energy_distress,
            energy_distress,
        )
        self.condition_distress_total += condition_distress
        self.condition_distress_count += 1
        self.maximum_condition_distress = max(
            self.maximum_condition_distress,
            condition_distress,
        )
        self.received_signal_percept_count += len(experience.signal_percepts)

    def sample_energy(self, value: float) -> None:
        self.sampled_energy_total += value
        self.sampled_energy_count += 1


@dataclass(frozen=True, slots=True)
class _BirthModules:
    encoder: tuple[Tensor, ...]
    recurrent_core: tuple[Tensor, ...]
    predictor: tuple[Tensor, ...] | None
    action_heads: tuple[Tensor, ...]


class ResearchRunObserver:
    """Read immutable runtime state after transitions without entering causality."""

    def __init__(
        self,
        runner: AutonomousSimulationRunner,
        condition: LearningCondition,
        seeds: SeedMapping,
        *,
        sample_every_ticks: int,
        timeseries_sink: Callable[[dict[str, Any]], None],
    ) -> None:
        self.runner = runner
        self.condition = condition
        self.seeds = seeds
        self.sample_every_ticks = sample_every_ticks
        self.timeseries_sink = timeseries_sink
        self.start_tick = runner.simulation.state.tick
        self.initial_population = len(runner.brain_ids)
        self._brains: dict[int, NoraletBrain] = {
            identity: runner.brain_for(identity) for identity in runner.brain_ids
        }
        self._birth = {
            identity: _BirthModules(
                encoder=_snapshot(brain.model.encoder) or (),
                recurrent_core=_snapshot(brain.model.recurrent_core) or (),
                predictor=_snapshot(brain.model.prediction_model),
                action_heads=tuple(
                    parameter.detach().cpu().clone()
                    for parameter in brain.model.action_head_parameters()
                ),
            )
            for identity, brain in self._brains.items()
        }
        self._lifetimes: dict[int, _LifetimeAccumulator] = {}
        self._last_predictive: dict[int, Any] = {}
        self._last_homeostatic: dict[int, Any] = {}
        self._last_sampled_tick: int | None = None
        for routed in runner.simulation.routed_experiences_for_all():
            body = runner.simulation.state.body(routed.noralet_id)
            accumulator = _LifetimeAccumulator(
                noralet_id=routed.noralet_id,
                initial_tick=self.start_tick,
                initial_energy=body.energy,
                initial_condition=body.condition,
                final_observed_tick=self.start_tick,
                final_observed_energy=body.energy,
                final_observed_condition=body.condition,
                minimum_energy=body.energy,
                minimum_condition=body.condition,
            )
            accumulator.observe_published(
                tick=self.start_tick,
                body=body,
                experience=routed.experience,
            )
            self._lifetimes[routed.noralet_id] = accumulator
        self._sample_current(self.start_tick)

    def observe(self, result: AutonomousTickResult) -> None:
        self._last_predictive = {
            item.noralet_id: item for item in result.learning_results
        }
        self._last_homeostatic = {
            item.noralet_id: item
            for item in result.homeostatic_learning_results
        }
        for identity, action in result.action_intents:
            lifetime = self._lifetimes[identity]
            acceleration = action.acceleration
            lifetime.action_count += 1
            lifetime.acceleration_total += acceleration
            lifetime.absolute_acceleration_total += abs(acceleration)
            if acceleration > NEAR_ZERO_ACCELERATION:
                lifetime.positive_acceleration_count += 1
            elif acceleration < -NEAR_ZERO_ACCELERATION:
                lifetime.negative_acceleration_count += 1
            else:
                lifetime.near_zero_acceleration_count += 1
            if action.consume:
                lifetime.consume_attempt_count += 1
                lifetime.consume_selected_count += 1
            else:
                lifetime.consume_not_selected_count += 1
            emission = action.signal_emission
            if emission is None:
                key = "NONE"
            else:
                direction = (
                    "LEFT"
                    if emission.direction is SignalDirection.LEFT
                    else "RIGHT"
                )
                key = f"{emission.signal_type.value}_{direction}"
            lifetime.signal_selection_counts[key] += 1

        for event in result.tick_result.events:
            if isinstance(event, NoraletMoved):
                self._lifetimes[event.noralet_id].total_absolute_distance += abs(
                    event.position_after - event.position_before
                )
            elif isinstance(event, EnergyConsumed):
                lifetime = self._lifetimes[event.noralet_id]
                lifetime.successful_consumption_count += 1
                lifetime.total_energy_consumed += event.energy_transferred
            elif isinstance(event, SignalEmitted):
                lifetime = self._lifetimes[event.noralet_id]
                lifetime.signal_emission_count += 1
                lifetime.signal_type_counts[event.signal_type.value] += 1
                lifetime.signal_direction_counts[event.emission_direction.name] += 1
            elif isinstance(event, NoraletDied):
                lifetime = self._lifetimes[event.noralet_id]
                lifetime.death_cause = event.cause.value
                lifetime.final_observed_tick = event.tick_after

        for item in result.learning_results:
            self._lifetimes[item.noralet_id].prediction.add(item.prediction_loss)
        for item in result.homeostatic_learning_results:
            self._lifetimes[item.noralet_id].homeostatic.add(
                drive_after=item.homeostatic_drive_after,
                modulation=item.modulation,
                eligibility_norm=item.eligibility_norm,
                update_norm=item.applied_update_norm,
            )

        for routed in self.runner.simulation.routed_experiences_for_all():
            body = self.runner.simulation.state.body(routed.noralet_id)
            self._lifetimes[routed.noralet_id].observe_published(
                tick=result.tick_result.tick_after,
                body=body,
                experience=routed.experience,
            )
        if result.tick_result.tick_after % self.sample_every_ticks == 0:
            self._sample_current(result.tick_result.tick_after)

    def finish(self, *, max_ticks: int) -> tuple[dict[str, Any], ...]:
        final_tick = self.runner.simulation.state.tick
        if self.runner.brain_ids and self._last_sampled_tick != final_tick:
            self._sample_current(final_tick)
        return tuple(
            self._lifetime_summary(identity, max_ticks=max_ticks)
            for identity in sorted(self._lifetimes)
        )

    def run_summary(
        self,
        *,
        max_ticks: int,
        runtime_seconds: float,
        status: str = "completed",
        technical_error: str | None = None,
    ) -> dict[str, Any]:
        causes = [value.death_cause for value in self._lifetimes.values()]
        final_tick = self.runner.simulation.state.tick
        survivors = len(self.runner.brain_ids)
        return {
            "condition": self.condition.value,
            "replicate_seed": self.seeds.replicate_seed,
            "simulation_seed": self.seeds.simulation_seed,
            "base_brain_seed": self.seeds.base_brain_seed,
            "status": status,
            "technical_error": technical_error,
            "start_tick": self.start_tick,
            "final_tick": final_tick,
            "max_ticks": max_ticks,
            "extinct": survivors == 0,
            "survivors_at_end": survivors,
            "runtime_seconds": runtime_seconds,
            "device": str(next(iter(self._brains.values())).device),
            "initial_population": self.initial_population,
            "death_count": sum(cause is not None for cause in causes),
            "boundary_death_count": causes.count("world_boundary"),
            "energy_depletion_death_count": causes.count("energy_depletion"),
            "natural_death_count": causes.count("natural"),
        }

    def _sample_current(self, tick: int) -> None:
        experiences = {
            routed.noralet_id: routed.experience
            for routed in self.runner.simulation.routed_experiences_for_all()
        }
        for body in self.runner.simulation.state.bodies:
            identity = body.noralet_id
            lifetime = self._lifetimes[identity]
            experience = experiences[identity]
            lifetime.sample_energy(body.energy)
            prediction = self._last_predictive.get(identity)
            homeostatic = self._last_homeostatic.get(identity)
            hidden_norm = float(
                torch.linalg.vector_norm(self.runner.brain_for(identity).hidden_state)
                .cpu()
                .item()
            )
            self.timeseries_sink(
                {
                    "condition": self.condition.value,
                    "replicate_seed": self.seeds.replicate_seed,
                    "tick": tick,
                    "noralet_id": identity,
                    "alive": True,
                    "position": body.position,
                    "velocity": body.velocity,
                    "stored_energy": body.energy,
                    "physiological_condition": body.condition,
                    "energy_distress": experience.interoception.energy_distress,
                    "condition_distress": experience.interoception.condition_distress,
                    "energetic_exertion": experience.interoception.energetic_exertion,
                    "prediction_loss": (
                        prediction.prediction_loss if prediction is not None else None
                    ),
                    "gradient_norm": (
                        prediction.gradient_norm if prediction is not None else None
                    ),
                    "homeostatic_drive_before": (
                        homeostatic.homeostatic_drive_before
                        if homeostatic is not None
                        else None
                    ),
                    "homeostatic_drive_after": (
                        homeostatic.homeostatic_drive_after
                        if homeostatic is not None
                        else None
                    ),
                    "homeostatic_modulation": (
                        homeostatic.modulation if homeostatic is not None else None
                    ),
                    "eligibility_norm": (
                        homeostatic.eligibility_norm if homeostatic is not None else None
                    ),
                    "homeostatic_update_norm": (
                        homeostatic.applied_update_norm
                        if homeostatic is not None
                        else None
                    ),
                    "hidden_state_norm": hidden_norm,
                    "cumulative_absolute_displacement": lifetime.total_absolute_distance,
                    "consume_attempt_count": lifetime.consume_attempt_count,
                    "successful_consumption_count": (
                        lifetime.successful_consumption_count
                    ),
                    "signal_emission_count": lifetime.signal_emission_count,
                    "received_signal_percept_count": (
                        lifetime.received_signal_percept_count
                    ),
                }
            )
        self._last_sampled_tick = tick

    def _lifetime_summary(self, identity: int, *, max_ticks: int) -> dict[str, Any]:
        lifetime = self._lifetimes[identity]
        brain = self._brains[identity]
        birth = self._birth[identity]
        action_count = lifetime.action_count
        death_occurred = lifetime.death_cause is not None
        final_tick = self.runner.simulation.state.tick
        row: dict[str, Any] = {
            "condition": self.condition.value,
            "replicate_seed": self.seeds.replicate_seed,
            "noralet_id": identity,
            "initial_tick": lifetime.initial_tick,
            "final_observed_tick": lifetime.final_observed_tick,
            "death_occurred": death_occurred,
            "death_cause": lifetime.death_cause,
            "right_censored": not death_occurred and final_tick >= max_ticks,
            "observed_lifetime_ticks": (
                lifetime.final_observed_tick - lifetime.initial_tick
            ),
            "initial_energy": lifetime.initial_energy,
            "final_observed_energy": lifetime.final_observed_energy,
            "minimum_energy": lifetime.minimum_energy,
            "mean_sampled_energy": (
                lifetime.sampled_energy_total / lifetime.sampled_energy_count
            ),
            "initial_condition": lifetime.initial_condition,
            "final_observed_condition": lifetime.final_observed_condition,
            "minimum_condition": lifetime.minimum_condition,
            "mean_energy_distress": (
                lifetime.energy_distress_total / lifetime.energy_distress_count
            ),
            "maximum_energy_distress": lifetime.maximum_energy_distress,
            "mean_condition_distress": (
                lifetime.condition_distress_total
                / lifetime.condition_distress_count
            ),
            "maximum_condition_distress": lifetime.maximum_condition_distress,
            "total_absolute_distance_travelled": lifetime.total_absolute_distance,
            "mean_absolute_velocity": (
                lifetime.absolute_velocity_total / lifetime.velocity_observation_count
            ),
            "consume_attempt_count": lifetime.consume_attempt_count,
            "successful_consumption_count": lifetime.successful_consumption_count,
            "total_energy_consumed": lifetime.total_energy_consumed,
            "signal_emission_count": lifetime.signal_emission_count,
            "signal_A_count": lifetime.signal_type_counts["A"],
            "signal_B_count": lifetime.signal_type_counts["B"],
            "signal_C_count": lifetime.signal_type_counts["C"],
            "signal_D_count": lifetime.signal_type_counts["D"],
            "signal_LEFT_count": lifetime.signal_direction_counts["LEFT"],
            "signal_RIGHT_count": lifetime.signal_direction_counts["RIGHT"],
            "received_signal_percept_count": lifetime.received_signal_percept_count,
            "mean_requested_acceleration": (
                lifetime.acceleration_total / action_count if action_count else None
            ),
            "mean_absolute_requested_acceleration": (
                lifetime.absolute_acceleration_total / action_count
                if action_count
                else None
            ),
            "positive_acceleration_count": lifetime.positive_acceleration_count,
            "negative_acceleration_count": lifetime.negative_acceleration_count,
            "near_zero_acceleration_count": lifetime.near_zero_acceleration_count,
            "consume_selected_count": lifetime.consume_selected_count,
            "consume_not_selected_count": lifetime.consume_not_selected_count,
            "online_encoder_parameter_drift_norm": parameter_drift_norm(
                brain.model.encoder,
                birth.encoder,
            ),
            "GRU_parameter_drift_norm": parameter_drift_norm(
                brain.model.recurrent_core,
                birth.recurrent_core,
            ),
            "predictor_parameter_drift_norm": parameter_drift_norm(
                brain.model.prediction_model,
                birth.predictor,
            ),
            "action_head_parameter_drift_norm": self._action_head_drift(
                brain,
                birth.action_heads,
            ),
        }
        row.update(
            {
                f"signal_{key}_count": lifetime.signal_selection_counts[key]
                for key in lifetime.signal_selection_counts
            }
        )
        row.update(lifetime.prediction.summary())
        row.update(lifetime.homeostatic.summary())
        return {column: row[column] for column in NORALET_SUMMARY_COLUMNS}

    @staticmethod
    def _action_head_drift(
        brain: NoraletBrain,
        birth_parameters: tuple[Tensor, ...],
    ) -> float:
        current = brain.model.action_head_parameters()
        squared = math.fsum(
            float(
                torch.sum(
                    (
                        parameter.detach().cpu().double() - birth.double()
                    ).square()
                ).item()
            )
            for parameter, birth in zip(current, birth_parameters, strict=True)
        )
        return math.sqrt(squared)
