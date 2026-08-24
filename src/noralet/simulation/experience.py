"""Deterministic transformation from objective world truth to experience."""

from __future__ import annotations

from dataclasses import dataclass
import math

from noralet.noralets.body import NoraletBodyState
from noralet.noralets.experience import (
    ExternalPercept,
    Interoception,
    NoraletExperience,
    NoraletExperienceConfig,
    SensorimotorFeedback,
)
from noralet.simulation.state import WorldState


_LARGEST_SIGNAL_BELOW_ONE = math.nextafter(1.0, 0.0)


@dataclass(frozen=True, slots=True)
class _TransitionFeedback:
    """Exact engine facts retained only until sensory transformation."""

    applied_acceleration: float = 0.0
    consume_attempt_executed: bool = False
    consumed_energy: float = 0.0
    actual_energy_expenditure: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "applied_acceleration",
            "consumed_energy",
            "actual_energy_expenditure",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, converted)
        if type(self.consume_attempt_executed) is not bool:
            raise TypeError("consume_attempt_executed must be a boolean")
        if self.consumed_energy < 0.0:
            raise ValueError("consumed_energy cannot be negative")
        if self.actual_energy_expenditure < 0.0:
            raise ValueError("actual_energy_expenditure cannot be negative")


_NEUTRAL_TRANSITION_FEEDBACK = _TransitionFeedback()


@dataclass(frozen=True, slots=True)
class _ExperienceBuilder:
    """Engine-side deterministic boundary hiding objective world semantics."""

    config: NoraletExperienceConfig
    energy_capacity: float
    left_boundary: float
    right_boundary: float

    def build(
        self,
        state: WorldState,
        body: NoraletBodyState,
        feedback: _TransitionFeedback,
    ) -> NoraletExperience:
        """Derive one complete sensory value without mutating world or RNG."""

        return NoraletExperience(
            external_percepts=self._external_percepts(state, body),
            interoception=self._interoception(body, feedback),
            sensorimotor_feedback=self._sensorimotor_feedback(feedback),
        )

    def _external_percepts(
        self,
        state: WorldState,
        observer: NoraletBodyState,
    ) -> tuple[ExternalPercept, ...]:
        radius = self.config.vision_radius
        zero_signature = (0.0,) * self.config.signature_length
        percepts: list[
            tuple[tuple[float, int, int], ExternalPercept]
        ] = []

        for point in state.energy_points:
            distance = abs(point.position - observer.position)
            if distance <= radius:
                percepts.append(
                    (
                        (point.position, 1, point.point_id),
                        self._external_percept(
                            appearance=(
                                *self.config.consumable_base_appearance,
                                *zero_signature,
                            ),
                            relative_position=point.position - observer.position,
                            distance=distance,
                        ),
                    )
                )

        for other in state.bodies:
            if other.noralet_id == observer.noralet_id:
                continue
            if len(other.perceptual_signature) != self.config.signature_length:
                raise ValueError("visible Noralet has an invalid perceptual signature")
            distance = abs(other.position - observer.position)
            if distance <= radius:
                percepts.append(
                    (
                        (other.position, 2, other.noralet_id),
                        self._external_percept(
                            appearance=(
                                *self.config.noralet_base_appearance,
                                *other.perceptual_signature,
                            ),
                            relative_position=other.position - observer.position,
                            distance=distance,
                        ),
                    )
                )

        boundary_sources = (
            (self.left_boundary, -1.0, 0),
            (self.right_boundary, 1.0, 1),
        )
        for position, direction, tie_identity in boundary_sources:
            distance = abs(position - observer.position)
            if distance <= radius:
                percepts.append(
                    (
                        (position, 0, tie_identity),
                        ExternalPercept(
                            appearance_pattern=(
                                *self.config.boundary_base_appearance,
                                *zero_signature,
                            ),
                            direction_signal=direction,
                            proximity_signal=self._proximity(distance),
                        ),
                    )
                )

        percepts.sort(key=lambda item: item[0])
        return tuple(percept for _, percept in percepts)

    def _external_percept(
        self,
        *,
        appearance: tuple[float, ...],
        relative_position: float,
        distance: float,
    ) -> ExternalPercept:
        if relative_position < 0.0:
            direction = -1.0
        elif relative_position > 0.0:
            direction = 1.0
        else:
            direction = 0.0
        return ExternalPercept(
            appearance_pattern=appearance,
            direction_signal=direction,
            proximity_signal=self._proximity(distance),
        )

    def _proximity(self, distance: float) -> float:
        proximity = 1.0 - distance / self.config.vision_radius
        return min(1.0, max(0.0, proximity))

    def _interoception(
        self,
        body: NoraletBodyState,
        feedback: _TransitionFeedback,
    ) -> Interoception:
        energy_ratio = min(1.0, max(0.0, body.energy / self.energy_capacity))
        energy_distress = (
            (1.0 - energy_ratio) ** self.config.energy_distress_exponent
        )
        condition_distress = (
            (1.0 - body.condition) ** self.config.condition_distress_exponent
        )
        return Interoception(
            energy_distress=min(1.0, max(0.0, energy_distress)),
            condition_distress=min(1.0, max(0.0, condition_distress)),
            energetic_exertion=self._saturating_sensation(
                feedback.actual_energy_expenditure,
                self.config.exertion_sensation_scale,
            ),
        )

    def _sensorimotor_feedback(
        self,
        feedback: _TransitionFeedback,
    ) -> SensorimotorFeedback:
        if feedback.applied_acceleration < 0.0:
            motor_direction = -1.0
        elif feedback.applied_acceleration > 0.0:
            motor_direction = 1.0
        else:
            motor_direction = 0.0
        return SensorimotorFeedback(
            motor_direction=motor_direction,
            motor_effort=self._saturating_sensation(
                abs(feedback.applied_acceleration),
                self.config.motor_effort_scale,
            ),
            consume_activation=(1.0 if feedback.consume_attempt_executed else 0.0),
            ingestion_signal=self._saturating_sensation(
                feedback.consumed_energy,
                self.config.ingestion_sensation_scale,
            ),
        )

    @staticmethod
    def _saturating_sensation(amount: float, scale: float) -> float:
        sensation = -math.expm1(-(amount / scale))
        return min(sensation, _LARGEST_SIGNAL_BELOW_ONE)
