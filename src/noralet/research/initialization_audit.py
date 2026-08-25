"""Fast observer-only audit of newborn BaseBrain action distributions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import statistics
from typing import Any

from noralet.brain import BaseBrain, base_brain_initialization_manifest
from noralet.brain.runtime import SignalMotorChoice
from noralet.noralets.experience import (
    Interoception,
    NoraletExperience,
    SensorimotorFeedback,
)
from noralet.research.config import (
    _actuator_config,
    _brain_config,
    _experience_config,
    _signal_config,
)


INITIALIZATION_AUDIT_ID = "003-basebrain-neutral-initialization-audit"
_AUDIT_SEED_DOMAIN = b"project-noralet:research-003:basebrain-init-audit:v1\0"


@dataclass(frozen=True, slots=True)
class InitializationAuditResult:
    audit_id: str
    sample_count: int
    audit_seed: int
    device: str
    initialization: dict[str, object]
    acceleration_mean: float
    acceleration_standard_deviation: float
    acceleration_positive_fraction: float
    acceleration_negative_fraction: float
    consume_activation_probability: float
    signal_emission_probability: float
    signal_none_probability: float
    conditional_emission_probabilities: dict[str, float]

    def state(self) -> dict[str, Any]:
        return asdict(self)


def initialization_audit_seed(audit_seed: int, sample_index: int) -> int:
    """Derive a distinct stable BaseBrain seed without Python hash()."""

    if type(audit_seed) is not int:
        raise TypeError("audit_seed must be an integer")
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    digest = hashlib.sha256()
    digest.update(_AUDIT_SEED_DOMAIN)
    digest.update(str(audit_seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(sample_index).encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def neutral_synthetic_experience() -> NoraletExperience:
    """Return content-free brain-facing input without constructing a world."""

    return NoraletExperience(
        external_percepts=(),
        signal_percepts=(),
        interoception=Interoception(
            energy_distress=0.0,
            condition_distress=0.0,
            energetic_exertion=0.0,
        ),
        sensorimotor_feedback=SensorimotorFeedback(
            motor_direction=0.0,
            motor_effort=0.0,
            consume_activation=0.0,
            ingestion_signal=0.0,
            signal_emission_activation=0.0,
            signal_emission_pattern=(0.0, 0.0, 0.0),
            signal_emission_direction=0.0,
        ),
    )


def run_initialization_audit(
    *,
    sample_count: int = 100,
    audit_seed: int = 1,
    device: str = "cpu",
) -> InitializationAuditResult:
    """Activate fresh brains once; run no world, learning, fitness or evolution."""

    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(audit_seed) is not int:
        raise TypeError("audit_seed must be an integer")
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    normalized_device = device.strip().lower()
    if normalized_device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda, or auto")

    experience = neutral_synthetic_experience()
    acceleration: list[float] = []
    consume: list[float] = []
    signal_none: list[float] = []
    conditional_emissions: list[tuple[float, ...]] = []
    for index in range(sample_count):
        brain_config = _brain_config(
            initialization_audit_seed(audit_seed, index),
            normalized_device,
        )
        base_brain = BaseBrain(
            brain_config,
            _experience_config(),
            _signal_config(),
            _actuator_config(),
        )
        parameters = base_brain.spawn().activate(experience)
        probabilities = parameters.signal_probabilities
        emission_probability = math.fsum(probabilities[1:])
        acceleration.append(parameters.acceleration_loc)
        consume.append(parameters.consume_probability)
        signal_none.append(probabilities[0])
        conditional_emissions.append(
            tuple(value / emission_probability for value in probabilities[1:])
        )

    category_names = tuple(value.value for value in SignalMotorChoice)[1:]
    return InitializationAuditResult(
        audit_id=INITIALIZATION_AUDIT_ID,
        sample_count=sample_count,
        audit_seed=audit_seed,
        device=normalized_device,
        initialization=base_brain_initialization_manifest(),
        acceleration_mean=math.fsum(acceleration) / sample_count,
        acceleration_standard_deviation=float(statistics.pstdev(acceleration)),
        acceleration_positive_fraction=(
            sum(value > 0.0 for value in acceleration) / sample_count
        ),
        acceleration_negative_fraction=(
            sum(value < 0.0 for value in acceleration) / sample_count
        ),
        consume_activation_probability=math.fsum(consume) / sample_count,
        signal_emission_probability=(
            1.0 - math.fsum(signal_none) / sample_count
        ),
        signal_none_probability=math.fsum(signal_none) / sample_count,
        conditional_emission_probabilities={
            name: math.fsum(row[column] for row in conditional_emissions)
            / sample_count
            for column, name in enumerate(category_names)
        },
    )
