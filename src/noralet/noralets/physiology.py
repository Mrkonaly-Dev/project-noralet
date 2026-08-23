"""Objective Noralet condition, ageing and natural-mortality mechanics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys


_MAX_FINITE_FLOAT = sys.float_info.max
_LOG_MAX_FINITE_FLOAT = math.log(_MAX_FINITE_FLOAT)
_LARGEST_PROBABILITY_BELOW_ONE = math.nextafter(1.0, 0.0)


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _condition_value(condition: float) -> float:
    converted = _finite_float("condition", condition)
    if not 0.0 <= converted <= 1.0:
        raise ValueError("condition must be in [0, 1]")
    return converted


def _age_value(age_ticks: int) -> int:
    if type(age_ticks) is not int:
        raise TypeError("age_ticks must be an integer")
    if age_ticks < 0:
        raise ValueError("age_ticks cannot be negative")
    return age_ticks


def _saturated_product(*values: float) -> float:
    result = 1.0
    for value in values:
        if value == 0.0:
            return 0.0
        if result > _MAX_FINITE_FLOAT / value:
            return _MAX_FINITE_FLOAT
        result *= value
    return result


def _saturated_sum(*values: float) -> float:
    result = 0.0
    for value in values:
        if value >= _MAX_FINITE_FLOAT - result:
            return _MAX_FINITE_FLOAT
        result += value
    return result


def _age_pressure(age_ticks: int, scale: float, exponent: float) -> float:
    if age_ticks == 0:
        return 0.0
    log_pressure = exponent * (math.log(age_ticks) - math.log(scale))
    if log_pressure >= _LOG_MAX_FINITE_FLOAT:
        return _MAX_FINITE_FLOAT
    return math.exp(log_pressure)


@dataclass(frozen=True, slots=True)
class NoraletPhysiologyConfig:
    """Slow condition wear and state-driven natural-mortality parameters."""

    low_energy_condition_threshold_ratio: float
    baseline_condition_loss_per_tick: float
    deprivation_condition_loss_scale: float
    deprivation_exponent: float
    base_mortality_hazard: float
    mortality_age_scale: float
    mortality_age_exponent: float
    condition_hazard_scale: float
    mortality_condition_exponent: float
    age_hazard_scale: float
    interaction_hazard_scale: float

    def __post_init__(self) -> None:
        for name in (
            "low_energy_condition_threshold_ratio",
            "baseline_condition_loss_per_tick",
            "deprivation_condition_loss_scale",
            "deprivation_exponent",
            "base_mortality_hazard",
            "mortality_age_scale",
            "mortality_age_exponent",
            "condition_hazard_scale",
            "mortality_condition_exponent",
            "age_hazard_scale",
            "interaction_hazard_scale",
        ):
            object.__setattr__(self, name, _finite_float(name, getattr(self, name)))

        if not 0.0 < self.low_energy_condition_threshold_ratio <= 1.0:
            raise ValueError(
                "low_energy_condition_threshold_ratio must be in (0, 1]"
            )
        if self.baseline_condition_loss_per_tick < 0.0:
            raise ValueError("baseline_condition_loss_per_tick cannot be negative")
        if self.deprivation_condition_loss_scale < 0.0:
            raise ValueError("deprivation_condition_loss_scale cannot be negative")
        if self.deprivation_exponent < 1.0:
            raise ValueError("deprivation_exponent must be at least 1")
        if self.base_mortality_hazard < 0.0:
            raise ValueError("base_mortality_hazard cannot be negative")
        if self.mortality_age_scale <= 0.0:
            raise ValueError("mortality_age_scale must be positive")
        if self.mortality_age_exponent <= 1.0:
            raise ValueError("mortality_age_exponent must be greater than 1")
        if self.condition_hazard_scale < 0.0:
            raise ValueError("condition_hazard_scale cannot be negative")
        if self.mortality_condition_exponent < 1.0:
            raise ValueError("mortality_condition_exponent must be at least 1")
        if self.age_hazard_scale < 0.0:
            raise ValueError("age_hazard_scale cannot be negative")
        if self.interaction_hazard_scale < 0.0:
            raise ValueError("interaction_hazard_scale cannot be negative")


def condition_after_tick(
    current_condition: float,
    stored_energy: float,
    energy_capacity: float,
    config: NoraletPhysiologyConfig,
) -> float:
    """Return irreversible condition after baseline and deprivation wear."""

    if not isinstance(config, NoraletPhysiologyConfig):
        raise TypeError("config must be a NoraletPhysiologyConfig")
    condition = _condition_value(current_condition)
    energy = _finite_float("stored_energy", stored_energy)
    capacity = _finite_float("energy_capacity", energy_capacity)
    if energy < 0.0:
        raise ValueError("stored_energy cannot be negative")
    if capacity <= 0.0:
        raise ValueError("energy_capacity must be positive")
    if energy > capacity:
        raise ValueError("stored_energy cannot exceed energy_capacity")

    energy_ratio = energy / capacity
    threshold = config.low_energy_condition_threshold_ratio
    deprivation = max(0.0, (threshold - energy_ratio) / threshold)
    deprivation_loss = _saturated_product(
        config.deprivation_condition_loss_scale,
        deprivation**config.deprivation_exponent,
    )
    condition_loss = _saturated_sum(
        config.baseline_condition_loss_per_tick,
        deprivation_loss,
    )
    return max(0.0, condition - condition_loss)


def mortality_hazard(
    age_ticks: int,
    condition: float,
    config: NoraletPhysiologyConfig,
) -> float:
    """Return the finite non-negative age/condition mortality hazard."""

    if not isinstance(config, NoraletPhysiologyConfig):
        raise TypeError("config must be a NoraletPhysiologyConfig")
    age = _age_value(age_ticks)
    resolved_condition = _condition_value(condition)
    age_pressure = _age_pressure(
        age,
        config.mortality_age_scale,
        config.mortality_age_exponent,
    )
    condition_pressure = (
        (1.0 - resolved_condition) ** config.mortality_condition_exponent
    )
    return _saturated_sum(
        config.base_mortality_hazard,
        _saturated_product(config.age_hazard_scale, age_pressure),
        _saturated_product(config.condition_hazard_scale, condition_pressure),
        _saturated_product(
            config.interaction_hazard_scale,
            age_pressure,
            condition_pressure,
        ),
    )


def natural_death_probability(
    age_ticks: int,
    condition: float,
    config: NoraletPhysiologyConfig,
) -> float:
    """Convert mortality hazard into a representable probability below one."""

    hazard = mortality_hazard(age_ticks, condition, config)
    probability = -math.expm1(-hazard)
    return min(probability, _LARGEST_PROBABILITY_BELOW_ONE)
