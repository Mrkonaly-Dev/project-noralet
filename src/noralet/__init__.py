"""Project Noralet simulation package."""

from importlib import import_module

from noralet.simulation import (
    ActionIntent,
    ActiveSignal,
    ConsumableEnergyPoint,
    DeterministicRandomStreams,
    EnergyConservationError,
    EnergyConsumed,
    EnergyEcologyConfig,
    EnergyPointDecayed,
    EnergyPointDissolved,
    EnergyPointFormed,
    EnergyTotals,
    EnvironmentalEnergyPool,
    ExternalPercept,
    FormationProbabilities,
    Interoception,
    NoraletActuatorConfig,
    NoraletAccelerated,
    NoraletBodyState,
    NoraletDeathCause,
    NoraletDied,
    NoraletEnergyConfig,
    NoraletPhysiologyConfig,
    NoraletEnergyExpenditureReason,
    NoraletEnergyReleased,
    NoraletEnergySpent,
    NoraletExperience,
    NoraletExperienceConfig,
    NoraletMoved,
    NoraletSignalConfig,
    SensorimotorFeedback,
    SignalDirection,
    SignalEmissionIntent,
    SignalEmitted,
    SignalPercept,
    SignalType,
    RegionDefinition,
    RegionKind,
    RoutedNoraletExperience,
    Simulation,
    SimulationConfig,
    SimulationEvent,
    TickAdvanced,
    TickResult,
    WorldState,
    condition_after_tick,
    mortality_hazard,
    natural_death_probability,
)

_BRAIN_EXPORTS = frozenset(
    (
        "ACTION_RANDOM_DRAW_ORDER",
        "AutonomousSimulationRunner",
        "AutonomousTickResult",
        "BaseBrain",
        "BrainActionParameters",
        "ExperienceEncoder",
        "NoraletBrain",
        "NoraletBrainConfig",
        "NoraletBrainModel",
        "SignalMotorChoice",
        "resolve_brain_device",
    )
)


def __getattr__(name: str) -> object:
    """Load the optional heavy neural API only when it is requested."""

    if name not in _BRAIN_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("noralet.brain"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set((*globals(), *_BRAIN_EXPORTS)))

__all__ = [
    "ACTION_RANDOM_DRAW_ORDER",
    "ActionIntent",
    "ActiveSignal",
    "AutonomousSimulationRunner",
    "AutonomousTickResult",
    "BaseBrain",
    "BrainActionParameters",
    "ConsumableEnergyPoint",
    "DeterministicRandomStreams",
    "EnergyConservationError",
    "EnergyConsumed",
    "EnergyEcologyConfig",
    "EnergyPointDecayed",
    "EnergyPointDissolved",
    "EnergyPointFormed",
    "EnergyTotals",
    "EnvironmentalEnergyPool",
    "ExternalPercept",
    "ExperienceEncoder",
    "FormationProbabilities",
    "Interoception",
    "NoraletActuatorConfig",
    "NoraletAccelerated",
    "NoraletBodyState",
    "NoraletBrain",
    "NoraletBrainConfig",
    "NoraletBrainModel",
    "NoraletDeathCause",
    "NoraletDied",
    "NoraletEnergyConfig",
    "NoraletEnergyExpenditureReason",
    "NoraletEnergyReleased",
    "NoraletEnergySpent",
    "NoraletExperience",
    "NoraletExperienceConfig",
    "NoraletMoved",
    "NoraletPhysiologyConfig",
    "NoraletSignalConfig",
    "RegionDefinition",
    "RegionKind",
    "RoutedNoraletExperience",
    "Simulation",
    "SimulationConfig",
    "SimulationEvent",
    "SensorimotorFeedback",
    "SignalDirection",
    "SignalEmissionIntent",
    "SignalEmitted",
    "SignalMotorChoice",
    "SignalPercept",
    "SignalType",
    "TickAdvanced",
    "TickResult",
    "WorldState",
    "condition_after_tick",
    "mortality_hazard",
    "natural_death_probability",
    "resolve_brain_device",
]
