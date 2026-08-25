"""Neural control substrate for autonomous Noralets."""

from noralet.brain.base import BaseBrain
from noralet.brain.config import (
    BASE_BRAIN_INITIALIZATION_VERSION,
    BaseBrainInitializationConfig,
    NoraletBrainConfig,
    base_brain_initialization_manifest,
    resolve_brain_device,
)
from noralet.brain.coordinator import (
    AutonomousSimulationRunner,
    AutonomousTickResult,
    NoraletHomeostaticLearningResult,
    NoraletLearningResult,
)
from noralet.brain.encoder import ExperienceEncoder
from noralet.brain.learning import (
    ActionEligibilityTraces,
    HomeostaticPlasticityResult,
    NoraletHomeostaticPlasticityConfig,
    NoraletLearningConfig,
    PredictiveLearningResult,
    homeostatic_drive,
    homeostatic_modulation,
)
from noralet.brain.model import (
    ACTION_VECTOR_SIZE,
    BrainActionParameters,
    NoraletBrainModel,
    PredictionModel,
)
from noralet.brain.runtime import (
    ACTION_RANDOM_DRAW_ORDER,
    BrainActionSelection,
    NoraletBrain,
    SignalMotorChoice,
)

__all__ = [
    "ACTION_RANDOM_DRAW_ORDER",
    "ACTION_VECTOR_SIZE",
    "ActionEligibilityTraces",
    "AutonomousSimulationRunner",
    "AutonomousTickResult",
    "BaseBrain",
    "BASE_BRAIN_INITIALIZATION_VERSION",
    "BaseBrainInitializationConfig",
    "BrainActionSelection",
    "BrainActionParameters",
    "ExperienceEncoder",
    "HomeostaticPlasticityResult",
    "NoraletBrain",
    "NoraletBrainConfig",
    "NoraletBrainModel",
    "NoraletHomeostaticLearningResult",
    "NoraletHomeostaticPlasticityConfig",
    "NoraletLearningConfig",
    "NoraletLearningResult",
    "PredictionModel",
    "PredictiveLearningResult",
    "SignalMotorChoice",
    "homeostatic_drive",
    "homeostatic_modulation",
    "base_brain_initialization_manifest",
    "resolve_brain_device",
]
