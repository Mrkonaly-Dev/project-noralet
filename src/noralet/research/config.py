"""Immutable protocol and explicit baseline construction for Research 001."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum, StrEnum
import hashlib
import math
from pathlib import Path
from typing import Any

from noralet.brain import (
    BaseBrain,
    NoraletBrainConfig,
    NoraletHomeostaticPlasticityConfig,
    NoraletLearningConfig,
)
from noralet.noralets.actuators import NoraletActuatorConfig
from noralet.noralets.body import NoraletBodyState
from noralet.noralets.energy import NoraletEnergyConfig
from noralet.noralets.experience import NoraletExperienceConfig
from noralet.noralets.physiology import NoraletPhysiologyConfig
from noralet.noralets.signals import NoraletSignalConfig
from noralet.simulation.config import SimulationConfig
from noralet.simulation.runtime import Simulation
from noralet.world.energy import (
    ConsumableEnergyPoint,
    EnergyEcologyConfig,
    EnvironmentalEnergyPool,
    FormationProbabilities,
)
from noralet.world.regions import RegionDefinition, RegionKind


EXPERIMENT_ID = "001-baseline-lifetime-adaptation"
EXPERIMENT_SCHEMA_VERSION = "1.0"
_SEED_DOMAIN = b"project-noralet:research-001:seed:v1\0"


class LearningCondition(StrEnum):
    """The complete naturally supported 2x2 lifetime-learning protocol."""

    NO_LEARNING = "no-learning"
    PREDICTIVE_ONLY = "predictive-only"
    FULL_CURRENT_BRAIN = "full-current-brain"
    HOMEOSTATIC_ONLY = "homeostatic-only"

    @property
    def predictive_enabled(self) -> bool:
        return self in (
            LearningCondition.PREDICTIVE_ONLY,
            LearningCondition.FULL_CURRENT_BRAIN,
        )

    @property
    def homeostatic_enabled(self) -> bool:
        return self in (
            LearningCondition.FULL_CURRENT_BRAIN,
            LearningCondition.HOMEOSTATIC_ONLY,
        )


PROTOCOL_CONDITIONS = tuple(LearningCondition)

CONDITION_DEFINITIONS = {
    condition.value: {
        "predictive_lifetime_learning": condition.predictive_enabled,
        "homeostatic_action_plasticity": condition.homeostatic_enabled,
    }
    for condition in PROTOCOL_CONDITIONS
}

PREDEFINED_HYPOTHESES = (
    {
        "id": "H1",
        "text": (
            "Lifetime learning may affect survival duration, Energy regulation, "
            "distress exposure, and consumption behaviour relative to no learning."
        ),
    },
    {
        "id": "H2",
        "text": (
            "Prediction loss may decrease with lived experience when predictive "
            "lifetime learning is active."
        ),
    },
    {
        "id": "H3",
        "text": (
            "Homeostatic action plasticity may change action statistics and "
            "action-head parameters without necessarily improving ecology."
        ),
    },
    {
        "id": "H4",
        "text": (
            "Initially shared inherited brains may diverge in hidden trajectories, "
            "parameters, actions, and observed lifetimes through different histories."
        ),
    },
    {
        "id": "H5",
        "text": (
            "Signal behaviour is measured descriptively; meaningful communication "
            "is not presumed to emerge in this baseline experiment."
        ),
    },
)


@dataclass(frozen=True, slots=True)
class SeedMapping:
    replicate_seed: int
    simulation_seed: int
    base_brain_seed: int


def _derived_seed(replicate_seed: int, role: str) -> int:
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(str(replicate_seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(role.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def seed_mapping(replicate_seed: int) -> SeedMapping:
    if type(replicate_seed) is not int:
        raise TypeError("replicate_seed must be an integer")
    return SeedMapping(
        replicate_seed=replicate_seed,
        simulation_seed=_derived_seed(replicate_seed, "simulation"),
        base_brain_seed=_derived_seed(replicate_seed, "base-brain"),
    )


@dataclass(frozen=True, slots=True)
class BaselineExperimentConfig:
    """Batch-level controls; pilot defaults are experimental, not architectural."""

    replicate_seeds: tuple[int, ...] = tuple(range(1, 11))
    max_ticks: int = 5_000
    sample_every_ticks: int = 10
    initial_population: int = 6
    device: str = "cuda"
    conditions: tuple[LearningCondition, ...] = PROTOCOL_CONDITIONS
    output_root: Path = Path("research-results")

    def __post_init__(self) -> None:
        if not isinstance(self.replicate_seeds, tuple):
            raise TypeError("replicate_seeds must be an immutable tuple")
        if len(self.replicate_seeds) < 2:
            raise ValueError("the research protocol requires at least two seeds")
        if any(type(seed) is not int for seed in self.replicate_seeds):
            raise TypeError("every replicate seed must be an integer")
        if len(set(self.replicate_seeds)) != len(self.replicate_seeds):
            raise ValueError("replicate seeds must be unique")
        for name in ("max_ticks", "sample_every_ticks", "initial_population"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        device = self.device.strip().lower()
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        if not isinstance(self.conditions, tuple) or not self.conditions:
            raise TypeError("conditions must be a non-empty immutable tuple")
        if not all(isinstance(value, LearningCondition) for value in self.conditions):
            raise TypeError("every condition must be a LearningCondition")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be unique")
        canonical_conditions = tuple(
            condition for condition in PROTOCOL_CONDITIONS if condition in self.conditions
        )
        output_root = Path(self.output_root)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "conditions", canonical_conditions)
        object.__setattr__(self, "output_root", output_root)

    @property
    def seed_mappings(self) -> tuple[SeedMapping, ...]:
        return tuple(seed_mapping(seed) for seed in self.replicate_seeds)

    @property
    def total_runs(self) -> int:
        return len(self.replicate_seeds) * len(self.conditions)

    def scientific_configuration(self) -> dict[str, Any]:
        return {
            "experiment_id": EXPERIMENT_ID,
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "replicate_seeds": list(self.replicate_seeds),
            "seed_mappings": [asdict(value) for value in self.seed_mappings],
            "max_ticks": self.max_ticks,
            "sample_every_ticks": self.sample_every_ticks,
            "initial_population": self.initial_population,
            "device": self.device,
            "conditions": [condition.value for condition in self.conditions],
            "baseline_configuration": baseline_configuration_manifest(
                population=self.initial_population,
                device=self.device,
                seeds=self.seed_mappings[0],
            ),
        }


def predictive_learning_config() -> NoraletLearningConfig:
    return NoraletLearningConfig(
        learning_rate=0.01,
        max_gradient_norm=1.0,
        predictor_hidden_size=8,
    )


def homeostatic_plasticity_config() -> NoraletHomeostaticPlasticityConfig:
    return NoraletHomeostaticPlasticityConfig(
        energy_distress_weight=1.0,
        condition_distress_weight=1.0,
        homeostatic_modulation_scale=0.2,
        eligibility_decay=0.8,
        action_learning_rate=0.05,
        max_homeostatic_update_norm=2.0,
    )


def _ecology_config() -> EnergyEcologyConfig:
    regions = (
        RegionDefinition("left-sparse", -100.0, -25.0, RegionKind.SPARSE),
        RegionDefinition("central-fertile", -25.0, 25.0, RegionKind.FERTILE),
        RegionDefinition("right-sparse", 25.0, 100.0, RegionKind.SPARSE),
    )
    return EnergyEcologyConfig(
        regions=regions,
        initial_environmental_energy=tuple(
            EnvironmentalEnergyPool(region.region_id, 500.0)
            for region in regions
        ),
        formation_probabilities=FormationProbabilities(
            infertile=0.001,
            sparse=0.004,
            fertile=0.012,
        ),
        formation_energy_min=4.0,
        formation_energy_max=8.0,
        decay_rate=0.002,
        point_removal_threshold=0.1,
        minimum_energy_point_spacing=3.0,
    )


def _experience_config() -> NoraletExperienceConfig:
    return NoraletExperienceConfig(
        vision_radius=12.0,
        consumable_base_appearance=(0.72, -0.11),
        noralet_base_appearance=(0.09, 0.83),
        boundary_base_appearance=(-0.44, 0.17),
        signature_length=2,
        energy_distress_exponent=2.0,
        condition_distress_exponent=1.5,
        motor_effort_scale=2.0,
        ingestion_sensation_scale=5.0,
        exertion_sensation_scale=3.0,
    )


def _signal_config() -> NoraletSignalConfig:
    return NoraletSignalConfig(
        signal_radius=20.0,
        signal_energy_cost=0.02,
        signal_pattern_a=(0.91, -0.13, 0.27),
        signal_pattern_b=(-0.22, 0.84, 0.31),
        signal_pattern_c=(0.18, 0.36, -0.77),
        signal_pattern_d=(-0.63, -0.24, 0.52),
    )


def _physiology_config() -> NoraletPhysiologyConfig:
    return NoraletPhysiologyConfig(
        low_energy_condition_threshold_ratio=0.4,
        baseline_condition_loss_per_tick=0.00002,
        deprivation_condition_loss_scale=0.001,
        deprivation_exponent=2.0,
        base_mortality_hazard=0.00002,
        mortality_age_scale=5_000.0,
        mortality_age_exponent=2.0,
        condition_hazard_scale=0.0002,
        mortality_condition_exponent=2.0,
        age_hazard_scale=0.00005,
        interaction_hazard_scale=0.0001,
    )


def _energy_config() -> NoraletEnergyConfig:
    return NoraletEnergyConfig(
        energy_capacity=100.0,
        existence_energy_cost_per_tick=0.02,
        acceleration_energy_cost_per_unit=0.1,
        consume_radius=1.0,
    )


def _actuator_config() -> NoraletActuatorConfig:
    return NoraletActuatorConfig(max_acceleration=0.25)


def _brain_config(base_brain_seed: int, device: str) -> NoraletBrainConfig:
    return NoraletBrainConfig(
        base_brain_seed=base_brain_seed,
        external_percept_embedding_size=4,
        signal_percept_embedding_size=4,
        interoception_embedding_size=3,
        sensorimotor_embedding_size=4,
        experience_embedding_size=6,
        hidden_size=7,
        acceleration_exploration_std=0.2,
        device=device,
    )


def initial_bodies(population: int) -> tuple[NoraletBodyState, ...]:
    if type(population) is not int or population <= 0:
        raise ValueError("population must be a positive integer")
    if population == 1:
        positions = (0.0,)
    else:
        positions = tuple(
            -30.0 + 60.0 * index / (population - 1)
            for index in range(population)
        )
    return tuple(
        NoraletBodyState(
            noralet_id=index + 1,
            position=position,
            velocity=0.0,
            energy=60.0,
            age_ticks=0,
            condition=1.0,
            perceptual_signature=(
                (index + 1) / (population + 1),
                -(index + 1) / (population + 1),
            ),
        )
        for index, position in enumerate(positions)
    )


def initial_energy_points() -> tuple[ConsumableEnergyPoint, ...]:
    return tuple(
        ConsumableEnergyPoint(point_id=index, position=position, energy=20.0)
        for index, position in enumerate((-60.0, -20.0, 20.0, 60.0))
    )


def build_run_components(
    config: BaselineExperimentConfig,
    condition: LearningCondition,
    seeds: SeedMapping,
) -> tuple[Simulation, BaseBrain]:
    """Construct one fair condition run without consuming any runtime RNG."""

    if not isinstance(config, BaselineExperimentConfig):
        raise TypeError("config must be a BaselineExperimentConfig")
    if not isinstance(condition, LearningCondition):
        raise TypeError("condition must be a LearningCondition")
    if not isinstance(seeds, SeedMapping):
        raise TypeError("seeds must be a SeedMapping")
    return build_baseline_components(
        initial_population=config.initial_population,
        device=config.device,
        condition=condition,
        simulation_seed=seeds.simulation_seed,
        base_brain_seed=seeds.base_brain_seed,
    )


def build_baseline_components(
    *,
    initial_population: int,
    device: str,
    condition: LearningCondition,
    simulation_seed: int,
    base_brain_seed: int,
) -> tuple[Simulation, BaseBrain]:
    """Build one baseline world/brain pair for research or observer tooling."""

    if type(initial_population) is not int or initial_population <= 0:
        raise ValueError("initial_population must be a positive integer")
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    normalized_device = device.strip().lower()
    if normalized_device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda, or auto")
    if not isinstance(condition, LearningCondition):
        raise TypeError("condition must be a LearningCondition")
    if type(simulation_seed) is not int:
        raise TypeError("simulation_seed must be an integer")
    if type(base_brain_seed) is not int:
        raise TypeError("base_brain_seed must be an integer")

    energy = _energy_config()
    experience = _experience_config()
    signals = _signal_config()
    actuator = _actuator_config()
    simulation = Simulation(
        SimulationConfig(
            master_seed=simulation_seed,
            left_boundary=-100.0,
            right_boundary=100.0,
            energy_ecology=_ecology_config(),
            noralet_energy=energy,
            noralet_physiology=_physiology_config(),
            noralet_experience=experience,
            noralet_signals=signals,
            noralet_actuators=actuator,
        ),
        initial_bodies=initial_bodies(initial_population),
        initial_energy_points=initial_energy_points(),
    )
    base_brain = BaseBrain(
        _brain_config(base_brain_seed, normalized_device),
        experience,
        signals,
        actuator,
        predictive_learning_config() if condition.predictive_enabled else None,
        (
            homeostatic_plasticity_config()
            if condition.homeostatic_enabled
            else None
        ),
    )
    return simulation, base_brain


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("manifest values must be finite")
    return value


def baseline_configuration_manifest(
    *,
    population: int,
    device: str,
    seeds: SeedMapping,
) -> dict[str, Any]:
    """Serialize every fixed numerical choice in the baseline protocol."""

    return _json_value(
        {
            "classification": "experimental parameters, not architecture constants",
            "world_boundaries": {"left": -100.0, "right": 100.0},
            "energy_ecology": _ecology_config(),
            "noralet_energy": _energy_config(),
            "noralet_physiology": _physiology_config(),
            "noralet_experience": _experience_config(),
            "noralet_signals": _signal_config(),
            "noralet_actuators": _actuator_config(),
            "brain": _brain_config(seeds.base_brain_seed, device),
            "predictive_learning": predictive_learning_config(),
            "homeostatic_plasticity": homeostatic_plasticity_config(),
            "initial_bodies": initial_bodies(population),
            "initial_energy_points": initial_energy_points(),
        }
    )
