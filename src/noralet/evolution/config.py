"""Immutable protocol for Evolution Bootstrap Environment v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any

from noralet.research.config import (
    LearningCondition,
    SeedMapping,
    baseline_configuration_manifest,
)


EVOLUTION_ID = "001-basebrain-bootstrap"
EVOLUTION_SCHEMA_VERSION = "1.0"
DEFAULT_TRAINING_WORLD_SEEDS = (1101, 2203, 3301, 4409)
DEFAULT_VALIDATION_WORLD_SEEDS = (5501, 6607, 7703, 8807)
_SEED_DOMAIN = b"project-noralet:evolution-001:seed:v1\0"


def derived_seed(initial_seed: int, *roles: object) -> int:
    """Derive a stable 63-bit seed without Python's randomized hash()."""

    if type(initial_seed) is not int:
        raise TypeError("initial_seed must be an integer")
    digest = hashlib.sha256()
    digest.update(_SEED_DOMAIN)
    digest.update(str(initial_seed).encode("ascii"))
    for role in roles:
        encoded = str(role).encode("utf-8")
        digest.update(b"\0")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def fixed_world_seeds(role: str, count: int) -> tuple[int, ...]:
    """Return a fixed prefix, deterministically extending it for smoke variants."""

    if role not in ("training", "validation"):
        raise ValueError("role must be training or validation")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    defaults = (
        DEFAULT_TRAINING_WORLD_SEEDS
        if role == "training"
        else DEFAULT_VALIDATION_WORLD_SEEDS
    )
    values = list(defaults[:count])
    while len(values) < count:
        values.append(derived_seed(0, "world", role, len(values)))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """External mutation-only viability-selection protocol."""

    generation_count: int = 50
    device: str = "cuda"
    population_size: int = 32
    elite_count: int = 4
    parent_pool_size: int = 8
    mutation_sigma: float = 0.02
    training_world_seeds: tuple[int, ...] = DEFAULT_TRAINING_WORLD_SEEDS
    validation_world_seeds: tuple[int, ...] = DEFAULT_VALIDATION_WORLD_SEEDS
    noralets_per_world: int = 6
    max_ticks: int = 2_000
    initial_body_energy: float = 10.0
    initial_seed: int = 1
    champion_checkpoint_interval: int = 5
    output_root: Path = Path("evolution-results")

    def __post_init__(self) -> None:
        for name in (
            "generation_count",
            "population_size",
            "elite_count",
            "parent_pool_size",
            "noralets_per_world",
            "max_ticks",
            "champion_checkpoint_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.elite_count > self.parent_pool_size:
            raise ValueError("elite_count cannot exceed parent_pool_size")
        if self.parent_pool_size > self.population_size:
            raise ValueError("parent_pool_size cannot exceed population_size")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        device = self.device.strip().lower()
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        for name in ("mutation_sigma", "initial_body_energy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            converted = float(value)
            if not math.isfinite(converted) or converted <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, converted)
        if self.initial_body_energy > 100.0:
            raise ValueError("initial_body_energy cannot exceed body capacity 100")
        if type(self.initial_seed) is not int:
            raise TypeError("initial_seed must be an integer")
        for name in ("training_world_seeds", "validation_world_seeds"):
            seeds = getattr(self, name)
            if not isinstance(seeds, tuple) or not seeds:
                raise TypeError(f"{name} must be a non-empty tuple")
            if any(type(seed) is not int for seed in seeds):
                raise TypeError(f"every {name} value must be an integer")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} must contain unique seeds")
        if set(self.training_world_seeds) & set(self.validation_world_seeds):
            raise ValueError("training and validation seeds must be disjoint")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def worlds_per_candidate(self) -> int:
        return len(self.training_world_seeds)

    @property
    def individuals_per_candidate(self) -> int:
        return self.worlds_per_candidate * self.noralets_per_world

    def state(self) -> dict[str, Any]:
        values = asdict(self)
        values["training_world_seeds"] = list(self.training_world_seeds)
        values["validation_world_seeds"] = list(self.validation_world_seeds)
        values["output_root"] = str(self.output_root)
        return values

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> EvolutionConfig:
        values = dict(state)
        values["training_world_seeds"] = tuple(values["training_world_seeds"])
        values["validation_world_seeds"] = tuple(values["validation_world_seeds"])
        values["output_root"] = Path(values["output_root"])
        return cls(**values)

    def with_resume_target(
        self,
        *,
        generation_count: int,
        device: str | None = None,
    ) -> EvolutionConfig:
        return replace(
            self,
            generation_count=generation_count,
            device=self.device if device is None else device,
        )

    def scientific_configuration(self) -> dict[str, Any]:
        manifest_seed = SeedMapping(
            replicate_seed=self.initial_seed,
            simulation_seed=self.training_world_seeds[0],
            base_brain_seed=derived_seed(self.initial_seed, "manifest-brain"),
        )
        return {
            "evolution_id": EVOLUTION_ID,
            "schema_version": EVOLUTION_SCHEMA_VERSION,
            "generation_count_target": self.generation_count,
            "device": self.device,
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "parent_pool_size": self.parent_pool_size,
            "mutation_sigma": self.mutation_sigma,
            "training_world_seeds": list(self.training_world_seeds),
            "validation_world_seeds": list(self.validation_world_seeds),
            "worlds_per_candidate": self.worlds_per_candidate,
            "noralets_per_world": self.noralets_per_world,
            "max_ticks": self.max_ticks,
            "initial_body_energy": self.initial_body_energy,
            "initial_seed": self.initial_seed,
            "champion_checkpoint_interval": self.champion_checkpoint_interval,
            "learning_mode": LearningCondition.FULL_CURRENT_BRAIN.value,
            "fitness": "mean observed lifetime ticks, truncated at max_ticks",
            "selection_interpretation": "viability proxy for evolutionary bootstrap",
            "validation_policy": "generation-best candidate on every generation",
            "evolution_environment_configuration": baseline_configuration_manifest(
                population=self.noralets_per_world,
                device=self.device,
                seeds=manifest_seed,
                initial_body_energy=self.initial_body_energy,
            ),
        }
