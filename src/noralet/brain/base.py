"""Deterministic CPU prototype initialization and independent brain cloning."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping

import torch
from torch import Tensor

from noralet.brain.config import (
    NoraletBrainConfig,
    base_brain_initialization_manifest,
    resolve_brain_device,
)
from noralet.brain.learning import (
    NoraletHomeostaticPlasticityConfig,
    NoraletLearningConfig,
)
from noralet.brain.model import SIGNAL_MOTOR_OUTCOME_COUNT, NoraletBrainModel
from noralet.noralets.actuators import NoraletActuatorConfig
from noralet.noralets.experience import NoraletExperienceConfig
from noralet.noralets.signals import NoraletSignalConfig


_BASE_BRAIN_SEED_DOMAIN = b"project-noralet:base-brain:v1\0"
_PREDICTOR_SEED_DOMAIN = b"project-noralet:base-brain:predictor:v1\0"
_RECURRENT_SEED_DOMAIN = b"project-noralet:base-brain:recurrent:neutral-v2\0"
_ACCELERATION_SEED_DOMAIN = b"project-noralet:base-brain:acceleration:neutral-v2\0"


class BaseBrain:
    """One deterministic prototype inherited by all initial Noralets."""

    def __init__(
        self,
        config: NoraletBrainConfig,
        experience_config: NoraletExperienceConfig,
        signal_config: NoraletSignalConfig,
        actuator_config: NoraletActuatorConfig,
        learning_config: NoraletLearningConfig | None = None,
        homeostatic_plasticity_config: (
            NoraletHomeostaticPlasticityConfig | None
        ) = None,
    ) -> None:
        if not isinstance(config, NoraletBrainConfig):
            raise TypeError("config must be a NoraletBrainConfig")
        if not isinstance(experience_config, NoraletExperienceConfig):
            raise TypeError(
                "experience_config must be a NoraletExperienceConfig"
            )
        if not isinstance(signal_config, NoraletSignalConfig):
            raise TypeError("signal_config must be a NoraletSignalConfig")
        if not isinstance(actuator_config, NoraletActuatorConfig):
            raise TypeError("actuator_config must be a NoraletActuatorConfig")
        if learning_config is not None and not isinstance(
            learning_config,
            NoraletLearningConfig,
        ):
            raise TypeError("learning_config must be a NoraletLearningConfig")
        if homeostatic_plasticity_config is not None and not isinstance(
            homeostatic_plasticity_config,
            NoraletHomeostaticPlasticityConfig,
        ):
            raise TypeError(
                "homeostatic_plasticity_config must be a "
                "NoraletHomeostaticPlasticityConfig"
            )
        if (
            homeostatic_plasticity_config is not None
            and config.acceleration_exploration_std <= 0.0
        ):
            raise ValueError(
                "homeostatic action plasticity requires positive "
                "acceleration_exploration_std"
            )

        self.config = config
        self.learning_config = learning_config
        self.homeostatic_plasticity_config = homeostatic_plasticity_config
        self.actuator_config = actuator_config
        self.external_pattern_length = experience_config.appearance_length
        self.signal_pattern_length = signal_config.signal_pattern_length
        self.device = resolve_brain_device(config.device)
        torch.use_deterministic_algorithms(True)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            prototype = NoraletBrainModel(
                config,
                external_pattern_length=self.external_pattern_length,
                signal_pattern_length=self.signal_pattern_length,
                learning_config=learning_config,
            )
        self._initialize_parameters(
            prototype.iteration_8_parameters(),
            config.base_brain_seed,
            _BASE_BRAIN_SEED_DOMAIN,
        )
        self._initialize_recurrent_core(
            prototype,
            config.base_brain_seed,
        )
        self._initialize_action_baselines(
            prototype,
            config,
        )
        if prototype.prediction_model is not None:
            self._initialize_parameters(
                tuple(prototype.prediction_model.parameters()),
                config.base_brain_seed,
                _PREDICTOR_SEED_DOMAIN,
            )
        prototype.eval()
        self._prototype = prototype

    @property
    def prototype_model(self) -> NoraletBrainModel:
        """Return the observer-side prototype module."""

        return self._prototype

    @property
    def initialization_manifest(self) -> dict[str, object]:
        """Describe the initializer used before any explicit genome load."""

        return base_brain_initialization_manifest(self.config.initialization)

    def parameter_snapshot(self) -> tuple[Tensor, ...]:
        """Return detached CPU copies of all prototype parameters."""

        return tuple(
            parameter.detach().cpu().clone()
            for parameter in self._prototype.parameters()
        )

    def inherited_parameter_state(self) -> dict[str, Tensor]:
        """Return named detached CPU copies of the complete inherited genome."""

        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self._prototype.named_parameters()
        }

    def load_inherited_parameter_state(
        self,
        state: Mapping[str, Tensor],
    ) -> None:
        """Replace the prototype genome while preserving its fixed architecture."""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a parameter mapping")
        parameters = dict(self._prototype.named_parameters())
        if set(state) != set(parameters):
            missing = sorted(set(parameters) - set(state))
            unexpected = sorted(set(state) - set(parameters))
            raise ValueError(
                "inherited parameter names do not match the BaseBrain: "
                f"missing={missing}, unexpected={unexpected}"
            )
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name]
                if not isinstance(value, Tensor):
                    raise TypeError(f"inherited parameter {name!r} must be a Tensor")
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError(
                        f"inherited parameter {name!r} shape/dtype does not match"
                    )
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"inherited parameter {name!r} must be finite")
                parameter.copy_(value.to(device=parameter.device))

    def spawn(self, *, action_random_source: object | None = None) -> NoraletBrain:
        """Clone an independent model and zero hidden state on the target device."""

        from noralet.brain.runtime import NoraletBrain

        model = copy.deepcopy(self._prototype).to(self.device)
        model.eval()
        target_encoder = None
        if self.learning_config is not None:
            target_encoder = copy.deepcopy(model.encoder)
            target_encoder.requires_grad_(False)
            target_encoder.eval()
        return NoraletBrain(
            model=model,
            config=self.config,
            learning_config=self.learning_config,
            homeostatic_plasticity_config=self.homeostatic_plasticity_config,
            actuator_config=self.actuator_config,
            device=self.device,
            action_random_source=action_random_source,
            target_experience_encoder=target_encoder,
        )

    def matches_simulation_configs(
        self,
        experience_config: NoraletExperienceConfig | None,
        signal_config: NoraletSignalConfig | None,
        actuator_config: NoraletActuatorConfig | None,
    ) -> bool:
        """Return whether a simulation presents the prototype's exact interface."""

        return (
            experience_config is not None
            and signal_config is not None
            and (
                experience_config.appearance_length
                == self.external_pattern_length
            )
            and signal_config.signal_pattern_length == self.signal_pattern_length
            and actuator_config == self.actuator_config
        )

    @staticmethod
    def _seeded_generator(seed: int, domain: bytes) -> torch.Generator:
        digest = hashlib.sha256()
        digest.update(domain)
        digest.update(str(seed).encode("ascii"))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)
        )
        return generator

    @classmethod
    def _initialize_parameters(
        cls,
        parameters: tuple[torch.nn.Parameter, ...],
        seed: int,
        domain: bytes,
    ) -> None:
        generator = cls._seeded_generator(seed, domain)
        with torch.no_grad():
            for parameter in parameters:
                fan = parameter.shape[-1] if parameter.ndim > 1 else parameter.numel()
                bound = 1.0 / math.sqrt(max(1, fan))
                parameter.uniform_(-bound, bound, generator=generator)

    @classmethod
    def _initialize_recurrent_core(
        cls,
        prototype: NoraletBrainModel,
        seed: int,
    ) -> None:
        """Use neutral biases and standard stable GRU matrix initialization."""

        generator = cls._seeded_generator(seed, _RECURRENT_SEED_DOMAIN)
        recurrent = prototype.recurrent_core
        with torch.no_grad():
            for gate in recurrent.weight_ih.chunk(3, dim=0):
                torch.nn.init.xavier_uniform_(gate, generator=generator)
            for gate in recurrent.weight_hh.chunk(3, dim=0):
                torch.nn.init.orthogonal_(gate, generator=generator)
            recurrent.bias_ih.zero_()
            recurrent.bias_hh.zero_()

    @classmethod
    def _initialize_action_baselines(
        cls,
        prototype: NoraletBrainModel,
        config: NoraletBrainConfig,
    ) -> None:
        """Set content-free motor priors without changing runtime sampling."""

        initialization = config.initialization
        generator = cls._seeded_generator(
            config.base_brain_seed,
            _ACCELERATION_SEED_DOMAIN,
        )
        consume_logit = math.log(
            initialization.initial_consume_probability
            / (1.0 - initialization.initial_consume_probability)
        )
        emission_count = SIGNAL_MOTOR_OUTCOME_COUNT - 1
        none_logit = math.log(1.0 - initialization.initial_signal_probability)
        emission_logit = math.log(
            initialization.initial_signal_probability / emission_count
        )
        with torch.no_grad():
            prototype.acceleration_head.weight.uniform_(
                -initialization.acceleration_output_weight_scale,
                initialization.acceleration_output_weight_scale,
                generator=generator,
            )
            prototype.acceleration_head.bias.zero_()
            prototype.consume_head.bias.fill_(consume_logit)
            prototype.signal_head.bias[0] = none_logit
            prototype.signal_head.bias[1:].fill_(emission_logit)
