"""Canonical inherited BaseBrain genomes and deterministic Gaussian mutation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from noralet.brain import BaseBrain


@dataclass(frozen=True, slots=True)
class BaseBrainGenome:
    """Detached CPU copy of every named inherited prototype parameter."""

    _items: tuple[tuple[str, Tensor], ...]

    def __post_init__(self) -> None:
        if not isinstance(self._items, tuple) or not self._items:
            raise TypeError("genome items must be a non-empty tuple")
        normalized: list[tuple[str, Tensor]] = []
        names: list[str] = []
        for name, value in self._items:
            if not isinstance(name, str) or not name:
                raise TypeError("genome parameter names must be non-empty strings")
            if not isinstance(value, Tensor):
                raise TypeError(f"genome parameter {name!r} must be a Tensor")
            if not value.is_floating_point():
                raise TypeError(f"genome parameter {name!r} must be floating-point")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"genome parameter {name!r} must be finite")
            names.append(name)
            normalized.append((name, value.detach().cpu().clone()))
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("genome parameter names must be unique and sorted")
        object.__setattr__(self, "_items", tuple(normalized))

    @classmethod
    def from_base_brain(cls, base_brain: BaseBrain) -> BaseBrainGenome:
        if not isinstance(base_brain, BaseBrain):
            raise TypeError("base_brain must be a BaseBrain")
        state = base_brain.inherited_parameter_state()
        return cls(tuple(sorted(state.items())))

    @classmethod
    def from_state(cls, state: Mapping[str, Tensor]) -> BaseBrainGenome:
        if not isinstance(state, Mapping):
            raise TypeError("state must be a parameter mapping")
        return cls(tuple(sorted(state.items())))

    def state(self) -> dict[str, Tensor]:
        return {name: value.clone() for name, value in self._items}

    def apply_to(self, base_brain: BaseBrain) -> None:
        if not isinstance(base_brain, BaseBrain):
            raise TypeError("base_brain must be a BaseBrain")
        base_brain.load_inherited_parameter_state(self.state())

    def tensors(self) -> tuple[Tensor, ...]:
        return tuple(value.clone() for _, value in self._items)

    def exactly_equals(self, other: BaseBrainGenome) -> bool:
        return (
            isinstance(other, BaseBrainGenome)
            and len(self._items) == len(other._items)
            and all(
                left_name == right_name and torch.equal(left, right)
                for (left_name, left), (right_name, right) in zip(
                    self._items,
                    other._items,
                    strict=True,
                )
            )
        )


def mutate_genome(
    genome: BaseBrainGenome,
    *,
    sigma: float,
    seed: int,
) -> BaseBrainGenome:
    """Apply reproducible independent additive CPU Gaussian noise."""

    if not isinstance(genome, BaseBrainGenome):
        raise TypeError("genome must be a BaseBrainGenome")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise TypeError("sigma must be a real number")
    sigma = float(sigma)
    if not torch.isfinite(torch.tensor(sigma)).item() or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mutated: list[tuple[str, Tensor]] = []
    with torch.no_grad():
        for name, value in genome._items:
            noise = torch.randn(
                value.shape,
                dtype=value.dtype,
                device=value.device,
                generator=generator,
            )
            mutated.append((name, value + sigma * noise))
    return BaseBrainGenome(tuple(mutated))
