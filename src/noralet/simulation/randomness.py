"""Stable, isolated random-number streams derived from one master seed."""

from __future__ import annotations

import hashlib
import random


class DeterministicRandomStreams:
    """Own independent named ``random.Random`` instances for one simulation.

    Each stream seed is derived from the master seed and exact stream name with
    SHA-256. Creating or consuming one stream therefore cannot advance another.
    """

    __slots__ = ("_master_seed", "_streams")

    _DOMAIN = b"project-noralet:random-stream:v1\0"

    def __init__(self, master_seed: int) -> None:
        if type(master_seed) is not int:
            raise TypeError("master_seed must be an integer")

        self._master_seed = master_seed
        self._streams: dict[str, random.Random] = {}

    @property
    def master_seed(self) -> int:
        """Return the immutable master seed used for stream derivation."""

        return self._master_seed

    def seed_for(self, name: str) -> int:
        """Derive a process-stable integer seed for an exact stream name."""

        self._validate_name(name)
        digest = hashlib.sha256()
        digest.update(self._DOMAIN)
        digest.update(str(self._master_seed).encode("ascii"))
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        return int.from_bytes(digest.digest(), byteorder="big", signed=False)

    def stream(self, name: str) -> random.Random:
        """Return the persistent independent generator for ``name``."""

        self._validate_name(name)
        if name not in self._streams:
            self._streams[name] = random.Random(self.seed_for(name))
        return self._streams[name]

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("stream name must be a string")
        if not name:
            raise ValueError("stream name cannot be empty")

