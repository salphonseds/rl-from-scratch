"""Named, reproducible, independent random streams.

Every stochastic result in this curriculum traces back to a SeedBank.
Two properties are non-negotiable:

  1. Independence. Streams with different names are statistically
     independent (numpy SeedSequence spawn-key guarantee). The testbed
     that generates a problem instance must not share entropy with the
     agent that acts on it.

  2. Stability. The same (master_seed, name) pair yields the same
     numbers across processes, machines and years. Python's builtin
     hash() is salted per process, so names are hashed with blake2b.
"""

from __future__ import annotations

import hashlib
from typing import Union

import numpy as np

__all__ = ["SeedBank", "name_to_int"]

_NamePart = Union[str, int]


def name_to_int(part: _NamePart) -> int:
    """Map a stream-name component to a stable non-negative 64-bit int.

    Integers pass through unchanged (must be >= 0). Strings are hashed
    with blake2b, which is stable across processes -- unlike hash().
    """
    if isinstance(part, (int, np.integer)):
        if part < 0:
            raise ValueError(f"integer name parts must be >= 0, got {part}")
        return int(part)
    if isinstance(part, str):
        digest = hashlib.blake2b(part.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")
    raise TypeError(f"name parts must be str or int, got {type(part).__name__}")


class SeedBank:
    """A master seed plus a deterministic map from names to Generators.

    Example
    -------
    >>> bank = SeedBank(42)
    >>> testbed_rng = bank.generator("testbed", 7)
    >>> agent_rng   = bank.generator("agent", "epsilon_greedy", 7)

    The two Generators above are independent: exhausting one does not
    affect the other, and neither depends on call order.
    """

    def __init__(self, master_seed: int, label: str | None = None) -> None:
        if not isinstance(master_seed, (int, np.integer)) or master_seed < 0:
            raise ValueError(f"master_seed must be a non-negative int, got {master_seed!r}")
        self.master_seed = int(master_seed)
        self.label = label

    # -- core API ---------------------------------------------------------

    def seed_sequence(self, *parts: _NamePart) -> np.random.SeedSequence:
        """The SeedSequence for a named stream."""
        if not parts:
            raise ValueError("a stream needs at least one name part")
        spawn_key = tuple(name_to_int(p) for p in parts)
        return np.random.SeedSequence(entropy=self.master_seed, spawn_key=spawn_key)

    def generator(self, *parts: _NamePart) -> np.random.Generator:
        """A fresh, independent Generator for a named stream."""
        return np.random.default_rng(self.seed_sequence(*parts))

    def seed_int(self, *parts: _NamePart) -> int:
        """A 64-bit int for libraries that only accept a plain seed."""
        return int(self.seed_sequence(*parts).generate_state(1, dtype=np.uint64)[0])

    # -- provenance -------------------------------------------------------

    def describe(self) -> dict:
        """Metadata for the run ledger."""
        return {
            "master_seed": self.master_seed,
            "label": self.label,
            "numpy_version": np.__version__,
        }

    def __repr__(self) -> str:
        tag = f", label={self.label!r}" if self.label else ""
        return f"SeedBank(master_seed={self.master_seed}{tag})"
