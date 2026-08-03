"""Canonical hashing: proving an artifact is the one you think it is.

Backs @Artifact checks. Used in NB11 to prove that PPO updated only the
adapters -- setting requires_grad=False expresses intent, but a hash
taken before and after training is what proves the base never moved.

Hashes are canonical: fixed byte order, sorted keys, explicit dtype and
shape. The same tensor content yields the same digest across sessions,
machines and numpy versions.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np

__all__ = ["hash_array", "hash_arrays", "fingerprint", "compare_fingerprints"]


def hash_array(x, *, digest_size: int = 16) -> str:
    """Stable digest of one array: dtype + shape + big-endian bytes."""
    a = np.ascontiguousarray(np.asarray(x))
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(a.astype(a.dtype.newbyteorder(">"), copy=False).tobytes())
    return h.hexdigest()


def hash_arrays(named: dict[str, Any], *, digest_size: int = 16) -> str:
    """Digest of a named collection, order-independent (keys sorted)."""
    h = hashlib.blake2b(digest_size=digest_size)
    for key in sorted(named):
        h.update(key.encode())
        h.update(hash_array(named[key]).encode())
    return h.hexdigest()


def fingerprint(named: dict[str, Any], *, digest_size: int = 16) -> dict:
    """Per-item digests plus a combined digest, for precise diffing."""
    per = {k: hash_array(v, digest_size=digest_size) for k, v in named.items()}
    return {"combined": hash_arrays(named, digest_size=digest_size),
            "items": per,
            "n_items": len(per)}


def compare_fingerprints(before: dict, after: dict, *, verbose: bool = True) -> dict:
    """Which named tensors changed between two fingerprints."""
    b, a = before["items"], after["items"]
    changed = sorted(k for k in b.keys() & a.keys() if b[k] != a[k])
    added = sorted(a.keys() - b.keys())
    removed = sorted(b.keys() - a.keys())
    identical = before["combined"] == after["combined"]

    if verbose:
        if identical:
            print(f"  fingerprint unchanged ({before['n_items']} tensors, "
                  f"{before['combined'][:12]})")
        else:
            print(f"  fingerprint CHANGED: {len(changed)} modified, "
                  f"{len(added)} added, {len(removed)} removed")
            for k in changed[:10]:
                print(f"      modified: {k}")
            if len(changed) > 10:
                print(f"      ... and {len(changed)-10} more")
    return {"identical": identical, "changed": changed,
            "added": added, "removed": removed}


def torch_fingerprint(module, *, trainable_only: bool = False) -> dict:
    """Fingerprint a torch module's parameters (NB09 onward)."""
    named = {}
    for name, p in module.named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        named[name] = p.detach().cpu().numpy()
    return fingerprint(named)
