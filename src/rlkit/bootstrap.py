"""Uncertainty over independent experimental units.

The unit of resampling is a ROW. Callers must aggregate within a unit
before calling anything here:

    bandits   unit = run seed
    DQN/PPO   unit = trained-agent seed (average its eval episodes first)
    GRPO      unit = prompt group, never an individual completion
    prefs     unit = prompt/conversation, never an individual pair

Resampling completions inside a shared-baseline group, or eval episodes
inside one trained agent, understates variance -- those observations are
not independent.

All resampling is seeded, so a reported CI is reproducible.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["iqm", "bootstrap_ci", "bootstrap_curve", "summarize"]


def iqm(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Interquartile mean: mean of the middle 50% of values.

    More robust than the mean, more sample-efficient than the median.
    (Agarwal et al., 'Deep RL at the Edge of the Statistical Precipice'.)
    """
    x = np.asarray(x, dtype=float)
    if axis is None:
        x = x.ravel()
        axis = 0
    lo = np.quantile(x, 0.25, axis=axis, keepdims=True)
    hi = np.quantile(x, 0.75, axis=axis, keepdims=True)
    mask = (x >= lo) & (x <= hi)
    return np.sum(np.where(mask, x, 0.0), axis=axis) / np.maximum(mask.sum(axis=axis), 1)


def _resample_idx(n_units: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_units, size=(n_boot, n_units))


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.median,
    *,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> dict:
    """Percentile bootstrap CI for a scalar statistic over independent units.

    Parameters
    ----------
    values : 1-D array, one entry per independent unit.
    statistic : reducer applied to a resampled 1-D array.
    rng : seeded Generator, e.g. SeedBank(42).generator("bootstrap", "regret").

    Returns dict with estimate, lo, hi, se, n_units, n_boot, ci.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    if n < 2:
        raise ValueError(f"need >= 2 independent units, got {n}")
    if n < 10:
        print(f"    (warning: bootstrap over only {n} units -- CI is unreliable)")

    idx = _resample_idx(n, n_boot, rng)
    boot = np.array([statistic(values[i]) for i in idx])

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "estimate": float(statistic(values)),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(boot.std(ddof=1)),
        "n_units": int(n),
        "n_boot": int(n_boot),
        "ci": ci,
    }


def bootstrap_curve(
    curves: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray] = np.median,
    *,
    rng: np.random.Generator,
    n_boot: int = 2_000,
    ci: float = 0.95,
) -> dict:
    """Pointwise CI band for learning curves.

    curves : (n_units, n_steps). WHOLE ROWS are resampled, preserving
    within-run temporal correlation. Resampling timesteps independently
    would be wrong.

    Note: the band is pointwise, not simultaneous -- it is not a
    confidence region for the entire curve at once.
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError(f"expected (n_units, n_steps), got shape {curves.shape}")
    n, T = curves.shape
    if n < 2:
        raise ValueError(f"need >= 2 independent units, got {n}")

    idx = _resample_idx(n, n_boot, rng)
    boot = np.empty((n_boot, T), dtype=float)
    for b in range(n_boot):
        boot[b] = statistic(curves[idx[b]], axis=0)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha], axis=0)
    return {
        "estimate": np.asarray(statistic(curves, axis=0), dtype=float),
        "lo": lo,
        "hi": hi,
        "n_units": int(n),
        "n_steps": int(T),
        "n_boot": int(n_boot),
        "ci": ci,
    }


def summarize(name: str, values: np.ndarray, *, rng: np.random.Generator,
              n_boot: int = 10_000) -> dict:
    """Median + IQM + mean, each with a bootstrap CI. Prints one block."""
    values = np.asarray(values, dtype=float).ravel()
    out = {"name": name, "n_units": int(values.size)}
    print(f"{name}  (n_units={values.size})")
    for label, fn in [("median", np.median), ("iqm", iqm), ("mean", np.mean)]:
        r = bootstrap_ci(values, fn, rng=rng, n_boot=n_boot)
        out[label] = r
        print(f"    {label:>6}: {r['estimate']:9.4f}   "
              f"95% CI [{r['lo']:8.4f}, {r['hi']:8.4f}]")
    return out
