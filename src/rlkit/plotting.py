"""Plots that show uncertainty and do not hide bimodality.

Rules encoded here:

  1. Individual runs are drawn faintly behind every aggregate. A median
     line alone conceals bimodal outcomes -- e.g. DQN seeds that never
     solve at all.
  2. Shaded bands are BOOTSTRAP CONFIDENCE INTERVALS on the aggregate,
     not IQR. IQR describes spread across units; a CI describes
     uncertainty in the estimate. Pass show_iqr=True to overlay both.
  3. Curves must share an x-grid before aggregation. Use interp_to_grid
     when runs have unequal lengths (episode index is not comparable
     across runs of differing episode length).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .bootstrap import bootstrap_curve, bootstrap_ci

__all__ = ["interp_to_grid", "plot_curves", "plot_parameter_study", "plot_scalar_comparison"]

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def interp_to_grid(xs: list[np.ndarray], ys: list[np.ndarray],
                   grid: np.ndarray) -> np.ndarray:
    """Interpolate ragged per-run curves onto one shared x-grid.

    Returns (n_runs, len(grid)). Points beyond a run's final x are NaN,
    not extrapolated -- a run that ended early has no value there.
    """
    out = np.full((len(xs), grid.size), np.nan)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x, y = np.asarray(x, float), np.asarray(y, float)
        inside = grid <= x[-1]
        out[i, inside] = np.interp(grid[inside], x, y)
    return out


def plot_curves(
    curves: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    x: np.ndarray | None = None,
    statistic=np.median,
    n_boot: int = 2_000,
    show_individual: int | bool = 12,
    show_iqr: bool = False,
    xlabel: str = "step",
    ylabel: str = "value",
    title: str = "",
    ax: plt.Axes | None = None,
    figsize: tuple = (9, 5),
):
    """Aggregate learning curves with CI bands and faint individual runs.

    curves : {label: (n_units, n_steps)}. Rows are independent units.
    show_individual : how many raw runs to draw per label (True = all).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, (label, arr) in enumerate(curves.items()):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{label}: expected (n_units, n_steps), got {arr.shape}")
        xs = np.arange(arr.shape[1]) if x is None else np.asarray(x)
        colour = _PALETTE[i % len(_PALETTE)]

        n_show = arr.shape[0] if show_individual is True else int(show_individual)
        for row in arr[:max(n_show, 0)]:
            ax.plot(xs, row, color=colour, alpha=0.06, lw=0.7, zorder=1)

        band = bootstrap_curve(arr, statistic, rng=rng, n_boot=n_boot)
        ax.fill_between(xs, band["lo"], band["hi"], color=colour, alpha=0.25,
                        lw=0, zorder=2)
        if show_iqr:
            q1, q3 = np.quantile(arr, [0.25, 0.75], axis=0)
            ax.plot(xs, q1, color=colour, lw=0.8, ls=":", alpha=0.7, zorder=2)
            ax.plot(xs, q3, color=colour, lw=0.8, ls=":", alpha=0.7, zorder=2)
        ax.plot(xs, band["estimate"], color=colour, lw=2.0, zorder=3,
                label=f"{label}  (n={band['n_units']})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax


def plot_parameter_study(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    rng: np.random.Generator,
    statistic=np.median,
    n_boot: int = 2_000,
    log2_x: bool = True,
    xlabel: str = "parameter",
    ylabel: str = "average reward over final steps",
    title: str = "",
    figsize: tuple = (9, 5),
):
    """Sutton & Barto Fig 2.6 style: performance vs each method's parameter.

    results : {label: (params, values)} where values is
              (n_params, n_units) -- one column per independent unit.
    """
    _, ax = plt.subplots(figsize=figsize)

    for i, (label, (params, values)) in enumerate(results.items()):
        params = np.asarray(params, dtype=float)
        values = np.asarray(values, dtype=float)
        colour = _PALETTE[i % len(_PALETTE)]

        est, lo, hi = [], [], []
        for j in range(values.shape[0]):
            r = bootstrap_ci(values[j], statistic, rng=rng, n_boot=n_boot)
            est.append(r["estimate"]); lo.append(r["lo"]); hi.append(r["hi"])
        est, lo, hi = map(np.asarray, (est, lo, hi))

        ax.fill_between(params, lo, hi, color=colour, alpha=0.2, lw=0)
        ax.plot(params, est, "o-", color=colour, lw=1.8, ms=4, label=label)

    if log2_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax


def plot_scalar_comparison(
    values: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    statistic=np.median,
    n_boot: int = 10_000,
    xlabel: str = "value",
    title: str = "",
    figsize: tuple = (8, 3.2),
):
    """Horizontal point estimates with CIs and jittered per-unit points."""
    _, ax = plt.subplots(figsize=figsize)
    labels = list(values)

    for i, label in enumerate(labels):
        v = np.asarray(values[label], dtype=float).ravel()
        colour = _PALETTE[i % len(_PALETTE)]
        y = len(labels) - 1 - i

        jitter = rng.uniform(-0.13, 0.13, size=v.size)
        ax.scatter(v, np.full(v.size, y) + jitter, s=7, color=colour,
                   alpha=0.25, lw=0, zorder=1)

        r = bootstrap_ci(v, statistic, rng=rng, n_boot=n_boot)
        ax.plot([r["lo"], r["hi"]], [y, y], color=colour, lw=2.5, zorder=2)
        ax.plot([r["estimate"]], [y], "o", color=colour, ms=8, zorder=3)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    return ax
