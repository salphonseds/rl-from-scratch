"""Four-tier check taxonomy.

The tier of a check determines whether its failure may halt execution:

  Deterministic  identity / invariant / hand-computed / enumerated oracle
                 -> raises GateFailure. Correct code cannot fail these.

  Artifact       checkpoint, fingerprint, parameter count, config
                 -> raises GateFailure. Proves you are standing on the
                    artifact you think you are.

  Calibration    statistical property expected over replications
                 -> warns, never halts. Correct code CAN fail these by
                    chance, or for reasons of capacity or sample size.

  Hypothesis     scientific prediction. Deliberately absent from this
                 module. Hypotheses are pre-registered as data in
                 predictions.yaml and scored -- never asserted.

Usage
-----
    @Deterministic("incremental mean matches hand computation")
    def gate_incremental_mean():
        assert update(0.0, 2.0, 1) == 2.0
        assert update(2.0, 4.0, 2) == 3.0

    gate_incremental_mean()          # prints PASS, or raises

    @Calibration("sample mean converges to true arm mean")
    def check_convergence():
        lo, hi = bootstrap_ci(errors)
        return {"passed": lo <= 0 <= hi, "estimate": errors.mean(),
                "ci": (lo, hi)}

    check_convergence()              # prints PASS or a loud warning
"""

from __future__ import annotations

import functools
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

__all__ = [
    "Tier",
    "GateFailure",
    "CheckResult",
    "Deterministic",
    "Artifact",
    "Calibration",
    "check_log",
    "check_summary",
    "reset_checks",
]


class Tier(str, Enum):
    DETERMINISTIC = "deterministic"
    ARTIFACT = "artifact"
    CALIBRATION = "calibration"

    @property
    def halting(self) -> bool:
        return self is not Tier.CALIBRATION


class GateFailure(AssertionError):
    """A halting check failed. Execution must not continue."""


@dataclass
class CheckResult:
    name: str
    tier: Tier
    passed: bool
    detail: dict = field(default_factory=dict)
    error: str | None = None

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.tier.halting else "WARN")
        line = f"[{self.tier.value:>13}] {mark} | {self.name}"
        if self.detail:
            bits = ", ".join(f"{k}={_fmt(v)}" for k, v in self.detail.items()
                             if k != "passed")
            if bits:
                line += f" | {bits}"
        return line


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, tuple) and all(isinstance(x, float) for x in v):
        return "(" + ", ".join(f"{x:.4g}" for x in v) + ")"
    return str(v)


_LOG: list[CheckResult] = []


def check_log() -> list[CheckResult]:
    """All results recorded so far, in execution order."""
    return list(_LOG)


def reset_checks() -> None:
    _LOG.clear()


def check_summary(verbose: bool = True) -> dict:
    """Counts by tier and outcome. Call this in a notebook's closing cell."""
    counts: dict[str, dict[str, int]] = {}
    for r in _LOG:
        slot = counts.setdefault(r.tier.value, {"pass": 0, "fail": 0})
        slot["pass" if r.passed else "fail"] += 1
    if verbose:
        print("=" * 60)
        print("CHECK SUMMARY")
        print("=" * 60)
        for tier, c in counts.items():
            print(f"  {tier:>13}: {c['pass']} passed, {c['fail']} failed")
        halting_fails = [r for r in _LOG if not r.passed and r.tier.halting]
        if halting_fails:
            print(f"\n  {len(halting_fails)} HALTING FAILURE(S) -- results are void")
        warns = [r for r in _LOG if not r.passed and not r.tier.halting]
        if warns:
            print(f"\n  {len(warns)} calibration warning(s) -- investigate, "
                  f"do not ignore, do not treat as a bug by default")
    return counts


def _make_decorator(tier: Tier) -> Callable:
    def decorator(name: str) -> Callable:
        def wrap(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def runner(*args, **kwargs) -> CheckResult:
                detail: dict = {}
                try:
                    out = fn(*args, **kwargs)
                    if isinstance(out, dict):
                        detail = out
                        passed = bool(out.get("passed", True))
                    elif isinstance(out, bool):
                        passed = out
                    else:
                        passed = True          # no exception raised
                    result = CheckResult(name, tier, passed, detail)
                except AssertionError as exc:
                    result = CheckResult(name, tier, False, detail,
                                         error=str(exc) or "assertion failed")
                except Exception:
                    result = CheckResult(name, tier, False, detail,
                                         error=traceback.format_exc(limit=3))

                _LOG.append(result)
                print(result)
                if result.error:
                    print(f"    -> {result.error.strip().splitlines()[-1]}")

                if not result.passed and tier.halting:
                    raise GateFailure(
                        f"{tier.value} check failed: {name}"
                        + (f" -- {result.error}" if result.error else "")
                    )
                if not result.passed:
                    print("    !! CALIBRATION WARNING -- not halting. This may be "
                          "chance, capacity, or sample size, not necessarily a bug.")
                return result
            return runner
        return wrap
    return decorator


Deterministic = _make_decorator(Tier.DETERMINISTIC)
Artifact = _make_decorator(Tier.ARTIFACT)
Calibration = _make_decorator(Tier.CALIBRATION)
