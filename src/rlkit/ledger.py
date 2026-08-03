"""Run provenance: what produced this number, on what, and when.

One JSON per notebook run. Six months from now, when a result fails to
reproduce, this is what distinguishes 'the code changed' from 'numpy
changed underneath me'.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

__all__ = ["RunLedger", "environment_snapshot", "git_state"]

_PACKAGES = ["numpy", "scipy", "matplotlib", "yaml", "torch", "gymnasium"]


def environment_snapshot() -> dict:
    """Python, platform, and versions of packages that affect results."""
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = getattr(__import__(name), "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def git_state(repo_dir: str = ".") -> dict:
    """Commit hash and dirty flag, so a result maps to a code state."""
    def run(*args):
        try:
            r = subprocess.run(["git", *args], cwd=repo_dir,
                               capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


class RunLedger:
    """Accumulates metadata, timings and results for one notebook run."""

    def __init__(self, notebook: str, seed_bank=None, repo_dir: str = ".") -> None:
        self.notebook = notebook
        self.started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._t0 = time.time()
        self.doc: dict[str, Any] = {
            "notebook": notebook,
            "started_utc": self.started_utc,
            "environment": environment_snapshot(),
            "git": git_state(repo_dir),
            "seed_bank": seed_bank.describe() if seed_bank is not None else None,
            "results": {},
            "timings": {},
            "notes": [],
        }

    def record(self, key: str, value: Any) -> None:
        self.doc["results"][key] = value

    def time(self, key: str, seconds: float) -> None:
        self.doc["timings"][key] = round(float(seconds), 3)

    def note(self, text: str) -> None:
        self.doc["notes"].append(text)

    def attach_checks(self, results) -> None:
        """Store the rlkit.gates check log."""
        self.doc["checks"] = [
            {"name": r.name, "tier": r.tier.value, "passed": r.passed,
             "detail": {k: _jsonable(v) for k, v in r.detail.items()}}
            for r in results
        ]

    def attach_predictions(self, scored: dict) -> None:
        self.doc["predictions"] = scored

    def save(self, path: str) -> str:
        self.doc["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.doc["wall_clock_s"] = round(time.time() - self._t0, 1)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.doc, f, indent=2, default=_jsonable)
        print(f"ledger written: {path}  ({self.doc['wall_clock_s']}s wall clock)")
        return path

    def summary(self) -> None:
        env, git = self.doc["environment"], self.doc["git"]
        print("=" * 60)
        print(f"RUN LEDGER -- {self.notebook}")
        print("=" * 60)
        print(f"  started : {self.started_utc}")
        print(f"  python  : {env['python']}")
        print(f"  numpy   : {env['packages'].get('numpy')}")
        if git["commit"]:
            dirty = " (uncommitted changes)" if git["dirty"] else ""
            print(f"  git     : {git['commit'][:8]} on {git['branch']}{dirty}")
        if self.doc["results"]:
            print("  results :")
            for k, v in self.doc["results"].items():
                print(f"      {k} = {_jsonable(v)}")


def _jsonable(v: Any) -> Any:
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, tuple):
        return list(v)
    return v
