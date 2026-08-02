"""Pre-registered predictions: data, never code.

A hypothesis is not a gate. There is deliberately no @Hypothesis
decorator anywhere in rlkit -- if a prediction could halt execution,
the notebook could only ever confirm what was expected.

Workflow
--------
  1. Early cell, BEFORE the first run:
         preds = Predictions.create("nb01", path, [...])
     -> writes YAML, refuses to overwrite. Commit it to git now; the
        commit timestamp is what makes pre-registration verifiable.

  2. Closing cell, after results exist:
         preds = Predictions.load(path)
         preds.score({"leader_1000": "ucb", ...})
     -> grades, reports accuracy by stated confidence, never raises.

The misses are the point. A curriculum where every prediction lands
was not teaching anything.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import yaml

__all__ = ["Prediction", "Predictions"]

_CONFIDENCE = ("low", "medium", "high")


@dataclass
class Prediction:
    key: str                      # short stable id, e.g. "leader_1000"
    claim: str                    # prose statement, unambiguous
    expected: Any                 # the value that will be compared
    confidence: str = "medium"    # low | medium | high
    reasoning: str = ""           # why -- the part worth re-reading on a miss

    def __post_init__(self) -> None:
        if self.confidence not in _CONFIDENCE:
            raise ValueError(f"confidence must be one of {_CONFIDENCE}, "
                             f"got {self.confidence!r}")


def _hash(items: list[dict]) -> str:
    payload = json.dumps(items, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


class Predictions:
    """A write-once collection of pre-registered predictions."""

    def __init__(self, notebook: str, path: str, items: list[Prediction],
                 created_utc: str, content_hash: str) -> None:
        self.notebook = notebook
        self.path = path
        self.items = items
        self.created_utc = created_utc
        self.content_hash = content_hash

    # -- construction -----------------------------------------------------

    @classmethod
    def create(cls, notebook: str, path: str, items: list[Prediction],
               *, overwrite: bool = False) -> "Predictions":
        """Write predictions to disk. Refuses to clobber by default."""
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"{path} already exists. Predictions are write-once: this file "
                f"was committed before the run. To re-register deliberately, "
                f"delete it by hand (git history will show it)."
            )
        raw = [asdict(p) for p in items]
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        doc = {
            "notebook": notebook,
            "created_utc": created,
            "content_hash": _hash(raw),
            "predictions": raw,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False,
                           allow_unicode=True)
        obj = cls(notebook, path, items, created, doc["content_hash"])
        obj.show()
        print(f"\n  written to {path}")
        print("  COMMIT THIS FILE NOW, before running anything else:")
        print(f"    git add {path} && git commit -m 'pre-register {notebook}' && git push")
        return obj

    @classmethod
    def load(cls, path: str) -> "Predictions":
        with open(path) as f:
            doc = yaml.safe_load(f)
        raw = doc["predictions"]
        recomputed = _hash(raw)
        if recomputed != doc.get("content_hash"):
            print(f"  !! content hash mismatch in {path}: predictions were edited "
                  f"after registration ({doc.get('content_hash')} -> {recomputed})")
        return cls(doc["notebook"], path,
                   [Prediction(**p) for p in raw],
                   doc["created_utc"], doc["content_hash"])

    # -- display ----------------------------------------------------------

    def show(self) -> None:
        print("=" * 70)
        print(f"PRE-REGISTERED PREDICTIONS -- {self.notebook}")
        print(f"registered {self.created_utc}   hash {self.content_hash[:12]}")
        print("=" * 70)
        for p in self.items:
            print(f"\n  [{p.key}]  ({p.confidence} confidence)")
            print(f"    claim    : {p.claim}")
            print(f"    expected : {p.expected}")
            if p.reasoning:
                print(f"    because  : {p.reasoning}")

    # -- scoring ----------------------------------------------------------

    def score(self, actual: dict[str, Any], *,
              comparators: dict[str, Any] | None = None,
              verbose: bool = True) -> dict:
        """Grade predictions against observed results. Never raises.

        actual : {key: observed value}
        comparators : optional {key: fn(expected, observed) -> bool}
                      for tolerances or set membership.
        """
        comparators = comparators or {}
        rows, hits, scored = [], 0, 0

        for p in self.items:
            if p.key not in actual:
                rows.append((p, None, "unscored"))
                continue
            obs = actual[p.key]
            cmp_fn = comparators.get(p.key, lambda e, o: e == o)
            try:
                ok = bool(cmp_fn(p.expected, obs))
            except Exception as exc:
                rows.append((p, obs, f"error: {exc}"))
                continue
            scored += 1
            hits += ok
            rows.append((p, obs, "HIT" if ok else "MISS"))

        if verbose:
            print("=" * 70)
            print(f"PREDICTION LEDGER -- {self.notebook}")
            print(f"registered {self.created_utc}")
            print("=" * 70)
            for p, obs, verdict in rows:
                print(f"\n  [{p.key}]  {verdict}   ({p.confidence} confidence)")
                print(f"    claim    : {p.claim}")
                print(f"    expected : {p.expected}")
                print(f"    observed : {obs}")
                if verdict == "MISS":
                    print(f"    reasoning was: {p.reasoning or '(none recorded)'}")
                    print(f"    -> WHY WAS THIS WRONG? Write it down.")

            print("\n" + "-" * 70)
            print(f"  {hits}/{scored} correct" if scored else "  nothing scored")
            for level in _CONFIDENCE:
                sub = [(p, v) for p, _, v in rows
                       if p.confidence == level and v in ("HIT", "MISS")]
                if sub:
                    n_hit = sum(1 for _, v in sub if v == "HIT")
                    print(f"    {level:>6} confidence: {n_hit}/{len(sub)}")
            if scored and hits == scored:
                print("\n  All predictions correct. Either the notebook was too easy,")
                print("  or the predictions were too safe. Predict harder next time.")

        return {
            "notebook": self.notebook,
            "n_scored": scored,
            "n_hits": hits,
            "rows": [{"key": p.key, "verdict": v, "expected": p.expected,
                      "observed": o, "confidence": p.confidence}
                     for p, o, v in rows],
        }
