"""Adaptive result cutoff: return as many results as the score curve justifies.

Spec forms (scores sorted best-first):
  "elbow"          keep the results before the knee (bend) of the score curve
  ("ratio", x)     keep scores >= x * best           (e.g. 0.5)
  ("cliff", x)     stop at the first drop > x of the previous score (relative)
  ("min", x)       absolute threshold (use with reranker or vector scores)
Always keeps at least one result when any exist.
"""
from __future__ import annotations

import numpy as np


def keep_count(scores: list[float], spec) -> int:
    n = len(scores)
    if n <= 1 or spec is None:
        return n
    s = np.asarray(scores, dtype=np.float64)
    if spec == "elbow":
        if n < 3 or s[0] == s[-1]:
            return n
        x = np.linspace(0.0, 1.0, n)
        y = (s - s[-1]) / (s[0] - s[-1])  # 1 at the top, 0 at the bottom
        knee = int(np.argmax((1.0 - x) - y))  # furthest below the straight line top->bottom
        return max(knee, 1)  # results before the bend; the knee point itself is already 'low'
    kind, x = spec
    if kind == "ratio":
        return max(int(np.sum(s >= x * s[0])), 1)
    if kind == "min":
        return max(int(np.sum(s >= x)), 1)
    if kind == "cliff":
        for i in range(1, n):
            if s[i - 1] > 0 and (s[i - 1] - s[i]) / s[i - 1] > x:
                return i
        return n
    raise ValueError(f"unknown cutoff {spec!r}")
