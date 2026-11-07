from __future__ import annotations

import numpy as np


class VectorMatrix:
    """Growing (n, dim) float32 matrix of frame embeddings, in frame-id order."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = np.zeros((1024, dim), dtype=np.float32)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    @property
    def vectors(self) -> np.ndarray:
        return self._vecs[: self._n]

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32).reshape(-1, self.dim)
        need = self._n + len(vecs)
        if need > len(self._vecs):
            grown = np.zeros((max(need, 2 * len(self._vecs)), self.dim), dtype=np.float32)
            grown[: self._n] = self._vecs[: self._n]
            self._vecs = grown
        self._vecs[self._n : need] = vecs
        self._n = need
