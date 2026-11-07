from __future__ import annotations

import numpy as np


class SegmentIndex:
    """Two-level index: frames are grouped into fixed-size timeline segments.

    Each segment keeps a centroid ("keyframe"). Search scores segment centroids
    first, then scans only the frames of the top segments, like scrubbing to a
    region of a video before decoding frames. Exact scan below `exact_below`.
    """

    def __init__(self, dim: int, segment_size: int = 64, exact_below: int = 2048):
        self.dim = dim
        self.segment_size = segment_size
        self.exact_below = exact_below
        self._vecs = np.zeros((1024, dim), dtype=np.float32)
        self._n = 0
        self._sums = np.zeros((16, dim), dtype=np.float32)
        self._centroids = np.zeros((16, dim), dtype=np.float32)

    def __len__(self) -> int:
        return self._n

    @property
    def n_segments(self) -> int:
        return (self._n + self.segment_size - 1) // self.segment_size

    @property
    def vectors(self) -> np.ndarray:
        return self._vecs[: self._n]

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32).reshape(-1, self.dim)
        m = len(vecs)
        if m == 0:
            return
        need = self._n + m
        if need > len(self._vecs):
            cap = max(need, 2 * len(self._vecs))
            grown = np.zeros((cap, self.dim), dtype=np.float32)
            grown[: self._n] = self._vecs[: self._n]
            self._vecs = grown
        self._vecs[self._n : need] = vecs
        first, last = self._n // self.segment_size, (need - 1) // self.segment_size
        if last >= len(self._sums):
            cap = max(last + 1, 2 * len(self._sums))
            for name in ("_sums", "_centroids"):
                old = getattr(self, name)
                new = np.zeros((cap, self.dim), dtype=np.float32)
                new[: len(old)] = old
                setattr(self, name, new)
        for seg in range(first, last + 1):
            lo, hi = seg * self.segment_size, min((seg + 1) * self.segment_size, need)
            self._sums[seg] = self._vecs[lo:hi].sum(axis=0)
            norm = np.linalg.norm(self._sums[seg])
            self._centroids[seg] = self._sums[seg] / norm if norm else 0
        self._n = need

    def search(
        self, q: np.ndarray, k: int, n_segments: int | None = None
    ) -> list[tuple[int, float]]:
        """Return up to k (frame_id, score) pairs, best first."""
        if self._n == 0:
            return []
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        if self._n <= self.exact_below:
            return self._topk(np.arange(self._n), self.vectors @ q, k)
        if n_segments is None:
            n_segments = max(4, int(np.ceil(k / self.segment_size)) + 3)
        seg_scores = self._centroids[: self.n_segments] @ q
        n_segments = min(n_segments, self.n_segments)
        top = np.argpartition(-seg_scores, n_segments - 1)[:n_segments]
        ids = np.concatenate(
            [
                np.arange(s * self.segment_size, min((s + 1) * self.segment_size, self._n))
                for s in top
            ]
        )
        return self._topk(ids, self._vecs[ids] @ q, k)

    @staticmethod
    def _topk(ids: np.ndarray, scores: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(ids))
        part = np.argpartition(-scores, k - 1)[:k]
        part = part[np.argsort(-scores[part])]
        return [(int(ids[i]), float(scores[i])) for i in part]

    def load_vectors(self, vecs: np.ndarray) -> None:
        self.__init__(self.dim, self.segment_size, self.exact_below)
        self.add(vecs)
