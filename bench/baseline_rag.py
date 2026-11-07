"""Standard RAG baselines (faiss flat / HNSW) over the same vectors."""
from __future__ import annotations

import faiss
import numpy as np


class FlatBaseline:
    name = "faiss-flat"

    def __init__(self, vecs: np.ndarray):
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, q: np.ndarray, k: int) -> list[int]:
        _, ids = self.index.search(q.reshape(1, -1), k)
        return ids[0].tolist()

    def nbytes(self) -> int:
        return self.index.ntotal * self.index.d * 4


class HNSWBaseline:
    name = "faiss-hnsw"

    def __init__(self, vecs: np.ndarray, m: int = 16, ef_search: int = 64):
        self.index = faiss.IndexHNSWFlat(vecs.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 100
        self.index.add(vecs)
        self.index.hnsw.efSearch = ef_search
        self._bytes = vecs.nbytes + len(vecs) * m * 2 * 4  # vectors + approx links

    def search(self, q: np.ndarray, k: int) -> list[int]:
        _, ids = self.index.search(q.reshape(1, -1), k)
        return ids[0].tolist()

    def nbytes(self) -> int:
        return self._bytes
