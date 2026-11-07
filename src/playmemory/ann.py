from __future__ import annotations

import numpy as np


def _faiss():
    try:
        import faiss
    except ImportError as e:  # pragma: no cover
        raise ImportError("ANN search needs the 'ann' extra: pip install playmemory[ann]") from e
    return faiss


class AnnIndex:
    """HNSW graph over the stored vectors (faiss backend, inner product).

    Ids are frame ids (insertion order). Deletions/time filters are applied by
    the caller after search, with over-fetch.
    """

    def __init__(self, index, ef_search: int = 64):
        self.index = index
        self.ef_search = ef_search

    @classmethod
    def build(cls, vecs: np.ndarray, m: int = 16, ef_construction: int = 100,
              ef_search: int = 64) -> "AnnIndex":
        faiss = _faiss()
        index = faiss.IndexHNSWFlat(vecs.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        if len(vecs):
            index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        return cls(index, ef_search)

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, vecs: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vecs, dtype=np.float32))

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self))
        if k == 0:
            return []
        self.index.hnsw.efSearch = max(self.ef_search, k)
        scores, ids = self.index.search(np.ascontiguousarray(q, dtype=np.float32).reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def to_bytes(self) -> bytes:
        return _faiss().serialize_index(self.index).tobytes()

    @classmethod
    def from_bytes(cls, blob: bytes, ef_search: int = 64) -> "AnnIndex":
        return cls(_faiss().deserialize_index(np.frombuffer(blob, dtype=np.uint8)), ef_search)
