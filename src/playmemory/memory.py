from __future__ import annotations

import bisect
import os
import time
from typing import Any

import numpy as np

from .embed import Embedder, HashEmbedder
from .frame import Frame, Hit
from .index import SegmentIndex
from .lexical import BM25Index
from .store import FrameStore

RRF_K = 60


class Memory:
    """Timeline memory with hybrid (BM25 + vector) retrieval.

    path=None keeps everything in RAM. With a path, everything (frames,
    embeddings, tombstones) lives in ONE file; BM25 and the vector matrix are
    rebuilt from it on open.
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        embedder: Embedder | None = None,
        segment_size: int = 64,
        approx: bool = False,
    ):
        self.embedder = embedder or HashEmbedder()
        self.store = FrameStore(path, self.embedder.dim, self.embedder.name)
        self.approx = approx
        self.index = SegmentIndex(
            self.embedder.dim, segment_size, exact_below=2048 if approx else 1 << 62
        )
        self.bm25 = BM25Index()
        if len(self.store):
            self.index.add(self.store.take_loaded_vectors())
            for f in self.store.range(0, len(self.store)):
                self.bm25.add(f.text)

    def __len__(self) -> int:
        return len(self.store)

    # writing ------------------------------------------------------------------
    def add(self, text: str, meta: dict | None = None, t: float | None = None) -> Frame:
        return self.add_many([text], [meta], [t])[0]

    def add_many(
        self,
        texts: list[str],
        metas: list[dict | None] | None = None,
        times: list[float | None] | None = None,
    ) -> list[Frame]:
        vecs = self.embedder.embed(texts)
        now = time.time()
        last = self.store.times[-1] if len(self.store) else now
        frames = []
        for i, text in enumerate(texts):
            t = times[i] if times and times[i] is not None else max(now, last)
            frames.append(
                self.store.append(t, text, metas[i] if metas else None, vecs[i])
            )
            self.bm25.add(text)
            last = t
        self.index.add(vecs)
        return frames

    def forget(self, fid: int, t: float | None = None) -> None:
        """Tombstone a frame: hidden from current search, kept for as_of queries."""
        self.store.delete(fid, time.time() if t is None else t)

    def supersede(self, old_id: int, text: str, meta: dict | None = None,
                  t: float | None = None) -> Frame:
        """Replace a stale fact: add the new frame and tombstone the old one."""
        now = time.time() if t is None else t
        new = self.add(text, {**(meta or {}), "supersedes": old_id}, now)
        self.store.delete(old_id, new.t)
        return new

    # reading ------------------------------------------------------------------
    def _mask(self, as_of: float | None, where: dict[str, Any] | None) -> np.ndarray:
        n = len(self.store)
        mask = np.ones(n, dtype=bool)
        if as_of is not None:
            mask[bisect.bisect_right(self.store.times, as_of) :] = False
        for fid, dt in self.store.deleted.items():
            if as_of is None or dt <= as_of:
                mask[fid] = False
        if where:
            metas = self.store.metas
            for i in np.flatnonzero(mask):
                m = metas[i]
                if any(m.get(k) != v for k, v in where.items()):
                    mask[i] = False
        return mask

    def search(
        self,
        query: str,
        k: int = 5,
        context: int = 0,
        *,
        mode: str = "hybrid",
        as_of: float | None = None,
        where: dict[str, Any] | None = None,
        recency_half_life: float | None = None,
        pool: int = 50,
    ) -> list[Hit]:
        """Top-k frames. mode: hybrid | vector | lexical.

        as_of: query the memory as it was at that time (later frames and later
        deletions are invisible). where: exact-match filter on frame meta.
        recency_half_life: multiply scores by 0.5**(age/half_life).
        """
        n = len(self.store)
        if n == 0:
            return []
        mask = self._mask(as_of, where)
        ranked: list[tuple[int, float]]
        vec_hits: list[tuple[int, float]] = []
        lex_hits: list[tuple[int, float]] = []
        if mode in ("hybrid", "vector"):
            q = self.embedder.embed([query])[0]
            if self.approx:
                vec_hits = [(i, s) for i, s in self.index.search(q, pool * 4) if mask[i]][:pool]
            else:
                scores = self.index.vectors @ q
                scores[~mask] = -np.inf
                top = min(pool, int(mask.sum()))
                if top:
                    idx = np.argpartition(-scores, top - 1)[:top]
                    idx = idx[np.argsort(-scores[idx])]
                    vec_hits = [(int(i), float(scores[i])) for i in idx]
        if mode in ("hybrid", "lexical"):
            lex_hits = self.bm25.search(query, lambda f: bool(mask[f]), pool)
        if mode == "vector":
            ranked = vec_hits
        elif mode == "lexical":
            ranked = lex_hits
        else:
            fused: dict[int, float] = {}
            for hits in (vec_hits, lex_hits):
                for rank, (fid, _) in enumerate(hits):
                    fused[fid] = fused.get(fid, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = list(fused.items())
        if recency_half_life:
            ref = as_of if as_of is not None else self.store.times[-1]
            times = self.store.times
            ranked = [(f, s * 0.5 ** (max(ref - times[f], 0) / recency_half_life))
                      for f, s in ranked]
        ranked.sort(key=lambda x: -x[1])
        return [self._hit(fid, score, context, as_of) for fid, score in ranked[:k]]

    def _hit(self, fid: int, score: float, context: int, as_of: float | None) -> Hit:
        s = self.store
        before = s.range(fid - context, fid, True, as_of) if context else []
        after = s.range(fid + 1, fid + 1 + context, True, as_of) if context else []
        if as_of is not None:
            after = [f for f in after if f.t <= as_of]
        return Hit(s.get(fid), score, tuple(before), tuple(after))

    def get(self, fid: int) -> Frame:
        return self.store.get(fid)

    def seek(self, t: float) -> Frame | None:
        """The live frame in effect at time t (last live frame with time <= t)."""
        i = self.store.seek_time(t)
        while i is not None and i >= 0 and not self.store.is_live(i, t):
            i -= 1
        return None if i is None or i < 0 else self.store.get(i)

    def window(self, t0: float, t1: float) -> list[Frame]:
        lo, hi = self.store.time_range(t0, t1)
        return self.store.range(lo, hi, True)

    def save(self) -> None:
        self.store.sync()

    def close(self) -> None:
        self.save()
        self.store.close()
