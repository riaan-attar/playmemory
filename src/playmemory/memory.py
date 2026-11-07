from __future__ import annotations

import bisect
import functools
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .embed import Embedder, HashEmbedder
from .frame import Frame, Hit
from .index import SegmentIndex
from .lexical import BM25Index
from .store import FrameStore, _fsync_dir

RRF_K = 60
MAX_TEXT_BYTES = 1 << 20  # 1 MiB per frame
MAX_META_BYTES = 64 << 10


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self._lock:
            return fn(self, *a, **kw)

    return wrapper


def _writes(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        if self.read_only:
            raise PermissionError("memory opened read-only")
        with self._lock:
            return fn(self, *a, **kw)

    return wrapper


class Memory:
    """Timeline memory with hybrid (BM25 + vector) retrieval.

    path=None keeps everything in RAM. With a path, everything (frames,
    embeddings, tombstones, index checkpoints) lives in ONE file.
    One writer process at a time (exclusive lock); any number of
    read_only=True openers. Thread-safe within a process.
    sync: "always" | "batch" (fsync on commit/close) | "never".
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        embedder: Embedder | None = None,
        segment_size: int = 64,
        approx: bool = False,
        *,
        read_only: bool = False,
        sync: str = "batch",
        sync_every: int = 0,
        max_text_bytes: int = MAX_TEXT_BYTES,
        max_meta_bytes: int = MAX_META_BYTES,
    ):
        self.path = Path(path) if path else None
        self.embedder = embedder or HashEmbedder()
        self.read_only = read_only
        self.segment_size = segment_size
        self.approx = approx
        self.max_text_bytes = max_text_bytes
        self.max_meta_bytes = max_meta_bytes
        self._sync, self._sync_every = sync, sync_every
        self._lock = threading.RLock()
        self._open()

    def _open(self) -> None:
        self.store = FrameStore(
            self.path, self.embedder.dim, self.embedder.name,
            read_only=self.read_only, sync=self._sync, sync_every=self._sync_every,
        )
        self.index = SegmentIndex(
            self.embedder.dim, self.segment_size,
            exact_below=2048 if self.approx else 1 << 62,
        )
        cp = self.store.checkpoint_arrays
        self.bm25 = BM25Index.from_arrays(cp) if cp is not None else BM25Index()
        for fid, text in self.store.tail_texts:
            assert len(self.bm25) == fid, "index/log out of step"
            self.bm25.add(text)
        self.store.tail_texts = []
        if len(self.store):
            self.index.add(self.store.load_vectors())

    def __len__(self) -> int:
        return len(self.store)

    # writing ------------------------------------------------------------------
    def add(self, text: str, meta: dict | None = None, t: float | None = None) -> Frame:
        return self.add_many([text], [meta], [t])[0]

    @_writes
    def add_many(
        self,
        texts: list[str],
        metas: list[dict | None] | None = None,
        times: list[float | None] | None = None,
    ) -> list[Frame]:
        import json

        for i, text in enumerate(texts):
            if len(text.encode()) > self.max_text_bytes:
                raise ValueError(f"text {i} exceeds {self.max_text_bytes} bytes")
            if metas and metas[i] and len(json.dumps(metas[i])) > self.max_meta_bytes:
                raise ValueError(f"meta {i} exceeds {self.max_meta_bytes} bytes")
        vecs = self.embedder.embed(texts)
        return self._append(texts, metas, times, vecs)

    def _append(self, texts, metas, times, vecs) -> list[Frame]:
        now = time.time()
        last = self.store.times[-1] if len(self.store) else now
        frames = []
        for i, text in enumerate(texts):
            if times and times[i] is not None:
                t = times[i]
            else:  # wall clock may step backwards: never go below the last frame
                t = max(now, last)
            frames.append(self.store.append(t, text, metas[i] if metas else None, vecs[i]))
            self.bm25.add(text)
            last = t
        self.index.add(vecs)
        return frames

    @_writes
    def forget(self, fid: int, t: float | None = None) -> None:
        """Tombstone a frame: hidden from current search, kept for as_of queries."""
        self.store.delete(fid, time.time() if t is None else t)

    @_writes
    def supersede(self, old_id: int, text: str, meta: dict | None = None,
                  t: float | None = None) -> Frame:
        """Replace a stale fact: add the new frame and tombstone the old one."""
        now = time.time() if t is None else t
        new = self.add(text, {**(meta or {}), "supersedes": old_id}, now)
        self.store.delete(old_id, new.t)
        return new

    # durability ---------------------------------------------------------------
    @_writes
    def commit(self) -> None:
        """Group commit: fsync everything written so far."""
        self.store.sync()

    @_writes
    def checkpoint(self) -> None:
        """Persist BM25 + offsets so the next open replays only the tail."""
        self.store.write_checkpoint(self.bm25.to_arrays())
        self.store.sync()

    @_writes
    def compact(self, keep_history: bool = False) -> dict[str, int]:
        """Rewrite the file atomically, dropping old checkpoints (and, unless
        keep_history, deleted frames). Stored embeddings are reused."""
        if self.path is None:
            raise ValueError("in-memory memory cannot be compacted")
        before = self.store.nbytes
        s = self.store
        keep = [i for i in range(len(s)) if keep_history or i not in s.deleted]
        remap = {old: new for new, old in enumerate(keep)}
        tmp = self.path.with_name(self.path.name + ".compact")
        if tmp.exists():
            tmp.unlink()
        new = Memory(tmp, self.embedder, self.segment_size, self.approx, sync="never")
        vecs = self.index.vectors[keep] if keep else np.zeros((0, self.embedder.dim), np.float32)
        for k, old in enumerate(keep):
            f = s.get(old)
            meta = dict(f.meta)
            if "supersedes" in meta:
                if meta["supersedes"] in remap:
                    meta["supersedes"] = remap[meta["supersedes"]]
                else:
                    del meta["supersedes"]
            new.store.append(f.t, f.text, meta, vecs[k])
            new.bm25.add(f.text)
        if len(keep):
            new.index.add(vecs)
        if keep_history:
            for old, dt in s.deleted.items():
                new.store.delete(remap[old], dt)
        new.checkpoint()
        new.store.close()
        s.close()
        os.replace(tmp, self.path)
        _fsync_dir(self.path)
        self._open()
        return {"bytes_before": before, "bytes_after": self.store.nbytes, "frames": len(keep)}

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

    @_locked
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

    @_locked
    def get(self, fid: int) -> Frame:
        return self.store.get(fid)

    @_locked
    def seek(self, t: float) -> Frame | None:
        """The live frame in effect at time t (last live frame with time <= t)."""
        i = self.store.seek_time(t)
        while i is not None and i >= 0 and not self.store.is_live(i, t):
            i -= 1
        return None if i is None or i < 0 else self.store.get(i)

    @_locked
    def window(self, t0: float, t1: float) -> list[Frame]:
        lo, hi = self.store.time_range(t0, t1)
        return self.store.range(lo, hi, True)

    @_locked
    def stats(self) -> dict[str, Any]:
        s = self.store
        return {
            "frames": len(s),
            "live": len(s) - len(s.deleted),
            "deleted": len(s.deleted),
            "bytes": s.nbytes,
            "checkpoints": s.checkpoints,
            "frames_since_checkpoint": s.tail_records,
            "dropped_bytes_on_open": s.dropped_bytes,
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
        }

    def save(self) -> None:  # backwards-compatible alias
        if not self.read_only:
            self.commit()

    def close(self) -> None:
        with self._lock:
            if self.store._fd is None and self.path is not None:
                return
            if not self.read_only and self.path is not None:
                # checkpoint only when the un-indexed tail is big: each one
                # adds a (compactable) record, so avoid one per open/close
                if self.store.tail_records >= max(1000, len(self.store) // 10):
                    self.checkpoint()
            self.store.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
