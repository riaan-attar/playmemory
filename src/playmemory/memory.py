from __future__ import annotations

import bisect
import functools
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .ann import AnnIndex
from .embed import Embedder, HashEmbedder, from_spec
from .episodes import EpisodeConfig, EpisodeIndex
from .frame import Frame, Hit, Span
from .index import VectorMatrix
from .lexical import TOKENIZER_VERSION, BM25Index, tokenize
from .store import FLAG_VEC_F16, FileChangedError, FrameStore, _fsync_dir, read_header

RRF_K = 60
MAX_TEXT_BYTES = 1 << 20  # 1 MiB per frame
MAX_META_BYTES = 64 << 10


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self._lock:
            if self.auto_refresh:
                self._refresh_impl()
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
        embedder: Embedder | str | None = None,
        *,
        ann_threshold: int | None = 200_000,
        episode_config: EpisodeConfig | None = None,
        read_only: bool = False,
        sync: str = "batch",
        sync_every: int = 100,
        vec_dtype: str = "float32",
        auto_refresh: bool = False,
        max_text_bytes: int = MAX_TEXT_BYTES,
        max_meta_bytes: int = MAX_META_BYTES,
    ):
        self.path = Path(path) if path else None
        if isinstance(embedder, str):
            embedder = from_spec(embedder)
        if embedder is None and path is not None and Path(path).exists() and Path(path).stat().st_size:
            embedder = from_spec(read_header(path)["spec"])  # reopen with the embedder the file was built with
        if embedder is None:
            import os as _os

            if _os.environ.get("PLAYMEMORY_ALLOW_DEBUG_EMBEDDER") == "1":
                embedder = HashEmbedder()
            else:
                raise ValueError(
                    "no embedder given: pass e.g. OnnxEmbedder() / 'onnx' / HTTPEmbedder(...). "
                    "Use 'debug' (non-semantic, tests only) or set PLAYMEMORY_ALLOW_DEBUG_EMBEDDER=1."
                )
        self.embedder = embedder
        self.read_only = read_only
        if vec_dtype not in ("float32", "float16"):
            raise ValueError("vec_dtype must be float32 or float16")
        self._flags = FLAG_VEC_F16 if vec_dtype == "float16" else 0
        self.auto_refresh = auto_refresh and read_only
        self.ann_threshold = ann_threshold
        self.episode_config = episode_config or EpisodeConfig()
        self.max_text_bytes = max_text_bytes
        self.max_meta_bytes = max_meta_bytes
        self._sync, self._sync_every = sync, sync_every
        self._lock = threading.RLock()
        self._open()

    def _open(self) -> None:
        self.store = FrameStore(
            self.path, self.embedder.dim, self.embedder.name,
            spec=getattr(self.embedder, "spec", self.embedder.name), flags=self._flags,
            read_only=self.read_only, sync=self._sync, sync_every=self._sync_every,
        )
        self.index = VectorMatrix(self.embedder.dim)
        cp = self.store.checkpoint_arrays
        bm_ok = cp is not None and "bm_tokver" in cp and int(cp["bm_tokver"][0]) == TOKENIZER_VERSION
        self.bm25 = BM25Index.from_arrays(cp) if bm_ok else BM25Index()
        if cp is not None and not bm_ok:  # tokenizer changed since the checkpoint: re-index all texts
            self.store.tail_texts = [(i, self.store.get(i).text) for i in range(len(self.store))]
        self.episodes = (EpisodeIndex.from_arrays(cp, self.episode_config)
                         if cp is not None and "ep_starts" in cp else EpisodeIndex(self.episode_config))
        for fid, text in self.store.tail_texts:
            assert len(self.bm25) == fid, "index/log out of step"
            self.bm25.add(text)
        self.store.tail_texts = []
        if len(self.store):
            self.index.add(self.store.load_vectors())
        self.ann: AnnIndex | None = None
        saved = self.store.read_ann() if self.ann_threshold is not None else None
        if saved and saved[0] <= len(self.store):
            try:
                self.ann = AnnIndex.from_bytes(saved[1])
                if len(self.ann) < len(self.store):  # frames added since it was saved
                    self.ann.add(self.index.vectors[len(self.ann):])
            except ImportError:
                self.ann = None
        # numpy mirrors used to build visibility masks without Python loops
        n = len(self.store)
        self._cap = max(1024, n * 2)
        self._del_np = np.full(self._cap, np.inf)  # deletion time, inf = alive
        for fid, dt in self.store.deleted.items():
            self._del_np[fid] = dt
        self._meta_ix: dict[str, dict[Any, list[int]]] = {}
        for fid, m in enumerate(self.store.metas):
            if m:
                self._index_meta(fid, m)

    def _index_meta(self, fid: int, meta: dict) -> None:
        for k, v in meta.items():
            try:
                self._meta_ix.setdefault(k, {}).setdefault(v, []).append(fid)
            except TypeError:  # unhashable value: left to the slow path
                pass

    def _grow(self, n: int) -> None:
        if n > self._cap:
            cap = max(n, self._cap * 2)
            new = np.full(cap, np.inf)
            new[: self._cap] = self._del_np
            self._del_np, self._cap = new, cap

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
            self._index_meta(frames[-1].id, frames[-1].meta)
            last = t
        self._grow(len(self.store))
        self.index.add(vecs)
        if self.ann is not None:
            self.ann.add(vecs)
        return frames

    @_writes
    def forget(self, fid: int, t: float | None = None) -> None:
        """Tombstone a frame: hidden from current search, kept for as_of queries."""
        dt = time.time() if t is None else t
        self.store.delete(fid, dt)
        self._del_np[fid] = self.store.deleted[fid]

    @_writes
    def supersede(self, old_id: int, text: str, meta: dict | None = None,
                  t: float | None = None) -> Frame:
        """Replace a stale fact: add the new frame and tombstone the old one."""
        now = time.time() if t is None else t
        new = self.add(text, {**(meta or {}), "supersedes": old_id}, now)
        self.store.delete(old_id, new.t)
        self._del_np[old_id] = self.store.deleted[old_id]
        return new

    # multi-process reads -------------------------------------------------------
    def _refresh_impl(self) -> int:
        try:
            first, newdel = self.store.refresh()
        except FileChangedError:  # writer compacted/replaced the file: start over
            self.store.close()
            self._open()
            return len(self.store)
        n = len(self.store)
        if n > first:
            vecs = self.store.load_vectors(first)
            for fid, text in self.store.tail_texts:
                self.bm25.add(text)
            self.store.tail_texts = []
            for fid in range(first, n):
                if self.store.metas[fid]:
                    self._index_meta(fid, self.store.metas[fid])
            self._grow(n)
            self.index.add(vecs)
            if self.ann is not None:
                self.ann.add(vecs)
        for fid in newdel:
            self._del_np[fid] = self.store.deleted[fid]
        return n - first

    def refresh(self) -> int:
        """Pick up what another process has written since this handle last looked.
        Returns the number of new frames (0 if nothing changed)."""
        with self._lock:
            return self._refresh_impl()

    # durability ---------------------------------------------------------------
    @_writes
    def commit(self) -> None:
        """Group commit: fsync everything written so far."""
        self.store.sync()

    @_writes
    def checkpoint(self) -> None:
        """Persist BM25 + offsets so the next open replays only the tail."""
        if self.ann is None and self.ann_threshold is not None and len(self.store) >= self.ann_threshold:
            self.build_ann()
        self._sync_episodes()
        self.store.write_checkpoint({**self.bm25.to_arrays(), **self.episodes.to_arrays()})
        if self.ann is not None:
            self.store.write_ann(self.ann.to_bytes(), len(self.ann))
        self.store.sync()

    @_writes
    def build_ann(self) -> None:
        """Build the HNSW graph now (needs the 'ann' extra). It is then kept
        up to date on every add and saved with each checkpoint."""
        self.ann = AnnIndex.build(self.index.vectors)

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
        new = Memory(tmp, self.embedder, vec_dtype="float16" if self._flags & FLAG_VEC_F16 else "float32",
                     ann_threshold=self.ann_threshold, episode_config=self.episode_config,
                     sync="never")
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
        d = self._del_np[:n]
        if as_of is None:
            mask &= ~np.isfinite(d)
        else:
            mask[bisect.bisect_right(self.store.times, as_of) :] = False
            mask &= ~(d <= as_of)
        for k, v in (where or {}).items():
            try:
                ids = self._meta_ix.get(k, {}).get(v, ())
            except TypeError:  # unhashable filter value: scan
                ids = [i for i, m in enumerate(self.store.metas) if (m or {}).get(k) == v]
            sel = np.zeros(n, dtype=bool)
            if len(ids):
                sel[np.asarray(ids, dtype=np.int64)] = True
            mask &= sel
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
        exact: bool = False,
    ) -> list[Hit]:
        """Top-k frames. mode: hybrid | vector | lexical.

        as_of: query the memory as it was at that time (later frames and later
        deletions are invisible). where: exact-match filter on frame meta.
        recency_half_life: multiply scores by 0.5**(age/half_life).
        """
        ranked = self._rank(query, mode, as_of, where, recency_half_life, pool, exact)
        return [self._hit(fid, score, context, as_of) for fid, score in ranked[:k]]

    def _rank(self, query: str, mode: str, as_of: float | None, where: dict[str, Any] | None,
              recency_half_life: float | None, pool: int, exact: bool) -> list[tuple[int, float]]:
        """Ranked (frame id, score) candidates, best first, at most ~pool of them."""
        n = len(self.store)
        if n == 0:
            return []
        mask = self._mask(as_of, where)
        ranked: list[tuple[int, float]]
        vec_hits: list[tuple[int, float]] = []
        lex_hits: list[tuple[int, float]] = []
        if mode in ("hybrid", "vector"):
            q = self.embedder.embed([query])[0]
            visible = int(mask.sum())
            if self.ann is not None and not exact and visible >= 0.5 * n and n > pool * 8:
                # ANN with over-fetch; if too many hits are masked, go exact
                fetch = min(n, pool * 4 + (n - visible) * 2)
                vec_hits = [(i, s) for i, s in self.ann.search(q, fetch) if mask[i]][:pool]
                if len(vec_hits) < min(pool, visible):
                    vec_hits = self._exact_vec(q, mask, pool, visible)
            else:
                vec_hits = self._exact_vec(q, mask, pool, visible)
        if mode in ("hybrid", "lexical"):
            lex_hits = self.bm25.search(query, mask, pool)
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
        return ranked

    def _exact_vec(self, q: np.ndarray, mask: np.ndarray, pool: int, visible: int):
        scores = self.index.vectors @ q
        scores[~mask] = -np.inf
        top = min(pool, visible)
        if not top:
            return []
        idx = np.argpartition(-scores, top - 1)[:top]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def _hit(self, fid: int, score: float, context: int, as_of: float | None) -> Hit:
        s = self.store
        before = s.range(fid - context, fid, True, as_of) if context else []
        after = s.range(fid + 1, fid + 1 + context, True, as_of) if context else []
        if as_of is not None:
            after = [f for f in after if f.t <= as_of]
        return Hit(s.get(fid), score, tuple(before), tuple(after))

    def _sync_episodes(self) -> None:
        if len(self.store) != getattr(self.episodes, "_n", -1) or not len(self.episodes):
            self.episodes.sync(self.index.vectors, self.store.metas, self.store.times)

    @_locked
    def episode_list(self) -> list[tuple[int, int]]:
        """[(start_id, end_id_inclusive)] for every scene on the timeline."""
        self._sync_episodes()
        return [(a, b - 1) for a, b in self.episodes.bounds()]

    @_locked
    def read(self, start_id: int, end_id: int, as_of: float | None = None) -> list[Frame]:
        """The frames of a span pointer (inclusive ids), minus forgotten/future ones."""
        return self.store.range(start_id, end_id + 1, True, as_of)

    @_locked
    def recall_span(self, query: str, k: int = 1, budget_chars: int = 4000, *,
                    mode: str = "hybrid", as_of: float | None = None,
                    where: dict[str, Any] | None = None, pool: int = 50) -> list[Span]:
        """Return up to k contiguous timeline spans ("watch from A to B") that
        best cover the query, each trimmed to its share of budget_chars.

        Episodes (scenes) are ranked by their best-matching frame; inside an episode the span is grown around the best frames
        toward whichever neighbour is more relevant until the budget is spent.
        """
        if len(self.store) == 0:
            return []
        self._sync_episodes()
        ranked = self._rank(query, mode, as_of, where, None, pool, False)
        if not ranked:
            return []
        fids = np.array([f for f, _ in ranked])
        eps = self.episodes.episode_of(fids)
        # a scene is ranked by its single best frame (summing many weak hits
        # measured far worse: 39% vs 60% of top scenes containing the evidence)
        ep_hits: dict[int, dict[int, float]] = {}
        for (fid, sc), e in zip(ranked, eps):
            ep_hits.setdefault(int(e), {})[fid] = sc
        ep_score = {e: max(h.values()) + 1e-3 * sum(h.values()) for e, h in ep_hits.items()}
        top = sorted(ep_score, key=lambda e: -ep_score[e])[:k]
        per = max(budget_chars // max(len(top), 1), 1)
        vis = self._mask(as_of, None)
        q = self.embedder.embed([query])[0]
        bounds = self.episodes.bounds()
        spans: list[Span] = []
        for e in top:
            s, end = bounds[e]  # [s, end)
            lo, hi = self._grow_span(s, end, ep_hits[e], vis, q, per)
            frames = tuple(self.store.range(lo, hi + 1, True, as_of))
            if not frames:
                continue
            spans.append(Span(lo, hi, frames[0].t, frames[-1].t, frames, ep_score[e], e, end - s,
                              self._keywords(frames)))
        spans.sort(key=lambda sp: sp.start_id)
        return spans

    def _grow_span(self, s: int, end: int, hits: dict[int, float], vis: np.ndarray,
              q: np.ndarray, budget: int) -> tuple[int, int]:
        """Pick [lo, hi] inside episode [s, end): cover the hit frames if they
        fit, then expand toward the more query-similar neighbour while budget lasts."""
        cost = {i: len(self.store.get(i).text) + 1 for i in range(s, end) if vis[i]}
        rel = self.index.vectors[s:end] @ q
        order = sorted((f for f in hits if f in cost), key=lambda f: -hits[f])
        if not order:
            order = [next(iter(cost), s)]
        lo = hi = order[0]
        used = cost.get(lo, 0)
        for f in order[1:]:  # extend to also cover other evidence frames if affordable
            nlo, nhi = min(lo, f), max(hi, f)
            extra = sum(cost.get(i, 0) for i in range(nlo, nhi + 1)) - sum(cost.get(i, 0) for i in range(lo, hi + 1))
            if used + extra <= budget:
                used, lo, hi = used + extra, nlo, nhi
        while True:
            left = next((i for i in range(lo - 1, s - 1, -1) if i in cost), None)
            right = next((i for i in range(hi + 1, end) if i in cost), None)
            cands = [(rel[i - s], i, side) for i, side in ((left, "L"), (right, "R"))
                     if i is not None and used + cost[i] <= budget]
            if not cands:
                break
            _, i, side = max(cands)
            used += cost[i]
            lo, hi = (i, hi) if side == "L" else (lo, i)
        return lo, hi

    _STOP = frozenset("the and was for that with this have you are but not they his her she him from "
                      "what when who how out about just like had been has were will would there their".split())

    def _keywords(self, frames, n: int = 5) -> str:
        tf: dict[str, int] = {}
        for f in frames:
            for t in tokenize(f.text):
                if len(t) > 2 and t not in self._STOP:
                    tf[t] = tf.get(t, 0) + 1
        N = max(len(self.bm25), 1)
        score = {t: c * np.log(1 + N / (1 + len(self.bm25.post_ids.get(t, ())))) for t, c in tf.items()}
        return ", ".join(sorted(score, key=lambda t: -score[t])[:n])

    @_locked
    def find(self, where: dict[str, Any], limit: int | None = None,
             as_of: float | None = None) -> list[Frame]:
        """Live frames whose meta matches `where`, in timeline order."""
        ids = np.flatnonzero(self._mask(as_of, where))
        if limit is not None:
            ids = ids[-limit:]
        return [self.store.get(int(i)) for i in ids]

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
            "episodes": len(self.episodes),
            "ann": self.ann is not None,
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
