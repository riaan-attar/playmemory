from __future__ import annotations

import bisect
import contextlib
import dataclasses
import functools
import uuid
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import acl as _acl
from .acl import Principal
from .ann import AnnIndex
from .cutoff import keep_count
from .embed import Embedder, HashEmbedder, from_spec
from .episodes import EpisodeConfig, EpisodeIndex
from .frame import Frame, Hit, Span
from .index import VectorMatrix
from .lexical import TOKENIZER_VERSION, BM25Index, tokenize
from .pii import mask_pii
from .temporal import parse_time_range
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


@dataclasses.dataclass
class ReplayDiff:
    event: dict
    original: list
    replayed: list

    @property
    def same(self) -> bool:
        return self.original == self.replayed

    @property
    def added(self) -> list:
        return [x for x in self.replayed if x not in self.original]

    @property
    def removed(self) -> list:
        return [x for x in self.original if x not in self.replayed]

    @property
    def reordered(self) -> bool:
        common = [x for x in self.original if x in self.replayed]
        return common != [x for x in self.replayed if x in self.original]


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
        password: str | bytes | None = None,
        principal: Principal | None = None,
        redact: bool | Any = False,
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
        self._session: dict | None = None
        self._seq = 0
        self._snapshot: tuple[int, int] | None = None  # (frames, tombstones) visible during replay
        self._replaying = False
        self._password = password
        self.principal = principal
        self.redact = redact
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
            password=self._password,
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
        vectors: np.ndarray | None = None,
    ) -> list[Frame]:
        """Add frames. `vectors` (n, dim) skips embedding the texts (precomputed/cached/image vectors)."""
        import json

        for i, text in enumerate(texts):
            if len(text.encode()) > self.max_text_bytes:
                raise ValueError(f"text {i} exceeds {self.max_text_bytes} bytes")
            if metas and metas[i] and len(json.dumps(metas[i])) > self.max_meta_bytes:
                raise ValueError(f"meta {i} exceeds {self.max_meta_bytes} bytes")
        vecs = self.embedder.embed(texts) if vectors is None else np.asarray(vectors, dtype=np.float32)
        if vecs.shape != (len(texts), self.embedder.dim):
            raise ValueError(f"vectors must have shape ({len(texts)}, {self.embedder.dim})")
        frames = self._append(texts, metas, times, vecs)
        self._log("add", {"count": len(frames)}, [f.id for f in frames])
        return frames

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
    def add_images(self, paths: list[str], captions: list[str | None] | None = None,
                   metas: list[dict | None] | None = None, times: list[float | None] | None = None) -> list[Frame]:
        """Add images as frames: the vector is the image embedding (needs an embedder with
        `embed_images`, e.g. ClipEmbedder); the searchable text is the caption or file name."""
        if not hasattr(self.embedder, "embed_images"):
            raise TypeError("this embedder cannot embed images; use ClipEmbedder (playmemory[images])")
        paths = [str(p) for p in paths]
        texts = [(captions[i] if captions and captions[i] else Path(p).name) for i, p in enumerate(paths)]
        metas = [{**(metas[i] or {} if metas else {}), "kind": "image", "source": str(Path(p).resolve())}
                 for i, p in enumerate(paths)]
        return self.add_many(texts, metas, times, vectors=self.embedder.embed_images(paths))

    @_writes
    def forget(self, fid: int, t: float | None = None) -> None:
        """Tombstone a frame: hidden from current search, kept for as_of queries."""
        dt = time.time() if t is None else t
        self.store.delete(fid, dt)
        self._del_np[fid] = self.store.deleted[fid]
        self._log("forget", {}, [fid])

    @_writes
    def supersede(self, old_id: int, text: str, meta: dict | None = None,
                  t: float | None = None) -> Frame:
        """Replace a stale fact: add the new frame and tombstone the old one."""
        now = time.time() if t is None else t
        new = self.add(text, {**(meta or {}), "supersedes": old_id}, now)
        self.store.delete(old_id, new.t)
        self._del_np[old_id] = self.store.deleted[old_id]
        return new

    @_writes
    def ingest(self, paths, **kw):
        """Chunk and add files/directories (txt, md, html, csv, json/jsonl; pdf, docx, xlsx,
        pptx with the `docs` extra). Idempotent per file content. See playmemory.ingest."""
        from .ingest import ingest_paths

        return ingest_paths(self, paths, **kw)

    # sessions: audit trail + deterministic replay ----------------------------------
    def _log(self, op: str, params: dict, result) -> None:
        if self._session is None or self._replaying or self.read_only:
            return
        who = None
        if self.principal is not None:
            who = {"tenant": self.principal.tenant, "roles": sorted(self.principal.roles)}
        self._seq += 1
        self.store.append_event({"s": self._session["id"], "q": self._seq, "op": op, "t": time.time(),
                                 "n": len(self.store), "nt": len(self.store.deleted), "who": who,
                                 "p": params, "r": result})

    @contextlib.contextmanager
    def session(self, name: str = "session"):
        """Record every search/span/add/forget inside the block into the file, with its
        parameters and results. Replay them later with `replay(session_id, **overrides)`."""
        sid = self.start_session(name)
        try:
            yield sid
        finally:
            self.end_session()

    @_writes
    def start_session(self, name: str = "session") -> str:
        if self._session is not None:
            raise RuntimeError("a session is already recording")
        sid = uuid.uuid4().hex[:10]
        self._session, self._seq = {"id": sid, "name": name}, 0
        self._log("session_start", {"name": name}, None)
        return sid

    @_writes
    def end_session(self) -> None:
        if self._session is not None:
            self._log("session_end", {}, None)
            self._session = None

    @_locked
    def events(self, session_id: str | None = None) -> list[dict]:
        return [e for e in self.store.events() if session_id is None or e["s"] == session_id]

    @_locked
    def sessions(self) -> list[dict]:
        out: dict[str, dict] = {}
        for e in self.store.events():
            d = out.setdefault(e["s"], {"id": e["s"], "name": "", "t": e["t"], "events": 0})
            d["events"] += 1
            if e["op"] == "session_start":
                d["name"] = e["p"]["name"]
        return list(out.values())

    @_locked
    def replay(self, session_id: str, **overrides) -> list["ReplayDiff"]:
        """Re-run the recorded searches/spans of a session against the memory exactly as it was
        when each was issued (later frames/deletions hidden) and diff the results. `overrides`
        replace search parameters (mode=..., k=..., rerank=..., graph=...) so you can ask "what
        would a different setting have returned?"."""
        diffs = []
        self._replaying = True
        try:
            for e in self.store.events():
                if e["s"] != session_id or e["op"] not in ("search", "recall_span"):
                    continue
                params = {**e["p"], **overrides}
                params.pop("rerank", None) if "rerank" not in overrides else None
                params.pop("graph", None) if "graph" not in overrides else None
                if params.get("when") is not None and params.get("now") is None:
                    params["now"] = e["t"]
                if isinstance(params.get("when"), list):
                    params["when"] = tuple(params["when"])
                if isinstance(params.get("cutoff"), list):
                    params["cutoff"] = tuple(params["cutoff"])
                self._snapshot = (e["n"], e["nt"])
                if e["op"] == "search":
                    got = [[h.frame.id, round(h.score, 5)] for h in self.search(**params)]
                    old, new = [r[0] for r in e["r"]], [r[0] for r in got]
                else:
                    got = [[sp.start_id, sp.end_id] for sp in self.recall_span(**params)]
                    old, new = [tuple(r) for r in e["r"]], [tuple(r) for r in got]
                diffs.append(ReplayDiff(e, old, new))
        finally:
            self._snapshot = None
            self._replaying = False
        return diffs

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
    def sign(self, key: bytes | str) -> None:
        """Append an HMAC-SHA256 signature over everything written so far."""
        self.store.sign(key.encode() if isinstance(key, str) else key)

    @_locked
    def verify_signature(self, key: bytes | str) -> tuple[bool, int]:
        """(valid, unsigned trailing bytes) for the last signature in the file."""
        return self.store.verify_signature(key.encode() if isinstance(key, str) else key)

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
                     password=self._password, sync="never")
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
            for ev in s.events():  # ids are unchanged when history is kept, so the audit trail stays valid
                new.store.append_event(ev)
        new.checkpoint()
        new.store.close()
        s.close()
        os.replace(tmp, self.path)
        _fsync_dir(self.path)
        self._open()
        return {"bytes_before": before, "bytes_after": self.store.nbytes, "frames": len(keep)}

    # reading ------------------------------------------------------------------
    def _mask(self, as_of: float | None, where: dict[str, Any] | None,
              window: tuple[float, float] | None = None,
              principal: Principal | None = None) -> np.ndarray:
        n = len(self.store)
        mask = np.ones(n, dtype=bool)
        principal = principal or self.principal
        if principal is not None:
            mask &= self._acl_mask(principal, n)
        if window is not None:  # keep frames with window[0] <= t < window[1]
            lo = bisect.bisect_left(self.store.times, window[0])
            hi = bisect.bisect_left(self.store.times, window[1])
            mask[:lo] = False
            mask[hi:] = False
        d = self._del_np[:n]
        if self._snapshot is not None:  # replay: the world as it was when the event was recorded
            n_snap, n_tomb = self._snapshot
            mask[n_snap:] = False
            d = np.full(n, np.inf)
            for fid, dt in list(self.store.deleted.items())[:n_tomb]:
                d[fid] = dt
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
        when: str | tuple[float, float] | None = None,
        now: float | None = None,
        rerank: Any = None,
        rerank_pool: int = 30,
        cutoff: Any = None,
        principal: Principal | None = None,
        graph: Any = None,
    ) -> list[Hit]:
        """Top-k frames. mode: hybrid | vector | lexical.

        as_of: query the memory as it was at that time (later frames and later
        deletions are invisible). where: exact-match filter on frame meta.
        recency_half_life: multiply scores by 0.5**(age/half_life).
        when: restrict to a time range: (t0, t1), a phrase like "last week" / "in May 2023",
        or "auto" to read one out of the query itself (relative to `now`, default wall clock).
        rerank: a Reranker rescoring the top `rerank_pool` candidates.
        graph: a playmemory.graph.Graph; frames about entities named in the query join the fusion.
        cutoff: adaptive result count, see playmemory.cutoff ("elbow", ("ratio", .5), ...).
        """
        ranked = self._rank(query, mode, as_of, where, recency_half_life, pool, exact,
                            when=when, now=now, rerank=rerank, rerank_pool=rerank_pool, principal=principal, graph=graph)
        if cutoff is not None:
            ranked = ranked[: keep_count([sc for _, sc in ranked[:k]], cutoff)]
        vis = self._mask(as_of, None, None, principal) if context else None
        hits = [self._hit(fid, score, context, as_of, vis) for fid, score in ranked[:k]]
        if self._session is not None and not self._replaying:
            self._log("search", {"query": query, "k": k, "context": context, "mode": mode, "as_of": as_of,
                                 "where": where, "recency_half_life": recency_half_life, "pool": pool,
                                 "exact": exact, "when": when if isinstance(when, (str, type(None))) else list(when),
                                 "now": now, "cutoff": cutoff, "rerank": getattr(rerank, "name", bool(rerank)) if rerank else None,
                                 "graph": graph is not None},
                      [[h.frame.id, round(h.score, 5)] for h in hits])
        return hits

    def _rank(self, query: str, mode: str, as_of: float | None, where: dict[str, Any] | None,
              recency_half_life: float | None, pool: int, exact: bool, *,
              when=None, now: float | None = None, rerank=None,
              rerank_pool: int = 30, principal=None, graph=None) -> list[tuple[int, float]]:
        """Ranked (frame id, score) candidates, best first, at most ~pool of them."""
        n = len(self.store)
        if n == 0:
            return []
        window = self._window(when, query, now)
        mask = self._mask(as_of, where, window, principal)
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
        graph_hits: list[tuple[int, float]] = []
        if graph is not None and mode == "hybrid":  # entity-linked retrieval as a third RRF list
            graph_hits = [(f, sc) for f, sc in graph.hits(query, as_of) if mask[f]]
        if mode == "vector":
            ranked = vec_hits
        elif mode == "lexical":
            ranked = lex_hits
        else:
            fused: dict[int, float] = {}
            for hits in (vec_hits, lex_hits, graph_hits):
                for rank, (fid, _) in enumerate(hits):
                    fused[fid] = fused.get(fid, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = list(fused.items())
        if recency_half_life:
            last = self._snapshot[0] if self._snapshot else len(self.store)
            ref = as_of if as_of is not None else self.store.times[max(last, 1) - 1]
            times = self.store.times
            ranked = [(f, s * 0.5 ** (max(ref - times[f], 0) / recency_half_life))
                      for f, s in ranked]
        ranked.sort(key=lambda x: -x[1])
        if rerank is not None and ranked:
            head, tail = ranked[:rerank_pool], ranked[rerank_pool:]
            scores = rerank.score(query, [self.store.get(f).text for f, _ in head])
            head = sorted(((f, float(sc)) for (f, _), sc in zip(head, scores)), key=lambda x: -x[1])
            floor = head[-1][1]  # keep reranked candidates above the un-reranked tail
            ranked = head + [(f, min(sc, floor) - 1e-6 * (i + 1)) for i, (f, sc) in enumerate(tail)]
        return ranked

    def _acl_mask(self, p: Principal, n: int) -> np.ndarray:
        ix = self._meta_ix
        tenanted = np.zeros(n, dtype=bool)
        for ids in ix.get(_acl.TENANT, {}).values():
            tenanted[np.asarray(ids, dtype=np.int64)] = True
        ok = np.zeros(n, dtype=bool)
        mine = ix.get(_acl.TENANT, {}).get(p.tenant, ())
        if len(mine):
            ok[np.asarray(mine, dtype=np.int64)] = True
        if p.allow_untenanted:
            ok |= ~tenanted
        for fid in ix.get(_acl.VISIBILITY, {}).get("restricted", ()):  # usually a small set
            if ok[fid]:
                ok[fid] = _acl.allowed(p, self.store.metas[fid] or {})
        return ok

    def _red(self, f: Frame) -> Frame:
        """Query-time PII masking of returned text (stored text is untouched)."""
        if not self.redact:
            return f
        fn = self.redact if callable(self.redact) else mask_pii
        return dataclasses.replace(f, text=fn(f.text))

    def _window(self, when, query: str, now: float | None):
        if when is None:
            return None
        now = time.time() if now is None else now
        if isinstance(when, tuple):
            return when
        return parse_time_range(query if when == "auto" else when, now)

    def _exact_vec(self, q: np.ndarray, mask: np.ndarray, pool: int, visible: int):
        scores = self.index.vectors @ q
        scores[~mask] = -np.inf
        top = min(pool, visible)
        if not top:
            return []
        idx = np.argpartition(-scores, top - 1)[:top]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def _hit(self, fid: int, score: float, context: int, as_of: float | None, vis=None) -> Hit:
        s = self.store
        if vis is None:
            vis = self._mask(as_of, None) if context else None
        before: list[Frame] = []
        after: list[Frame] = []
        if context:  # walk outward collecting `context` VISIBLE neighbours per side
            i = fid - 1
            while i >= 0 and len(before) < context:
                if vis[i]:
                    before.append(s.get(i))
                i -= 1
            before.reverse()
            i, n = fid + 1, len(s)
            while i < n and len(after) < context:
                if vis[i]:
                    after.append(s.get(i))
                i += 1
        return Hit(self._red(s.get(fid)), score, tuple(map(self._red, before)), tuple(map(self._red, after)))

    def _sync_episodes(self) -> None:
        if len(self.store) != getattr(self.episodes, "_n", -1) or not len(self.episodes):
            self.episodes.sync(self.index.vectors, self.store.metas, self.store.times)

    @_locked
    def episode_list(self) -> list[tuple[int, int]]:
        """[(start_id, end_id_inclusive)] for every scene on the timeline."""
        self._sync_episodes()
        return [(a, b - 1) for a, b in self.episodes.bounds()]

    @_locked
    def read(self, start_id: int, end_id: int, as_of: float | None = None,
             principal: Principal | None = None) -> list[Frame]:
        """The frames of a span pointer (inclusive ids), minus forgotten/future ones."""
        vis = self._mask(as_of, None, None, principal)
        return [self._red(f) for f in self.store.range(start_id, end_id + 1, True, as_of) if vis[f.id]]

    @_locked
    def recall_span(self, query: str, k: int = 1, budget_chars: int = 4000, *,
                    mode: str = "hybrid", as_of: float | None = None,
                    where: dict[str, Any] | None = None, pool: int = 50,
                    when=None, now: float | None = None, rerank: Any = None,
                    rerank_pool: int = 30, principal: Principal | None = None) -> list[Span]:
        """Return up to k contiguous timeline spans ("watch from A to B") that
        best cover the query, each trimmed to its share of budget_chars.

        Episodes (scenes) are ranked by their best-matching frame; inside an episode the span is grown around the best frames
        toward whichever neighbour is more relevant until the budget is spent.
        """
        if len(self.store) == 0:
            return []
        self._sync_episodes()
        ranked = self._rank(query, mode, as_of, where, None, pool, False,
                            when=when, now=now, rerank=rerank, rerank_pool=rerank_pool, principal=principal)
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
        vis = self._mask(as_of, None, self._window(when, query, now), principal)  # spans stay inside `when` + ACL
        q = self.embedder.embed([query])[0]
        bounds = self.episodes.bounds()
        spans: list[Span] = []
        for e in top:
            s, end = bounds[e]  # [s, end)
            lo, hi = self._grow_span(s, end, ep_hits[e], vis, q, per)
            frames = tuple(self._red(f) for f in self.store.range(lo, hi + 1, True, as_of) if vis[f.id])
            if not frames:
                continue
            spans.append(Span(lo, hi, frames[0].t, frames[-1].t, frames, ep_score[e], e, int(vis[s:end].sum()),
                              self._keywords(frames)))
        spans.sort(key=lambda sp: sp.start_id)
        if self._session is not None and not self._replaying:
            self._log("recall_span", {"query": query, "k": k, "budget_chars": budget_chars, "mode": mode,
                                    "as_of": as_of, "where": where, "pool": pool,
                                    "when": when if isinstance(when, (str, type(None))) else list(when), "now": now},
                      [[sp.start_id, sp.end_id] for sp in spans])
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
             as_of: float | None = None, principal: Principal | None = None) -> list[Frame]:
        """Live frames whose meta matches `where`, in timeline order."""
        ids = np.flatnonzero(self._mask(as_of, where, None, principal))
        if limit is not None:
            ids = ids[-limit:]
        return [self._red(self.store.get(int(i))) for i in ids]

    @_locked
    def get(self, fid: int, principal: Principal | None = None) -> Frame:
        p = principal or self.principal
        if p is not None and not self._acl_mask(p, len(self.store))[fid]:
            raise PermissionError(f"frame {fid} is not visible to this principal")
        return self._red(self.store.get(fid))

    @_locked
    def seek(self, t: float) -> Frame | None:
        """The live frame in effect at time t (last live frame with time <= t)."""
        i = self.store.seek_time(t)
        vis = self._mask(t, None, None, None) if self.principal is not None else None
        while i is not None and i >= 0 and not (self.store.is_live(i, t) and (vis is None or vis[i])):
            i -= 1
        return None if i is None or i < 0 else self._red(self.store.get(i))

    @_locked
    def window(self, t0: float, t1: float) -> list[Frame]:
        lo, hi = self.store.time_range(t0, t1)
        vis = self._mask(None, None, None, None) if self.principal is not None else None
        return [self._red(f) for f in self.store.range(lo, hi, True) if vis is None or vis[f.id]]

    def ask(self, question: str, synthesizer, *, k: int = 8, context: int = 1,
            budget_chars: int = 6000, spans: bool = False, **search_kw):
        """Retrieve memory, have `synthesizer(question, context_text)` answer it, and
        return an Answer whose `citations` are the frames the answer cited.
        spans=True feeds contiguous scenes (recall_span) instead of top-k hits."""
        from .ask import Answer, build_context, extract_citations

        if spans:
            frames = [f for sp in self.recall_span(question, 2, budget_chars, **search_kw) for f in sp.frames]
        else:
            seen: dict[int, Frame] = {}
            used = 0
            for h in self.search(question, k, context, **search_kw):
                for f in h.context():
                    if f.id not in seen and used + len(f.text) <= budget_chars:
                        seen[f.id] = f
                        used += len(f.text)
            frames = sorted(seen.values(), key=lambda f: f.id)
        ctx = build_context(frames)
        text = synthesizer(question, ctx)
        return Answer(question, text, extract_citations(text, frames), frames, ctx)

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
            "encrypted": self.store._cipher is not None,
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

