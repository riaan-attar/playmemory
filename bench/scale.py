"""Scale test with synthetic vectors (no embedding cost): build, file size, open time,
exact vs HNSW search latency/recall, and hybrid latency at n frames.

    python -m bench.scale --n 1000000 --dim 128
"""
from __future__ import annotations

import argparse
import os
import resource
import tempfile
import time

import numpy as np

from playmemory import Memory


class _Emb:
    def __init__(self, dim, rng):
        self.dim, self.name, self.spec, self._rng = dim, f"synthetic-{dim}", f"synthetic-{dim}", rng

    def embed(self, texts):  # only used for queries
        v = self._rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--no-ann", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    emb = _Emb(a.dim, rng)
    # clustered vectors so neighbours are meaningful: 1,000 cluster centres + noise
    centres = rng.normal(size=(1000, a.dim)).astype(np.float32)
    path = os.path.join(tempfile.mkdtemp(prefix="pm_scale_"), "big.pmem")
    m = Memory(path, emb, ann_threshold=None, sync="never")
    t0 = time.perf_counter()
    for start in range(0, a.n, a.chunk):
        k = min(a.chunk, a.n - start)
        c = rng.integers(0, 1000, k)
        v = centres[c] + 0.35 * rng.normal(size=(k, a.dim)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        texts = [f"doc {start+i} topic{c[i]} word{(start+i) % 5000} common" for i in range(k)]
        m.add_many(texts, vectors=v, times=[float(start + i) for i in range(k)])
    build = time.perf_counter() - t0
    t0 = time.perf_counter(); m.checkpoint(); ckpt = time.perf_counter() - t0
    size = os.path.getsize(path)
    print(f"n={a.n:,} dim={a.dim}: add {build:.1f}s ({a.n/build:,.0f}/s), checkpoint {ckpt:.1f}s, file {size/1e6:.0f} MB, peak RSS {rss_mb():.0f} MB", flush=True)
    m.close()

    t0 = time.perf_counter(); m = Memory(path, emb, ann_threshold=None, read_only=True); op = time.perf_counter() - t0
    print(f"open (with checkpoint): {op:.2f}s, RSS {rss_mb():.0f} MB", flush=True)

    Q = emb.embed([""] * a.queries)
    q_texts = [f"topic{rng.integers(0,1000)} word{rng.integers(0,5000)}" for _ in range(a.queries)]

    def timed(fn):
        lat = []
        for i in range(a.queries):
            t = time.perf_counter(); fn(i); lat.append(time.perf_counter() - t)
        return np.percentile(lat, 50) * 1000, np.percentile(lat, 95) * 1000

    V = m.index.vectors
    truth = [set(np.argsort(-(V @ q))[:10].tolist()) for q in Q[: min(a.queries, 30)]]
    p50, p95 = timed(lambda i: m._exact_vec(Q[i], np.ones(len(m), bool), 10, len(m)))
    print(f"exact vector scan   p50 {p50:7.1f} ms  p95 {p95:7.1f} ms")
    emb._rng = np.random.default_rng(1)
    p50, p95 = timed(lambda i: m.bm25.search(q_texts[i], None, 10))
    print(f"BM25 (numpy)        p50 {p50:7.1f} ms  p95 {p95:7.1f} ms")
    if not a.no_ann:
        t0 = time.perf_counter(); m.read_only = False; m.build_ann(); build_ann = time.perf_counter() - t0
        rec = np.mean([len({i for i, _ in m.ann.search(Q[j], 10)} & truth[j]) / 10 for j in range(len(truth))])
        p50, p95 = timed(lambda i: m.ann.search(Q[i], 10))
        print(f"HNSW (faiss) ef=64  p50 {p50:7.1f} ms  p95 {p95:7.1f} ms  recall@10 {rec:.3f}  (build {build_ann:.0f}s)")
        for ef in (128, 256, 512):  # wider search = better recall, slower
            m.ann.ef_search = ef
            rec = np.mean([len({i for i, _ in m.ann.search(Q[j], 10)} & truth[j]) / 10 for j in range(len(truth))])
            p50, p95 = timed(lambda i: m.ann.search(Q[i], 10))
            print(f"HNSW (faiss) ef={ef:<3} p50 {p50:7.1f} ms  p95 {p95:7.1f} ms  recall@10 {rec:.3f}")
        m.ann.ef_search = 64
    m.embedder.embed = lambda texts: Q[[hash(t) % len(Q) for t in texts]]  # cheap deterministic query vectors
    p50, p95 = timed(lambda i: m.search(q_texts[i], k=10))
    print(f"full hybrid search  p50 {p50:7.1f} ms  p95 {p95:7.1f} ms  (BM25 + {'HNSW' if m.ann is not None else 'exact'} + fusion + fetch)")
    print(f"peak RSS {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
