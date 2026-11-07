"""Benchmark PlayMemory's segment index against faiss RAG baselines.

    python -m bench.run [--sizes 10000 100000] [--llm-ms 500]

Two datasets: "clustered" (topics persist over the timeline, like a real
conversation or agent log) and "shuffled" (same frames, timeline order
destroyed), which shows where the hierarchical index depends on locality.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from bench.baseline_rag import FlatBaseline, HNSWBaseline
from playmemory import Memory
from playmemory.index import SegmentIndex

K = 10
N_QUERIES = 200


def make_texts(n: int, shuffled: bool, rng: np.random.Generator) -> list[str]:
    n_topics = max(20, n // 200)
    vocab = [f"w{i}" for i in range(20000)]
    topic_words = [rng.choice(vocab, 60, replace=False) for _ in range(n_topics)]
    texts, topic, left = [], 0, 0
    for _ in range(n):
        if left == 0:  # topic runs of ~100-300 frames
            topic, left = int(rng.integers(n_topics)), int(rng.integers(100, 300))
        left -= 1
        words = rng.choice(topic_words[topic], 8)
        texts.append(" ".join(words))
    if shuffled:
        rng.shuffle(texts)
    return texts


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) * 1000  # ms


def timed(fn, queries) -> tuple[list[list[int]], list[float]]:
    out, lat = [], []
    for q in queries:
        t0 = time.perf_counter()
        out.append(fn(q))
        lat.append(time.perf_counter() - t0)
    return out, lat


def run(n: int, shuffled: bool, llm_ms: float) -> None:
    rng = np.random.default_rng(42)
    texts = make_texts(n, shuffled, rng)
    mem = Memory(segment_size=64)
    t0 = time.perf_counter()
    mem.add_many(texts, times=[float(i) for i in range(n)])
    build_pm = time.perf_counter() - t0
    vecs = mem.index.vectors.copy()

    # queries: half of the words of a random frame
    picks = rng.choice(n, N_QUERIES, replace=False)
    qtexts = [" ".join(texts[i].split()[:4]) for i in picks]
    qvecs = mem.embedder.embed(qtexts)

    # Tie-tolerant ground truth: a result counts if its exact score reaches
    # the k-th best exact score (hash embeddings produce many exact ties).
    kth = [np.sort(vecs @ q)[-K] - 1e-6 for q in qvecs]
    scores = [vecs @ q for q in qvecs]

    systems = {}
    t0 = time.perf_counter(); flat = FlatBaseline(vecs); b_flat = time.perf_counter() - t0
    t0 = time.perf_counter(); hnsw = HNSWBaseline(vecs); b_hnsw = time.perf_counter() - t0
    systems["faiss-flat"] = (lambda q: flat.search(q, K), b_flat, flat.nbytes())
    systems["faiss-hnsw"] = (lambda q: hnsw.search(q, K), b_hnsw, hnsw.nbytes())
    approx = SegmentIndex(vecs.shape[1], 64, exact_below=2048)
    approx.add(vecs)
    systems["pm-exact"] = (
        lambda q: [i for i, _ in mem.index.search(q, K)],
        build_pm,
        vecs.nbytes,
    )
    systems["pm-centroid"] = (
        lambda q: [i for i, _ in approx.search(q, K)],
        0.0,
        vecs.nbytes + approx.n_segments * vecs.shape[1] * 4,
    )

    label = "shuffled" if shuffled else "clustered"
    print(f"\n== n={n:,} ({label}), k={K}, {N_QUERIES} queries ==")
    print(f"{'system':<12} {'recall@10':>9} {'p50 ms':>8} {'p95 ms':>8} {'build s':>8} {'MB':>7} {'% of '+str(int(llm_ms))+'ms LLM':>14}")
    for name, (fn, build, nbytes) in systems.items():
        res, lat = timed(fn, qvecs)
        recall = np.mean([sum(sc[i] >= th for i in r) / K for r, sc, th in zip(res, scores, kth)])
        p50 = pct(lat, 50)
        print(f"{name:<12} {recall:>9.3f} {p50:>8.3f} {pct(lat, 95):>8.3f} {build:>8.2f} {nbytes/1e6:>7.1f} {100*p50/(p50+llm_ms):>13.3f}%")

    # full Memory.search path (embed text + search + fetch frames w/ context)
    for mode in ("vector", "hybrid"):
        _, lat = timed(lambda q: mem.search(q, k=K, context=1, mode=mode), qtexts)
        print(f"{'pm '+mode+' API':<12} {'':>9} {pct(lat, 50):>8.3f} {pct(lat, 95):>8.3f}   (embed + search + fetch frames with +-1 context)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    ap.add_argument("--llm-ms", type=float, default=500.0)
    args = ap.parse_args()
    for n in args.sizes:
        for shuffled in (False, True):
            run(n, shuffled, args.llm_ms)
