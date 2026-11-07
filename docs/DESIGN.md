# PlayMemory design and findings

## Idea
Memory as a seekable timeline ("video-like"): frames in time order, grouped into
segments with a centroid "keyframe". Query -> pick segments -> scan their frames.
Seeking by time/id is O(log n) / O(1); search results carry temporal neighbours.

## What the benchmark showed (HashEmbedder, k=10, 200 queries, `python -m bench.run`)
| n | data | system | recall@10 | p50 ms |
|---|------|--------|-----------|--------|
| 10k | clustered | faiss-flat | 1.00 | 0.84 |
| 10k | clustered | faiss-hnsw | 1.00 | 0.18 |
| 10k | clustered | playmemory | 0.75 | 0.10 |
| 10k | shuffled | playmemory | 0.20 | 0.12 |
| 100k | clustered | faiss-flat | 1.00 | 14.7 |
| 100k | clustered | faiss-hnsw | 1.00 | 0.40 |
| 100k | clustered | playmemory | 0.38 | 0.27 |

Segment sweep at 100k (clustered): recall 0.57 @ 8 segments (0.4 ms), 0.93 @ 64 (6 ms),
0.97 @ 128 (6 ms).

## Conclusions
1. The speed claim does not hold. At matching recall (~0.97+), the segment index
   (~6 ms) is slower than HNSW (~0.4 ms). At speeds where it wins, recall is poor.
2. Retrieval is a tiny share of an LLM call: even brute-force at 100k is ~3% of a
   500 ms generation; HNSW is ~0.08%. Faster retrieval barely moves end-to-end latency.
3. Centroid routing depends on temporal locality. With shuffled order recall collapses.
4. Where the timeline model genuinely helps: temporal context (`search(context=n)`,
   `seek`, `window`), a single portable append-only file, cheap incremental append,
   and replay of ordered agent history. These are features, not speed.

## Suggested next step
Use the timeline as the storage/context layer and put an ANN index (HNSW) on top for
routing, instead of centroids. Evaluate with a real embedder (`SentenceTransformerEmbedder`)
and real conversation data; hash embeddings are lexical only.

## Limitations
No BM25 fallback yet; reads use pread instead of mmap; no deletion/compaction;
benchmark data is synthetic.


## v0.2 update (after studying Memvid, see ROADMAP.md)
- Default retrieval is now hybrid: BM25 + exact vector search fused by RRF. The centroid
  index was later removed (see CHANGELOG 0.4.0).
- Exact numpy search at 100k: recall 1.00, p50 7.5 ms (faiss-flat 11.7 ms, HNSW 0.7 ms).
  Hybrid full API (embed + BM25 + vector + fetch frames with context): ~10 ms at 100k.
  Centroid routing stays at recall 0.38 (clustered) / 0.06 (shuffled): not recommended.
- Everything lives in one file (embeddings and tombstones inside CRC-checked records).
  Embedder name+dim are bound to the file; mismatch raises `ModelMismatchError`.
- `search(as_of=t)` is true time travel: frames and deletions after t are invisible.
- Past ~1M frames exact search needs an ANN index (HNSW); not built yet.
