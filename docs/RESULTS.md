# Measured results

All numbers are from this repo's harnesses, run on LoCoMo (`locomo10.json`: 10 long
conversations, 1,535 questions with gold evidence turns; adversarial category skipped),
embedder `onnx:BAAI/bge-small-en-v1.5` (384-d), CPU. Metric is **retrieval only**
(did the returned memory contain the gold dialogue turns?). There is no LLM judge, so
nothing here measures final answer quality. Reproduce:

    python -m bench.quality --data locomo10.json --embedder onnx     # chunk baselines
    python -m bench.budget  --data locomo10.json --embedder onnx     # span vs chunks, equal size

## 1. Chunk retrieval baselines (evidence-recall@k, all questions)
| system | k=1 | k=5 | k=10 |
|---|---|---|---|
| bm25 only | 0.262 | 0.464 | 0.537 |
| vector only | 0.267 | 0.496 | 0.594 |
| **hybrid (RRF)** | 0.309 | 0.546 | 0.638 |
| hybrid + 1 neighbour each side | 0.388 | 0.661 | 0.749 |

Hybrid beats either alone at every k; temporal context is worth ~+0.1 recall at equal k
(but costs 3x the text, see section 2). Query latency p50 about 21 ms, almost all of it
embedding the query (BM25 alone: 1.3 ms).

## 2. "Watch from A to B": timeline spans vs top-k chunks at equal size
Each system may return at most B characters. `span1` = one scene (episode) trimmed to B,
`span2` = two scenes sharing B. Evidence-recall, all questions:

| system | B=500 | B=1000 | B=2000 | B=4000 |
|---|---|---|---|---|
| chunks (top hits, whole frames) | **0.406** | **0.542** | **0.647** | **0.757** |
| chunks + context | 0.378 | 0.518 | 0.630 | 0.721 |
| span1 | 0.397 | 0.497 | 0.537 | 0.547 |
| span2 | 0.379 | 0.470 | 0.594 | 0.643 |

Per category at B=500 (chunks / span1 / span2): single-hop 0.480 / **0.495** / 0.424,
temporal 0.516 / 0.463 / **0.519**, multi-hop 0.131 / 0.110 / **0.158**,
open-domain 0.188 / 0.149 / 0.163.

**Honest reading.** On this metric spans do **not** beat top-k chunks. They match them at
small budgets (and win a few cells at B=500) but plateau once the chosen scene is
exhausted (a scene is ~8 turns), while chunks keep adding the next-best turn from anywhere.
Why this benchmark is hard on spans: 76% of questions have exactly one gold turn
(precision per character dominates), and 78% have all gold turns inside one scene, so
scenes are coherent, but the top-ranked scene held the evidence only 60% of the time
(it was 39% when scenes were ranked by summed hit scores; ranking by best hit fixed most of that).

What spans give that this metric cannot see: contiguous, readable context (a scene, not
a bag of lines), a pointer the model can expand (`read_span`), and a smaller number of
distinct places to read. Whether that improves **answers** needs an LLM-judged QA eval
(not run yet).

### 2b. Reranker and more spans (full 1,535 questions, same setup)
`+rr` = the ONNX cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescoring the top 30 candidates.
`spanK` = K scenes sharing the budget. Evidence-recall, all questions:

| system | B=500 | B=1000 | B=2000 | B=4000 |
|---|---|---|---|---|
| chunks | 0.406 | 0.542 | 0.647 | 0.757 |
| **chunks + rerank** | **0.564** | **0.670** | **0.737** | **0.778** |
| span1 | 0.397 | 0.497 | 0.537 | 0.547 |
| span2 / span3 | 0.379 / 0.396 | 0.470 / 0.426 | 0.594 / 0.567 | 0.643 / 0.675 |
| span1 + rerank | 0.521 | 0.621 | 0.652 | 0.659 |
| span2 + rerank | 0.526 | 0.590 | 0.709 | 0.748 |
| span3 + rerank | 0.545 | 0.564 | 0.683 | 0.776 |

**The reranker is the largest single improvement in this repo**: +16 points of evidence recall at 500
characters and +2 at 4,000. Spans with several scenes (span3+rerank) catch up to reranked chunks only at
the largest budget (0.776 vs 0.778). The one place spans beat reranked chunks is multi-hop at the smallest
budget (B=500: span3+rr 0.338 vs chunks+rr 0.251); by B=1000 chunks are ahead again (0.387 vs 0.345).
Cost: scoring 30 passages with the cross-encoder takes about 30 ms on this CPU, versus about 1 ms for
the search itself. Cutoff, `ask`, graph and replay are covered by tests, not by this benchmark.

## 3. Head-to-head with Memvid (memvid-sdk 2.0.160, identical vectors)

Setup (`bench/vs_memvid.py`): the same 10 conversations, one turn per document/frame, the same
BGE-small vectors for turns and questions fed to both systems, `k` = 1/5/10, no neighbour context.
Memvid: `put_many(..., embeddings=...)`, hybrid `find(mode="sem", query_embedding=...)` and
`mode="lex"`, default settings, telemetry disabled. Evidence-recall@k, all 1,535 questions:

| system | k=1 | k=5 | k=10 | hit@10 |
|---|---|---|---|---|
| PlayMemory lexical (BM25) | 0.262 | 0.464 | 0.537 | 0.493 |
| PlayMemory vector (exact) | 0.267 | 0.496 | 0.594 | 0.534 |
| **PlayMemory hybrid** | **0.309** | **0.546** | **0.638** | **0.576** |
| Memvid `sem` (its hybrid) | 0.107 | 0.201 | 0.233 | 0.207 |
| Memvid `lex` | 0.001 | 0.002 | 0.002 | 0.001 |

| | PlayMemory | Memvid |
|---|---|---|
| search latency p50 / p95 (query vector given) | 0.71 / 0.91 ms (hybrid), 0.31 ms (vector) | 0.28 / 0.37 ms (`sem`), 1.64 / 2.75 ms (`lex`) |
| file size, 10 conversations | 12.3 MB (float32), 7.8 MB (float16) | 34.3 MB |
| build (1.6k-turn corpus, vectors supplied) | 0.8 s total | 353 s total |
| open | 9 ms/file | 15 ms/file |

**How to read this, honestly.** This is a retrieval comparison with Memvid in its default configuration
through its Python SDK, not a verdict on Memvid. What I verified: Memvid's `lex` mode treats a whole
natural-language question as an implicit AND, so it returns almost nothing for questions (and errored on
4 of 1,535 queries, counted as empty); and on identical vectors its `sem` results share only 37% of their
top-10 with exact cosine ranking, and the true top-10 never appears even when asking for 100 results
(flat at 37% from rank 10 to rank 100). Its HNSW source (Euclidean, M=16, efSearch=50) should do better
on ~600 vectors, so I could not explain the gap and found no option that closes it; it may be a
configuration or usage issue on my side (for example how externally supplied embeddings are wired), or
the way its hybrid mode restricts candidates. Memvid's own headline numbers use an LLM-judged answer
accuracy, which this does not measure, and I did not test its built-in embedding path. Treat the table as
"what you get from memvid-sdk 2.0.160 with supplied BGE vectors and defaults", reproducible with the
commands in `bench/vs_memvid.py`. Search speed is comparable (both well under 2 ms); PlayMemory is
smaller on disk and about 400x faster to build here, with the caveat that Memvid's build also creates a
Tantivy index and HNSW graph from scratch.

## 4. Scale / speed (`python -m bench.scale`, synthetic clustered vectors)
| frames x dim | add | open (checkpointed) | exact scan | BM25 | HNSW | full hybrid API | file | peak RSS |
|---|---|---|---|---|---|---|---|---|
| 200k x 128 | 28k/s | 1.2 s | 9.9 ms | 0.5 ms | 0.5 ms, recall@10 0.40 (ef=64) | 1.4 ms | 139 MB | 1.3 GB |
| 500k x 64 | 29k/s | 3.1 s | 26.8 ms | 1.2 ms | 0.4-1.0 ms, recall@10 0.35 / 0.43 / 0.53 / 0.74 at ef 64 / 128 / 256 / 512 | 2.7 ms | 220 MB | 1.7 GB |

HNSW recall here is a hard case (Gaussian noise around 1,000 centres, so many near-ties); on the earlier
100k clustered-text benchmark it was 0.996. `Memory(ann_ef_search=...)` trades latency for recall.
The exact scan is linear (about 27 ms at 500k), which is why HNSW takes over above `ann_threshold`.
1M frames was not run on this 15 GB machine.

## Known limits
Single embedder and dataset for quality; no answer-quality (LLM-judged) benchmark; scene boundaries use a
similarity heuristic with `session` metadata as forced boundaries (LoCoMo provides sessions); the Memvid
comparison covers its default hybrid mode via the Python SDK only.
