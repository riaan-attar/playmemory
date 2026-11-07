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

## 3. Scale / speed (synthetic clustered text, hash embedder; `python -m bench.run`)
| n | exact (numpy) | HNSW (faiss, persisted) | full hybrid API |
|---|---|---|---|
| 100k | recall 1.00, p50 ~7-10 ms | recall 0.996, p50 ~0.6 ms | ~2.2 ms |

Open time at 100k frames: 2.25 s without a checkpoint, 0.5 s with one.

## Known limits
Single embedder and dataset; no answer-quality metric; scene boundaries use a similarity
heuristic with `session` metadata as forced boundaries (LoCoMo provides sessions);
Memvid was not run (no Rust toolchain), so no head-to-head numbers exist yet.
