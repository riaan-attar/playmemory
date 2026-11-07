# PlayMemory vs Memvid, and the plan

Source studied: github.com/memvid/memvid (Apache-2.0, Rust, spec MV2 v2.1, read from
README, MV2_SPEC.md and src/). Not run: no Rust toolchain here, so the comparison is
by design, not by measured numbers.

## What Memvid is (and is not)
"Video" is a metaphor there too: no video codec. Memory is an append-only sequence of
immutable "Smart Frames" in ONE `.mv2` file: 4 KB header, embedded WAL (crash
recovery), compressed frame segments, Tantivy BM25 index, HNSW vector index, time
index, TOC footer with checksums. Extras: time-travel replay of agent sessions,
structured memory cards (fact/preference/event...), entity graph, adaptive result
cutoff, embedding-model binding, encryption, ACL/PII, PDF/xlsx/audio/CLIP ingestion.
Hybrid retrieval, not video-speed tricks: its speed comes from BM25 + HNSW.
This confirms our benchmark conclusion: the win is structure, not a faster lookup.

## Where we stand today (v0.1)
| Area | Memvid | PlayMemory v0.1 |
|---|---|---|
| Single file | yes (all indexes inside) | no: sidecar `.emb.npy` |
| Crash safety | WAL + checksums | torn-tail truncate only, no checksums |
| Retrieval | BM25 + HNSW hybrid | centroid routing, recall 0.38 at 100k: weak |
| Lexical search | Tantivy | none |
| Temporal | time index, replay | seek/window, neighbour context |
| Deletes/updates | tombstones | none |
| Language | Rust (+ Node/Python SDKs) | pure Python + numpy |

## Where we can be better (our angle)
1. **Python-native, tiny, hackable**: `pip install`, numpy only; Memvid's Python is an SDK over Rust.
2. **Temporal context in every hit**: neighbours before/after, not isolated chunks.
3. **True time travel on queries**: `search(as_of=t)` sees frames AND deletions as they were at t.
4. **Supersede, don't pile up**: `supersede(old_id, new_text)` keeps history but hides stale facts from current search.
5. **Honest, reproducible benchmarks**: recall measured against exact search, tie-tolerant, with locality ablation.
6. Hybrid retrieval done right: BM25 + vector fused by RRF, optional recency decay.

## Milestones
- **M1 Format v2** (single file): header (magic, version, dim, embedder name), CRC32 per record,
  embeddings inside the frame record, tombstone records with timestamps. Model-mismatch error.
- **M2 Hybrid retrieval**: BM25 inverted index, exact vector search, RRF fusion, metadata filter,
  `as_of`, recency decay. Demote centroid index to optional approximate mode.
- **M3 Lifecycle**: `forget`, `supersede`.
- **M4 Bench/docs**: update benchmark with hybrid, document honestly.
- Later: structured memory cards (pluggable LLM/rule extractor), adaptive cutoff, ANN (hnswlib)
  for >1M frames, ingestion adapters, encryption, MCP server wrapper.

## Non-goals
Re-implementing Memvid's multimodal ingestion, PDF/audio, or Rust performance.
