# Graph Report - playmemory  (2026-11-07)

## Corpus Check
- Corpus is ~15,056 words - fits in a single context window. You may not need a graph.

## Summary
- 395 nodes · 902 edges · 13 communities (8 shown, 5 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 136 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Agent Log & Storage IO
- Memory Cards
- Memory Core Lifecycle
- Frames & Hits
- Memory Tests
- Budget Benchmark
- Episode Segmentation
- Baseline RAG Bench
- Design Docs & Findings
- BM25 Lexical Index
- ANN Index
- Package Root

## God Nodes (most connected - your core abstractions)
1. `Memory` - 98 edges
2. `FrameStore` - 37 edges
3. `Frame` - 22 edges
4. `run()` - 17 edges
5. `remember()` - 14 edges
6. `EpisodeConfig` - 14 edges
7. `SegmentIndex` - 14 edges
8. `MemoryServer` - 14 edges
9. `EpisodeIndex` - 13 edges
10. `from_spec()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `Memory`  [INFERRED]
  bench/budget.py → src/playmemory/memory.py
- `main()` --calls--> `from_spec()`  [INFERRED]
  bench/budget.py → src/playmemory/embed.py
- `main()` --calls--> `EpisodeConfig`  [INFERRED]
  bench/budget.py → src/playmemory/episodes.py
- `evaluate()` --calls--> `Memory`  [INFERRED]
  bench/quality.py → src/playmemory/memory.py
- `main()` --calls--> `from_spec()`  [INFERRED]
  bench/quality.py → src/playmemory/embed.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Roadmap milestones M1-M4** — docs_roadmap_m1_format_v2, docs_roadmap_m2_hybrid, docs_roadmap_m3_lifecycle, docs_roadmap_m4_bench_docs [EXTRACTED 1.00]
- **Hybrid retrieval components** — readme_hybrid_search, docs_design_exact_search, docs_design_hybrid_default [INFERRED 0.75]

## Communities (13 total, 5 thin omitted)

### Community 0 - "Agent Log & Storage IO"
Cohesion: 0.06
Nodes (12): Embedder, from_spec(), HashEmbedder, HTTPEmbedder, OnnxEmbedder, SentenceTransformerEmbedder, Span, CorruptFileError (+4 more)

### Community 1 - "Memory Cards"
Cohesion: 0.08
Nodes (20): Card, Extractor, LLMExtractor, profile(), remember(), RuleExtractor, _frame(), MemoryServer (+12 more)

### Community 3 - "Frames & Hits"
Cohesion: 0.08
Nodes (4): Frame, Hit, _locked(), wrapper()

### Community 4 - "Memory Tests"
Cohesion: 0.10
Nodes (33): Memory, test_ann_persist_incremental_and_recall(), test_compact_rebuilds_ann(), test_lexical_scores_rank_rare_terms_higher(), test_where_filter_index_matches_scan(), _texts(), _fill(), test_checkpoint_roundtrip_and_tail_replay() (+25 more)

### Community 5 - "Budget Benchmark"
Cohesion: 0.12
Nodes (16): CachedEmbedder, main(), pack_chunks(), run(), evaluate(), load_turns(), main(), _frame_dict() (+8 more)

### Community 6 - "Episode Segmentation"
Cohesion: 0.11
Nodes (17): EpisodeConfig, EpisodeIndex, _forced(), segment(), _valleys(), _window_sims(), build(), stream() (+9 more)

### Community 7 - "Baseline RAG Bench"
Cohesion: 0.11
Nodes (7): FlatBaseline, HNSWBaseline, make_texts(), pct(), run(), timed(), SegmentIndex

### Community 8 - "Design Docs & Findings"
Cohesion: 0.11
Nodes (22): Benchmark results (bench.run), Centroid segment index (approx mode), PlayMemory design and findings, Exact numpy vector search, HNSW / faiss-hnsw, Hybrid retrieval as default (v0.2), Embedder binding / ModelMismatchError, Timeline memory model (segments, keyframes) (+14 more)

## Knowledge Gaps
- **5 isolated node(s):** `playmemory`, `Memory API (add, search, seek, window)`, `Embedder binding / ModelMismatchError`, `M4 Bench and docs`, `Non-goals`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 100 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Memory` connect `Memory Tests` to `Agent Log & Storage IO`, `Memory Cards`, `Memory Core Lifecycle`, `Frames & Hits`, `Budget Benchmark`, `Episode Segmentation`, `Baseline RAG Bench`, `BM25 Lexical Index`, `ANN Index`?**
  _High betweenness centrality (0.453) - this node is a cross-community bridge._
- **Are the 56 inferred relationships involving `Memory` (e.g. with `run()` and `evaluate()`) actually correct?**
  _`Memory` has 56 INFERRED edges - model-reasoned connections that need verification._
- **What connects `playmemory`, `Memory API (add, search, seek, window)`, `Embedder binding / ModelMismatchError` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Agent Log & Storage IO` be split into smaller, more focused modules?**
  _Cohesion score 0.061952861952861954 - nodes in this community are weakly interconnected._
- **Why does `FrameStore` connect `Memory Core Lifecycle` to `Agent Log & Storage IO`, `Frames & Hits`, `Memory Tests`, `Budget Benchmark`, `Baseline RAG Bench`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `FrameStore` (e.g. with `Memory` and `Frame`) actually correct?**
  _`FrameStore` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Memory Cards` be split into smaller, more focused modules?**
  _Cohesion score 0.07890070921985816 - nodes in this community are weakly interconnected._