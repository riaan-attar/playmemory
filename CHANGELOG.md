# Changelog

## 0.4.0
Search quality
- Unicode-aware tokenizer (NFKC, casefold, Latin accent folding, CJK/Hangul bigrams); old checkpoints are re-indexed.
- ONNX cross-encoder reranker (+16 points evidence recall at a 500-char budget on LoCoMo), adaptive cutoff
  (elbow/ratio/cliff/min), natural-language date filters (`when=`), `ask()` with citations (Claude / OpenAI-compatible).
- Spans: scenes ranked by best hit, `k` scenes per budget, spans honour `when`/ACL.
Documents and media
- `ingest()` for txt/md/html/csv/json/jsonl and pdf/docx/xlsx/pptx (extra `docs`): heading-aware chunking, tables kept
  together, idempotent by content hash; `--defer-embedding` + `enrich()`; audio transcription and CLIP images (experimental).
Security and operations
- AES-256-GCM encryption (extra `crypto`), HMAC signatures, tenant/role ACLs in the single visibility gate, query-time PII masking.
- Entity graph + predicate schema, session recording and deterministic replay, `doctor` repair, reader `refresh()`.
Format and portability
- File format v3 (embedder spec + flags in the header, frame ids, zlib records, float16 vectors, vector-update/event/signature
  records; v2 files still open); Windows-safe I/O; a failed open no longer leaves the file locked.
- Reopen any file without flags; `--api-key` / `PLAYMEMORY_API_KEY`; `sync_every=100`; `ann_ef_search`.
- Removed the centroid `SegmentIndex`; `Memory()` without an embedder is an error (`"debug"` for tests).
Benchmarks: LoCoMo budget/rerank study, head-to-head with memvid-sdk on identical vectors, scale test, LLM-judged QA harness.

## 0.3.0
- Single-file CRC-checked format with embeddings, tombstones and index checkpoints inside.
- Durability: exclusive writer lock, lock-free read-only openers, fsync modes, atomic `compact()`.
- Hybrid BM25 + vector search (RRF), `as_of` time travel, `forget`/`supersede`, `where=` filters.
- Optional persisted HNSW (faiss) for large memories.
- Episodes (scenes) and `recall_span()`: contiguous timeline spans with start/end pointers.
- Memory cards with auto-superseding facts; MCP stdio server; `playmemory` CLI.
- Embedders: torch-free ONNX (BGE-small), OpenAI-compatible HTTP, sentence-transformers.
- LoCoMo benchmarks (`bench/quality.py`, `bench/budget.py`); results in `docs/RESULTS.md`.
