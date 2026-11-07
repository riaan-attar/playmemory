# Changelog

## 0.4.0 (in progress)
- Search: Unicode-aware tokenizer (NFKC, casefold, Latin accent folding, CJK/Hangul bigrams); old checkpoints are re-indexed.
- Portability: pread/pwrite shim with short-write handling, Windows-safe lock offset; a failed open no longer leaves the file locked.
- File format v3: header stores the embedder spec and flags; v2 files still open. Frames carry their id, long records
  are zlib-compressed when it helps, optional float16 vectors, empty metadata is not materialised.
- Reopen any file without flags (embedder rebuilt from the header); `--api-key` / `PLAYMEMORY_API_KEY` for HTTP embedders.
- `refresh()` / `auto_refresh` for read-only readers (survives writer compaction); `sync_every=100` default.
- `playmemory FILE doctor`: salvage intact records after damage anywhere in the file.
- Removed the centroid `SegmentIndex`; `Memory()` without an embedder is now an error (explicit `"debug"` for tests).

## 0.3.0
- Single-file CRC-checked format with embeddings, tombstones and index checkpoints inside.
- Durability: exclusive writer lock, lock-free read-only openers, fsync modes, atomic `compact()`.
- Hybrid BM25 + vector search (RRF), `as_of` time travel, `forget`/`supersede`, `where=` filters.
- Optional persisted HNSW (faiss) for large memories.
- Episodes (scenes) and `recall_span()`: contiguous timeline spans with start/end pointers.
- Memory cards with auto-superseding facts; MCP stdio server; `playmemory` CLI.
- Embedders: torch-free ONNX (BGE-small), OpenAI-compatible HTTP, sentence-transformers.
- LoCoMo benchmarks (`bench/quality.py`, `bench/budget.py`); results in `docs/RESULTS.md`.
