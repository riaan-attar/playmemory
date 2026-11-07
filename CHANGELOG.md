# Changelog

## 0.3.0
- Single-file CRC-checked format with embeddings, tombstones and index checkpoints inside.
- Durability: exclusive writer lock, lock-free read-only openers, fsync modes, atomic `compact()`.
- Hybrid BM25 + vector search (RRF), `as_of` time travel, `forget`/`supersede`, `where=` filters.
- Optional persisted HNSW (faiss) for large memories.
- Episodes (scenes) and `recall_span()`: contiguous timeline spans with start/end pointers.
- Memory cards with auto-superseding facts; MCP stdio server; `playmemory` CLI.
- Embedders: torch-free ONNX (BGE-small), OpenAI-compatible HTTP, sentence-transformers.
- LoCoMo benchmarks (`bench/quality.py`, `bench/budget.py`); results in `docs/RESULTS.md`.
