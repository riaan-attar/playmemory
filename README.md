# PlayMemory

Lightweight, single-file memory for AI agents. Memories live on a **timeline**; recall returns
the right *stretch* of it ("watch from 1:30 to 2:10"), not just scattered chunks.

- **One file**, CRC-checked, crash-safe tail, one writer + many lock-free readers
- **Hybrid search**: BM25 + vectors (RRF), metadata filters, recency decay
- **Time travel**: `as_of=t` sees the memory (and what had been forgotten) at that time
- **Scenes**: `recall_span()` returns a contiguous span with `start_id`/`end_id` pointers
- **Cards**: facts like "lives in Paris" auto-supersede when they change
- **MCP server** + CLI; core install is just `numpy`

```bash
pip install playmemory              # core (numpy only)
pip install "playmemory[onnx]"      # + local BGE-small embedder, no torch
pip install "playmemory[all]"       # + HNSW for huge memories
```

## Python
```python
from playmemory import Memory, OnnxEmbedder

with Memory("agent.pmem", OnnxEmbedder()) as m:          # use a real embedder in production
    m.add("I'm vegetarian and I'm moving to Kyoto in April", t=1.0)
    m.add("Kyoto temples serve shojin ryori", t=2.0)

    m.search("what can the user eat?", k=3, context=1)    # hits + neighbouring turns
    m.search("where does the user live?", as_of=1.5)      # time travel
    for sp in m.recall_span("Kyoto trip", k=1, budget_chars=2000):
        print(sp.start_id, sp.end_id, sp.summary)         # pointer ...
        print(sp.text())                                  # ... and the text
    m.read(0, 5)                                          # "press play" on a pointer
```
Facts that change: `from playmemory.cards import remember, profile`.

## CLI
```bash
playmemory agent.pmem add "Alice lives in Paris"
playmemory agent.pmem search "where is alice" -k 3 --context 1
playmemory agent.pmem story  "alice's move" --budget 2000
playmemory agent.pmem stats | verify | compact | checkpoint | export
```

## MCP (Claude Desktop / Claude Code / Cursor)
```json
{ "mcpServers": { "memory": { "command": "playmemory", "args": ["/path/agent.pmem", "serve"] } } }
```
Tools: `remember`, `recall`, `recall_story`, `read_span`, `forget`, `seek`, `timeline`, `profile`, `stats`.

## Honest status
Measured on LoCoMo (`docs/RESULTS.md`): hybrid search clearly beats BM25-only or vector-only.
Span retrieval matches top-k chunks at small budgets but does **not** beat them on evidence
recall at larger ones; whether scenes improve final answers is untested. The default
`HashEmbedder` is non-semantic (tests only). Compared with Memvid (Rust, multimodal, encryption,
graph) this is a smaller Python engine; see `docs/ROADMAP.md`.

Node.js, npm CLI and Rust packages are planned after the Python package.
