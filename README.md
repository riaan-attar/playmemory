# PlayMemory

Timeline ("video-like") memory layer for AI systems: append frames, then `search`
(with temporal context), `seek(t)`, or `window(t0, t1)`.

```python
from playmemory import Memory
m = Memory("mem.pmem")          # or Memory() for in-memory
m.add("user likes Kyoto", t=1.0)
m.search("travel", k=3, context=1)
m.seek(1.5); m.window(0, 2)
old = m.add("alice lives in paris", t=2.0)
m.supersede(old.id, "alice lives in berlin", t=5.0)
m.search("where does alice live")             # -> berlin
m.search("where does alice live", as_of=3.0)  # -> paris (time travel)
m.close()
```

Features: single file (CRC-checked, crash-safe tail), hybrid BM25 + vector search (RRF),
temporal context per hit, `where=` meta filters, recency decay, `forget`/`supersede`,
`as_of` time travel, embedder binding. See [docs/ROADMAP.md](docs/ROADMAP.md) for how this
compares with Memvid.

Setup: `python -m venv .venv && .venv/bin/pip install -e '.[dev,bench]'`
Test: `pytest`. Benchmark: `python -m bench.run`. Findings: [docs/DESIGN.md](docs/DESIGN.md).
