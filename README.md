# PlayMemory

Lightweight, single-file memory for AI agents. Memories live on a **timeline**, and recall can
return the right *stretch* of it ("watch from 1:30 to 2:10"), not just scattered chunks.

- **One file.** Frames, embeddings, tombstones and index checkpoints live in a single
  CRC-checked `.pmem` file with a crash-safe tail.
- **One writer, many readers.** An exclusive lock for the writer; any number of lock-free
  `read_only` openers. Thread-safe inside a process.
- **Hybrid search.** BM25 plus vectors fused with RRF, exact-match metadata filters, optional
  recency decay.
- **Time travel.** `as_of=t` shows the memory as it was at time `t`, including what had not
  yet been forgotten.
- **Scenes.** `recall_span()` returns a contiguous span of the timeline with `start_id` /
  `end_id` pointers you can expand later with `read()`.
- **Cards.** Facts such as "lives in Paris" supersede the old value automatically when they change.
- **MCP server and CLI.** The core install depends only on `numpy`.

> Status: beta (0.3.0), Python only. See [Honest status](#honest-status) for what has and has
> not been measured.

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Core concepts](#core-concepts)
4. [Python API](#python-api)
5. [Embedders](#embedders)
6. [Memory cards](#memory-cards)
7. [Ingesting documents](#ingesting-documents)
8. [Rerank, cutoff, dates and `ask`](#rerank-cutoff-dates-and-ask)
9. [Security: encryption, signatures, ACL, PII](#security)
10. [Entity graph and schema](#entity-graph-and-schema)
11. [Sessions and replay](#sessions-and-replay)
12. [Progressive enrichment](#progressive-enrichment)
13. [CLI](#cli)
14. [MCP server](#mcp-server)
15. [Durability and concurrency](#durability-and-concurrency)
16. [Scaling](#scaling)
17. [Benchmarks](#benchmarks)
18. [Honest status](#honest-status)
19. [Project layout](#project-layout)
20. [Development](#development)
21. [Roadmap](#roadmap)

## Install

```bash
pip install playmemory              # core: numpy only
pip install "playmemory[onnx]"      # + local BGE-small embedder (onnxruntime), no torch
pip install "playmemory[st]"        # + sentence-transformers
pip install "playmemory[ann]"       # + faiss HNSW for very large memories
pip install "playmemory[docs]"      # + PDF / DOCX / XLSX / PPTX ingestion
pip install "playmemory[crypto]"    # + AES-256-GCM encryption
pip install "playmemory[audio]"     # + faster-whisper transcription (experimental)
pip install "playmemory[images]"    # + CLIP image frames (experimental)
pip install "playmemory[ner]"       # + spaCy entity co-occurrence extractor (experimental)
pip install "playmemory[all]"       # onnx + ann + docs + crypto
```

Requires Python 3.10 or newer. Node.js, npm CLI and Rust packages are not available; they are
only planned (see [Roadmap](#roadmap)).

## Quick start

```python
from playmemory import Memory, OnnxEmbedder

with Memory("agent.pmem", OnnxEmbedder()) as m:
    m.add("I'm vegetarian and I'm moving to Kyoto in April", t=1.0)
    m.add("Kyoto temples serve shojin ryori", t=2.0)

    # hits plus one neighbouring turn on each side
    for hit in m.search("what can the user eat?", k=3, context=1):
        print(hit.score, hit.frame.text)

    # the same question, as the memory stood at t=1.5
    m.search("where does the user live?", as_of=1.5)

    # a contiguous span of the timeline, as a pointer plus text
    for sp in m.recall_span("Kyoto trip", k=1, budget_chars=2000):
        print(sp.start_id, sp.end_id, sp.summary)
        print(sp.text())

    m.read(0, 5)     # "press play" on a pointer: frames 0..5 inclusive
```

`Memory` requires an embedder: pass an object, a spec string (`"onnx"`, `"st:MODEL"`, `"http:URL|MODEL"`), or reopen an
existing file, which rebuilds the embedder it was created with from the spec stored in its header (API keys are never
stored; use `--api-key` or `PLAYMEMORY_API_KEY`). `"debug"` (or `PLAYMEMORY_ALLOW_DEBUG_EMBEDDER=1`) selects a lexical,
non-semantic embedder for tests only.

## Core concepts

| Concept | Meaning |
|---|---|
| **Frame** | One memory unit on the timeline: `id`, timestamp `t`, `text`, JSON `meta`. Ids are assigned in append order. |
| **Hit** | A search result: the matched frame, its score, and optionally its temporal neighbours. |
| **Span** | A contiguous run of frames `start_id..end_id` with the text that fits a character budget. |
| **Episode (scene)** | A group of adjacent frames found by a similarity heuristic. Forced boundaries come from a metadata key (default `session`) or a time gap. |
| **Tombstone** | `forget()` and `supersede()` never rewrite history. They record a deletion time, so `as_of` queries still see the old state. `compact()` can drop it for good. |
| **Checkpoint** | A saved copy of the BM25 and episode indexes (and HNSW, if built) inside the file, so reopening a large memory does not rebuild them. |

## Python API

### `Memory(path=None, embedder=None, ...)`

| Parameter | Default | Notes |
|---|---|---|
| `path` | `None` | `None` keeps everything in RAM. |
| `embedder` | required for new files | An object with `embed(list[str]) -> ndarray`, or a spec string. The file is bound to the embedder that created it; opening with a different one raises `ModelMismatchError`. |
| `vec_dtype` | `"float32"` | `"float16"` stores vectors at half the size (new files only; recorded in the header). |
| `auto_refresh` | `False` | With `read_only=True`, every call first picks up what a writer process has appended. `refresh()` does it on demand. |
| `ann_threshold` | `200_000` | Frame count at which the HNSW index is used, if faiss is installed. |
| `episode_config` | `EpisodeConfig()` | Scene segmentation settings. |
| `read_only` | `False` | Open without taking the writer lock. |
| `sync` | `"batch"` | `"always"` fsyncs every write, `"batch"` on commit and close, `"never"` leaves it to the OS. |
| `sync_every` | `100` | With `"batch"`, fsync every N writes, so a power loss can lose at most about N records. |
| `max_text_bytes`, `max_meta_bytes` | module limits | Per-frame size caps. |

### Writing

```python
m.add(text, meta=None, t=None)            # -> Frame
m.add_many(texts, metas=None, times=None) # bulk append -> list[Frame]
m.forget(fid, t=None)                     # tombstone a frame
m.supersede(old_id, text, meta=None, ...) # forget old_id and add its replacement
m.commit()                                # flush and fsync (per sync mode)
m.checkpoint()                            # persist indexes for fast reopen
m.build_ann()                             # build the persisted HNSW index
m.compact(keep_history=False)             # atomic rewrite; returns counts
```

### Reading

```python
m.search(query, k=5, context=0, *,
         mode="hybrid",                   # hybrid | vector | lexical
         as_of=None,                      # time travel
         where=None,                      # exact-match filter on meta, e.g. {"user": "ana"}
         recency_half_life=None,          # score *= 0.5 ** (age / half_life), seconds
         pool=50, exact=False)            # -> list[Hit]

m.recall_span(query, k=1, budget_chars=4000, *,
              mode="hybrid", as_of=None, where=None, pool=50)   # -> list[Span]

m.read(start_id, end_id, as_of=None)      # frames start..end inclusive
m.get(fid)                                # one frame
m.find(where, limit=None, ...)            # metadata-only lookup
m.seek(t)                                 # frame at or before time t
m.window(t0, t1)                          # frames in a time range
m.episode_list()                          # [(start_id, end_id), ...]
m.stats()                                 # sizes, counts, embedder, index state
```

### How `recall_span` works

Episodes are ranked by their best-matching frame. Inside the chosen episode, the span is
grown around the best frames toward whichever neighbour is more relevant until the character
budget is spent. With `k > 1` the budget is shared between spans.

### Errors

| Exception | Raised when |
|---|---|
| `LockedError` | Another process holds the writer lock. |
| `ModelMismatchError` | The file was written with a different embedder. |
| `CorruptFileError` | A CRC check fails outside the recoverable tail. |

## Embedders

| Class | Spec string | Needs |
|---|---|---|
| `HashEmbedder(dim=256)` | `hash` or `hash-DIM` | nothing (non-semantic, for tests) |
| `OnnxEmbedder(repo=...)` | `onnx` or `onnx:REPO` | `playmemory[onnx]`; BGE-small by default |
| `SentenceTransformerEmbedder(model)` | `st` or `st:MODEL` | `playmemory[st]` |
| `HTTPEmbedder(url, model, key)` | `http:URL\|MODEL[\|APIKEY]` | any OpenAI-compatible endpoint |

`from_spec("onnx")` builds one from a string, which is what the CLI `--embedder` flag uses.
Custom embedders only need an `embed(texts) -> np.ndarray` method that returns a 2-D float array.

## Memory cards

Cards turn free text into structured facts that live in the same file as ordinary frames
(`meta = {"kind": "card", ...}`). `remember()` stores the raw text as a frame too and returns
`(raw_frame, card_frames)`.

```python
from playmemory import Memory
from playmemory.cards import remember, profile

with Memory("agent.pmem", embedder) as m:
    remember(m, "I live in Paris", t=1.0)
    remember(m, "I moved to Kyoto", t=5.0)     # supersedes lives_in=Paris

    profile(m)                                 # {"lives_in": ["Kyoto"], ...}
    profile(m, as_of=2.0)                      # {"lives_in": ["Paris"], ...}
```

Kinds are `fact`, `preference`, `event`, `profile`, `relationship`, `goal` and `other`. Facts,
profile entries and relationships are *exclusive slots*: a new value replaces the old one
through `supersede`, so current searches see only the latest while `as_of` still answers "what
did we believe then?".

Extraction is pluggable. `RuleExtractor` is the default (regexes for name, residence, employer,
diet, likes and dislikes, goals, relationships). `LLMExtractor(fn)` wraps a callable returning
rows of `kind/subject/predicate/value[/exclusive]` (for example an LLM call that returns JSON), and any object with `extract(text) -> list[Card]` works.

## Ingesting documents

```python
rep = m.ingest("notes/")                      # file or directory (recursive; dot-directories skipped)
rep = m.ingest("report.pdf", max_chars=800, overlap=100, defer_embedding=False)
rep.files, rep.chunks, rep.skipped_unchanged, rep.replaced, rep.unsupported, rep.failed
```

Text, Markdown, HTML, CSV/TSV and JSON/JSONL need nothing extra; PDF, DOCX, XLSX and PPTX need
`playmemory[docs]`. The chunker is heading-aware (each chunk carries `heading`, `section`, `page`,
`offset`, `source`), keeps tables together and repeats the header row when it must split one, splits
long paragraphs on sentences, and sets `meta.session` per section so every heading becomes its own
scene for `recall_span`. Ingest is idempotent: an unchanged file (same SHA-256) is skipped; a changed
file has its old chunks forgotten first (history stays reachable with `as_of`), or pass
`on_change="append"`. One unreadable file is reported in `rep.failed` and never aborts a directory.

Audio/video: `m.ingest("talk.mp3", transcriber=WhisperTranscriber("base"))` (extra `audio`) stores one
frame per segment with `meta.start`/`meta.end`, so a span is a literal "watch 1:30 to 2:10".
Images: use `ClipEmbedder` as the embedder and `m.add_images(paths, captions)` (extra `images`).
Both are experimental and only tested with fakes in CI.

## Rerank, cutoff, dates and `ask`

```python
from playmemory import OnnxCrossEncoder, ClaudeSynthesizer

m.search("where does alice work", k=5, rerank=OnnxCrossEncoder())      # rescore the top 30 (extra onnx)
m.search("alice", k=20, mode="vector", cutoff=("ratio", 0.5))          # adaptive result count
m.search("what did we decide last week", when="auto")                  # "yesterday", "last Tuesday", "in March 2023"...
ans = m.ask("where does alice live?", ClaudeSynthesizer(), k=8)        # answer + citations
ans.text; ans.citations            # frames the answer cited as [12]; ans.context = everything shown to the model
```

`cutoff` takes `"elbow"`, `("ratio", x)`, `("cliff", x)` or `("min", x)`. `when` accepts a `(t0, t1)`
tuple, a phrase, or `"auto"`; spans stay inside the window. `ask(spans=True)` feeds whole scenes instead
of top-k hits. Synthesizers are any `fn(question, context) -> str`; `ClaudeSynthesizer`
(Anthropic Messages API, `ANTHROPIC_API_KEY`) and `OpenAICompatSynthesizer` use plain HTTP.

## Security

- **Encryption** (`playmemory[crypto]`): `Memory(path, embedder, password=...)`. Every record payload is
  AES-256-GCM sealed under a scrypt-derived key; the record's offset is authenticated data, so tampering
  or swapping records is detected. Frames, tombstones, checkpoints, ANN graphs, events and vector
  updates are all sealed; only the header (embedder spec, dimension) is readable. CLI: `--password-file`
  or `PLAYMEMORY_PASSWORD`. A wrong password raises `WrongPasswordError`; none, `EncryptedFileError`.
- **Signatures**: `m.sign(key)` appends an HMAC-SHA256 over everything before it;
  `m.verify_signature(key)` returns `(valid, unsigned_trailing_bytes)`.
- **ACL**: reserved meta keys `tenant`, `visibility` (`public`/`restricted`), `roles`, `groups`. Give the
  handle (or a call) a `Principal(tenant="acme", roles={"eng"})` and every read path - search, spans,
  neighbour context, `get`, `seek`, `window`, `read`, `find`, `ask` - honours it. No principal = admin view.
  `serve --tenant acme --role eng` runs the MCP server as that principal.
- **PII masking**: `Memory(redact=True)` (or your own `fn(text) -> text`) masks emails, SSNs, IPs,
  Luhn-valid card numbers and phone numbers in returned text; stored text is untouched. Regex-based, not a guarantee.

## Entity graph and schema

```python
from playmemory.graph import Graph, Schema
from playmemory.cards import remember

schema = Schema().declare("lives_in", "one", "entity").declare("likes", "many")
remember(m, "Alice lives in Paris and loves tea", schema=schema)
g = Graph(m, schema)
g.triples(subject="alice"); g.neighbors("alice"); g.related("alice", hops=2)   # time-aware: as_of= on each
m.search("tell me about Alice", graph=g)     # frames about entities named in the query join the fusion
```

The graph is a view over memory cards, so it stays consistent with `supersede`, `forget` and `as_of`.
The schema validates value types (`int`, `float`, `date`, `bool`, `entity`), allowed card kinds, and
decides one-vs-many cardinality. `NerExtractor` (spaCy, extra `ner`) is experimental.

## Sessions and replay

```python
with m.session("tuning run") as sid:
    m.search("topic1 word3", k=5)
    ...
m.sessions(); m.events(sid)
diffs = m.replay(sid, mode="vector", rerank=my_reranker)    # same questions, other settings
for d in diffs: print(d.event["p"]["query"], d.same, d.added, d.removed, d.reordered)
```

Inside a session every search, span, add and forget is logged in the file (sealed if the file is
encrypted) with its parameters, results, principal and the frame/tombstone counts at that moment.
`replay` re-runs the reads against the memory *as it was then*, even after later writes, so it
reproduces the original results exactly and shows what a different setting would have changed.
`compact(keep_history=True)` keeps the trail; plain `compact()` drops it because ids change.

## Progressive enrichment

`m.add_many(texts, defer_embedding=True)` (or `ingest(..., defer_embedding=True)` / `--defer-embedding`)
stores frames at once - keyword-searchable immediately - and `m.enrich()` / `playmemory FILE enrich`
embeds them in batches later. Enriched vectors are append-only update records; they survive reopen,
checkpoint, compaction, encryption, float16 and `doctor`, and reader handles pick them up on `refresh()`.

## CLI

```bash
playmemory agent.pmem add "Alice lives in Paris"
playmemory agent.pmem add --meta '{"session": 3}' --t 12.5 "Moved to Kyoto"
echo "line one" | playmemory agent.pmem add          # one frame per stdin line

playmemory agent.pmem search "where is alice" -k 3 --context 1
playmemory agent.pmem search "alice" --mode lexical --as-of 10 --where '{"session": 3}'
playmemory agent.pmem story  "alice's move" --budget 2000 [--pointers]
playmemory agent.pmem read 0 5
playmemory agent.pmem seek 12.5
playmemory agent.pmem window 10 20
playmemory agent.pmem forget 7
playmemory agent.pmem stats | verify | checkpoint | export
playmemory agent.pmem doctor [--write [--replace]]   # salvage intact records after mid-file damage
playmemory agent.pmem ingest docs/ report.pdf [--defer-embedding] [--transcribe base]
playmemory agent.pmem enrich                         # embed frames ingested with --defer-embedding
playmemory agent.pmem search "standup" --when "last week" --rerank --cutoff ratio:0.5
playmemory agent.pmem ask "where does alice live?" --llm claude[:MODEL] [--spans]
playmemory agent.pmem compact [--keep-history]
playmemory agent.pmem serve                          # MCP over stdio
```

Output is one JSON document per result, so it pipes into `jq`. `--embedder SPEC` overrides the
embedder; by default the CLI reads the one recorded in the file header. Read-only commands
(`search`, `story`, `read`, `seek`, `window`, `stats`, `export`) open the file without taking the writer lock.

## MCP server

A dependency-free stdio server, so it works with Claude Desktop, Claude Code, Cursor and any
other MCP client:

```json
{
  "mcpServers": {
    "memory": { "command": "playmemory", "args": ["/path/agent.pmem", "serve"] }
  }
}
```

Tools: `remember`, `recall` (with `when`), `recall_story`, `read_span`, `forget`, `seek`, `timeline`,
`profile`, `stats`, and `ingest` only when started with `--ingest-root DIR` (paths outside the roots are refused).
`serve --tenant T --role R --redact` runs it as an ACL principal with PII masking.

## Durability and concurrency

- Records are CRC-checked. A torn write at the tail is detected and ignored on open, so a crash
  loses at most the uncommitted tail.
- `compact()` writes a new file and swaps it in atomically.
- One writer at a time (`LockedError` otherwise). Readers never block and never lock.
- `playmemory FILE verify` re-checks every record; `doctor` resynchronises past damage anywhere in the file and writes
  a salvaged copy (`--write`, or `--replace` to keep the original as `FILE.corrupt`). Lost frames shift ids, so
  tombstones and `supersedes`/`source` pointers are remapped.
- Size limits on text and metadata guard against runaway frames.

## Scaling

| Frames | Search path |
|---|---|
| up to ~200k | Exact vectorised numpy scan plus BM25. |
| above `ann_threshold` | Persisted HNSW via faiss (`playmemory[ann]`), rebuilt with `build_ann()` and saved by `checkpoint()`. |

Measured (`python -m bench.scale`, synthetic clustered vectors, 16-core CPU): at 500k frames x 64-d,
adds run at about 29k/s, opening a checkpointed file takes 3.1 s, an exact vector scan takes 27 ms,
BM25 1.2 ms and the full hybrid API 2.7 ms with HNSW. At 200k x 128-d: exact 9.9 ms, hybrid 1.4 ms.
HNSW recall depends on the data and on `ann_ef_search` (default 128): on this deliberately noisy
synthetic data recall@10 climbs from 0.35 (ef=64) to 0.74 (ef=512) at about 1 ms, while on the
clustered text benchmark at 100k it was 0.996. Peak RSS was 1.7 GB at 500k. 1M frames was not run
(it would not fit comfortably in the 15 GB test machine's free memory).

The older centroid segment index was removed: it was slower than HNSW at equal recall and
collapsed when the data was not temporally clustered (`docs/DESIGN.md`).

## Benchmarks

Everything is reproducible from this repo. LoCoMo, retrieval only, BGE-small, CPU:

| System | evidence-recall@1 | @5 | @10 |
|---|---|---|---|
| BM25 only | 0.262 | 0.464 | 0.537 |
| Vector only | 0.267 | 0.496 | 0.594 |
| **Hybrid (RRF)** | 0.309 | 0.546 | 0.638 |
| Hybrid + 1 neighbour each side | 0.388 | 0.661 | 0.749 |

```bash
python -m bench.quality --data locomo10.json --embedder onnx   # chunk baselines
python -m bench.budget  --data locomo10.json --embedder onnx   # spans vs chunks at equal size
python -m bench.run                                            # scale and speed
```

Full tables, per-category numbers and caveats are in [docs/RESULTS.md](docs/RESULTS.md).

## Honest status

- Hybrid search clearly beats BM25-only or vector-only on LoCoMo; the ONNX cross-encoder reranker is
  the largest single gain (+16 points of evidence recall at a 500-character budget).
- Span retrieval **matches** top-k chunks at small budgets but does **not** beat them on evidence recall
  overall (it wins only on multi-hop at the smallest budget). Whether contiguous scenes improve *final
  answers* is untested: there is no LLM-judged QA benchmark yet (it needs an API key).
- Head-to-head with `memvid-sdk` 2.0.160 on identical vectors: PlayMemory hybrid 0.638 evidence-recall@10
  vs 0.233 for Memvid's default hybrid mode, with comparable search speed and smaller files. I could not
  explain Memvid's low recall (it returns only 37% of the exact top-10), so treat it as "defaults through
  the Python SDK", not a verdict on Memvid; details and caveats in [docs/RESULTS.md](docs/RESULTS.md).
- The metric is retrieval only, on one dataset and one embedder.
- Experimental, tested only with fakes in CI: audio transcription, CLIP images, spaCy NER.
- Not done: Node.js / npm / Rust packages, an HTTP server, and a "sketch" candidate prefilter (an exact
  scan costs 27 ms at 500k frames and HNSW answers in about 1 ms, so it was not worth the complexity).
- `HashEmbedder` (`"debug"`) is non-semantic; production code must pass a real embedder.

## Project layout

```
src/playmemory/
  memory.py     Memory: public API, search, spans, lifecycle
  store.py      FrameStore: single-file format, locking, recovery
  lexical.py    BM25Index (numpy)
  ann.py        AnnIndex (faiss HNSW, persisted)
  index.py      VectorMatrix (growing embedding matrix)
  doctor.py     diagnose/salvage a damaged file
  ingest/       chunker, loaders (text/html/csv/json/pdf/docx/xlsx/pptx), audio
  crypto.py     scrypt + AES-256-GCM, HMAC
  acl.py pii.py tenant/role access control, PII masking
  graph.py      entity graph + predicate schema
  rerank.py cutoff.py temporal.py ask.py   retrieval extras
  episodes.py   scene segmentation and EpisodeIndex
  cards.py      memory cards, extractors, remember(), profile()
  embed.py      embedders and from_spec()
  frame.py      Frame, Hit, Span
  server.py     MCP stdio server
  cli.py        playmemory command
bench/          LoCoMo and scale benchmarks, baselines
docs/           DESIGN.md, RESULTS.md, ROADMAP.md
examples/       agent_log.py (conversation log as timeline memory)
tests/          pytest suite
```

## Development

```bash
git clone git@github.com:riaan-attar/playmemory.git && cd playmemory
pip install -e ".[dev,all]"
pytest
python -m build          # sdist and wheel into dist/
```

## Roadmap

Planned after the Python package: an npm CLI, a Node.js SDK and a Rust crate. None of them exist
yet. Other open items, including an LLM-judged QA evaluation and a head-to-head against Memvid,
are in [docs/ROADMAP.md](docs/ROADMAP.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
