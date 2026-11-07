from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Block, Chunk, chunk_blocks
from .loaders import AUDIO_EXT, SUPPORTED, MissingDependency, load

__all__ = ["Block", "Chunk", "chunk_blocks", "load", "ingest_paths", "IngestReport", "SUPPORTED"]


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    skipped_unchanged: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    frame_ids: list[int] = field(default_factory=list)


def _files(paths, recursive: bool):
    for p in map(Path, paths):
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            yield from sorted(f for f in it if f.is_file() and not any(part.startswith(".") for part in f.relative_to(p).parts))
        else:
            yield p


def ingest_paths(mem, paths, *, max_chars: int = 800, overlap: int = 100, recursive: bool = True,
                 meta: dict | None = None, t: float | None = None, on_change: str = "replace",
                 transcriber=None, defer_embedding: bool = False) -> IngestReport:
    """Chunk and add files (or directories) to `mem`. Idempotent: an unchanged file
    (same content hash) is skipped; a changed one has its old chunks forgotten first
    (on_change="replace") or is added again (on_change="append")."""
    rep = IngestReport()
    base_t = time.time() if t is None else t
    base_t = max(base_t, mem.store.times[-1] if len(mem) else 0.0)  # timeline never goes backwards
    step = 0
    for f in _files([paths] if isinstance(paths, (str, Path)) else paths, recursive):
        ext = f.suffix.lower()
        if ext in AUDIO_EXT:
            if transcriber is None:
                rep.unsupported.append(str(f))
                continue
        elif ext not in SUPPORTED:
            rep.unsupported.append(str(f))
            continue
        try:
            src = str(f.resolve())
            sha = hashlib.sha256(f.read_bytes()).hexdigest()
            if mem.find({"source": src, "file_sha": sha}, limit=1):
                rep.skipped_unchanged.append(src)
                continue
            if ext in AUDIO_EXT:
                texts, metas, times = _audio_frames(f, src, transcriber, meta, base_t)
            else:
                chunks = chunk_blocks(load(f), source=src, max_chars=max_chars, overlap=overlap, base_meta=meta)
                texts = [c.text for c in chunks]
                metas = [c.meta for c in chunks]
                times = [base_t + (step + i) * 1e-6 for i in range(len(chunks))]
                step += len(chunks)
            for m in metas:
                m["file_sha"] = sha
            if on_change == "replace":
                old = mem.find({"source": src})
                for fr in old:
                    mem.forget(fr.id)
                if old:
                    rep.replaced.append(src)
            frames = mem.add_many(texts, metas, times, defer_embedding=defer_embedding) if texts else []
            rep.frame_ids += [fr.id for fr in frames]
            rep.files += 1
            rep.chunks += len(frames)
        except MissingDependency as e:
            rep.failed[str(f)] = str(e)
        except Exception as e:  # one bad file must not abort a directory
            rep.failed[str(f)] = f"{type(e).__name__}: {e}"
    return rep


def _audio_frames(f: Path, src: str, transcriber, meta, base_t: float):
    """transcriber(path) -> [(start_s, end_s, text)]. Frame time = ingest time + start offset
    (monotonic); meta.start/meta.end hold the offsets in the recording, so a span over these
    frames is a literal 'watch from 1:30 to 2:10'."""
    texts, metas, times = [], [], []
    for start, end, text in transcriber(str(f)):
        if text.strip():
            texts.append(text.strip())
            metas.append({**(meta or {}), "source": src, "kind": "audio", "start": float(start),
                          "end": float(end), "session": src})
            times.append(base_t + float(start))
    return texts, metas, times
