"""Structure-aware chunking: headings are tracked, paragraphs are packed, tables
stay together (header row repeated when a table must be split), long paragraphs
are split on sentences, then hard-split with overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENT = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass
class Block:
    """A unit from a loader: heading | para | table (text rows joined by newlines)."""

    kind: str  # "heading" | "para" | "table"
    text: str
    level: int = 1  # heading level
    page: int | None = None
    extra: dict = field(default_factory=dict)  # e.g. {"sheet": "Q3"}


@dataclass
class Chunk:
    text: str
    meta: dict


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for sent in _SENT.split(text):
        if len(sent) > max_chars:  # no sentence boundary: hard split with overlap
            if cur:
                parts.append(cur)
                cur = ""
            step = max(max_chars - overlap, 1)
            parts += [sent[i : i + max_chars] for i in range(0, len(sent), step)]
        elif len(cur) + len(sent) + 1 > max_chars:
            parts.append(cur)
            cur = (cur[-overlap:] + " " if overlap and cur else "") + sent
            if len(cur) > max_chars:
                cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def _split_table(text: str, max_chars: int) -> list[str]:
    rows = text.split("\n")
    if len(text) <= max_chars or len(rows) < 3:
        return [text]
    header, body, out, cur = rows[0], rows[1:], [], []
    size = len(header)
    for r in body:
        if cur and size + len(r) + 1 > max_chars:
            out.append("\n".join([header, *cur]))
            cur, size = [], len(header)
        cur.append(r)
        size += len(r) + 1
    if cur:
        out.append("\n".join([header, *cur]))
    return out


def chunk_blocks(blocks: list[Block], *, source: str | None = None, max_chars: int = 800,
                 overlap: int = 100, base_meta: dict | None = None) -> list[Chunk]:
    """Pack blocks into chunks of about max_chars, never crossing a heading or page change.
    meta per chunk: source, heading (path), section (index), page, chunk, offset, kind."""
    base = dict(base_meta or {})
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []  # (level, title)
    section, offset = 0, 0
    cur: list[str] = []
    cur_page: int | None = None
    cur_extra: dict = {}

    def meta(kind: str, page, extra) -> dict:
        m = {**base, "kind": kind, "section": section, "chunk": len(chunks), "offset": offset}
        if source:
            m["source"] = source
            m["session"] = f"{source}#{section}"  # scene boundary at every section
        if path:
            m["heading"] = " > ".join(t for _, t in path)
        if page is not None:
            m["page"] = page
        m.update(extra)
        return m

    def flush():
        nonlocal cur, offset
        if cur:
            text = "\n\n".join(cur)
            chunks.append(Chunk(text, meta("text", cur_page, cur_extra)))
            offset += len(text) + 2
            cur = []

    for b in blocks:
        text = b.text.strip()
        if not text:
            continue
        if b.kind == "heading":
            flush()
            path = [(lv, t) for lv, t in path if lv < b.level] + [(b.level, text)]
            section += 1
            continue
        if cur and (b.page != cur_page or b.extra != cur_extra):
            flush()
        if b.kind == "table":
            flush()
            for piece in _split_table(text, max_chars):
                chunks.append(Chunk(piece, meta("table", b.page, b.extra)))
                offset += len(piece) + 2
            continue
        cur_page, cur_extra = b.page, b.extra
        for piece in _split_long(text, max_chars, overlap):
            if cur and sum(len(c) for c in cur) + len(piece) + 2 > max_chars:
                flush()
                cur_page, cur_extra = b.page, b.extra
            cur.append(piece)
    flush()
    return chunks
