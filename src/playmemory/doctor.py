"""Salvage a damaged memory file.

`FrameStore` already drops a torn tail on open. The doctor handles damage in the
MIDDLE of the file: it resynchronises on the next record whose length and CRC both
check out (length, type and CRC32 all valid), then copies
every intact frame and tombstone into a fresh file. Frame ids shift when frames are
lost, so tombstones and meta pointers (`supersedes`, `source`) are remapped; a
tombstone whose frame was lost is dropped (nothing is left to hide).
"""
from __future__ import annotations

import mmap
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .store import (
    _CRC, _LEN, _TOMB, FLAG_VEC_F16, REC_ANN, REC_CHECKPOINT, REC_FRAME, REC_FRAME_Z,
    REC_TOMBSTONE, FrameStore, read_header, _fsync_dir,
)

_KNOWN = (REC_FRAME, REC_TOMBSTONE, REC_CHECKPOINT, REC_ANN, REC_FRAME_Z)
_WINDOW = 1 << 24
_POINTER_KEYS = ("supersedes", "source")


@dataclass
class Report:
    path: str
    ok: bool = True
    records_ok: int = 0
    frames_salvaged: int = 0
    frames_lost_at_least: int = 0
    tombstones_applied: int = 0
    tombstones_dropped: int = 0
    gaps: list[tuple[int, int]] = field(default_factory=list)  # damaged byte ranges
    bytes_dropped: int = 0
    repaired_path: str | None = None
    backup_path: str | None = None


def _try(data, pos: int, size: int) -> int | None:
    """End offset of a valid record at pos, else None."""
    if pos + 9 > size:
        return None
    (n,) = _LEN.unpack_from(data, pos)
    end = pos + 4 + n + 4
    if n < 1 or end > size or data[pos + 4] not in _KNOWN:
        return None
    if zlib.crc32(bytes(data[pos + 4 : pos + 4 + n])) != _CRC.unpack_from(data, pos + 4 + n)[0]:
        return None
    return end


def _next_valid(data, start: int, size: int) -> int | None:
    pos = start
    arr = np.frombuffer(data, dtype=np.uint8)
    while pos < size:
        hi = min(pos + _WINDOW, size)
        types = arr[pos + 4 : hi + 4 if hi + 4 <= size else size]
        cand = np.flatnonzero((types >= 1) & (types <= 5)) + pos
        for c in cand.tolist():
            if _try(data, c, size) is not None:  # length + type + CRC32 all check out
                return c
        pos = hi
    return None


def walk(data, pos0: int):
    """Yield ('rec', pos, end) for valid records and ('gap', start, end) for damage."""
    size, pos = len(data), pos0
    while pos < size:
        end = _try(data, pos, size)
        if end is not None:
            yield "rec", pos, end
            pos = end
            continue
        nxt = _next_valid(data, pos + 1, size)
        if nxt is None:
            yield "gap", pos, size
            return
        yield "gap", pos, nxt
        pos = nxt


def _header_end(path: str) -> int:
    h = read_header(path)
    n = struct.calcsize("<4sHHH") + len(h["name"].encode())
    return n + (2 + len(h["spec"].encode()) + 4 if h["version"] >= 3 else 0)


def diagnose(path: str | os.PathLike) -> Report:
    return _run(str(path), None)


def repair(path: str | os.PathLike, dest: str | os.PathLike | None = None,
           replace: bool = False) -> Report:
    """Write the salvaged file to dest (default `<path>.repaired`). With replace=True the
    damaged original is kept as `<path>.corrupt` and the repaired file takes its place."""
    path = str(path)
    dest = str(dest or path + ".repaired")
    rep = _run(path, dest)
    if replace and rep.repaired_path:
        backup = path + ".corrupt"
        os.replace(path, backup)
        os.replace(dest, path)
        _fsync_dir(Path(path))
        rep.backup_path, rep.repaired_path = backup, path
    return rep


def _run(path: str, dest: str | None) -> Report:
    h = read_header(path)
    pos0 = _header_end(path)
    vsz = h["dim"] * (2 if h["flags"] & FLAG_VEC_F16 else 4)
    vdt = np.float16 if h["flags"] & FLAG_VEC_F16 else np.float32
    rep = Report(path)
    out = None
    if dest:
        if os.path.exists(dest):
            os.unlink(dest)
        out = FrameStore(dest, h["dim"], h["name"], spec=h["spec"], flags=h["flags"], sync="never")
    idmap: dict[int, int] = {}
    seen_frames = 0
    max_id = -1
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if size else b""
        try:
            for kind, a, b in walk(data, pos0):
                if kind == "gap":
                    rep.gaps.append((a, b))
                    rep.bytes_dropped += b - a
                    continue
                rep.records_ok += 1
                rtype = data[a + 4]
                body = bytes(data[a + 4 : b - 4])
                if rtype in (REC_FRAME, REC_FRAME_Z):
                    jl = struct.unpack_from("<I", body, 1)[0]
                    if len(body) != 1 + 4 + jl + vsz:
                        continue
                    raw = body[5 : 5 + jl]
                    if rtype == REC_FRAME_Z:
                        raw = zlib.decompress(raw)
                    import json

                    rec = json.loads(raw)
                    old = rec.get("i", seen_frames)  # v2 records have no id: assume no gaps
                    seen_frames += 1
                    max_id = max(max_id, old)
                    rep.frames_salvaged += 1
                    if out is not None:
                        meta = dict(rec.get("meta") or {})
                        for k in _POINTER_KEYS:
                            if k in meta:
                                if meta[k] in idmap:
                                    meta[k] = idmap[meta[k]]
                                else:
                                    del meta[k]
                        vec = np.frombuffer(body, vdt, h["dim"], 5 + jl).astype(np.float32)
                        idmap[old] = out.append(rec["t"], rec["text"], meta, vec).id
                    else:
                        idmap[old] = old
                elif rtype == REC_TOMBSTONE:
                    old, t = _TOMB.unpack(body[1:])
                    if old in idmap:
                        rep.tombstones_applied += 1
                        if out is not None:
                            out.delete(idmap[old], t)
                    else:
                        rep.tombstones_dropped += 1
        finally:
            if isinstance(data, mmap.mmap):
                try:
                    data.close()
                except BufferError:
                    pass
    rep.frames_lost_at_least = max(max_id + 1 - rep.frames_salvaged, 0)
    rep.ok = not rep.gaps
    if out is not None:
        out.sync()
        out.close()
        rep.repaired_path = dest
    return rep
