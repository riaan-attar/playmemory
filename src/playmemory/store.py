from __future__ import annotations

import bisect
import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np

from .frame import Frame

MAGIC = b"PMEM"
VERSION = 2
_HDR = struct.Struct("<4sHHH")  # magic, version, dim, name_len
_LEN = struct.Struct("<I")
_CRC = struct.Struct("<I")
_TOMB = struct.Struct("<Qd")  # frame id, time of deletion
_JLEN = struct.Struct("<I")

REC_FRAME = 1
REC_TOMBSTONE = 2


class ModelMismatchError(ValueError):
    pass


class CorruptFileError(ValueError):
    pass


class FrameStore:
    """Single-file, append-only frame log with embeddings inside the records.

    Layout:  header | record*
      header : "PMEM" u16 version, u16 dim, u16 name_len, embedder name bytes
      record : u32 length | u8 type | payload | u32 crc32(type+payload)
        frame     : u32 json_len | json{t,text,meta} | float32[dim]
        tombstone : u64 frame id | f64 deletion time
    Frame id == order of frame records. On open, one sequential scan rebuilds
    offsets/times/tombstones; a torn or corrupt tail is truncated.
    In-memory mode (path=None) uses the same encoding in a bytearray.
    """

    def __init__(self, path: str | os.PathLike | None, dim: int, model: str = ""):
        self.path = Path(path) if path else None
        self.dim = dim
        self.model = model
        self._offsets: list[int] = []
        self._times: list[float] = []
        self.metas: list[dict] = []
        self.deleted: dict[int, float] = {}
        self._vecs: list[np.ndarray] = []  # populated only during load
        self._buf = bytearray()
        self._fd: int | None = None
        if self.path:
            self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        if self._size() == 0:
            name = model.encode()
            self._write(_HDR.pack(MAGIC, VERSION, dim, len(name)) + name, 0)
        self._end = self._scan()

    # raw io ---------------------------------------------------------------
    def _size(self) -> int:
        return os.fstat(self._fd).st_size if self._fd is not None else len(self._buf)

    def _read(self, off: int, n: int) -> bytes:
        if self._fd is not None:
            return os.pread(self._fd, n, off)
        return bytes(self._buf[off : off + n])

    def _write(self, data: bytes, off: int) -> None:
        if self._fd is not None:
            os.pwrite(self._fd, data, off)
        else:
            self._buf[off : off + len(data)] = data

    def _truncate(self, size: int) -> None:
        if self._fd is not None:
            os.ftruncate(self._fd, size)
        else:
            del self._buf[size:]

    # open/scan ---------------------------------------------------------------
    def _scan(self) -> int:
        data = self._read(0, self._size())
        if len(data) < _HDR.size:
            raise CorruptFileError("truncated header")
        magic, ver, dim, nlen = _HDR.unpack_from(data, 0)
        if magic != MAGIC:
            raise CorruptFileError("not a PlayMemory file")
        if ver != VERSION:
            raise CorruptFileError(f"unsupported version {ver}")
        model = data[_HDR.size : _HDR.size + nlen].decode()
        if dim != self.dim or (self.model and model and model != self.model):
            raise ModelMismatchError(
                f"file built with embedder {model!r} dim={dim}; "
                f"opened with {self.model!r} dim={self.dim}"
            )
        self.model = model or self.model
        pos = _HDR.size + nlen
        while pos + 4 <= len(data):
            (n,) = _LEN.unpack_from(data, pos)
            end = pos + 4 + n + 4
            if n < 1 or end > len(data):
                break
            body = data[pos + 4 : pos + 4 + n]
            (crc,) = _CRC.unpack_from(data, pos + 4 + n)
            if zlib.crc32(body) != crc:
                break
            rtype, payload = body[0], body[1:]
            if rtype == REC_FRAME:
                jl = _JLEN.unpack_from(payload, 0)[0]
                rec = json.loads(payload[4 : 4 + jl])
                self._times.append(rec["t"])
                self.metas.append(rec["meta"])
                self._vecs.append(np.frombuffer(payload, np.float32, self.dim, 4 + jl))
                self._offsets.append(pos)
            elif rtype == REC_TOMBSTONE:
                fid, t = _TOMB.unpack(payload)
                self.deleted[fid] = t
            pos = end
        if pos < len(data):
            self._truncate(pos)  # drop torn/corrupt tail
        return pos

    def take_loaded_vectors(self) -> np.ndarray:
        """Embeddings read during open (frame order); releases the buffer."""
        out = np.vstack(self._vecs) if self._vecs else np.zeros((0, self.dim), np.float32)
        self._vecs = []
        return out

    # records ------------------------------------------------------------------
    def _append_record(self, rtype: int, payload: bytes) -> int:
        body = bytes([rtype]) + payload
        rec = _LEN.pack(len(body)) + body + _CRC.pack(zlib.crc32(body))
        off = self._end
        self._write(rec, off)
        self._end += len(rec)
        return off

    def __len__(self) -> int:
        return len(self._times)

    @property
    def times(self) -> list[float]:
        return self._times

    def append(self, t: float, text: str, meta: dict | None, vec: np.ndarray) -> Frame:
        if self._times and t < self._times[-1]:
            raise ValueError("timestamps must be non-decreasing")
        j = json.dumps({"t": t, "text": text, "meta": meta or {}}, separators=(",", ":")).encode()
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"vector dim {v.shape[0]} != {self.dim}")
        off = self._append_record(REC_FRAME, _JLEN.pack(len(j)) + j + v.tobytes())
        fid = len(self._times)
        self._offsets.append(off)
        self._times.append(t)
        self.metas.append(meta or {})
        return Frame(fid, t, text, meta or {})

    def delete(self, fid: int, t: float) -> None:
        if not 0 <= fid < len(self._times):
            raise IndexError(fid)
        if fid in self.deleted:
            return
        self._append_record(REC_TOMBSTONE, _TOMB.pack(fid, t))
        self.deleted[fid] = t

    def is_live(self, fid: int, as_of: float | None = None) -> bool:
        d = self.deleted.get(fid)
        return d is None or (as_of is not None and d > as_of)

    def get(self, fid: int) -> Frame:
        """Fetch a frame (including deleted ones; history is never lost)."""
        if not 0 <= fid < len(self._times):
            raise IndexError(fid)
        off = self._offsets[fid]
        (n,) = _LEN.unpack(self._read(off, 4))
        payload = self._read(off + 5, n - 1)
        jl = _JLEN.unpack_from(payload, 0)[0]
        rec = json.loads(payload[4 : 4 + jl])
        return Frame(fid, rec["t"], rec["text"], rec["meta"])

    def range(self, start: int, stop: int, live_only: bool = False,
              as_of: float | None = None) -> list[Frame]:
        start, stop = max(start, 0), min(stop, len(self))
        return [self.get(i) for i in range(start, stop)
                if not live_only or self.is_live(i, as_of)]

    def seek_time(self, t: float) -> int | None:
        i = bisect.bisect_right(self._times, t) - 1
        return i if i >= 0 else None

    def time_range(self, t0: float, t1: float) -> tuple[int, int]:
        return bisect.bisect_left(self._times, t0), bisect.bisect_right(self._times, t1)

    def sync(self) -> None:
        if self._fd is not None:
            os.fsync(self._fd)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
