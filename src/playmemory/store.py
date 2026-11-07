from __future__ import annotations

import bisect
import io
import json
import mmap
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
REC_CHECKPOINT = 3  # derived index state; safe to ignore or drop (compaction)


class ModelMismatchError(ValueError):
    pass


class CorruptFileError(ValueError):
    pass


class LockedError(RuntimeError):
    """Another process holds the write lock on this file."""


def _try_lock(fd: int) -> None:
    try:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise LockedError("file is locked by another writer") from e
    except ImportError:  # Windows
        import msvcrt

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise LockedError("file is locked by another writer") from e


def _fsync_dir(path: Path) -> None:
    try:
        dfd = os.open(path.parent, os.O_RDONLY)
    except OSError:  # not supported (Windows)
        return
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class FrameStore:
    """Single-file, append-only frame log with embeddings inside the records.

    Layout:  header | record*
      header : "PMEM" u16 version, u16 dim, u16 name_len, embedder name bytes
      record : u32 length | u8 type | payload | u32 crc32(type+payload)
        frame      : u32 json_len | json{t,text,meta} | float32[dim]
        tombstone  : u64 frame id | f64 deletion time
        checkpoint : npz of derived index state (offsets, times, metas, BM25)
    Frame id == order of frame records. Open loads the last valid checkpoint
    and replays only the records after it; a torn or corrupt tail is dropped
    (and truncated unless read_only).

    Concurrency: one writer (exclusive lock) and any number of lock-free
    read-only openers, which simply ignore a partially written tail record.
    sync: "always" fsyncs every write, "batch" on sync()/close (and every
    `sync_every` records if set), "never" leaves it to the OS.
    In-memory mode (path=None) uses the same encoding in a bytearray.
    """

    def __init__(
        self,
        path: str | os.PathLike | None,
        dim: int,
        model: str = "",
        *,
        read_only: bool = False,
        sync: str = "batch",
        sync_every: int = 0,
    ):
        if sync not in ("always", "batch", "never"):
            raise ValueError("sync must be always|batch|never")
        self.path = Path(path) if path else None
        self.dim = dim
        self.model = model
        self.read_only = read_only
        self.sync_mode = sync
        self.sync_every = sync_every
        self._unsynced = 0
        self._offsets: list[int] = []
        self._voffs: list[int] = []
        self._times: list[float] = []
        self.metas: list[dict] = []
        self.deleted: dict[int, float] = {}
        self.checkpoint_arrays: dict[str, np.ndarray] | None = None
        self.tail_texts: list[tuple[int, str]] = []  # frames not covered by checkpoint
        self.dropped_bytes = 0
        self.checkpoints = 0
        self._cp_frames = 0  # frames covered by the last checkpoint
        self._buf = bytearray()
        self._fd: int | None = None
        created = False
        if self.path:
            if read_only:
                self._fd = os.open(self.path, os.O_RDONLY)
            else:
                self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
                try:
                    _try_lock(self._fd)
                except Exception:
                    os.close(self._fd)
                    self._fd = None
                    raise
        if self._size() == 0:
            if read_only:
                raise CorruptFileError("empty file")
            name = model.encode()
            self._write(_HDR.pack(MAGIC, VERSION, dim, len(name)) + name, 0)
            created = True
        self._end = self._scan()
        if created and self._fd is not None:
            os.fsync(self._fd)
            _fsync_dir(self.path)

    # raw io ---------------------------------------------------------------
    def _size(self) -> int:
        return os.fstat(self._fd).st_size if self._fd is not None else len(self._buf)

    def _read(self, off: int, n: int) -> bytes:
        if self._fd is not None:
            return os.pread(self._fd, n, off)
        return bytes(self._buf[off : off + n])

    def _write(self, data: bytes, off: int) -> None:
        if self.read_only:
            raise PermissionError("store opened read-only")
        if self._fd is not None:
            os.pwrite(self._fd, data, off)
        else:
            self._buf[off : off + len(data)] = data

    def _truncate(self, size: int) -> None:
        if self._fd is not None:
            os.ftruncate(self._fd, size)
        else:
            del self._buf[size:]

    def _data(self):
        """Read-only bytes-like view of the whole file (mmap when file-backed)."""
        if self._fd is not None:
            if os.fstat(self._fd).st_size == 0:
                return b""
            return mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        return bytes(self._buf)

    # open/scan ---------------------------------------------------------------
    def _scan(self) -> int:
        data = self._data()
        try:
            return self._scan_data(data)
        finally:
            if isinstance(data, mmap.mmap):
                try:
                    data.close()
                except BufferError:
                    pass

    def _scan_data(self, data) -> int:
        size = len(data)
        if size < _HDR.size:
            raise CorruptFileError("truncated header")
        magic, ver, dim, nlen = _HDR.unpack_from(data, 0)
        if magic != MAGIC:
            raise CorruptFileError("not a PlayMemory file")
        if ver != VERSION:
            raise CorruptFileError(
                f"unsupported file version {ver} (this build reads {VERSION})"
            )
        model = bytes(data[_HDR.size : _HDR.size + nlen]).decode()
        if dim != self.dim or (self.model and model and model != self.model):
            raise ModelMismatchError(
                f"file built with embedder {model!r} dim={dim}; "
                f"opened with {self.model!r} dim={self.dim}"
            )
        self.model = model or self.model
        pos0 = _HDR.size + nlen

        # 1) hop the length chain to find the last checkpoint (no CRC work)
        cp = None
        pos = pos0
        while pos + 5 <= size:
            (n,) = _LEN.unpack_from(data, pos)
            end = pos + 4 + n + 4
            if n < 1 or end > size:
                break
            if data[pos + 4] == REC_CHECKPOINT:
                cp = (pos, end)
            pos = end

        # 2) load it if its CRC holds, else fall back to a full scan
        start = pos0
        if cp:
            try:
                self._load_checkpoint(data, cp)
                start = cp[1]
            except Exception:
                self._reset_state()

        # 3) replay everything after the checkpoint with full CRC checks
        pos = start
        while pos + 4 <= size:
            (n,) = _LEN.unpack_from(data, pos)
            end = pos + 4 + n + 4
            if n < 1 or end > size:
                break
            body = bytes(data[pos + 4 : pos + 4 + n])
            (crc,) = _CRC.unpack_from(data, pos + 4 + n)
            if zlib.crc32(body) != crc:
                break
            rtype = body[0]
            if rtype == REC_FRAME:
                jl = _JLEN.unpack_from(body, 1)[0]
                if len(body) != 1 + 4 + jl + self.dim * 4:
                    break
                rec = json.loads(body[5 : 5 + jl])
                fid = len(self._times)
                self._offsets.append(pos)
                self._voffs.append(pos + 4 + 1 + 4 + jl)
                self._times.append(rec["t"])
                self.metas.append(rec["meta"])
                self.tail_texts.append((fid, rec["text"]))
            elif rtype == REC_TOMBSTONE:
                fid, t = _TOMB.unpack(body[1:])
                self.deleted[fid] = t
            elif rtype == REC_CHECKPOINT:
                pass  # superseded or corrupt-skipped older checkpoint
            pos = end
        if pos < size:
            self.dropped_bytes = size - pos
            if not self.read_only:
                self._truncate(pos)
        return pos

    def _reset_state(self) -> None:
        self._offsets, self._voffs, self._times, self.metas = [], [], [], []
        self.deleted = {}
        self.checkpoint_arrays = None
        self.tail_texts = []
        self._cp_frames = 0

    def _load_checkpoint(self, data, cp: tuple[int, int]) -> None:
        pos, end = cp
        body = bytes(data[pos + 4 : end - 4])
        if zlib.crc32(body) != _CRC.unpack_from(data, end - 4)[0]:
            raise CorruptFileError("bad checkpoint crc")
        z = np.load(io.BytesIO(body[1:]), allow_pickle=False)
        arrays = {k: z[k] for k in z.files}
        n = len(arrays["times"])
        if len(arrays["offsets"]) != n or len(arrays["voffs"]) != n:
            raise CorruptFileError("inconsistent checkpoint")
        self._offsets = arrays["offsets"].tolist()
        self._voffs = arrays["voffs"].tolist()
        self._times = arrays["times"].tolist()
        self.metas = json.loads(arrays["metas"].tobytes())
        self.deleted = dict(zip(arrays["del_ids"].tolist(), arrays["del_t"].tolist()))
        self.checkpoint_arrays = arrays
        self._cp_frames = n
        self.checkpoints += 1

    # vectors ------------------------------------------------------------------
    def load_vectors(self) -> np.ndarray:
        """All embeddings in frame order, read straight from the file."""
        n = len(self._voffs)
        out = np.empty((n, self.dim), dtype=np.float32)
        if n == 0:
            return out
        data = self._data()
        try:
            for i, off in enumerate(self._voffs):
                out[i] = np.frombuffer(data, np.float32, self.dim, off)
        finally:
            if isinstance(data, mmap.mmap):
                try:
                    data.close()
                except BufferError:
                    pass
        return out

    # records ------------------------------------------------------------------
    def _append_record(self, rtype: int, payload: bytes) -> int:
        body = bytes([rtype]) + payload
        rec = _LEN.pack(len(body)) + body + _CRC.pack(zlib.crc32(body))
        off = self._end
        self._write(rec, off)
        self._end += len(rec)
        self._unsynced += 1
        if self._fd is not None:
            if self.sync_mode == "always" or (
                self.sync_mode == "batch" and self.sync_every and self._unsynced >= self.sync_every
            ):
                self.sync()
        return off

    def __len__(self) -> int:
        return len(self._times)

    @property
    def times(self) -> list[float]:
        return self._times

    @property
    def tail_records(self) -> int:
        """Frames written since the last checkpoint."""
        return len(self._times) - self._cp_frames

    @property
    def nbytes(self) -> int:
        return self._end

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
        self._voffs.append(off + 4 + 1 + 4 + len(j))
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

    def write_checkpoint(self, extra: dict[str, np.ndarray]) -> None:
        """Persist derived index state so the next open skips the full replay."""
        arrays = {
            "offsets": np.asarray(self._offsets, dtype=np.int64),
            "voffs": np.asarray(self._voffs, dtype=np.int64),
            "times": np.asarray(self._times, dtype=np.float64),
            "metas": np.frombuffer(json.dumps(self.metas, separators=(",", ":")).encode(), np.uint8),
            "del_ids": np.asarray(list(self.deleted), dtype=np.int64),
            "del_t": np.asarray(list(self.deleted.values()), dtype=np.float64),
            **extra,
        }
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        self._append_record(REC_CHECKPOINT, buf.getvalue())
        self._cp_frames = len(self._times)
        self.checkpoints += 1

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
        if self._fd is not None and not self.read_only:
            os.fsync(self._fd)
        self._unsynced = 0

    def verify(self) -> tuple[int, int]:
        """Re-read the whole file checking every CRC. Returns (records, bad_bytes)."""
        data = self._data()
        try:
            size = len(data)
            _, _, _, nlen = _HDR.unpack_from(data, 0)
            pos, count = _HDR.size + nlen, 0
            while pos + 4 <= size:
                (n,) = _LEN.unpack_from(data, pos)
                end = pos + 4 + n + 4
                if n < 1 or end > size or zlib.crc32(bytes(data[pos + 4 : pos + 4 + n])) != _CRC.unpack_from(data, pos + 4 + n)[0]:
                    break
                pos, count = end, count + 1
            return count, size - pos
        finally:
            if isinstance(data, mmap.mmap):
                try:
                    data.close()
                except BufferError:
                    pass

    def close(self) -> None:
        if self._fd is not None:
            if not self.read_only:
                try:
                    os.fsync(self._fd)
                except OSError:
                    pass
            os.close(self._fd)  # also releases the lock
            self._fd = None
