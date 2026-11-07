from __future__ import annotations

import bisect
import io
import json
import mmap
import os
import struct
import threading
import zlib
from pathlib import Path

import numpy as np

from .frame import Frame

MAGIC = b"PMEM"
VERSION = 3  # v3 = v2 + embedder spec + flags in the header; record formats are identical
READ_VERSIONS = (2, 3)
FLAG_VEC_F16 = 1  # vectors stored as float16 (half the size), read back as float32
Z_MIN_JSON = 200  # don't bother compressing tiny records
_HDR = struct.Struct("<4sHHH")  # magic, version, dim, name_len
_SPEC = struct.Struct("<H")
_FLAGS = struct.Struct("<I")
_LEN = struct.Struct("<I")
_CRC = struct.Struct("<I")
_TOMB = struct.Struct("<Qd")  # frame id, time of deletion
_JLEN = struct.Struct("<I")

REC_FRAME = 1
REC_TOMBSTONE = 2
REC_CHECKPOINT = 3  # derived index state; safe to ignore or drop (compaction)
REC_FRAME_Z = 5  # frame whose JSON part is zlib-compressed (used only when it shrinks)
REC_ANN = 4  # serialized ANN graph: u64 frames covered | blob (derived, droppable)


_HAS_PREAD = hasattr(os, "pread")  # absent on Windows; tests flip this to exercise the fallback
_SEEK_LOCK = threading.Lock()
_O_BINARY = getattr(os, "O_BINARY", 0)


def _pread(fd: int, n: int, off: int) -> bytes:
    if _HAS_PREAD:
        return os.pread(fd, n, off)
    with _SEEK_LOCK:
        os.lseek(fd, off, os.SEEK_SET)
        return os.read(fd, n)


def _pwrite(fd: int, data: bytes, off: int) -> None:
    """Write all of data at off (handles short writes)."""
    view = memoryview(data)
    with _SEEK_LOCK if not _HAS_PREAD else _NullLock():
        while view:
            if _HAS_PREAD:
                n = os.pwrite(fd, view, off)
            else:
                os.lseek(fd, off, os.SEEK_SET)
                n = os.write(fd, view)
            view, off = view[n:], off + n


class _NullLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class ModelMismatchError(ValueError):
    pass


class CorruptFileError(ValueError):
    pass


class FileChangedError(RuntimeError):
    """The file under an open store was replaced (e.g. compacted by the writer)."""


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

        try:  # lock a byte far past EOF so readers can still read the real data
            os.lseek(fd, 1 << 40, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise LockedError("file is locked by another writer") from e


def read_header(path: str | os.PathLike) -> dict:
    """Parse just the header: {version, dim, name, spec, flags}."""
    with open(path, "rb") as f:
        head = f.read(_HDR.size)
        if len(head) < _HDR.size:
            raise CorruptFileError("truncated header")
        magic, ver, dim, nlen = _HDR.unpack(head)
        if magic != MAGIC:
            raise CorruptFileError("not a PlayMemory file")
        name = f.read(nlen).decode()
        spec, flags = name, 0
        if ver >= 3:
            (slen,) = _SPEC.unpack(f.read(2))
            spec = f.read(slen).decode()
            (flags,) = _FLAGS.unpack(f.read(4))
    return {"version": ver, "dim": dim, "name": name, "spec": spec, "flags": flags}


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
        spec: str = "",
        flags: int = 0,
        read_only: bool = False,
        sync: str = "batch",
        sync_every: int = 0,
    ):
        if sync not in ("always", "batch", "never"):
            raise ValueError("sync must be always|batch|never")
        self.path = Path(path) if path else None
        self.dim = dim
        self.model = model
        self.spec = spec
        self.flags = flags
        self.version = VERSION
        self.read_only = read_only
        self.sync_mode = sync
        self.sync_every = sync_every
        self._unsynced = 0
        self._set_vec_dtype()
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
        self._ann_loc: tuple[int, int] | None = None
        self._buf = bytearray()
        self._fd: int | None = None
        created = False
        if self.path:
            if read_only:
                self._fd = os.open(self.path, os.O_RDONLY | _O_BINARY)
            else:
                self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | _O_BINARY)
                try:
                    _try_lock(self._fd)
                except Exception:
                    os.close(self._fd)
                    self._fd = None
                    raise
        try:
            if self._size() == 0:
                if read_only:
                    raise CorruptFileError("empty file")
                name, sp = model.encode(), (spec or model).encode()
                self._write(_HDR.pack(MAGIC, VERSION, dim, len(name)) + name
                            + _SPEC.pack(len(sp)) + sp + _FLAGS.pack(flags), 0)
                created = True
            self._end = self._scan()
            if created and self._fd is not None:
                os.fsync(self._fd)
                _fsync_dir(self.path)
        except BaseException:  # a failed open must not leave the file locked
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raise

    def _set_vec_dtype(self) -> None:
        self._vdt = np.float16 if self.flags & FLAG_VEC_F16 else np.float32
        self._vsz = self.dim * np.dtype(self._vdt).itemsize

    # raw io ---------------------------------------------------------------
    def _size(self) -> int:
        return os.fstat(self._fd).st_size if self._fd is not None else len(self._buf)

    def _read(self, off: int, n: int) -> bytes:
        if self._fd is not None:
            return _pread(self._fd, n, off)
        return bytes(self._buf[off : off + n])

    def _write(self, data: bytes, off: int) -> None:
        if self.read_only:
            raise PermissionError("store opened read-only")
        if self._fd is not None:
            _pwrite(self._fd, data, off)
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
        if ver not in READ_VERSIONS:
            raise CorruptFileError(
                f"unsupported file version {ver} (this build reads {READ_VERSIONS})"
            )
        self.version = ver
        model = bytes(data[_HDR.size : _HDR.size + nlen]).decode()
        if dim != self.dim or (self.model and model and model != self.model):
            raise ModelMismatchError(
                f"file built with embedder {model!r} dim={dim}; "
                f"opened with {self.model!r} dim={self.dim}"
            )
        self.model = model or self.model
        pos0 = _HDR.size + nlen
        if ver >= 3:
            (slen,) = _SPEC.unpack_from(data, pos0)
            self.spec = bytes(data[pos0 + 2 : pos0 + 2 + slen]).decode()
            (self.flags,) = _FLAGS.unpack_from(data, pos0 + 2 + slen)
            pos0 += 2 + slen + 4
        else:
            self.spec = self.spec or model
        self._pos0 = pos0  # offset of the first record
        self._set_vec_dtype()

        # 1) hop the length chain to find the last checkpoint (no CRC work)
        cp = None
        ann = None
        pos = pos0
        while pos + 5 <= size:
            (n,) = _LEN.unpack_from(data, pos)
            end = pos + 4 + n + 4
            if n < 1 or end > size:
                break
            if data[pos + 4] == REC_CHECKPOINT:
                cp = (pos, end)
            elif data[pos + 4] == REC_ANN:
                ann = (pos, end)
            pos = end
        self._ann_loc = ann

        # 2) load it if its CRC holds, else fall back to a full scan
        start = pos0
        if cp:
            try:
                self._load_checkpoint(data, cp)
                start = cp[1]
            except Exception:
                self._reset_state()

        # 3) replay everything after the checkpoint with full CRC checks
        pos = self._replay(data, start, size)
        if pos < size:
            self.dropped_bytes = size - pos
            if not self.read_only:
                self._truncate(pos)
        return pos

    def _parse_frame(self, body: bytes, pos: int):
        """-> (record dict, absolute vector offset) or None when malformed."""
        jl = _JLEN.unpack_from(body, 1)[0]
        if len(body) != 1 + 4 + jl + self._vsz:
            return None
        raw = body[5 : 5 + jl]
        if body[0] == REC_FRAME_Z:
            raw = zlib.decompress(raw)
        return json.loads(raw), pos + 4 + 1 + 4 + jl

    def _replay(self, data, start: int, size: int) -> int:
        """Apply every valid record in data[start:size]; return the end of the valid prefix."""
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
            if rtype in (REC_FRAME, REC_FRAME_Z):
                try:
                    parsed = self._parse_frame(body, pos)
                except (ValueError, zlib.error, struct.error):
                    parsed = None
                if parsed is None:
                    break
                rec, voff = parsed
                fid = len(self._times)
                self._offsets.append(pos)
                self._voffs.append(voff)
                self._times.append(rec["t"])
                self.metas.append(rec["meta"] or None)
                self.tail_texts.append((fid, rec["text"]))
            elif rtype == REC_TOMBSTONE:
                fid, t = _TOMB.unpack(body[1:])
                self.deleted[fid] = t
            # checkpoint / ann records carry no frame state here
            pos = end
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
    def load_vectors(self, start: int = 0) -> np.ndarray:
        """Embeddings of frames start.. in order (float32), read straight from the file."""
        n = len(self._voffs) - start
        out = np.empty((max(n, 0), self.dim), dtype=np.float32)
        if n <= 0:
            return out
        data = self._data()
        try:
            for i, off in enumerate(self._voffs[start:]):
                out[i] = np.frombuffer(data, self._vdt, self.dim, off)
        finally:
            if isinstance(data, mmap.mmap):
                try:
                    data.close()
                except BufferError:
                    pass
        return out

    def meta_of(self, fid: int) -> dict:
        return self.metas[fid] or {}

    def refresh(self) -> tuple[int, set[int]]:
        """Pick up records appended (by a writer in another process) since we
        last looked. Returns (first new frame id, ids newly tombstoned).
        Raises FileChangedError if the file was replaced/truncated (compaction)."""
        if self._fd is None:
            return len(self._times), set()
        size = os.fstat(self._fd).st_size
        try:
            same = os.path.samestat(os.fstat(self._fd), os.stat(self.path))
        except OSError:
            same = False
        if not same or size < self._end:
            raise FileChangedError("file was replaced or truncated")
        first, before = len(self._times), set(self.deleted)
        if size == self._end:
            return first, set()
        data = self._data()
        try:
            self._end = self._replay(data, self._end, size)
        finally:
            if isinstance(data, mmap.mmap):
                try:
                    data.close()
                except BufferError:
                    pass
        return first, set(self.deleted) - before

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
        fid = len(self._times)
        j = json.dumps({"i": fid, "t": t, "text": text, "meta": meta or {}},
                       separators=(",", ":"), ensure_ascii=False).encode()
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"vector dim {v.shape[0]} != {self.dim}")
        rtype = REC_FRAME
        if len(j) >= Z_MIN_JSON:
            z = zlib.compress(j, 3)
            if len(z) < 0.9 * len(j):
                j, rtype = z, REC_FRAME_Z
        off = self._append_record(rtype, _JLEN.pack(len(j)) + j + v.astype(self._vdt).tobytes())
        self._offsets.append(off)
        self._voffs.append(off + 4 + 1 + 4 + len(j))
        self._times.append(t)
        self.metas.append(meta or None)
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

    def write_ann(self, blob: bytes, covered: int) -> None:
        off = self._append_record(REC_ANN, struct.pack("<Q", covered) + blob)
        self._ann_loc = (off, self._end)

    def read_ann(self) -> tuple[int, bytes] | None:
        """(frames covered, serialized graph) of the last intact ANN record."""
        if self._ann_loc is None:
            return None
        pos, end = self._ann_loc
        raw = self._read(pos, end - pos)
        body = raw[4:-4]
        if len(raw) != end - pos or zlib.crc32(body) != _CRC.unpack(raw[-4:])[0]:
            return None
        return struct.unpack_from("<Q", body, 1)[0], body[9:]

    def is_live(self, fid: int, as_of: float | None = None) -> bool:
        d = self.deleted.get(fid)
        return d is None or (as_of is not None and d > as_of)

    def get(self, fid: int) -> Frame:
        """Fetch a frame (including deleted ones; history is never lost)."""
        if not 0 <= fid < len(self._times):
            raise IndexError(fid)
        off = self._offsets[fid]
        head = self._read(off, 5)
        (n,) = _LEN.unpack_from(head)
        payload = self._read(off + 5, n - 1)
        jl = _JLEN.unpack_from(payload, 0)[0]
        raw = payload[4 : 4 + jl]
        rec = json.loads(zlib.decompress(raw) if head[4] == REC_FRAME_Z else raw)
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
            pos, count = self._pos0, 0
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
