import os
import subprocess
import sys
import textwrap

import pytest

from playmemory import Memory
from playmemory.store import CorruptFileError, LockedError


def _fill(m, n, start=0):
    m.add_many([f"doc {i} token{i % 7}" for i in range(start, start + n)],
               times=[float(i) for i in range(start, start + n)])


def test_checkpoint_roundtrip_and_tail_replay(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    _fill(m, 50)
    m.forget(3, t=100.0)
    m.checkpoint()
    _fill(m, 10, start=50)  # tail after the checkpoint
    m.forget(4, t=101.0)
    m.close()

    m2 = Memory(p)
    assert m2.store.checkpoints >= 1
    assert len(m2) == 60
    assert m2.store.deleted.keys() == {3, 4}
    assert len(m2.bm25) == 60
    assert m2.search("token3", k=100, mode="lexical")  # lexical index restored
    assert 3 not in [h.frame.id for h in m2.search("doc 3 token3", k=100)]
    assert m2.get(55).text.startswith("doc 55")
    m2.close()


def test_corrupt_checkpoint_falls_back_to_full_scan(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    _fill(m, 20)
    m.checkpoint()
    m.close()
    raw = bytearray(p.read_bytes())
    raw[-10] ^= 0xFF  # flip a byte inside the checkpoint record
    p.write_bytes(bytes(raw))
    m2 = Memory(p)
    # the corrupt checkpoint is dropped as a bad tail; all frames survive
    assert len(m2) == 20
    assert m2.search("token3", k=3, mode="lexical")
    m2.close()


def test_truncation_at_every_byte_of_last_record(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    _fill(m, 5)
    good = m.store.nbytes
    m.add("the last record", t=99.0)
    full = m.store.nbytes
    m.close()
    raw = p.read_bytes()
    assert len(raw) == full
    for cut in range(good, full):
        p.write_bytes(raw[:cut])
        m2 = Memory(p)
        assert len(m2) == 5, cut
        assert m2.store.dropped_bytes == cut - good
        m2.add("after recovery", t=100.0)  # file stays appendable
        assert len(m2) == 6
        m2.close()
        with Memory(p, sync="never") as m3:
            assert len(m3) == 6


def test_single_writer_lock_and_read_only(tmp_path):
    p = tmp_path / "m.pmem"
    w = Memory(p)
    _fill(w, 5)
    w.commit()
    with pytest.raises(LockedError):
        Memory(p)
    r = Memory(p, read_only=True)  # lock-free readers work alongside the writer
    assert len(r) == 5
    with pytest.raises(PermissionError):
        r.add("nope")
    r.close()
    w.close()
    Memory(p).close()  # lock released


def test_read_only_never_modifies_file(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    _fill(m, 3)
    m.close()
    p.write_bytes(p.read_bytes() + b"\x05\x00\x00")  # torn tail
    before = p.read_bytes()
    r = Memory(p, read_only=True)
    assert len(r) == 3 and r.store.dropped_bytes == 3
    r.close()
    assert p.read_bytes() == before


def test_kill9_mid_write_recovers(tmp_path):
    p = tmp_path / "m.pmem"
    script = textwrap.dedent(f"""
        import sys, os
        from playmemory import Memory
        m = Memory({str(p)!r}, sync="always")
        i = 0
        while True:
            m.add(f"event number {{i}}", t=float(i))
            i += 1
            if i == 200:
                print("ready", flush=True)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE,
                            env={**os.environ, "PYTHONPATH": "src"})
    assert proc.stdout.readline().strip() == b"ready"
    proc.kill()
    proc.wait()
    m = Memory(p)
    assert len(m) >= 200
    n = len(m)
    assert m.get(n - 1).text == f"event number {n - 1}"
    m.add("post-crash", t=1e9)
    m.close()
    with Memory(p, sync="never") as m2:
        assert len(m2) == n + 1


def test_compact_drops_deleted_and_checkpoints(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    _fill(m, 100)
    for i in range(0, 100, 2):
        m.forget(i, t=500.0)
    new = m.supersede(1, "replacement for one", t=600.0)
    m.checkpoint()
    m.checkpoint()
    info = m.compact()
    assert info["frames"] == 100 - 50 - 1 + 1
    assert info["bytes_after"] < info["bytes_before"]
    assert len(m) == info["frames"] and not m.store.deleted
    hits = m.search("replacement for one", k=1)
    assert hits[0].frame.text == "replacement for one"
    assert "supersedes" not in hits[0].frame.meta  # target was dropped
    m.add("still writable", t=700.0)
    m.close()
    with Memory(p) as m2:
        assert len(m2) == info["frames"] + 1
    assert not (tmp_path / "m.pmem.compact").exists()


def test_compact_keep_history_preserves_time_travel(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    old = m.add("alice lives in paris", t=1.0)
    m.supersede(old.id, "alice lives in berlin", t=5.0)
    m.compact(keep_history=True)
    assert [h.frame.text for h in m.search("alice lives", k=3, as_of=3.0)] == ["alice lives in paris"]
    assert [h.frame.text for h in m.search("alice lives", k=3)] == ["alice lives in berlin"]
    m.close()


def test_limits_and_clock_clamp():
    m = Memory(max_text_bytes=10, max_meta_bytes=20)
    with pytest.raises(ValueError):
        m.add("x" * 11)
    with pytest.raises(ValueError):
        m.add("ok", {"k": "v" * 50})
    m = Memory()
    m.add("a", t=1e12)  # far-future frame
    f = m.add("b")  # wall clock is "earlier": clamped, not an error
    assert f.t == 1e12


def test_unknown_version_and_garbage_rejected(tmp_path):
    p = tmp_path / "x.pmem"
    p.write_bytes(b"not a memory file at all")
    with pytest.raises(CorruptFileError):
        Memory(p)
    p2 = tmp_path / "y.pmem"
    Memory(p2).close()
    raw = bytearray(p2.read_bytes())
    raw[4] = 99  # version
    p2.write_bytes(bytes(raw))
    with pytest.raises(CorruptFileError, match="unsupported file version"):
        Memory(p2)


def test_verify_and_stats(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    _fill(m, 10)
    assert m.store.verify() == (10, 0)
    s = m.stats()
    assert s["frames"] == 10 and s["live"] == 10
    m.close()


def test_context_manager_and_threads(tmp_path):
    import threading

    with Memory(tmp_path / "m.pmem") as m:
        def work(k):
            for i in range(50):
                m.add(f"thread {k} item {i}")
                m.search("item", k=3)
        ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert len(m) == 200
        assert m.store.verify()[1] == 0
