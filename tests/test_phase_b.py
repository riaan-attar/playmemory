import os
import random
import subprocess
import sys
import textwrap

import pytest

from playmemory import Memory
from playmemory import store as store_mod
from playmemory.cli import main
from playmemory.doctor import diagnose, repair
from playmemory.store import REC_FRAME, REC_FRAME_Z, read_header


def fill(m, n, start=0, prefix="doc"):
    m.add_many([f"{prefix} {i} token{i % 5}" for i in range(start, start + n)],
               times=[float(i) for i in range(start, start + n)])


# ---- stale readers -> refresh --------------------------------------------------
def test_refresh_sees_writes_deletes_and_compaction(tmp_path):
    p = tmp_path / "m.pmem"
    w = Memory(p, "debug")
    fill(w, 5)
    w.commit()
    r = Memory(p, read_only=True, auto_refresh=True)
    assert len(r) == 5
    fill(w, 5, start=5)
    w.forget(2, t=100.0)
    w.commit()
    assert len(r) == 5  # nothing refreshed yet (len is a plain count)
    assert r.refresh() == 5 and len(r) == 10
    assert 2 not in [h.frame.id for h in r.search("doc 2 token2", k=20)]
    # auto_refresh: a search sees a brand-new frame without an explicit refresh()
    w.add("freshly written unique zebra", t=300.0)
    w.commit()
    assert r.search("zebra", k=1, mode="lexical")[0].frame.text == "freshly written unique zebra"
    # the writer compacts (replaces the file): the reader reopens transparently
    w.compact()
    w.add("after compaction yak", t=400.0)
    w.commit()
    assert r.search("yak", k=1, mode="lexical")[0].frame.text == "after compaction yak"
    assert r.refresh() == 0
    r.close()
    w.close()


def test_refresh_across_processes(tmp_path):
    p = tmp_path / "m.pmem"
    Memory(p, "debug").close()
    r = Memory(p, read_only=True)
    script = textwrap.dedent(f"""
        from playmemory import Memory
        with Memory({str(p)!r}, "debug") as m:
            m.add_many([f"from the other process {{i}}" for i in range(20)])
    """)
    subprocess.run([sys.executable, "-c", script], check=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.refresh() == 20
    assert len(r.search("other process", k=30, mode="lexical")) == 20
    r.close()


# ---- durability knobs ----------------------------------------------------------
@pytest.mark.parametrize("mode,kw,expect", [("always", {}, lambda n: n), ("batch", {"sync_every": 50}, lambda n: n // 50),
                                            ("never", {}, lambda n: 0)])
def test_fsync_counts_by_mode(tmp_path, monkeypatch, mode, kw, expect):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    m = Memory(tmp_path / "m.pmem", "debug", sync=mode, **kw)
    base = len(calls)  # header fsync at creation
    for i in range(200):
        m.add(f"x {i}")
    assert len(calls) - base == expect(200)
    m.close()


def test_default_sync_every_bounds_loss_window(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    m = Memory(tmp_path / "m.pmem", "debug")  # defaults: batch, sync_every=100
    base = len(calls)
    for i in range(250):
        m.add(f"x {i}")
    assert len(calls) - base == 2  # at most 100 unsynced records can be at risk
    m.close()


# ---- compression and float16 vectors -------------------------------------------
def test_long_records_compress_short_do_not(tmp_path):
    p = tmp_path / "m.pmem"
    long = ("the quick brown fox jumps over the lazy dog " * 40).strip()
    with Memory(p, "debug") as m:
        m.add("short note", t=1.0)
        m.add(long, {"k": "v"}, t=2.0)
        raw = [m.store._read(m.store._offsets[i] + 4, 1)[0] for i in range(2)]
        assert raw == [REC_FRAME, REC_FRAME_Z]
        assert m.get(1).text == long and m.get(1).meta == {"k": "v"}
    with Memory(p, "debug") as m2:  # compressed records survive reopen, scan and search
        assert m2.get(1).text == long
        assert m2.search("lazy dog", k=1, mode="lexical")[0].frame.id == 1
        assert m2.store.verify()[1] == 0


def test_float16_vectors_halve_vector_bytes(tmp_path):
    texts = [f"entry {i} about topic{i % 20} and thing{i % 7}" for i in range(300)]
    sizes = {}
    for dt in ("float32", "float16"):
        p = tmp_path / f"{dt}.pmem"
        with Memory(p, "debug", vec_dtype=dt) as m:
            m.add_many(texts, times=[float(i) for i in range(300)])
            top = m.search("entry 17 about topic17", k=1, mode="vector")[0].frame.id
        sizes[dt] = p.stat().st_size
        with Memory(p, "debug") as m2:  # dtype comes from the file header
            assert m2.search("entry 17 about topic17", k=1, mode="vector")[0].frame.id == top == 17
            assert m2.store.flags == (1 if dt == "float16" else 0)
    assert sizes["float16"] < sizes["float32"] - 300 * 256 * 2 * 0.9


def test_empty_meta_is_not_materialised():
    m = Memory(embedder="debug")
    m.add_many(["a", "b"], metas=[None, {"k": 1}])
    assert m.store.metas == [None, {"k": 1}]
    assert m.get(0).meta == {} and m.find({"k": 1})[0].id == 1


# ---- embedder policy -----------------------------------------------------------
def test_no_embedder_is_an_error_unless_debug_allowed(monkeypatch):
    monkeypatch.delenv("PLAYMEMORY_ALLOW_DEBUG_EMBEDDER", raising=False)
    with pytest.raises(ValueError, match="no embedder given"):
        Memory()
    Memory(embedder="debug")  # explicit opt-in works
    monkeypatch.setenv("PLAYMEMORY_ALLOW_DEBUG_EMBEDDER", "1")
    Memory()


# ---- doctor --------------------------------------------------------------------
def _damaged(tmp_path, kill=(10,)):
    p = tmp_path / "d.pmem"
    with Memory(p, "debug") as m:
        fill(m, 30)
        m.supersede(20, "replacement for twenty", t=40.0)  # frame 30, supersedes 20
        m.forget(5, t=50.0)
        m.forget(12, t=51.0)
        offsets = list(m.store._offsets)
    raw = bytearray(p.read_bytes())
    for k in kill:
        raw[offsets[k] + 12] ^= 0xFF  # corrupt inside frame k's record
    p.write_bytes(bytes(raw))
    return p


def test_doctor_diagnose_and_repair_middle_corruption(tmp_path):
    p = _damaged(tmp_path, kill=(10, 12))
    rep = diagnose(p)
    assert not rep.ok and len(rep.gaps) == 2 and rep.frames_salvaged == 29
    assert rep.frames_lost_at_least == 2 and rep.tombstones_dropped == 1  # frame 12's tombstone
    assert not os.path.exists(str(p) + ".repaired")  # diagnose never writes
    rep = repair(p)
    assert rep.repaired_path.endswith(".repaired")
    with Memory(rep.repaired_path, "debug") as m:
        assert len(m) == 29 and m.store.verify()[1] == 0
        texts = [m.get(i).text for i in range(len(m))]
        assert "doc 10 token0" not in texts and "doc 12 token2" not in texts
        assert m.search("doc 5 token0", k=30, mode="lexical") == [] or all(
            h.frame.text != "doc 5 token0" for h in m.search("doc 5 token0", k=30, mode="lexical"))  # still forgotten
        new = [f for f in (m.get(i) for i in range(len(m))) if f.text == "replacement for twenty"][0]
        old = m.get(new.meta["supersedes"])
        assert old.text == "doc 20 token0"  # pointer remapped to the shifted id
        assert m.store.deleted[old.id] == 40.0


def test_doctor_replace_keeps_backup_and_cli(tmp_path, capsys):
    p = _damaged(tmp_path, kill=(3,))
    assert main([str(p), "doctor"]) == 2
    assert main([str(p), "doctor", "--replace"]) == 2  # report still says damage was found
    assert os.path.exists(str(p) + ".corrupt")
    assert main([str(p), "verify"]) == 0
    with Memory(p, "debug") as m:
        assert len(m) == 30


def test_doctor_healthy_file_is_ok(tmp_path):
    p = tmp_path / "h.pmem"
    with Memory(p, "debug") as m:
        fill(m, 10)
    rep = diagnose(p)
    assert rep.ok and rep.frames_salvaged == 10 and rep.gaps == []


def test_doctor_random_corruption_fuzz(tmp_path):
    rnd = random.Random(7)
    for trial in range(12):
        p = tmp_path / f"f{trial}.pmem"
        with Memory(p, "debug") as m:
            fill(m, 60)
            m.forget(rnd.randrange(60), t=500.0)
            lo, hi = m.store._offsets[0], m.store.nbytes
        raw = bytearray(p.read_bytes())
        for _ in range(rnd.randint(1, 4)):
            raw[rnd.randrange(lo, hi)] ^= 0xFF
        p.write_bytes(bytes(raw))
        rep = repair(p)
        with Memory(rep.repaired_path, "debug") as m2:
            assert m2.store.verify()[1] == 0
            assert len(m2) == rep.frames_salvaged
            assert rep.frames_salvaged >= 40, (trial, rep)  # a few flips never destroy most of the data
