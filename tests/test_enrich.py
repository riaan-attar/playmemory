import json

import numpy as np
import pytest

from playmemory import Memory
from playmemory.cli import main
from playmemory.doctor import repair

TEXTS = [f"document {i} concerning topic{i % 5} and thing{i % 8}" for i in range(30)]


def test_deferred_frames_are_keyword_searchable_then_vector_searchable():
    m = Memory(embedder="debug")
    m.add_many(TEXTS, defer_embedding=True, times=[float(i) for i in range(30)])
    assert m.stats()["pending_embeddings"] == 30
    assert m.search("topic3 thing6", k=3, mode="lexical")  # keyword search works immediately
    assert m.search("topic3 thing6", k=3, mode="vector")[0].score == 0.0  # no embeddings yet
    assert m.enrich(batch=7) == 30 and m.stats()["pending_embeddings"] == 0
    top = m.search(TEXTS[13], k=1, mode="vector")[0]
    assert top.frame.id == 13 and top.score > 0.99
    assert m.enrich() == 0


def test_enriched_vectors_persist_through_reopen_checkpoint_and_compaction(tmp_path):
    p = tmp_path / "e.pmem"
    with Memory(p, "debug") as m:
        m.add_many(TEXTS[:10], times=list(map(float, range(10))))  # embedded normally
        m.add_many(TEXTS[10:], defer_embedding=True, times=[float(i) for i in range(10, 30)])
        assert m.enrich(limit=5) == 5
        m.checkpoint()
        m.enrich()  # the rest AFTER the checkpoint: replayed from the tail
    with Memory(p, "debug") as m2:
        assert m2.stats()["pending_embeddings"] == 0
        for i in (3, 12, 25):
            h = m2.search(TEXTS[i], k=1, mode="vector")[0]
            assert h.frame.id == i and h.score > 0.99
        m2.compact()
        assert m2.search(TEXTS[27], k=1, mode="vector")[0].frame.id == 27 and m2.stats()["pending_embeddings"] == 0
    with Memory(p, "debug") as m3:
        assert m3.search(TEXTS[21], k=1, mode="vector")[0].frame.id == 21


def test_pending_survives_reopen(tmp_path):
    p = tmp_path / "p.pmem"
    with Memory(p, "debug") as m:
        m.add_many(TEXTS[:6], defer_embedding=True)
    with Memory(p, "debug") as m2:
        assert m2.stats()["pending_embeddings"] == 6
        assert m2.enrich() == 6


def test_reader_sees_enrichment_via_refresh(tmp_path):
    p = tmp_path / "r.pmem"
    w = Memory(p, "debug")
    w.add_many(TEXTS[:10], defer_embedding=True)
    w.commit()
    r = Memory(p, read_only=True)
    assert r.search(TEXTS[4], k=1, mode="vector")[0].score == 0.0
    w.enrich()
    w.commit()
    r.refresh()
    assert r.search(TEXTS[4], k=1, mode="vector")[0].frame.id == 4 and r.stats()["pending_embeddings"] == 0
    r.close()
    w.close()


def test_encrypted_enrichment(tmp_path):
    pytest.importorskip("cryptography")
    p = tmp_path / "x.pmem"
    with Memory(p, "debug", password="pw") as m:
        m.add_many(TEXTS[:12], defer_embedding=True, times=list(map(float, range(12))))
        m.enrich()
    with Memory(p, password="pw") as m2:
        assert m2.search(TEXTS[7], k=1, mode="vector")[0].frame.id == 7


def test_float16_enrichment(tmp_path):
    p = tmp_path / "h.pmem"
    with Memory(p, "debug", vec_dtype="float16") as m:
        m.add_many(TEXTS[:12], defer_embedding=True)
        m.enrich()
    with Memory(p) as m2:
        assert m2.search(TEXTS[9], k=1, mode="vector")[0].frame.id == 9


def test_doctor_keeps_enriched_vectors(tmp_path):
    p = tmp_path / "d.pmem"
    with Memory(p, "debug") as m:
        m.add_many(TEXTS[:20], defer_embedding=True, times=list(map(float, range(20))))
        m.enrich()
        off = m.store._offsets[3]
    raw = bytearray(p.read_bytes())
    raw[off + 12] ^= 0xFF  # damage frame 3's record
    p.write_bytes(bytes(raw))
    rep = repair(p)
    with Memory(rep.repaired_path, "debug") as m2:
        assert len(m2) == 19 and m2.stats()["pending_embeddings"] == 0  # vectors were not lost
        assert m2.search(TEXTS[10], k=1, mode="vector")[0].score > 0.99


def test_ingest_defer_and_cli_enrich(tmp_path, capsys):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# A\n\nalpha zebra text here\n")
    f = str(tmp_path / "m.pmem")
    assert main([f, "--embedder", "hash-256", "ingest", str(d), "--defer-embedding"]) == 0
    capsys.readouterr()
    assert main([f, "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["pending_embeddings"] == 1
    assert main([f, "search", "zebra", "--mode", "lexical", "-k", "1"]) == 0
    assert "alpha zebra" in capsys.readouterr().out
    assert main([f, "enrich"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["embedded"] == 1 and out["pending_embeddings"] == 0
