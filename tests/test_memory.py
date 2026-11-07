import numpy as np
import pytest

from playmemory import Memory


def test_add_and_seek_in_memory():
    m = Memory()
    for i in range(10):
        m.add(f"event {i}", t=float(i))
    assert m.seek(3.5).text == "event 3"
    assert m.seek(-1) is None
    assert [f.text for f in m.window(2, 4)] == ["event 2", "event 3", "event 4"]


def test_monotonic_time_enforced():
    m = Memory()
    m.add("a", t=5.0)
    with pytest.raises(ValueError):
        m.add("b", t=1.0)


def test_search_with_context():
    m = Memory()
    texts = ["the cat sat", "dogs bark loudly", "pizza with cheese", "stars in the sky", "rain on roof"]
    m.add_many(texts, times=[float(i) for i in range(5)])
    (hit,) = m.search("cheese pizza", k=1, context=1)
    assert hit.frame.text == "pizza with cheese"
    assert [f.text for f in hit.before] == ["dogs bark loudly"]
    assert [f.text for f in hit.after] == ["stars in the sky"]


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "mem.pmem"
    m = Memory(p)
    m.add_many(["alpha one", "beta two", "gamma three"], metas=[{"a": 1}, None, None], times=[1.0, 2.0, 3.0])
    m.close()
    m2 = Memory(p)
    assert len(m2) == 3
    assert m2.get(0).meta == {"a": 1}
    assert m2.search("beta", k=1)[0].frame.text == "beta two"
    m2.add("delta four", t=4.0)
    assert m2.seek(10).text == "delta four"
    m2.close()
    assert [x.name for x in tmp_path.iterdir()] == ["mem.pmem"]  # single file


def _facts():
    m = Memory()
    m.add_many(
        ["alice lives in paris", "bob likes tea", "carol works at acme"],
        metas=[{"who": "alice"}, {"who": "bob"}, {"who": "carol"}],
        times=[1.0, 2.0, 3.0],
    )
    return m


def test_hybrid_lexical_and_vector_modes():
    m = _facts()
    for mode in ("hybrid", "vector", "lexical"):
        assert m.search("tea", k=1, mode=mode)[0].frame.text == "bob likes tea"


def test_where_filter():
    m = _facts()
    hits = m.search("lives works likes", k=3, where={"who": "carol"})
    assert [h.frame.meta["who"] for h in hits] == ["carol"]


def test_supersede_and_time_travel():
    m = Memory()
    old = m.add("alice lives in paris", t=1.0)
    new = m.supersede(old.id, "alice lives in berlin", t=5.0)
    assert new.meta["supersedes"] == old.id
    now = m.search("where does alice live", k=3)
    assert [h.frame.text for h in now] == ["alice lives in berlin"]
    past = m.search("where does alice live", k=3, as_of=3.0)
    assert [h.frame.text for h in past] == ["alice lives in paris"]
    assert m.get(old.id).text == "alice lives in paris"  # history kept


def test_forget_and_seek_skip_deleted():
    m = _facts()
    m.forget(1, t=2.2)
    assert m.seek(2.1).text == "bob likes tea"  # still alive then
    assert m.seek(2.5).text == "alice lives in paris"
    assert [f.text for f in m.window(0, 5)] == ["alice lives in paris", "carol works at acme"]
    assert m.search("tea", k=3, mode="lexical") == []


def test_recency_decay_prefers_newer():
    m = Memory()
    m.add("project status update", t=0.0)
    m.add("project status update", t=100.0)
    hits = m.search("project status", k=2, recency_half_life=10.0)
    assert hits[0].frame.t == 100.0


def test_tombstones_persist(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    a = m.add("keep me", t=1.0)
    b = m.add("drop me", t=2.0)
    m.forget(b.id, t=3.0)
    m.close()
    m2 = Memory(p)
    assert [h.frame.id for h in m2.search("me", k=5)] == [a.id]
    assert [h.frame.id for h in m2.search("me", k=5, as_of=2.5)][0:2] in ([a.id, b.id], [b.id, a.id])


def test_corrupt_tail_is_truncated(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p)
    m.add_many(["one", "two", "three"], times=[1.0, 2.0, 3.0])
    m.close()
    raw = p.read_bytes()
    p.write_bytes(raw[:-3] + b"\x00\x00\x00")  # corrupt last record's crc area
    m2 = Memory(p)
    assert len(m2) == 2
    m2.add("four", t=4.0)
    m2.close()
    assert len(Memory(p)) == 3


def test_embedder_mismatch_rejected(tmp_path):
    from playmemory import HashEmbedder
    from playmemory.store import ModelMismatchError

    p = tmp_path / "m.pmem"
    Memory(p, HashEmbedder(128)).close()
    with pytest.raises(ModelMismatchError):
        Memory(p, HashEmbedder(256))
