import numpy as np
import pytest

from playmemory import Memory

faiss = pytest.importorskip("faiss")


def _texts(n, rng):
    words = [f"w{i}" for i in range(500)]
    return [" ".join(rng.choice(words, 6)) for _ in range(n)]


def test_ann_persist_incremental_and_recall(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "m.pmem"
    texts = _texts(3000, rng)
    m = Memory(p, ann_threshold=1000)
    m.add_many(texts, times=list(map(float, range(3000))))
    assert m.ann is None
    m.checkpoint()  # crosses the threshold -> builds and persists the graph
    assert m.stats()["ann"]
    m.add_many(_texts(100, rng), times=[4000.0 + i for i in range(100)])
    m.close()

    m2 = Memory(p, ann_threshold=1000)
    assert m2.ann is not None and len(m2.ann) == 3100  # loaded + tail added
    hits_ann = m2.search(texts[42], k=5, mode="vector")
    hits_exact = m2.search(texts[42], k=5, mode="vector", exact=True)
    assert hits_ann[0].frame.id == hits_exact[0].frame.id
    # tie-tolerant: every ANN hit must score within eps of the exact 5th best
    kth = hits_exact[-1].score - 1e-5
    exact_all = {h.frame.id: h.score for h in m2.search(texts[42], k=3100, mode="vector", exact=True)}
    assert sum(exact_all[h.frame.id] >= kth for h in hits_ann) >= 4
    # deletions are honoured even though the graph still contains the node
    top = hits_ann[0].frame.id
    m2.forget(top, t=9999.0)
    assert top not in [h.frame.id for h in m2.search(texts[42], k=5, mode="vector")]
    m2.close()


def test_compact_rebuilds_ann(tmp_path):
    rng = np.random.default_rng(1)
    m = Memory(tmp_path / "m.pmem", ann_threshold=300)
    m.add_many(_texts(800, rng), times=list(map(float, range(800))))
    for i in range(0, 800, 2):
        m.forget(i, t=1000.0)
    m.compact()
    assert m.ann is not None and len(m.ann) == 400
    m.close()


def test_where_filter_index_matches_scan():
    m = Memory()
    m.add_many([f"item {i}" for i in range(30)],
               metas=[{"k": i % 3, "tags": ["x"]} for i in range(30)],
               times=list(map(float, range(30))))
    ids = sorted(h.frame.id for h in m.search("item", k=30, where={"k": 1}))
    assert ids == [i for i in range(30) if i % 3 == 1]
    assert len(m.search("item", k=30, where={"k": 1, "tags": ["x"]})) == 10  # unhashable path
    assert m.search("item", k=30, where={"missing": 1}) == []
    m.forget(1, t=100.0)
    assert 1 not in [h.frame.id for h in m.search("item", k=30, where={"k": 1})]


def test_lexical_scores_rank_rare_terms_higher():
    m = Memory()
    m.add_many(["common common common", "common rare", "other words"])
    hits = m.search("rare common", k=3, mode="lexical")
    assert hits[0].frame.text == "common rare"
