import numpy as np

from playmemory import EpisodeConfig, Memory
from playmemory.episodes import EpisodeIndex

TOPICS = {
    "murder": "detective killed victim knife blood suspect police mansion night body".split(),
    "cooking": "recipe oven flour sugar butter bake dough kitchen cake spoon".split(),
    "space": "rocket orbit planet astronaut launch moon telescope galaxy comet star".split(),
    "garden": "tomato soil seeds watering roses weeds compost greenhouse sprout leaf".split(),
}


def stream(order, seed=0, per=(14, 18, 12, 16)):
    """Frames made of mostly-topic words; returns texts and the true start of each run."""
    rng = np.random.default_rng(seed)
    texts, starts = [], []
    for name, n in zip(order, per):
        starts.append(len(texts))
        for _ in range(n):
            texts.append(" ".join(rng.choice(TOPICS[name], 6)) + " " + rng.choice(["ok", "so", "then", "and"]))
    return texts, starts


def build(order=("murder", "cooking", "space", "garden"), **cfg):
    texts, truth = stream(order)
    m = Memory(episode_config=EpisodeConfig(boundary_key=None, **cfg))
    m.add_many(texts, times=[float(i) for i in range(len(texts))])
    return m, truth, texts


def test_segmentation_finds_topic_boundaries():
    m, truth, _ = build()
    starts = [a for a, _ in m.episode_list()]
    for t in truth[1:]:
        assert any(abs(s - t) <= 2 for s in starts), (t, starts)
    assert len(starts) <= len(truth) + 2  # no wild over-segmentation


def test_forced_boundary_on_session_change():
    m = Memory(episode_config=EpisodeConfig(boundary_key="session"))
    m.add_many([f"same topic words {i}" for i in range(20)],
               metas=[{"session": i // 10} for i in range(20)], times=list(map(float, range(20))))
    assert 10 in [a for a, _ in m.episode_list()]


def test_max_len_and_min_len_respected():
    m, _, _ = build(max_len=8, min_len=3)
    sizes = [b - a + 1 for a, b in m.episode_list()]
    assert max(sizes) <= 8 + 3  # split at weakest valley within [min_len, max_len]
    assert min(sizes) >= 3


def test_recall_span_returns_contiguous_scene_within_budget():
    m, truth, texts = build()
    cooking = range(truth[1], truth[2])
    spans = m.recall_span("recipe oven flour dough bake", k=1, budget_chars=400)
    (sp,) = spans
    ids = [f.id for f in sp.frames]
    assert ids == list(range(sp.start_id, sp.end_id + 1))  # contiguous
    assert sp.chars <= 400 + 80
    assert sum(i in cooking for i in ids) >= 0.8 * len(ids)  # stays in the right scene
    assert sp.t0 == sp.frames[0].t and sp.t1 == sp.frames[-1].t
    assert sp.summary


def test_big_budget_returns_whole_scene_small_budget_is_partial():
    m, truth, _ = build()
    (big,) = m.recall_span("detective knife victim suspect", k=1, budget_chars=100_000)
    (small,) = m.recall_span("detective knife victim suspect", k=1, budget_chars=120)
    assert big.end_id - big.start_id + 1 == big.episode_len
    assert len(small.frames) < len(big.frames)
    assert small.start_id >= big.start_id and small.end_id <= big.end_id


def test_multi_span_for_multi_hop():
    m, truth, _ = build()
    spans = m.recall_span("rocket orbit moon and also tomato soil seeds", k=2, budget_chars=3000)
    assert len(spans) == 2 and spans[0].start_id < spans[1].start_id
    covered = {f.id for sp in spans for f in sp.frames}
    assert covered & set(range(truth[2], truth[3])) and covered & set(range(truth[3], 60))


def test_spans_respect_forget_and_as_of():
    m, truth, _ = build()
    (first,) = m.recall_span("recipe oven flour dough bake", k=1, budget_chars=100_000)
    gone = first.frames[len(first.frames) // 2].id
    m.forget(gone, t=1000.0)
    (sp,) = m.recall_span("recipe oven flour dough bake", k=1, budget_chars=100_000)
    assert gone not in [f.id for f in sp.frames]
    assert sp.start_id <= gone <= sp.end_id  # pointers still span the gap
    (past,) = m.recall_span("recipe oven flour dough bake", k=1, budget_chars=100_000, as_of=999.0)
    assert gone in [f.id for f in past.frames]
    assert m.read(sp.start_id, sp.end_id) == list(sp.frames)
    cutoff = float(first.start_id)
    early = m.recall_span("rocket orbit moon", k=1, budget_chars=100_000, as_of=cutoff)
    assert all(f.t <= cutoff for sp2 in early for f in sp2.frames)


def test_incremental_sync_close_to_full_rebuild():
    texts, truth = stream(("murder", "cooking", "space", "garden"))
    cfg = EpisodeConfig(boundary_key=None)
    m = Memory(episode_config=cfg)
    for i in range(0, len(texts), 5):  # grow in small batches
        m.add_many(texts[i : i + 5], times=[float(j) for j in range(i, min(i + 5, len(texts)))])
        m.episode_list()
    inc = [a for a, _ in m.episode_list()]
    full = EpisodeIndex(cfg)
    full.sync(m.index.vectors, m.store.metas, m.store.times)
    for t in truth[1:]:
        assert any(abs(s - t) <= 2 for s in inc)
    assert len(inc) <= len(full.starts) + 2


def test_episodes_persist_through_checkpoint_and_compaction(tmp_path):
    texts, truth = stream(("murder", "cooking", "space", "garden"))
    cfg = EpisodeConfig(boundary_key=None)
    p = tmp_path / "m.pmem"
    m = Memory(p, episode_config=cfg)
    m.add_many(texts, times=[float(i) for i in range(len(texts))])
    before = m.episode_list()
    m.checkpoint()
    m.close()
    m2 = Memory(p, episode_config=cfg)
    assert m2.episodes.starts == [a for a, _ in before]  # restored, not recomputed
    assert m2.episode_list() == before
    m2.forget(0, t=500.0)
    m2.compact()
    assert len(m2) == len(texts) - 1
    assert m2.episode_list()[0][0] == 0
    (sp,) = m2.recall_span("recipe oven flour dough bake", k=1, budget_chars=300)
    assert sp.frames
    m2.close()
