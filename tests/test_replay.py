import pytest

from playmemory import Memory

TEXTS = [f"item {i} about topic{i % 4} and word{i % 9}" for i in range(40)]


def seeded(path=None, **kw):
    m = Memory(path, "debug", **kw)
    m.add_many(TEXTS, times=[float(i) for i in range(40)])
    return m


def run_session(m):
    with m.session("debug run") as sid:
        m.search("topic1 word3", k=5, mode="lexical")
        m.add_many(["brand new topic1 word3 arrival"], times=[100.0])
        m.search("topic1 word3", k=5, mode="lexical", context=1)
        m.forget(1, t=101.0)
        m.recall_span("topic2 word7", k=1, budget_chars=300)
        m.search("topic1 word3", k=5, mode="hybrid", as_of=50.0)
    return sid


def test_replay_reproduces_results_even_after_later_changes():
    m = seeded()
    sid = run_session(m)
    # the world moves on AFTER the session: more frames, more deletions
    m.add_many([f"later topic1 word3 doc {i}" for i in range(10)], times=[200.0 + i for i in range(10)])
    m.forget(5, t=300.0)
    m.forget(10, t=301.0)
    diffs = m.replay(sid)
    assert [d.event["op"] for d in diffs] == ["search", "search", "recall_span", "search"]
    assert all(d.same for d in diffs), [(d.original, d.replayed) for d in diffs if not d.same]
    # sanity: without the snapshot the live memory really does answer differently now
    now = [h.frame.id for h in m.search("topic1 word3", k=5, mode="lexical")]
    assert now != diffs[0].original


def test_replay_with_overrides_shows_diff():
    m = seeded()
    sid = run_session(m)
    same = m.replay(sid)
    vec = m.replay(sid, mode="vector")
    assert all(d.event["op"] in ("search", "recall_span") for d in vec)
    changed = [d for d in vec if d.event["op"] == "search" and not d.same]
    assert changed  # a different retrieval mode changes at least one result list
    d = changed[0]
    assert set(d.added) | set(d.removed) or d.reordered
    k2 = m.replay(sid, k=2)
    assert all(len(x.replayed) <= 2 for x in k2 if x.event["op"] == "search")
    assert all(x.same for x in same)


def test_events_logged_with_params_results_and_snapshot_counters():
    m = seeded()
    sid = run_session(m)
    ev = m.events(sid)
    assert [e["op"] for e in ev] == ["session_start", "search", "add", "search", "forget", "recall_span", "search", "session_end"]
    s0 = ev[1]
    assert s0["p"]["query"] == "topic1 word3" and s0["p"]["mode"] == "lexical" and s0["n"] == 40 and s0["nt"] == 0
    assert ev[3]["n"] == 41 and ev[5]["nt"] == 1  # snapshot counters advance with writes
    assert all(isinstance(r[0], int) for r in s0["r"])
    sess = m.sessions()
    assert sess == [{"id": sid, "name": "debug run", "t": sess[0]["t"], "events": 8}]


def test_nothing_logged_outside_sessions_and_double_start_rejected():
    m = seeded()
    m.search("topic1", k=3)
    assert m.events() == [] and m.sessions() == []
    m.start_session("a")
    with pytest.raises(RuntimeError):
        m.start_session("b")
    m.end_session()
    m.end_session()  # idempotent
    assert len(m.sessions()) == 1


def test_sessions_persist_across_reopen_checkpoint_and_tail(tmp_path):
    p = tmp_path / "r.pmem"
    with seeded(p) as m:
        sid1 = run_session(m)
        m.checkpoint()
        with m.session("after checkpoint") as sid2:
            m.search("topic2", k=3, mode="lexical")
    with Memory(p) as m2:
        assert {s["id"] for s in m2.sessions()} == {sid1, sid2}
        assert all(d.same for d in m2.replay(sid1)) and all(d.same for d in m2.replay(sid2))
        assert m2.store.verify()[1] == 0


def test_replay_with_when_uses_recorded_clock():
    m = Memory(embedder="debug")
    day = 86400.0
    t0 = 1683000000.0
    m.add_many(["standup notes"] * 3, times=[t0, t0 + 3 * day, t0 + 6 * day])
    with m.session() as sid:
        m.search("standup yesterday", k=5, when="auto", now=t0 + 4 * day)
    d = m.replay(sid)[0]
    assert d.same and d.original == [1]  # 'yesterday' relative to the RECORDED clock is day 3, i.e. frame 1


def test_replay_ignores_read_only_handles_and_writes():
    m = seeded()
    sid = run_session(m)
    ops = [d.event["op"] for d in m.replay(sid)]
    assert "add" not in ops and "forget" not in ops


def test_compact_keep_history_keeps_events_and_replay_still_matches(tmp_path):
    p = tmp_path / "c.pmem"
    with seeded(p) as m:
        sid = run_session(m)
        m.compact(keep_history=True)
        assert [e["op"] for e in m.events(sid)][0] == "session_start"
        assert all(d.same for d in m.replay(sid))
        m.compact()  # dropping history renumbers frames: the audit trail is discarded, by design
        assert m.sessions() == []


def test_encrypted_events_are_unreadable_but_replayable(tmp_path):
    pytest.importorskip("cryptography")
    p = tmp_path / "e.pmem"
    with Memory(p, "debug", password="pw") as m:
        m.add_many(TEXTS, times=[float(i) for i in range(40)])
        with m.session("secret session") as sid:
            m.search("topic1 secretquery", k=3, mode="lexical")
    raw = p.read_bytes()
    assert b"secretquery" not in raw and b"secret session" not in raw
    with Memory(p, password="pw") as m2:
        assert m2.sessions()[0]["name"] == "secret session" and all(d.same for d in m2.replay(sid))


def test_principal_recorded_for_audit():
    from playmemory import Principal
    m = Memory(embedder="debug", principal=Principal("acme", {"eng"}))
    m.add("acme note", {"tenant": "acme"})
    with m.session() as sid:
        m.search("note", k=1, mode="lexical")
    assert m.events(sid)[1]["who"] == {"tenant": "acme", "roles": ["eng"]}
