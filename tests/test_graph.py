import pytest

from playmemory import Memory
from playmemory.cards import Card, LLMExtractor, remember
from playmemory.graph import Graph, Schema, SchemaError, norm


def facts():
    """Cards fed through the LLM-extractor adapter so the test controls them precisely."""
    rows = {
        "t1": [dict(kind="fact", subject="Alice", predicate="works_at", value="Acme")],
        "t2": [dict(kind="fact", subject="Acme", predicate="located_in", value="Berlin")],
        "t3": [dict(kind="relationship", subject="Alice", predicate="manager_is", value="Bob", exclusive=True)],
        "t4": [dict(kind="fact", subject="Bob", predicate="lives_in", value="Paris")],
        "t5": [dict(kind="preference", subject="Alice", predicate="likes", value="tea", exclusive=False),
               dict(kind="preference", subject="Alice", predicate="likes", value="rust", exclusive=False)],
    }
    mem = Memory(embedder="debug")
    ex = LLMExtractor(lambda text: rows[text])
    for i, key in enumerate(rows):
        remember(mem, key, ex, t=float(i + 1))
    return mem


def test_triples_filters_and_time_travel():
    g = Graph(facts())
    assert {(t.subject, t.predicate, t.object) for t in g.triples(subject="alice")} == {
        ("Alice", "works_at", "Acme"), ("Alice", "manager_is", "Bob"), ("Alice", "likes", "tea"), ("Alice", "likes", "rust")}
    assert [t.object for t in g.triples(predicate="located_in")] == ["Berlin"]
    assert [t.subject for t in g.triples(obj="ACME")] == ["Alice"]
    assert g.triples(as_of=0.5) == []
    assert {t.subject for t in g.triples(as_of=2.5)} == {"Alice", "Acme"}


def test_superseded_facts_leave_the_graph_but_not_history():
    m = facts()
    ex = LLMExtractor(lambda t: [dict(kind="fact", subject="Alice", predicate="works_at", value="Globex")])
    remember(m, "moved jobs", ex, t=20.0)
    g = Graph(m)
    assert [t.object for t in g.triples(subject="alice", predicate="works_at")] == ["Globex"]
    assert [t.object for t in g.triples(subject="alice", predicate="works_at", as_of=10.0)] == ["Acme"]


def test_entities_neighbors_related():
    g = Graph(facts())
    ents = g.entities()
    assert {"alice", "acme", "bob"} <= ents and "tea" not in ents and "berlin" not in ents
    nb = g.neighbors("Acme")
    assert {(n["direction"], n["predicate"], n["entity"]) for n in nb} == {("out", "located_in", "Berlin"), ("in", "works_at", "Alice")}
    rel = g.related("Alice", hops=2)
    assert set(rel) >= {"acme", "bob"}
    assert rel["acme"] == [("alice", "works_at", "acme")]
    assert rel["bob"] == [("alice", "manager_is", "bob")]
    assert "paris" not in rel and "berlin" not in rel  # values that are not entities are not nodes
    deeper = g.related("Acme", hops=2)
    assert deeper["bob"][-1] == ("alice", "manager_is", "bob") and len(deeper["bob"]) == 2


def test_schema_cardinality_types_and_domain():
    s = (Schema().declare("lives_in", "one", "entity", domain=["fact"])
         .declare("likes", "many").declare("age", "one", "int").declare("born", "one", "date"))
    mem = Memory(embedder="debug")
    ex = lambda rows: LLMExtractor(lambda t: rows)
    remember(mem, "a", ex([dict(kind="fact", subject="Ann", predicate="lives_in", value="Rome")]), t=1.0, schema=s)
    remember(mem, "b", ex([dict(kind="fact", subject="Ann", predicate="lives_in", value="Oslo")]), t=2.0, schema=s)
    remember(mem, "c", ex([dict(kind="fact", subject="Ann", predicate="likes", value="tea", exclusive=True)]), t=3.0, schema=s)
    remember(mem, "d", ex([dict(kind="fact", subject="Ann", predicate="likes", value="jazz")]), t=4.0, schema=s)
    g = Graph(mem, s)
    assert [t.object for t in g.triples(subject="ann", predicate="lives_in")] == ["Oslo"]  # cardinality one: superseded
    assert sorted(t.object for t in g.triples(subject="ann", predicate="likes")) == ["jazz", "tea"]  # many: kept
    assert {"ann", "oslo"} <= g.entities()  # schema says lives_in values are entities
    for bad in (dict(kind="fact", subject="Ann", predicate="age", value="forty"),
                dict(kind="fact", subject="Ann", predicate="born", value="May 3rd"),
                dict(kind="goal", subject="Ann", predicate="lives_in", value="Rome")):
        with pytest.raises(SchemaError):
            remember(mem, "x", ex([bad]), t=9.0, schema=s)
    remember(mem, "ok", ex([dict(kind="profile", subject="Ann", predicate="age", value="41")]), t=10.0, schema=s)
    strict = Schema(strict=True)
    with pytest.raises(SchemaError, match="not declared"):
        strict.validate(Card("fact", "x", "whatever", "y"))
    with pytest.raises(ValueError):
        Schema().declare("p", cardinality="several")


def test_mentioned_and_graph_aware_search():
    m = facts()
    m.add("unrelated gossip about the weather", t=30.0)
    g = Graph(m)
    assert g.mentioned("where does Alice's manager live? ask bob") == ["alice", "bob"]
    assert g.mentioned("nothing here") == []
    # lexical search for 'manager' alone finds only the manager card; the graph pulls in Bob's facts
    plain = {h.frame.meta.get("predicate") for h in m.search("what is known about Bob", k=6)}
    boosted = {h.frame.meta.get("predicate") for h in m.search("what is known about Bob", k=6, graph=g)}
    assert "lives_in" in boosted and "manager_is" in boosted
    assert len(boosted & {"lives_in", "manager_is"}) >= len(plain & {"lives_in", "manager_is"})
    ids = [f for f, _ in g.hits("tell me about Bob")]
    assert ids and all(m.store.is_live(i) for i in ids)


def test_norm():
    assert norm("  Alice   SMITH ") == "alice smith"
