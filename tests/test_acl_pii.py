import pytest

from playmemory import Memory, Principal, mask_pii
from playmemory.acl import allowed

ACME = Principal(tenant="acme", roles={"engineer"})
GLOBEX = Principal(tenant="globex")
ADMIN_ACME = Principal(tenant="acme", roles={"admin", "engineer"})


def build():
    m = Memory(embedder="debug")
    rows = [
        ("acme roadmap q3 public note", {"tenant": "acme"}),
        ("acme salary bands restricted to admins", {"tenant": "acme", "visibility": "restricted", "roles": ["admin"]}),
        ("acme engineering design doc", {"tenant": "acme", "visibility": "restricted", "roles": ["engineer"]}),
        ("globex roadmap q3 secret plans", {"tenant": "globex"}),
        ("shared company handbook roadmap", {}),  # untenanted
        ("acme standup roadmap notes", {"tenant": "acme"}),
    ]
    m.add_many([t for t, _ in rows], metas=[x for _, x in rows], times=[float(i) for i in range(len(rows))])
    return m


def ids(hits):
    return sorted(h.frame.id for h in hits)


def test_allowed_rule():
    assert allowed(ACME, {"tenant": "acme"})
    assert not allowed(ACME, {"tenant": "globex"})
    assert not allowed(ACME, {})  # untenanted hidden unless allow_untenanted
    assert allowed(Principal(tenant="acme", allow_untenanted=True), {})
    assert not allowed(ACME, {"tenant": "acme", "visibility": "restricted", "roles": ["admin"]})
    assert allowed(Principal(tenant="acme", groups={"g1"}), {"tenant": "acme", "visibility": "restricted", "groups": ["g1"]})


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_search_is_tenant_and_role_isolated(mode):
    m = build()
    mine = set(ids(m.search("roadmap", k=10, mode=mode, principal=ACME)))
    assert mine <= {0, 2, 5} and {0, 5} <= mine
    got = set(ids(m.search("roadmap salary design secret handbook", k=10, mode=mode, principal=ACME)))
    assert got <= {0, 2, 5} and 1 not in got and 3 not in got and 4 not in got
    got = set(ids(m.search("roadmap salary design secret handbook", k=10, mode=mode, principal=ADMIN_ACME)))
    assert got <= {0, 1, 2, 5} and 3 not in got
    assert set(ids(m.search("roadmap secret", k=10, mode=mode, principal=GLOBEX))) <= {3}
    unrestricted = set(ids(m.search("roadmap salary design secret handbook", k=10, mode=mode)))
    assert {1, 3, 4} <= unrestricted  # no principal = admin view


def test_neighbours_context_does_not_leak():
    m = build()
    (hit,) = m.search("standup roadmap notes", k=1, context=3, mode="lexical", principal=ACME)
    leaked = {f.id for f in hit.context()} & {1, 3, 4}
    assert not leaked and hit.frame.id == 5
    assert [f.id for f in hit.context()] == [0, 2, 5]  # walks past hidden frames to find 3 visible ones, in order


def test_spans_get_seek_window_read_respect_acl():
    m = Memory(embedder="debug", principal=ACME)
    m.add_many(["roadmap item one", "secret globex roadmap", "roadmap item two", "roadmap item three"],
               metas=[{"tenant": "acme"}, {"tenant": "globex"}, {"tenant": "acme"}, {"tenant": "acme"}],
               times=[1.0, 2.0, 3.0, 4.0], )
    sp = m.recall_span("roadmap", k=1, budget_chars=10_000)[0]
    assert 1 not in [f.id for f in sp.frames] and sp.episode_len == 3
    with pytest.raises(PermissionError):
        m.get(1)
    assert m.get(0).text == "roadmap item one"
    assert m.seek(2.5).id == 0  # frame 1 invisible: falls back to the last visible one
    assert [f.id for f in m.window(0, 10)] == [0, 2, 3]
    assert [f.id for f in m.read(0, 3)] == [0, 2, 3]
    assert [f.id for f in m.find({"tenant": "globex"})] == []
    assert [f.id for f in m.find({"tenant": "globex"}, principal=GLOBEX)] == [1]  # explicit principal overrides


def test_acl_composes_with_as_of_and_forget():
    m = build()
    m.forget(0, t=100.0)
    assert 0 not in ids(m.search("roadmap", k=10, mode="lexical", principal=ACME))
    assert 0 in ids(m.search("roadmap", k=10, mode="lexical", principal=ACME, as_of=50.0))
    assert 3 not in ids(m.search("roadmap", k=10, mode="lexical", principal=ACME, as_of=50.0))


def test_cards_and_ask_honour_principal():
    m = build()
    seen = {}
    ans = m.ask("what is the roadmap", lambda q, c: seen.setdefault("c", c) or "ok", k=10, context=2,
                mode="lexical", principal=ACME)
    assert "globex" not in seen["c"] and "salary" not in seen["c"]
    assert not any(f.id in (1, 3, 4) for f in ans.context)


# ---- PII ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("mail bob@example.com now", "mail [EMAIL] now"),
    ("ssn 123-45-6789 ok", "ssn [SSN] ok"),
    ("server 192.168.1.20 down", "server [IP] down"),
    ("card 4111 1111 1111 1111 on file", "card [CARD] on file"),
    ("card 4111-1111-1111-1111", "card [CARD]"),
    ("call +1 (415) 555-0100 today", "call [PHONE] today"),
    ("call 020 7946 0958", "call [PHONE]"),
    ("order 1234567890123456 shipped", "order 1234567890123456 shipped"),  # 16 digits but fails Luhn: not a card
    ("version 1.2.3 and year 2024", "version 1.2.3 and year 2024"),
    ("999.1.1.1 is not an ip", "999.1.1.1 is not an ip"),
    ("nothing sensitive here", "nothing sensitive here"),
])
def test_mask_pii(text, expected):
    assert mask_pii(text) == expected


def test_mask_pii_kinds_subset():
    t = "bob@example.com 123-45-6789"
    assert mask_pii(t, kinds=("email",)) == "[EMAIL] 123-45-6789"


def test_redact_applies_to_every_output_path_but_not_storage():
    m = Memory(embedder="debug", redact=True)
    m.add_many(["contact alice@example.com about the zebra project", "phone +1 415 555 0100 for zebra support"],
               times=[1.0, 2.0])
    h = m.search("alice zebra", k=1, mode="lexical", context=1)[0]  # searchable by the real value...
    assert h.frame.text == "contact [EMAIL] about the zebra project"  # ...but returned masked
    assert "[PHONE]" in h.after[0].text
    assert "[EMAIL]" in m.recall_span("zebra", 1, 1000)[0].frames[0].text
    assert m.get(0).text == "contact [EMAIL] about the zebra project"
    assert "[PHONE]" in m.seek(5.0).text and "[EMAIL]" in m.window(0, 5)[0].text
    assert "alice@example.com" in m.store.get(0).text  # stored text untouched
    custom = Memory(embedder="debug", redact=lambda t: t.upper())
    custom.add("quiet note")
    assert custom.get(0).text == "QUIET NOTE"


def test_mcp_server_principal_and_redaction():
    import json
    from playmemory.server import MemoryServer
    m = build()
    m.add("mail me at bob@example.com about the roadmap", {"tenant": "acme"}, t=10.0)
    m.principal, m.redact = ACME, True
    srv = MemoryServer(m)
    r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "recall", "arguments": {"query": "roadmap", "k": 10, "mode": "lexical"}}})
    hits = json.loads(r["result"]["content"][0]["text"])
    texts = " ".join(h["frame"]["text"] for h in hits)
    assert "globex" not in texts and "[EMAIL]" in texts and "bob@example.com" not in texts
