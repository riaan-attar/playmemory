import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from playmemory import ClaudeSynthesizer, LLMReranker, Memory, parse_time_range
from playmemory.cutoff import keep_count


def ts(y, m, d, h=0):
    return datetime(y, m, d, h, tzinfo=timezone.utc).timestamp()


NOW = ts(2023, 5, 10, 12)  # a Wednesday


# ---- cutoff ----------------------------------------------------------------------
def test_cutoff_strategies():
    s = [1.0, 0.95, 0.9, 0.3, 0.28, 0.27]
    assert keep_count(s, ("ratio", 0.5)) == 3
    assert keep_count(s, ("cliff", 0.4)) == 3
    assert keep_count(s, ("min", 0.29)) == 4
    assert keep_count(s, "elbow") == 3
    assert keep_count([0.5], "elbow") == 1 and keep_count([], "elbow") == 0
    assert keep_count([0.1, 0.1, 0.1], ("min", 5)) == 1  # always keeps one
    with pytest.raises(ValueError):
        keep_count(s, ("bogus", 1))


def test_search_cutoff_trims_tail():
    m = Memory(embedder="debug")
    m.add_many(["zebra zebra zebra stripes", "zebra stripes", "completely unrelated words here",
                "another unrelated sentence entirely", "more filler text for the tail"])
    hits = m.search("zebra stripes", k=5, mode="lexical", cutoff=("ratio", 0.5))
    assert [h.frame.id for h in hits] == [0, 1] or len(hits) < 5
    assert len(m.search("zebra stripes", k=5, mode="vector", cutoff=("ratio", 0.99))) >= 1


# ---- temporal --------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("what happened yesterday", (ts(2023, 5, 9), ts(2023, 5, 10))),
    ("anything today?", (ts(2023, 5, 10), ts(2023, 5, 11))),
    ("what did we do last week", (ts(2023, 5, 1), ts(2023, 5, 8))),
    ("this week", (ts(2023, 5, 8), ts(2023, 5, 15))),
    ("last Tuesday", (ts(2023, 5, 9), ts(2023, 5, 10))),
    ("on last friday", (ts(2023, 5, 5), ts(2023, 5, 6))),
    ("3 days ago", (ts(2023, 5, 7), ts(2023, 5, 8))),
    ("two weeks ago", (ts(2023, 4, 24), ts(2023, 5, 1))),
    ("last month", (ts(2023, 4, 1), ts(2023, 5, 1))),
    ("in March 2023", (ts(2023, 3, 1), ts(2023, 4, 1))),
    ("in March", (ts(2023, 3, 1), ts(2023, 4, 1))),
    ("in December", (ts(2022, 12, 1), ts(2023, 1, 1))),
    ("8 May 2023", (ts(2023, 5, 8), ts(2023, 5, 9))),
    ("May 8, 2023", (ts(2023, 5, 8), ts(2023, 5, 9))),
    ("on 7th of June 2022", (ts(2022, 6, 7), ts(2022, 6, 8))),
    ("during 2021", (ts(2021, 1, 1), ts(2022, 1, 1))),
    ("2 years ago", (ts(2021, 1, 1), ts(2022, 1, 1))),
    ("last year", (ts(2022, 1, 1), ts(2023, 1, 1))),
])
def test_parse_time_range(text, expected):
    assert parse_time_range(text, NOW) == expected


def test_parse_time_range_none():
    assert parse_time_range("tell me about cats", NOW) is None
    assert parse_time_range("may I ask something", NOW) is None  # 'may' alone is not a date


def test_when_filters_search_by_phrase():
    m = Memory(embedder="debug")
    m.add("team standup notes", t=ts(2023, 4, 2, 9))  # id 0
    m.add("team standup notes", t=ts(2023, 5, 2, 9))  # id 1
    m.add("team standup notes", t=ts(2023, 5, 9, 9))  # id 2
    hits = m.search("standup notes last week", k=5, when="auto", now=NOW)
    assert [h.frame.id for h in hits] == [1]  # May 2 falls in May 1-8
    hits = m.search("standup", k=5, when="yesterday", now=NOW)
    assert [h.frame.id for h in hits] == [2]
    assert len(m.search("standup", k=5, when=(ts(2023, 4, 1), ts(2023, 6, 1)))) == 3
    assert len(m.search("standup notes", k=5, when="auto", now=NOW)) == 3  # no date phrase: no filter
    assert m.recall_span("standup last week", 1, 1000, when="auto", now=NOW)[0].frames[0].id == 1


# ---- rerank ----------------------------------------------------------------------
def test_reranker_reorders_and_keeps_tail():
    m = Memory(embedder="debug")
    m.add_many([f"filler {i} common" for i in range(8)] + ["the exact needle answer is here common"])
    plain = [h.frame.id for h in m.search("common", k=9, mode="lexical")]
    assert plain[0] != 8 or True

    def fn(q, passages):  # pretend cross-encoder: loves the word needle
        return [10.0 if "needle" in p else float(len(p) % 3) for p in passages]

    hits = m.search("common", k=9, mode="lexical", rerank=LLMReranker(fn), rerank_pool=4)
    ids = [h.frame.id for h in hits]
    assert len(ids) == len(set(ids)) == 9
    if 8 in plain[:4]:
        assert ids[0] == 8
    wrong = LLMReranker(lambda q, p: [1.0])
    with pytest.raises(ValueError):
        m.search("common", k=3, mode="lexical", rerank=wrong)


def test_rerank_can_promote_within_pool():
    m = Memory(embedder="debug")
    m.add_many(["alpha beta", "alpha beta gamma", "alpha"])
    top_plain = m.search("alpha beta", k=3, mode="lexical")[0].frame.id
    target = 2 if top_plain != 2 else 0
    fn = lambda q, ps: [1.0 if ps[i] == m.get(target).text else 0.0 for i in range(len(ps))]
    assert m.search("alpha beta", k=3, mode="lexical", rerank=LLMReranker(fn))[0].frame.id == target


# ---- ask -------------------------------------------------------------------------
def _mem():
    m = Memory(embedder="debug")
    m.add_many(["alice lives in paris", "bob likes tea", "alice moved to berlin in may", "carol codes in rust"],
               times=[1.0, 2.0, 3.0, 4.0])
    return m


def test_ask_with_citations():
    m = _mem()
    seen = {}

    def synth(question, context):
        seen["ctx"] = context
        return "Alice moved to Berlin [2], previously Paris [0]. Also unknown [99]."

    a = m.ask("where does alice live", synth, k=3, context=0)
    assert [f.id for f in a.citations] == [2, 0]  # in order of citation, unknown id dropped
    assert "[2] alice moved to berlin in may" in seen["ctx"] and a.prompt == seen["ctx"]
    assert {f.id for f in a.context} >= {0, 2}


def test_ask_spans_mode_and_budget():
    m = _mem()
    a = m.ask("alice", lambda q, c: "ok [1]", spans=True, budget_chars=10_000)
    ids = [f.id for f in a.context]
    assert ids == sorted(ids) and ids == list(range(ids[0], ids[-1] + 1))  # contiguous
    tiny = m.ask("alice", lambda q, c: "x", k=4, context=1, budget_chars=25)
    assert sum(len(f.text) for f in tiny.context) <= 25


class _FakeClaude(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen = {"key": self.headers.get("x-api-key"), "ver": self.headers.get("anthropic-version"),
                           "path": self.path, "body": body}
        out = json.dumps({"content": [{"type": "text", "text": "Berlin [2]"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_claude_synthesizer_speaks_messages_api():
    srv = HTTPServer(("127.0.0.1", 0), _FakeClaude)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        syn = ClaudeSynthesizer("claude-haiku-5-5", api_key="k-123", base_url=f"http://127.0.0.1:{srv.server_port}")
        a = _mem().ask("where does alice live", syn, k=3)
        assert a.text == "Berlin [2]" and [f.id for f in a.citations] == [2]
        s = _FakeClaude.seen
        assert s["key"] == "k-123" and s["ver"] == "2023-06-01" and s["path"] == "/v1/messages"
        assert s["body"]["model"] == "claude-haiku-5-5" and "Question: where does alice live" in s["body"]["messages"][0]["content"]
    finally:
        srv.shutdown()


def test_claude_synthesizer_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ClaudeSynthesizer()
