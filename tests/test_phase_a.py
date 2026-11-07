import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from playmemory import HashEmbedder, HTTPEmbedder, Memory, from_spec
from playmemory import store as store_mod
from playmemory.cli import main
from playmemory.lexical import tokenize
from playmemory.store import MAGIC, read_header


# ---- A1: unicode lexical search ----------------------------------------------
def test_tokenizer_unicode():
    assert tokenize("Café NAÏVE") == ["cafe", "naive"]
    assert tokenize("привет мир") == ["привет", "мир"]
    assert tokenize("東京に行きたい") == ["東京", "京に", "に行", "行き", "きた", "たい"]
    assert tokenize("東") == ["東"]
    assert tokenize("hello東京world") == ["hello", "東京", "world"]
    assert tokenize("ｆｕｌｌｗｉｄｔｈ ① ") == ["fullwidth", "1"]  # NFKC
    assert tokenize("한국어") == ["한국", "국어"]


def test_lexical_search_non_english():
    m = Memory(embedder=HashEmbedder())
    m.add_many(["東京に行きたい", "привет мир", "un café à Paris", "plain english text"])
    for q, want in [("東京", 0), ("привет", 1), ("cafe", 2), ("CAFÉ", 2)]:
        hits = m.search(q, mode="lexical")
        assert hits and hits[0].frame.id == want, q


def test_old_checkpoint_tokenizer_triggers_reindex(tmp_path):
    p = tmp_path / "m.pmem"
    m = Memory(p, HashEmbedder())
    m.add_many(["東京に行きたい", "hello world"])
    m.checkpoint()
    m.close()
    # simulate a checkpoint written by an older tokenizer: patch the version and reopen
    import playmemory.memory as mm
    old = mm.TOKENIZER_VERSION
    mm.TOKENIZER_VERSION = old + 1
    try:
        with Memory(p, HashEmbedder()) as m2:
            assert m2.search("東京", mode="lexical")[0].frame.id == 0  # rebuilt from the log
    finally:
        mm.TOKENIZER_VERSION = old


# ---- A2: windows-style IO fallback -------------------------------------------
def test_store_works_without_pread(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "_HAS_PREAD", False)
    p = tmp_path / "m.pmem"
    with Memory(p, HashEmbedder()) as m:
        m.add_many([f"line {i} token{i % 3}" for i in range(50)], times=[float(i) for i in range(50)])
        m.forget(3, t=100.0)
        m.checkpoint()
        m.add("tail record", t=200.0)
        assert m.get(7).text.startswith("line 7")
    with Memory(p, HashEmbedder()) as m2:
        assert len(m2) == 51 and m2.store.verify() == (m2.store.verify()[0], 0)
        assert m2.search("tail record", k=1, mode="lexical")[0].frame.text == "tail record"
        assert 3 in m2.store.deleted


def test_pwrite_handles_short_writes(tmp_path, monkeypatch):
    import os
    real = os.pwrite
    monkeypatch.setattr(os, "pwrite", lambda fd, data, off: real(fd, bytes(data)[:5], off))  # always short
    p = tmp_path / "x.bin"
    fd = os.open(p, os.O_RDWR | os.O_CREAT)
    store_mod._pwrite(fd, b"0123456789abcdef", 3)
    os.close(fd)
    assert p.read_bytes()[3:] == b"0123456789abcdef"


# ---- A3: embedder spec in header ---------------------------------------------
class _FakeEmbeddingServer(BaseHTTPRequestHandler):
    keys_seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).keys_seen.append(self.headers.get("Authorization"))
        vecs = HashEmbedder(64).embed(body["input"])
        out = json.dumps({"data": [{"index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def embed_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeEmbeddingServer)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1/embeddings"
    srv.shutdown()


def test_http_embedder_file_reopens_from_spec_without_storing_key(tmp_path, embed_server, capsys):
    f = tmp_path / "h.pmem"
    emb = HTTPEmbedder(embed_server, "fake-model", api_key="SECRET-KEY-123")
    with Memory(f, emb) as m:
        m.add_many(["alice lives in paris", "bob likes tea"])
    raw = f.read_bytes()
    assert b"SECRET-KEY-123" not in raw  # secrets never reach the file
    hdr = read_header(f)
    assert hdr["version"] == 3 and hdr["spec"] == f"http:{embed_server}|fake-model"
    # reopen with NO embedder argument: rebuilt from the header (key from env/flag)
    with Memory(f, read_only=True) as m2:
        assert m2.search("tea", k=1, mode="lexical")[0].frame.text == "bob likes tea"
        assert m2.embedder.dim == 64
    assert main([str(f), "--api-key", "K2", "search", "tea", "-k", "1"]) == 0
    assert "bob likes tea" in capsys.readouterr().out
    assert "Bearer K2" in _FakeEmbeddingServer.keys_seen


def test_from_spec_roundtrips():
    for e in (HashEmbedder(128),):
        assert from_spec(e.spec).name == e.name
    with pytest.raises(ValueError):
        from_spec("nonsense")


def test_embedder_string_arg_and_mismatch(tmp_path):
    p = tmp_path / "s.pmem"
    Memory(p, "hash-128").close()
    from playmemory.store import ModelMismatchError
    with pytest.raises(ModelMismatchError):
        Memory(p, "hash-256")
    assert Memory(p).embedder.dim == 128  # no arg: taken from the file


# ---- v2 files still open -------------------------------------------------------
def test_v2_file_opens_and_is_appendable(tmp_path):
    p = tmp_path / "v3.pmem"
    with Memory(p, HashEmbedder()) as m:
        m.add_many(["old record one", "old record two"], times=[1.0, 2.0])
    hdr = read_header(p)
    raw = p.read_bytes()
    nlen = len(hdr["name"].encode())
    from playmemory.store import _HDR
    pos0 = _HDR.size + nlen
    v3_ext = 2 + len(hdr["spec"].encode()) + 4
    v2 = struct.pack("<4sHHH", MAGIC, 2, hdr["dim"], nlen) + hdr["name"].encode() + raw[pos0 + v3_ext:]
    q = tmp_path / "v2.pmem"
    q.write_bytes(v2)
    with Memory(q, HashEmbedder()) as m2:
        assert m2.store.version == 2 and len(m2) == 2
        assert m2.search("record two", k=1)[0].frame.id == 1
        m2.add("new record three", t=3.0)
    with Memory(q, HashEmbedder()) as m3:
        assert len(m3) == 3 and m3.store.verify()[1] == 0
        m3.compact()  # migration path: rewritten as current version
    assert read_header(q)["version"] == 3
