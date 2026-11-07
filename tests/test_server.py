import json
import os
import subprocess
import sys

from playmemory import Memory
from playmemory.server import MemoryServer


def rpc(srv, method, params=None, mid=1):
    return srv.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def call(srv, name, **args):
    r = rpc(srv, "tools/call", {"name": name, "arguments": args})["result"]
    return json.loads(r["content"][0]["text"]), r.get("isError", False)


def test_handshake_and_tools():
    srv = MemoryServer(Memory())
    assert rpc(srv, "initialize")["result"]["serverInfo"]["name"] == "playmemory"
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {t["name"] for t in rpc(srv, "tools/list")["result"]["tools"]}
    assert {"remember", "recall", "forget", "seek", "timeline", "profile", "stats"} <= names
    assert rpc(srv, "nope")["error"]["code"] == -32601


def test_remember_recall_profile_forget():
    srv = MemoryServer(Memory())
    out, err = call(srv, "remember", text="I live in Paris.")
    assert not err and out["cards"][0]["meta"]["value"] == "Paris"
    call(srv, "remember", text="I moved to Berlin.")
    prof, _ = call(srv, "profile")
    assert prof["lives_in"] == ["Berlin"]
    hits, _ = call(srv, "recall", query="where does the user live", k=1, where={"kind": "card"}, context=1)
    assert hits[0]["frame"]["meta"]["value"] == "Berlin"
    fid = hits[0]["frame"]["id"]
    assert call(srv, "forget", id=fid)[0] == {"forgot": fid}
    stats, _ = call(srv, "stats")
    assert stats["deleted"] >= 2


def test_tool_errors_reported_in_band():
    srv = MemoryServer(Memory())
    r = rpc(srv, "tools/call", {"name": "bogus", "arguments": {}})["result"]
    assert r["isError"] and "unknown tool" in r["content"][0]["text"]
    r = rpc(srv, "tools/call", {"name": "forget", "arguments": {"id": 99}})["result"]
    assert r["isError"]


def test_stdio_subprocess(tmp_path):
    f = tmp_path / "m.pmem"
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "remember", "arguments": {"text": "The deploy key rotates on Fridays"}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "recall", "arguments": {"query": "when does the key rotate", "k": 1}}},
    ]
    p = subprocess.run([sys.executable, "-m", "playmemory.cli", str(f), "serve"],
                       input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": "src"}, timeout=60)
    lines = [json.loads(x) for x in p.stdout.splitlines()]
    assert [l["id"] for l in lines] == [1, 2, 3]
    hit = json.loads(lines[2]["result"]["content"][0]["text"])[0]
    assert "Fridays" in hit["frame"]["text"]
    with Memory(f, read_only=True) as m:  # persisted by the server process
        assert len(m) >= 1
