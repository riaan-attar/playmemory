import io
import json
import sys

from playmemory.cli import main


def run(capsys, *argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main(list(argv))
    out = capsys.readouterr().out.strip().splitlines()
    return code, [json.loads(x) for x in out]


def test_cli_roundtrip(tmp_path, capsys, monkeypatch):
    f = str(tmp_path / "m.pmem")
    code, out = run(capsys, f, "add", stdin="alice lives in paris\nbob likes tea\n", monkeypatch=monkeypatch)
    assert out == [{"added": [0, 1]}]
    code, out = run(capsys, f, "search", "tea", "-k", "1")
    assert out[0]["text"] == "bob likes tea"
    code, out = run(capsys, f, "search", "tea", "--mode", "lexical", "--context", "1")
    assert out[0]["before"][0]["text"] == "alice lives in paris"
    code, out = run(capsys, f, "forget", "1")
    code, out = run(capsys, f, "search", "tea", "--mode", "lexical")
    assert out == []
    code, out = run(capsys, f, "stats")
    assert out[0]["deleted"] == 1 and out[0]["frames"] == 2
    code, out = run(capsys, f, "verify")
    assert code == 0 and out[0]["ok"]
    code, out = run(capsys, f, "export")
    assert [o["deleted_at"] is not None for o in out] == [False, True]
    code, out = run(capsys, f, "compact")
    assert out[0]["frames"] == 1
    code, out = run(capsys, f, "seek", "1e18")
    assert out[0]["text"] == "alice lives in paris"


def test_cli_verify_detects_corruption(tmp_path, capsys, monkeypatch):
    f = tmp_path / "m.pmem"
    run(capsys, str(f), "add", "one", "two")
    raw = bytearray(f.read_bytes())
    raw[-2] ^= 0xFF
    f.write_bytes(bytes(raw))
    code, out = run(capsys, str(f), "verify")
    assert code == 2 and not out[0]["ok"]


def test_cli_story_and_read(tmp_path, capsys, monkeypatch):
    f = str(tmp_path / "m.pmem")
    lines = [f"the detective inspected knife {i} at the mansion" for i in range(8)] + \
            [f"mix flour sugar butter step {i} for the cake" for i in range(8)]
    run(capsys, f, "add", stdin="\n".join(lines) + "\n", monkeypatch=monkeypatch)
    code, out = run(capsys, f, "story", "detective knife mansion", "--budget", "100000")
    sp = out[0]
    assert sp["frames"][0]["text"].startswith("the detective") and sp["scene_frames"] >= len(sp["frames"])
    code, out = run(capsys, f, "story", "detective knife mansion", "--pointers")
    assert "frames" not in out[0]
    code, out = run(capsys, f, "read", str(sp["start_id"]), str(sp["end_id"]))
    assert [o["id"] for o in out] == [x["id"] for x in sp["frames"]]
